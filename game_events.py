#!/usr/bin/env python3
"""
Find and label Jenga moves in a recording, with no hand-kept log.

  reference scan (first N s, intact tower) -> per-slot expected end-face depth
  every frame -> their read_view() -> offset vs reference
  settled + hand-free + unchanged for K frames -> a "stable state"
  stable state changes -> REMOVE / ADD events; a block stuck partway -> PUSH
  every event -> before|after RGB image with the slot's box drawn on it, for review

Assumes the phone is fixed for a game (pose taken once from the reference
frames).  Use --pose frame for a handheld capture.

Run isolated:  .venv/bin/python -I game_events.py [--out review]
"""
import argparse, collections, glob, json, os, sys
from dataclasses import dataclass
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "third_party", "STEPWISE"))
sys.path.insert(0, HERE)

from thirdeye.perception.tower import STANDARD, DepthView, EndReading, OCCLUDED_MM
from thirdeye.events import Slot

TOL = 6.0        # mm: |offset - reference| inside this is "the block is where it was"
ALONE = 12.0     # mm: recess beyond this vs reference = hole.  Their uncalibrated value is 20;
                 #     a per-slot reference removes the tower's own jitter, so a lower bar is safe.
BLOCK_ABS = 8.0  # mm: a block now sitting where the reference saw air
T_RGB = 20.0     # grey levels: face looks different from the reference.  On an intact tower the
                 #   body scores median 1-2, p90 < 4, none above 35 (room behind it is noisier).
HAND_CHANGED = 5 # that many faces changed at once is a hand or a shadow, not one block moving


def read_ends(view, geom=STANDARD, min_valid=0.5, pct=75.0):
    """Their read_view(), but with one fixed percentile.  Theirs leans towards the
    camera-side tail and switches tail with the sign of the median, so a flush
    face (median ~ 0) flips between two values; against a per-slot reference that
    reads as a 7 mm 'push'.  A fixed percentile is shift-equivariant."""
    h, w = view.depth_mm.shape
    out = {}
    for layer in range(geom.layers):
        for slot in range(geom.slots):
            for face, (centre, normal) in geom.ends(layer, slot).items():
                if not view.faces(normal, centre):
                    continue
                uv, expected = view.project(geom.samples(layer, slot, face))
                u = np.round(uv[:, 0]).astype(int)
                v = np.round(uv[:, 1]).astype(int)
                if not ((u >= 0) & (u < w) & (v >= 0) & (v < h) & (expected > 0)).all():
                    out[Slot(layer, slot)] = EndReading("unseen")
                    continue
                measured = view.depth_mm[v, u]
                ok = np.isfinite(measured) & (measured > 0)
                if view.mask is not None:
                    ok &= ~view.mask[v, u].astype(bool)
                if ok.mean() < min_valid:
                    # LiDAR dropout / low confidence -- NOT something in front of the tower
                    out[Slot(layer, slot)] = EndReading("invalid")
                    continue
                delta = measured[ok] - expected[ok]
                if np.mean(delta < -OCCLUDED_MM) > 0.5:
                    out[Slot(layer, slot)] = EndReading("occluded")
                    continue
                out[Slot(layer, slot)] = EndReading("seen", float(np.percentile(delta, pct)))
    return out


@dataclass
class Event:
    kind: str          # REMOVE | ADD | PUSH
    layer: int         # 0 = bottom
    slot: int
    t: float
    before: int        # frame index of the last stable frame before the change
    after: int         # frame index where the new state was confirmed
    note: str = ""


class Analyzer:
    def __init__(self, ref_ends, stable=3, hand_frac=0.06, ref_rgb=None, layers=None):
        votes = collections.defaultdict(list)
        for ends in ref_ends:
            for slot, e in ends.items():
                if e.state == "seen":
                    votes[slot].append(e.offset_mm)
        need = max(1, len(ref_ends) // 2)
        raw = {s: float(np.median(v)) for s, v in votes.items() if len(v) >= need}
        near = [o for o in raw.values() if abs(o) <= 15]
        self.bias = float(np.median(near)) if near else 0.0
        # a block end sits at the face; clearly further is air.  The first empty layer above the
        # tower looks only a few mm recessed from a corner (the layer below shows through), so
        # "not near the face" alone is not enough -- the in-between slots are left out.
        self.is_block, self.ref = {}, {}
        for s, o in raw.items():
            if abs(o - self.bias) <= 12:
                self.is_block[s], self.ref[s] = True, o
            elif o - self.bias > 25:
                self.is_block[s], self.ref[s] = False, o
        self.current = {s: int(b) for s, b in self.is_block.items()}
        votes_rgb = collections.defaultdict(list)
        for smp in (ref_rgb or []):
            for sl, v in smp.items():
                votes_rgb[sl].append(v)
        self.rgb_ref = {sl: np.median(np.array(v), axis=0) for sl, v in votes_rgb.items()}
        # Faces the LiDAR could not read in the reference (steep angle -> low confidence) are still
        # perfectly visible in colour: track them by appearance alone.
        if layers is not None:
            for sl in self.rgb_ref:
                if sl not in self.is_block and sl.layer < layers:
                    self.is_block[sl], self.ref[sl], self.current[sl] = True, None, 1
        self.stable, self.hand_frac = stable, hand_frac
        self._cand, self._held = None, 0
        self._prev_idx = len(ref_ends) - 1
        self._mv = collections.Counter()
        self._pushed = set()

    def classify(self, ends, smp=None):
        """Fuse the depth offset (geometry: recess / push) with the colour change of the same
        face (appearance).  Depth drops out at range; colour rescues those slots."""
        states, rel, occl = {}, {}, 0
        rgb = {}
        for sl, v in (smp or {}).items():
            rr = self.rgb_ref.get(sl)
            if rr is not None and self.is_block.get(sl):
                rgb[sl] = float(np.median(np.abs(v - rr)))
        for slot in set(ends) | set(rgb):
            if slot not in self.ref:
                continue
            e = ends.get(slot)
            if e is not None and e.state == "occluded":
                occl += 1
                states[slot] = "occluded"
                continue
            seen = e is not None and e.state == "seen"
            if not self.is_block[slot]:
                if not seen:
                    states[slot] = "unknown"
                    continue
                o = e.offset_mm
                rel[slot] = o - self.bias
                # a corner camera sees a neighbour's side in front of an empty slot, so
                # anything between "block here" and "far air" is simply not a verdict
                states[slot] = ("block" if abs(o - self.bias) <= BLOCK_ABS else
                                "air" if o - self.bias > 25 else "unknown")
                continue
            r = rgb.get(slot)
            changed = r is not None and r > T_RGB
            d = (e.offset_mm - self.ref[slot]) if (seen and self.ref[slot] is not None) else None
            if d is not None:
                rel[slot] = d
            if d is not None and d < -TOL:
                states[slot] = "moving"
            elif (d is not None and d >= ALONE) or (changed and (d is None or d >= TOL)):
                states[slot] = "empty"
            elif changed:
                states[slot] = "unknown"      # depth says in place, colour says changed: a shadow
            elif (d is not None and abs(d) <= TOL) or (d is None and r is not None):
                states[slot] = "block"
            else:
                states[slot] = "unknown"
        # a hand is "something nearer than the tower" in the depth reading.  (Counting colour changes
        # against the original reference would fire forever once the tower has genuinely changed.)
        hand = occl > self.hand_frac * max(len(self.ref), 1)
        return states, rel, hand

    def step(self, idx, t, ends, smp=None):
        states, rel, hand = self.classify(ends, smp)
        if hand:
            self._cand, self._held = None, 0
            self._mv.clear()
            return [], "hand"
        events = []
        moving = [s for s, st in states.items() if st == "moving"]
        for s in list(self._mv):
            if s not in moving:
                del self._mv[s]
        for s in moving:
            self._mv[s] += 1
            if self._mv[s] == self.stable and s not in self._pushed:
                self._pushed.add(s)
                events.append(Event("PUSH", s.layer, s.slot, t, self._prev_idx, idx,
                                    f"{rel.get(s, 0):+.0f} mm vs reference"))
        if moving:
            self._cand, self._held = None, 0
            return events, "moving"
        self._pushed.clear()
        cand = dict(self.current)
        for s, st in states.items():
            if st == "block":
                cand[s] = 1
            elif st in ("empty", "air"):
                cand[s] = 0
        if cand == self._cand:
            self._held += 1
        else:
            self._cand, self._held = cand, 1
        if self._held == self.stable and cand != self.current:
            for s in sorted(cand, key=lambda x: (x.layer, x.slot)):
                if cand[s] != self.current.get(s, 0):
                    events.append(Event("ADD" if cand[s] else "REMOVE", s.layer, s.slot, t,
                                        self._prev_idx, idx,
                                        f"{rel.get(s, 0):+.0f} mm vs reference"))
            self.current = cand
        if self._held >= self.stable and cand == self.current:
            self._prev_idx = idx
        return events, "ok"


def state_at(an, records, tm, ref_n, before=1.2, after=0.4):
    """Occupancy around a MARK tap: majority over the hand-free frames in a short window."""
    votes, used = collections.defaultdict(list), []
    for idx in range(ref_n, len(records)):
        tt = records[idx][1]
        if tt < tm - before:
            continue
        if tt > tm + after:
            break
        states, _rel, hand = an.classify(records[idx][2], records[idx][6])
        if hand:
            continue
        used.append(idx)
        for sl, st in states.items():
            if st == "block":
                votes[sl].append(1)
            elif st in ("empty", "air"):
                votes[sl].append(0)
    occ = {sl: int(np.mean(v) >= 0.5) for sl, v in votes.items() if len(v) >= 2}
    return occ, (used[len(used) // 2] if used else None), len(used)


def analyze_marks(an, records, marks, ref_n):
    prev, prev_idx, events = dict(an.current), ref_n - 1, []
    print(f"\n{len(marks)} MARK taps -> tower state at each tap, diffed against the previous tap")
    for i, tm in enumerate(marks, 1):
        occ, idx, n = state_at(an, records, tm, ref_n)
        if idx is None:
            print(f"  move {i:2d} t={tm:6.1f}s  no hand-free frames within 1.6 s of the tap -- skipped")
            continue
        removed = sorted((sl for sl, v in occ.items() if prev.get(sl, 0) == 1 and v == 0),
                         key=lambda x: (x.layer, x.slot))
        added = sorted((sl for sl, v in occ.items() if prev.get(sl, 0) == 0 and v == 1),
                       key=lambda x: (x.layer, x.slot))
        fmt = lambda L: ", ".join(f"L{x.layer + 1}/s{x.slot}" for x in L) or "-"
        flag = "" if (len(removed) <= 1 and len(added) <= 1) else "   <-- more than one block changed: check images"
        print(f"  move {i:2d} t={tm:6.1f}s ({n} frames)  removed: {fmt(removed):12s} added: {fmt(added)}{flag}")
        for sl in removed:
            events.append(Event("REMOVE", sl.layer, sl.slot, tm, prev_idx, idx, f"move {i}"))
        for sl in added:
            events.append(Event("ADD", sl.layer, sl.slot, tm, prev_idx, idx, f"move {i}"))
        prev.update(occ)
        prev_idx = idx
    return events


FLIP = np.diag([1.0, -1.0, -1.0, 1.0])    # OpenCV camera axes <-> ARKit camera axes


def hybrid_poses(stamps, layout, tp, max_err=7.0, min_markers=3):
    """Tower-in-camera pose for every frame.  Frames that see enough markers get a marker
    solve; the rest are carried by ARKit's own tracked pose from the nearest marker frame:
        world<-tower = P_arkit . FLIP . C_marker        (from an anchor frame)
        C_j          = FLIP . inv(P_j) . (world<-tower)  (any other frame)
    ARKit is only trusted over the seconds between marker sightings."""
    anchors, direct, P = [], {}, {}
    for i, ts in enumerate(stamps):
        meta = json.load(open(f"{tp.SAVE_DIR}/{ts}_intrinsics.json"))
        Pj = np.array(meta["pose_column_major"], np.float64).reshape(4, 4, order="F")
        Pj[:3, 3] *= 1000.0                    # ARKit metres -> tower millimetres
        P[ts] = Pj
        fr = tp.load_frame(ts)
        if fr is None:
            continue
        dets = tp.detect(fr["rgb"])
        sol = tp.solve(dets, fr["K"], layout=layout) if len(dets) >= 2 else None
        if sol and sol[2] <= 7.0:
            direct[ts] = (sol[0], sol[1])
            if len(dets) >= min_markers and sol[2] <= max_err:
                C = np.eye(4); C[:3, :3] = sol[0]; C[:3, 3] = sol[1]
                anchors.append((i, Pj @ FLIP @ C))
    out, chained = {}, 0
    for i, ts in enumerate(stamps):
        if ts in direct:
            out[ts] = direct[ts]
            continue
        if not anchors:
            continue
        a_i, W = min(anchors, key=lambda a: abs(a[0] - i))
        if abs(a_i - i) > 80:                  # ~25 s at 3 fps: too far to trust ARKit
            continue
        C = FLIP @ np.linalg.inv(P[ts]) @ W
        out[ts] = (C[:3, :3].copy(), C[:3, 3].copy())
        chained += 1
    print(f"poses: {len(direct)} from markers, {chained} carried by ARKit, "
          f"{len(stamps) - len(out)} unavailable  ({len(anchors)} anchor frames)")
    return out


def pair_moves(events):
    """A move is a REMOVE followed by the ADD that places the block on top."""
    moves, pending = [], None
    for e in events:
        if e.kind == "REMOVE":
            pending = e
        elif e.kind == "ADD" and pending is not None:
            moves.append((pending, e))
            pending = None
    return moves, pending


# --- files -------------------------------------------------------------------------------


def project(K, R, t, pts):
    cam = pts @ R.T + t
    uvw = cam @ K.T
    return uvw[:, :2] / uvw[:, 2:3]


def draw_review(path_before, path_after, ev, K, R, t, out_path, width=1500):
    import cv2
    box_lo, box_hi = STANDARD.box(ev.layer, ev.slot)
    corners = np.array([[x, y, z] for x in (box_lo[0], box_hi[0])
                        for y in (box_lo[1], box_hi[1]) for z in (box_lo[2], box_hi[2])])
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6),
             (5, 7), (6, 7)]
    panels = []
    for p, label in ((path_before, "BEFORE"), (path_after, "AFTER")):
        img = cv2.imread(p)
        uv = project(K, R, t, corners)
        col = (0, 0, 255) if ev.kind == "REMOVE" else (0, 200, 0) if ev.kind == "ADD" else (0, 200, 255)
        for a, b in edges:
            cv2.line(img, tuple(int(v) for v in uv[a]), tuple(int(v) for v in uv[b]), col, 3)
        cv2.putText(img, label, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 6)
        cv2.putText(img, label, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 3)
        panels.append(img)
    both = np.hstack(panels)
    s = width / both.shape[1]
    both = cv2.resize(both, None, fx=s, fy=s)
    cv2.putText(both, f"{ev.kind} layer {ev.layer + 1} (from bottom) slot {ev.slot}  t={ev.t:.1f}s  {ev.note}",
                (10, both.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(both, f"{ev.kind} layer {ev.layer + 1} (from bottom) slot {ev.slot}  t={ev.t:.1f}s  {ev.note}",
                (10, both.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    cv2.imwrite(out_path, both)


def main():
    import tower_points as tp
    import rgb_cue
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-seconds", type=float, default=3.0)
    ap.add_argument("--stable", type=int, default=3, help="consecutive identical frames")
    ap.add_argument("--min-conf", type=int, default=2)
    ap.add_argument("--layers", type=int, default=10, help="layers in the tower")
    ap.add_argument("--absolute", action="store_true",
                    help="no per-slot reference: judge faces against the ideal tower (moving camera)")
    ap.add_argument("--pose", choices=["fixed", "frame", "hybrid"], default="hybrid")
    ap.add_argument("--out", default=os.path.join(HERE, "review"))
    args = ap.parse_args()

    layout = tp.LAYOUTS["A"]
    stamps = sorted(os.path.basename(f).split("_")[0]
                    for f in glob.glob(f"{tp.SAVE_DIR}/*_intrinsics.json"))
    if not stamps:
        raise SystemExit("no frames")
    t0 = int(stamps[0])
    times = [(int(s) - t0) / 1000.0 for s in stamps]
    print(f"{len(stamps)} frames over {times[-1]:.1f} s")

    hybrid = hybrid_poses(stamps, layout, tp) if args.pose == "hybrid" else None
    # fixed pose from the reference window
    fixed = None
    if args.pose == "fixed":
        best = None
        for ts, tt in zip(stamps, times):
            if tt > args.ref_seconds:
                break
            fr = tp.load_frame(ts)
            if fr is None:
                continue
            s = tp.solve(tp.detect(fr["rgb"]), fr["K"], layout=layout)
            if s and (best is None or s[2] < best[2]):
                best = s
        if best is None:
            raise SystemExit("could not find the markers in the reference frames")
        fixed = best
        print(f"fixed pose from reference window: reproj {best[2]:.1f}px "
              f"camera at {np.round(-best[0].T @ best[1] / 10, 1)} cm (tower frame)")

    records = []   # (ts, t, ends, hand/moving, R, t, K_rgb)
    drift = 0
    for i, (ts, tt) in enumerate(zip(stamps, times)):
        fr = tp.load_frame(ts)
        if fr is None or fr["conf"] is None:
            continue
        if args.pose == "hybrid":
            if ts not in hybrid:
                continue
            R, tv = hybrid[ts]
        elif args.pose == "frame":
            s = tp.solve(tp.detect(fr["rgb"]), fr["K"], layout=layout)
            if not s or s[2] > 6:
                continue
            R, tv, _ = s
        else:
            R, tv, _ = fixed
            if i % 8 == 0:
                s = tp.solve(tp.detect(fr["rgb"]), fr["K"], layout=layout)
                if s and s[2] < 6:
                    moved = np.linalg.norm((-s[0].T @ s[1]) - (-R.T @ tv))
                    drift += moved > 8.0
        depth = fr["depth"].astype(np.float64) * 1000.0
        depth[fr["conf"] < args.min_conf] = 0.0
        h, w = depth.shape
        K = np.diag([w / fr["rgb"].shape[1], h / fr["rgb"].shape[0], 1.0]) @ fr["K"]
        view = DepthView(depth, K, R, tv)
        ends = read_ends(view)
        smp = rgb_cue.rgb_samples(rgb_cue.grey_blur(fr["rgb"]), fr["K"], R, tv, view)
        records.append((ts, tt, ends, R, tv, fr["K"], smp))
    print(f"{len(records)} frames analysed" + (f"; {drift} drift checks saw the phone move >8 mm"
                                              if drift else ""))

    ref_n = sum(1 for r in records if r[1] <= args.ref_seconds)
    if ref_n < 2:
        raise SystemExit("not enough reference frames")
    an = Analyzer([r[2] for r in records[:ref_n]], stable=args.stable,
                  ref_rgb=[r[6] for r in records[:ref_n]], layers=args.layers)
    if args.absolute:
        for sl, isb in an.is_block.items():
            if isb:
                an.ref[sl] = an.bias
        an.rgb_ref = {}
    print(f"reference: {ref_n} frames, {len(an.ref)} slot ends, "
          f"{sum(1 for v in an.current.values() if v)} blocks, bias {an.bias:+.1f} mm")

    mark_files = sorted(glob.glob(os.path.join(tp.SAVE_DIR, "*_mark.json")))
    marks = []
    for f in mark_files:
        try:
            marks.append((json.load(open(f))["t_ms"] - t0) / 1000.0)
        except Exception:
            pass
    marks.sort()
    if marks:
        events = analyze_marks(an, records, marks, ref_n)
        status = collections.Counter()
    else:
      events, status = [], collections.Counter()
      for idx in range(ref_n, len(records)):
        ts, tt, ends, _R, _t, _K, smp = records[idx]
        ev, st = an.step(idx, tt, ends, smp)
        status[st] += 1
        events += ev
    if not marks:
        print(f"frame status: {dict(status)}")

    os.makedirs(args.out, exist_ok=True)
    for f in glob.glob(os.path.join(args.out, "event_*.png")):
        os.remove(f)
    print(f"\n{len(events)} events")
    for n, e in enumerate(events, 1):
        print(f"  #{n:2d}  t={e.t:6.1f}s  {e.kind:6s} layer {e.layer + 1:2d} slot {e.slot}   {e.note}")
        if e.before < len(records) and e.after < len(records):
            _, _, _, R, tv, K, _ = records[e.after]
            draw_review(os.path.join(tp.SAVE_DIR, f"{records[e.before][0]}_rgb.jpg"),
                        os.path.join(tp.SAVE_DIR, f"{records[e.after][0]}_rgb.jpg"),
                        e, K, R, tv, os.path.join(args.out, f"event_{n:02d}.png"))
    moves, dangling = pair_moves(events)
    if moves:
        print("\nMOVES (pull -> place):")
        for n, (a, b) in enumerate(moves, 1):
            print(f"  {n}. removed layer {a.layer + 1} slot {a.slot}  ->  placed layer {b.layer + 1} "
                  f"slot {b.slot}   ({a.t:.1f}s -> {b.t:.1f}s)")
    if dangling:
        print(f"  (block pulled from layer {dangling.layer + 1} slot {dangling.slot} "
              f"at {dangling.t:.1f}s, not yet placed)")
    print(f"\nreview images: {args.out}/event_NN.png")


if __name__ == "__main__":
    main()
