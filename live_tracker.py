#!/usr/bin/env python3
"""
Live tower tracker: frames and MARK taps in -> which blocks are present, updated per move.

  START      -> reset; the first `ref_seconds` of frames are the reference scan (tower intact, hands out)
  every frame-> depth offset + colour of each visible block face (read_ends / rgb_samples)
  MARK tap   -> read the tower at that moment (hand-free frames only), diff against the last
                state, update the set of blocks that are present

Needs a still phone (pose is fixed from the reference window).  Pure logic: no server, no UI.
"""
import collections
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "third_party", "STEPWISE"))
sys.path.insert(0, HERE)

import game_events as ge
import rgb_cue
import tower_points as tp
from thirdeye.perception.tower import DepthView

LAYOUT = tp.LAYOUTS["A"]


class LiveTracker:
    def __init__(self, layers=10, ref_seconds=5.0, min_conf=2, settle=0.6, give_up=4.0):
        self.ref_seconds, self.min_conf = ref_seconds, min_conf
        self.settle, self.give_up = settle, give_up
        self.reset(layers)

    # ---- state -------------------------------------------------------------------------
    def reset(self, layers=None):
        if layers is not None:
            self.layers = int(layers)
        self.t0 = None
        self.pose = None                  # (R, t, reproj)
        self.pending_ref = []             # raw frames of the reference window
        self.records = []                 # (ts_ms, t_sec, ends, R, t, K_rgb, smp)
        self.an = None
        self.ref_n = 0
        self.prev = {}
        self.present = {(l, s) for l in range(self.layers) for s in range(3)}
        self.pending_marks = []           # (t_sec, wall_arrival)
        self.moves = []
        self.collapsed = False
        self.msg = "waiting for the first frames"

    def tower(self):
        top = max([l for l, _ in self.present], default=-1)
        layers = max(self.layers, top + 1)
        removed = [[l, s] for l in range(layers) for s in range(3) if (l, s) not in self.present]
        return {"layers": layers, "removed": removed}

    @property
    def ready(self):
        return self.an is not None

    # ---- frames ------------------------------------------------------------------------
    def _prep(self, fr):
        depth = fr["depth"].astype(np.float64) * 1000.0
        if fr["conf"] is not None:
            depth[fr["conf"] < self.min_conf] = 0.0
        h, w = depth.shape
        Kd = np.diag([w / fr["rgb"].shape[1], h / fr["rgb"].shape[0], 1.0]) @ fr["K"]
        return depth, Kd

    def _read(self, depth, Kd, K_rgb, grey):
        R, t, _ = self.pose
        view = DepthView(depth, Kd, R, t)
        return ge.read_ends(view), rgb_cue.rgb_samples(grey, K_rgb, R, t, view)

    def add_frame(self, ts_ms, fr, wall=None):
        if fr is None or fr.get("conf") is None:
            return
        if self.t0 is None:
            self.t0 = ts_ms
        tt = (ts_ms - self.t0) / 1000.0
        grey = rgb_cue.grey_blur(fr["rgb"])
        depth, Kd = self._prep(fr)

        if self.an is None:                       # still collecting the reference
            sol = tp.solve(tp.detect(fr["rgb"]), fr["K"], layout=LAYOUT)
            if sol and sol[2] <= 7.0 and (self.pose is None or sol[2] < self.pose[2]):
                self.pose = sol
            self.pending_ref.append((ts_ms, tt, depth, Kd, fr["K"], grey))
            self.msg = f"reference scan {min(tt, self.ref_seconds):.0f}/{self.ref_seconds:.0f} s - hands out"
            if tt >= self.ref_seconds:
                self._finalize_reference()
            return

        sol = tp.solve(tp.detect(fr["rgb"]), fr["K"], layout=LAYOUT)     # follow small camera shifts
        if sol and sol[2] <= 7.0:
            self.pose = sol
        ends, smp = self._read(depth, Kd, fr["K"], grey)
        R, t, _ = self.pose
        self.records.append((ts_ms, tt, ends, R, t, fr["K"], smp))
        if len(self.records) > 1500:
            del self.records[: len(self.records) - 1200]
            self.ref_n = 0                         # reference lives in self.an; indices are relative now

    def _finalize_reference(self):
        if self.pose is None:
            self.msg = "reference window passed but no markers were found - check the camera view"
            self.pending_ref.clear()
            return
        recs = []
        for ts_ms, tt, depth, Kd, Krgb, grey in self.pending_ref:
            ends, smp = self._read(depth, Kd, Krgb, grey)
            recs.append((ts_ms, tt, ends, *self.pose[:2], Krgb, smp))
        self.pending_ref.clear()
        self.an = ge.Analyzer([r[2] for r in recs], ref_rgb=[r[6] for r in recs], layers=self.layers)
        self.prev = dict(self.an.current)
        self.records = recs
        self.ref_n = len(recs)
        n_seen = sum(1 for v in self.an.is_block.values() if v)
        self.msg = f"ready - {n_seen} block faces calibrated. Press MARK after each move."

    # ---- marks -------------------------------------------------------------------------
    def mark(self, t_ms, wall):
        if self.t0 is None:
            return
        self.pending_marks.append(((t_ms - self.t0) / 1000.0, wall))

    def poll(self, wall):
        """Evaluate marks whose 'after' frames have arrived.  Returns a list of result dicts."""
        out = []
        if not self.ready or not self.pending_marks or not self.records:
            return out
        newest = self.records[-1][1]
        keep = []
        for tm, arrived in self.pending_marks:
            if newest < tm + self.settle and wall - arrived < self.give_up:
                keep.append((tm, arrived))
                continue
            out.append(self._evaluate(tm))
        self.pending_marks = keep
        return out

    def _evaluate(self, tm):
        occ, idx, n = ge.state_at(self.an, self.records, tm, self.ref_n)
        if idx is None:
            self.msg = "could not read that move (hand in view). Move the hand away and press MARK again"
            return {"ok": False, "t": tm, "msg": self.msg, "removed": [], "added": []}
        removed = [(sl.layer, sl.slot) for sl, v in occ.items() if self.prev.get(sl, 0) == 1 and v == 0]
        added = [(sl.layer, sl.slot) for sl, v in occ.items() if self.prev.get(sl, 0) == 0 and v == 1]
        if len(removed) >= 5:      # one move changes one or two blocks; five at once is a collapse
            self.collapsed = True
            self.msg = f"TOWER COLLAPSED ({len(removed)} blocks gone at once). Rebuild it, then press START again"
            return {"ok": True, "collapsed": True, "t": tm, "msg": self.msg, "removed": [], "added": []}
        self.prev.update(occ)
        for rc in removed:
            self.present.discard(rc)
        for rc in added:
            self.present.add(rc)
        self.moves.append({"t": tm, "removed": removed, "added": added, "frames": n})
        fmt = lambda L: ", ".join(f"L{l + 1}S{s}" for l, s in sorted(L)) or "-"
        self.msg = f"move {len(self.moves)}: removed {fmt(removed)} | added {fmt(added)}"
        return {"ok": True, "t": tm, "msg": self.msg, "removed": removed, "added": added, "frames": n}
