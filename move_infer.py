#!/usr/bin/env python3
"""
Which layer was the block pulled from?  The player says left / middle / right; the faces say the rest.

At a MARK tap the move is finished and the hand is away.  We compare the tower just BEFORE the move (a few
seconds after the previous tap) with just AFTER it (a few seconds after this tap):

   for every visible block face, register the "after" picture to the "before" picture (phase correlation, so a
   tower that merely shifted a millimetre does not count), and measure what is still different, lighting-
   normalised.  Faces whose change stands out from the median change of ALL faces are the ones that really
   changed.  Of the 9-10 blocks in the typed column (one per layer), the one that changed most is the pulled one.

Pure cv2 + numpy, so it runs inside the viewer server.  Needs the face-crop dataset (data capture ON).
"""
import collections
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import face_crops as fc

_HANN = {}


def _regions(h, w):
    """registration region (the face plus a little context) and the face core, as fractions of the crop"""
    return (int(.10 * h), int(.90 * h), int(.10 * w), int(.90 * w)), (int(.217 * h), int(.783 * h), int(.20 * w), int(.80 * w))
MIN_COS = 0.30


def _norm(g):
    g = g.astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-3)


def slots_for_positions(R, t, K, layers, rot=0):
    """{layer: {'left': slot, 'middle': slot, 'right': slot}} as seen in the iPhone picture AS THE PLAYER SEES IT:
    `rot` is how far the viewer has turned that picture (0, 90, 180 or 270 degrees clockwise), so "left" is left on
    the player's screen, not left in the phone's own frame."""
    cam = -R.T @ t
    a = np.radians(rot)
    ca, sa = float(np.cos(a)), float(np.sin(a))
    out = {}
    for layer in range(layers):
        xs = []
        for slot in range(3):
            best = None
            for sign in (1, -1):
                _c, c, n = fc.face_geometry(layer, slot, sign)
                to = cam - c
                cos = float(n @ to) / (np.linalg.norm(to) + 1e-9)
                if best is None or cos > best[0]:
                    best = (cos, c)
            uvw = (best[1] @ R.T + t) @ K.T
            du, dv = uvw[0] / uvw[2] - K[0, 2], uvw[1] / uvw[2] - K[1, 2]
            xs.append(du * ca - dv * sa)                   # horizontal position on the turned picture
        order = np.argsort(xs)
        out[layer] = {"left": int(order[0]), "middle": int(order[1]), "right": int(order[2])}
    return out


def _rows(run_dir):
    rows = []
    with open(os.path.join(run_dir, "index.jsonl")) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r["cos"] >= MIN_COS:
                rows.append(r)
    return rows


def _window_means(run_dir, rows, lo_ms, hi_ms):
    """{(cam, layer, slot, sign): mean normalised grey crop} over a time window, needing >= 2 crops."""
    acc = collections.defaultdict(list)
    for r in rows:
        if lo_ms <= r["ts_ms"] <= hi_ms:
            acc[(r["cam"], r["layer"], r["slot"], r["sign"], r.get("kind", "end"))].append(r["file"])
    out = {}
    for k, files in acc.items():
        if len(files) < 2:
            continue
        ims = []
        for fn in files[:12]:
            im = cv2.imread(os.path.join(run_dir, fn))
            if im is not None:
                ims.append(_norm(cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (3, 3), 0)))
        if len(ims) >= 2:
            out[k] = np.mean(ims, axis=0)
    return out


def _change(a, b):
    """(residual after aligning b onto a, slide in px)"""
    h, w = a.shape
    (fy0, fy1, fx0, fx1), (cy0, cy1, cx0, cx1) = _regions(h, w)
    if (h, w) not in _HANN:
        _HANN[(h, w)] = cv2.createHanningWindow((fx1 - fx0, fy1 - fy0), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a[fy0:fy1, fx0:fx1], b[fy0:fy1, fx0:fx1], _HANN[(h, w)])
    dx, dy = float(np.clip(dx, -0.12 * w, 0.12 * w)), float(np.clip(dy, -0.13 * h, 0.13 * h))
    al = cv2.warpAffine(b, np.float32([[1, 0, -dx], [0, 1, -dy]]), (w, h), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)
    return float(np.abs(a[cy0:cy1, cx0:cx1] - al[cy0:cy1, cx0:cx1]).mean()), (dx, dy)


def block_scores(run_dir, mark_ms, prev_mark_ms, start_ms, settle=0.8, after=4.5, use_sides=True, a_window=None, b_window=None):
    """{(layer, slot): z} for EVERY block seen both before and after the tap: how strongly its picture changed.
    Same measure as infer() uses (best face per camera, averaged over the cameras that saw it)."""
    rows = _rows(run_dir)
    a_lo = (prev_mark_ms + 1000) if prev_mark_ms else (start_ms + 500)
    a_hi = min(a_lo + 3500, mark_ms - 2000)
    if a_window is not None:
        a_lo, a_hi = a_window
    elif a_hi - a_lo < 1500:
        return None
    A = _window_means(run_dir, rows, a_lo, a_hi)
    if b_window is not None:
        B = _window_means(run_dir, rows, b_window[0], b_window[1])
    else:
        B = _window_means(run_dir, rows, mark_ms + settle * 1000, mark_ms + after * 1000)
    common = sorted(k for k in set(A) & set(B) if use_sides or k[4] == "end")
    if len(common) < 10:
        return None
    res = {k: _change(A[k], B[k]) for k in common}
    base = {}
    for kind in {k[4] for k in common}:
        v = np.array([res[k][0] for k in common if k[4] == kind])
        md = float(np.median(v))
        base[kind] = (md, float(1.4826 * np.median(np.abs(v - md)) + 1e-3))
    per = collections.defaultdict(dict)
    for k in common:
        z = (res[k][0] - base[k[4]][0]) / base[k[4]][1]
        per[(k[1], k[2])][k[0]] = max(per[(k[1], k[2])].get(k[0], -99.0), float(z))
    return {b: (float(np.mean(list(c.values()))) if len(c) >= 2 else max(c.values()) - 2.0) for b, c in per.items()}


def infer(run_dir, mark_ms, prev_mark_ms, start_ms, pos, slots, candidates, settle=0.8, after=4.5, use_sides=True,
          a_window=None, b_window=None):
    """
    pos        'left' | 'middle' | 'right'
    slots      slots_for_positions(...) output
    candidates layer indices a block could have been pulled from (existing layers only)
    Returns a dict; ok=False carries a reason.
    """
    rows = _rows(run_dir)
    if not rows:
        return {"ok": False, "reason": "no face crops yet (is data capture ON?)"}
    a_lo = (prev_mark_ms + 1000) if prev_mark_ms else (start_ms + 500)
    a_hi = min(a_lo + 3500, mark_ms - 2000)
    if a_window is None:
        if prev_mark_ms and a_hi - a_lo < 1500:
            return {"ok": False, "reason": "this tap came too soon after the previous one to see the tower before the move"}
        if a_hi - a_lo < 1500:
            return {"ok": False, "reason": "no calm intact-tower period before the first move"}
    else:
        a_lo, a_hi = a_window                                   # the calm stretch just before the hand came
    A = _window_means(run_dir, rows, a_lo, a_hi)
    if b_window is not None:                                    # the first calm stretch after the hand left
        B = _window_means(run_dir, rows, b_window[0], b_window[1])
    else:
        B = _window_means(run_dir, rows, mark_ms + settle * 1000, mark_ms + after * 1000)
    common = sorted(k for k in set(A) & set(B) if use_sides or k[4] == "end")
    if len(common) < 10:
        return {"ok": False, "reason": f"only {len(common)} faces seen both before and after (hand in the way?)"}
    res = {k: _change(A[k], B[k]) for k in common}
    # each crop type has its own natural noise level, so each gets its own baseline
    base = {}
    for kind in {k[4] for k in common}:
        v = np.array([res[k][0] for k in common if k[4] == kind])
        s = np.array([np.hypot(*res[k][1]) for k in common if k[4] == kind])
        md = float(np.median(v)); sm = float(np.median(s))
        base[kind] = (md, float(1.4826 * np.median(np.abs(v - md)) + 1e-3), sm, float(1.4826 * np.median(np.abs(s - sm)) + 0.5))
    scores = {}
    for layer in candidates:
        slot = slots.get(layer, {}).get(pos)
        if slot is None:
            continue
        best = None
        for k in common:
            if k[1] == layer and k[2] == slot:
                md, mad, sm, smad = base[k[4]]
                z = (res[k][0] - md) / mad
                zs = (np.hypot(*res[k][1]) - sm) / smad
                if best is None or z > best["z"]:
                    best = {"layer": layer + 1, "slot": slot, "z": float(z), "slide_z": float(zs),
                            "cam": k[0], "faces": 0}
        if best:
            best["faces"] = sum(1 for k in common if k[1] == layer and k[2] == slot)
            best["by_cam"] = {}
            for k in common:
                if k[1] == layer and k[2] == slot:
                    z = (res[k][0] - base[k[4]][0]) / base[k[4]][1]
                    best["by_cam"][k[0]] = max(best["by_cam"].get(k[0], -99.0), float(z))
            # a pulled block leaves a hole that BOTH cameras see; a hand or shadow usually fools only one.
            zs = list(best["by_cam"].values())
            best["z_strongest_face"] = best["z"]
            best["z"] = float(np.mean(zs)) if len(zs) >= 2 else float(max(zs)) - 2.0
            scores[layer] = best
    if len(scores) < 2:
        return {"ok": False, "reason": "too few faces of that column were visible both before and after"}
    ranked = sorted(scores.values(), key=lambda s: -s["z"])
    best, second = ranked[0], ranked[1]
    suspicious = False
    return {"ok": True, "pos": pos, "layer": best["layer"], "slot": best["slot"], "z": best["z"],
            "suspicious": suspicious,
            "runner_up": second["layer"], "runner_up_z": second["z"],
            "gap": best["z"] - second["z"], "faces_compared": len(common),
            "confident": bool(best["z"] >= 3.0 and best["z"] - second["z"] >= 2.0),
            "confidence_pct": confidence_pct(best["z"], best["z"] - second["z"]),
            "top": [{"layer": s["layer"], "z": round(s["z"], 2)} for s in ranked[:4]],
            "detail": {s["layer"]: s["by_cam"] for s in ranked}}


# How often the finder's answer was right, by how strongly it stood out: measured on every move with a known layer
# (57 moves in 12 games: the player's corrections, plus answers it was confident about and the player never corrected).
CONFIDENCE_TIERS = ((10.0, 5.0, 92), (8.0, 4.0, 89), (5.0, 3.0, 85), (3.0, 2.0, 82))     # (min score, min gap to 2nd, % right)
CONFIDENCE_LOW = 44                                                                       # below the weakest tier


def confidence_pct(z, gap):
    """About how likely the answer is to be exactly right, from how that kind of answer did in past games."""
    for zmin, gmin, pct in CONFIDENCE_TIERS:
        if z >= zmin and gap >= gmin:
            return pct
    return CONFIDENCE_LOW


# The same measurement when the player does NOT say which column: the best block anywhere, and how far ahead of the best block
# of any OTHER layer it is.  57 moves in 12 games (layer exactly right): z>=12 gap>=6 -> 95%, z>=10 gap>=5 -> 91%, z>=8 gap>=4 -> 88%,
# z>=6 gap>=3 -> 84%, anything weaker about 50%.  Every answer in the top tiers was within one layer of the truth.
FREE_TIERS = ((12.0, 6.0, 95), (10.0, 5.0, 91), (8.0, 4.0, 88), (6.0, 3.0, 84))
FREE_LOW = 50


def free_confidence_pct(z, gap):
    for zmin, gmin, pct in FREE_TIERS:
        if z >= zmin and gap >= gmin:
            return pct
    return FREE_LOW


def infer_free(run_dir, mark_ms, prev_mark_ms, start_ms, candidates, a_window=None, b_window=None, settle=0.8, after=2.5):
    """Which block was pulled, without being told the column?  `candidates` = {(layer, slot)} that could have been pulled
    (blocks that are there and may legally be pulled).  Returns a dict like infer(): ok, layer (1-based), slot, z, gap,
    confidence_pct, runner_up, top."""
    sc = block_scores(run_dir, mark_ms, prev_mark_ms, start_ms, settle=settle, after=after, a_window=a_window, b_window=b_window)
    if not sc:
        return {"ok": False, "reason": "too few faces seen both before and after (hand in the way, or data capture was off)"}
    sc = {k: v for k, v in sc.items() if k in candidates}
    if len(sc) < 3:
        return {"ok": False, "reason": "too few of the blocks that could have been pulled were visible"}
    ranked = sorted(sc.items(), key=lambda kv: -kv[1])
    (l1, s1), z1 = ranked[0]
    other = [z for (l, s), z in ranked[1:] if l != l1]
    z2 = max(other) if other else -5.0
    second_layer = next((l for (l, s), z in ranked[1:] if l != l1), None)
    gap = z1 - z2
    pct = free_confidence_pct(z1, gap)
    return {"ok": True, "layer": l1 + 1, "slot": int(s1), "z": float(z1), "gap": float(gap), "confidence_pct": pct,
            "runner_up": (second_layer + 1) if second_layer is not None else None, "runner_up_z": float(z2),
            "top": [{"layer": l + 1, "slot": int(s), "z": round(float(z), 1)} for (l, s), z in ranked[:4]]}


def parse_text(text):
    """'middle' / 'left 5' / 'r' -> (position, layer or None).  Layer is 1-based as on the overlay."""
    words = text.lower().replace(",", " ").split()
    names = {"left": "left", "l": "left", "middle": "middle", "mid": "middle", "m": "middle", "centre": "middle",
             "center": "middle", "c": "middle", "right": "right", "r": "right"}
    pos = next((names[w] for w in words if w in names), None)
    layer = next((int(w) for w in words if w.isdigit()), None)
    return pos, layer


def already_removed(run_dir, before_seq):
    """{(position, layer index)} of blocks taken out by EARLIER moves of this game.  Nothing is ever put back, so
    none of them can be the answer again.  Only trusted answers count: the player's own corrections, and
    answers the model itself was confident about."""
    path = os.path.join(run_dir, "labels.jsonl")
    if not os.path.exists(path):
        return set()
    model, corr = {}, {}
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "correction" in r:
            corr[r["seq"]] = r["correction"]
        elif r.get("result", {}).get("ok") and r["seq"] < before_seq:
            res = r["result"]
            model[r["seq"]] = (r["pos"], res["layer"], bool(res.get("confident")), r.get("user_layer"))
    out = set()
    for seq, (pos, layer, confident, user_layer) in model.items():
        if seq in corr:
            out.add((pos, corr[seq] - 1))
        elif user_layer:
            out.add((pos, user_layer - 1))
        elif confident:
            out.add((pos, layer - 1))
    return out
