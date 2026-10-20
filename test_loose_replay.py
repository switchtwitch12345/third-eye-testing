#!/usr/bin/env python3
"""
Replays recorded depth frames (game_04) through the server's loose-block detector, then fakes an 8 mm push-out
of one block to see whether it turns amber, how fast, and whether anything else does by mistake.
    MARKER_MAP=off MARKER_HALF=91 LIDAR_SAVE_DIR=game_04 .venv/bin/python test_loose_replay.py [push_mm]
"""
import glob
import os
import sys
import time

import numpy as np

import depth_protrusion as dp
import face_crops as fc
import iphone_saved as srv
import tower_points as tp

PUSH = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
stamps = sorted(os.path.basename(f).split("_")[0] for f in glob.glob(os.path.join(tp.SAVE_DIR, "*_intrinsics.json")))
srv._state["tower"] = {"layers": 9, "removed": []}
target, push, orig = None, {"on": False}, dp.depth_points


def patched(depth, conf, K, rw, rh, R, t, **kw):
    P = orig(depth, conf, K, rw, rh, R, t, **kw)
    if push["on"] and target:
        l, s, sign = target
        _c, c, n = fc.face_geometry(l, s, sign)
        along = np.array([n[1], -n[0], 0.0])
        m = (np.abs(P[:, 2] - fc.HEIGHT * (l + .5)) < 7) & (np.abs((P - c) @ along) < 12.5) & (np.abs((P - c) @ n) < 30)
        P[m] += n * PUSH
    return P


dp.depth_points = patched
used, log, first_push = 0, [], None
for ts in stamps[::3]:
    fr = tp.load_frame(ts)
    if fr is None:
        continue
    dets = tp.detect(fr["rgb"])
    if len(dets) < 3:
        continue
    sol = tp.solve(dets, fr["K"], layout=tp.LAYOUTS["A"])
    if not sol or sol[2] > 7:
        continue
    R, t, _ = sol
    srv._last_pose["iphone"] = (R, t, time.time())
    rw, rh = fr["rgb"].shape[1], fr["rgb"].shape[0]
    if target is None:
        offs = dp.face_offsets(orig(fr["depth"], fr["conf"], fr["K"], rw, rh, R, t), -R.T @ t, 9)
        if len(offs) < 3:
            continue
        target = max(offs, key=lambda k: offs[k][1])
        print(f"pushing face L{target[0] + 1} slot {target[1]} by {PUSH} mm after frame 60")
    used += 1
    push["on"] = used > 60
    K = fr["K"]
    h = {"X-W": str(fr["w"]), "X-H": str(fr["h"]), "X-Fx": str(K[0, 0]), "X-Fy": str(K[1, 1]), "X-Cx": str(K[0, 2]),
         "X-Cy": str(K[1, 2]), "X-RgbW": str(rw), "X-RgbH": str(rh)}
    conf = fr["conf"] if fr["conf"] is not None else np.full((fr["h"], fr["w"]), 2, np.uint8)
    srv._process_depth(h, fr["depth"].astype(np.float32).tobytes() + conf.tobytes())
    log.append((used, push["on"], [tuple(b) for b in srv._state["loose"]]))
    if used >= 120:
        break
exp = (target[0], target[1])
before = [x for x in log if not x[1]]
after = [x for x in log if x[1]]
false = sorted({b for x in before for b in x[2]}); fcount = {b: sum(1 for x in before if b in x[2]) for b in false}
hit = next((x[0] - 60 for x in after if exp in x[2]), None)
print(f"before the push ({len(before)} frames): blocks amber: {fcount or "none"} (frames out of {len(before)})")
print(f"after the push: {exp} turned amber after {hit} frames" if hit else f"after the push: {exp} never turned amber")
print(f"other blocks amber at the end: {[b for b in after[-1][2] if b != exp] or 'none'}")
