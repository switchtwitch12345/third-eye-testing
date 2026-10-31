#!/usr/bin/env python3
"""
Replays a recorded game's depth frames through the server's LiDAR tower fit and height watch, the way the live
server does: marker pose -> pose lock (raw marker frame) -> depth -> fit the real tower -> game over if the top drops.
    .venv/bin/python test_height_replay.py live_20261020_170044 [every_nth_frame]
Needs the recordings in iphone_saved/ and the run's dataset folder (for the time range).
"""
import glob
import json
import os
import sys
import time

import numpy as np

import iphone_saved as srv
import tower_points as tp

run = sys.argv[1]
every = int(sys.argv[2]) if len(sys.argv) > 2 else 1
rows = [json.loads(l) for l in open(os.path.join("dataset", run, "index.jsonl"))]
t0, t1 = rows[0]["ts_ms"], rows[-1]["ts_ms"]
stamps = sorted(os.path.basename(f).split("_")[0] for f in glob.glob(os.path.join(tp.SAVE_DIR, "*_intrinsics.json")))
stamps = [s for s in stamps if t0 <= int(s) <= t1][::every]
tp.reset_correction()
srv._state["tower"] = {"layers": 16, "removed": []}
fired, used, log = None, 0, []
for ts in stamps:
    fr = tp.load_frame(ts)
    if fr is None:
        continue
    d = tp.detect(fr["rgb"])
    s = tp.solve(d, fr["K"], layout=tp.LAYOUTS["A"]) if len(d) >= 3 else None
    if s and s[2] <= 10:
        R, t, _ = s
        Rc, tc = tp.get_correction()
        Rm = R @ Rc.T
        srv._pose_lock["iphone"] = (Rm, t - Rm @ tc)          # exactly what the overlay stores
        srv._lock_valid["iphone"] = time.time()
        srv._last_pose["iphone"] = (R, t, time.time())
    elif "iphone" not in srv._pose_lock:
        continue
    else:
        srv._lock_valid["iphone"] = time.time()               # markers hidden: the lock stays valid, as when the phone is still
    K = fr["K"]
    h = {"X-W": str(fr["w"]), "X-H": str(fr["h"]), "X-Fx": str(K[0, 0]), "X-Fy": str(K[1, 1]), "X-Cx": str(K[0, 2]),
         "X-Cy": str(K[1, 2]), "X-RgbW": str(fr["rgb"].shape[1]), "X-RgbH": str(fr["rgb"].shape[0])}
    conf = fr["conf"] if fr["conf"] is not None else np.full((fr["h"], fr["w"]), 2, np.uint8)
    srv._process_depth(h, fr["depth"].astype(np.float32).tobytes() + conf.tobytes())
    used += 1
    tf = srv._state["tower_fit"]
    if tf and used % 10 == 0:
        log.append((int(ts) - t0) / 1000)
    if srv._state["game_over"] and fired is None:
        fired = (int(ts) - t0) / 1000
        print(f"GAME OVER at {fired:.0f} s into the run: {srv._state['game_over']['reason']}")
print(f"{run}: {used} depth frames replayed")
print("fit        :", srv._state["tower_fit"])
print("height     :", srv._state["height_info"])
Rc, tc = tp.get_correction()
print(f"correction : shift {np.round(tc, 1)} mm, tilt {np.degrees(np.linalg.norm(np.cross(Rc[:, 2], [0, 0, 1]))):.2f} deg")
if fired is None:
    print("no game over declared")
