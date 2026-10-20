#!/usr/bin/env python3
"""
Replays a recorded game's depth frames through the server's tower-height watch and says when (if ever) it
declares GAME OVER.   .venv/bin/python test_height_replay.py live_20261020_170044 16 [seconds_to_skip_before_start]
Needs the recordings in iphone_saved/ and the run's dataset folder (for the time range).
"""
import glob
import json
import os
import sys
import time

import iphone_saved as srv
import tower_points as tp

run, layers = sys.argv[1], int(sys.argv[2])
rows = [json.loads(l) for l in open(os.path.join("dataset", run, "index.jsonl"))]
t0, t1 = rows[0]["ts_ms"], rows[-1]["ts_ms"]
stamps = sorted(os.path.basename(f).split("_")[0] for f in glob.glob(os.path.join(tp.SAVE_DIR, "*_intrinsics.json")))
stamps = [s for s in stamps if t0 <= int(s) <= t1]
srv._state["tower"] = {"layers": layers, "removed": []}
fired, used = None, 0
for ts in stamps:
    fr = tp.load_frame(ts)
    if fr is None:
        continue
    d = tp.detect(fr["rgb"])
    s = tp.solve(d, fr["K"], layout=tp.LAYOUTS["A"]) if len(d) >= 3 else None
    if s and s[2] <= 7:
        srv._last_pose["iphone"] = (s[0], s[1], time.time())
    elif "iphone" not in srv._last_pose:
        continue
    K = fr["K"]
    h = {"X-W": str(fr["w"]), "X-H": str(fr["h"]), "X-Fx": str(K[0, 0]), "X-Fy": str(K[1, 1]), "X-Cx": str(K[0, 2]),
         "X-Cy": str(K[1, 2]), "X-RgbW": str(fr["rgb"].shape[1]), "X-RgbH": str(fr["rgb"].shape[0])}
    import numpy as np
    conf = fr["conf"] if fr["conf"] is not None else np.full((fr["h"], fr["w"]), 2, np.uint8)
    srv._process_depth(h, fr["depth"].astype(np.float32).tobytes() + conf.tobytes())
    used += 1
    if srv._state["game_over"] and fired is None:
        fired = (int(ts) - t0) / 1000
        print(f"GAME OVER at {fired:.0f} s into the run: {srv._state['game_over']['reason']}")
hi = srv._state["height_info"]
print(f"{run}: {used} depth frames replayed; final height reading {hi}")
if fired is None:
    print("no game over declared")
