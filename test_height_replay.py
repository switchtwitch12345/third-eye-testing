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
COVER = [float(x) for x in os.environ.get("COVER_S", "").split(":")] if os.environ.get("COVER_S") else None    # simulate a hand over the phone between these seconds
COVER_MODE = os.environ.get("COVER_MODE", "empty")
BLIND_AFTER = float(sys.argv[3]) if len(sys.argv) > 3 else None      # seconds: from then on the markers are 'gone' and only the kept pose is used
rows = [json.loads(l) for l in open(os.path.join("dataset", run, "index.jsonl"))]
t0, t1 = rows[0]["ts_ms"], rows[-1]["ts_ms"]
stamps = sorted(os.path.basename(f).split("_")[0] for f in glob.glob(os.path.join(tp.SAVE_DIR, "*_intrinsics.json")))
stamps = [s for s in stamps if t0 <= int(s) <= t1][::every]
tp.reset_correction()
if os.environ.get("NO_BLOCK_GUARD"):
    srv.BLOCKED_ALL, srv.BLOCKED_CLOSE = 0.0, 9.0          # what the check did before it could tell a covered lens from a fall
srv._state["tower"] = {"layers": int(os.environ.get("LAYERS", 16)), "removed": []}
fired, used, log = None, 0, []
for ts in stamps:
    fr = tp.load_frame(ts)
    if fr is None:
        continue
    blind = BLIND_AFTER is not None and (int(ts) - t0) / 1000 >= BLIND_AFTER
    d = [] if blind else tp.detect(fr["rgb"])
    s = tp.solve(d, fr["K"], layout=tp.LAYOUTS["A"]) if len(d) >= 3 else None
    if s and s[2] <= 10:
        R, t, _ = s
        Rc, tc = tp.get_correction()
        Rm = R @ Rc.T
        srv._pose_lock["iphone"] = (Rm, t - Rm @ tc)          # exactly what the overlay stores
        srv._pose_keep["iphone"] = srv._pose_lock["iphone"]
        srv._lock_valid["iphone"] = time.time()
        srv._last_pose["iphone"] = (R, t, time.time())
    elif "iphone" not in srv._pose_lock and "iphone" not in srv._pose_keep:
        continue
    elif not blind:
        srv._lock_valid["iphone"] = time.time()               # markers hidden: the lock stays valid, as when the phone is still
    K = fr["K"]
    h = {"X-W": str(fr["w"]), "X-H": str(fr["h"]), "X-Fx": str(K[0, 0]), "X-Fy": str(K[1, 1]), "X-Cx": str(K[0, 2]),
         "X-Cy": str(K[1, 2]), "X-RgbW": str(fr["rgb"].shape[1]), "X-RgbH": str(fr["rgb"].shape[0])}
    conf = fr["conf"] if fr["conf"] is not None else np.full((fr["h"], fr["w"]), 2, np.uint8)
    dep = fr["depth"].astype(np.float32)
    if COVER and COVER[0] <= (int(ts) - t0) / 1000 <= COVER[1]:
        if COVER_MODE == "empty":
            conf = np.zeros_like(conf)                              # lens covered: nothing confident
        else:
            dep = np.full_like(dep, 0.12); conf = np.full_like(conf, 2)   # a hand 12 cm from the lens
    srv._process_depth(h, dep.tobytes() + conf.tobytes())
    used += 1
    if BLIND_AFTER is not None and (int(ts) - t0) / 1000 >= BLIND_AFTER:
        srv._pose_lock.pop("iphone", None)                       # as if the markers vanished and the lock was dropped
        srv._lock_valid["iphone"] = 0.0
    tf = srv._state["tower_fit"]
    if tf and used % 10 == 0:
        log.append((int(ts) - t0) / 1000)
    if COVER and srv._state["game_over"] and COVER[0] <= (int(ts) - t0) / 1000 <= COVER[1] + 6:
        print("  (game over raised while covered, at", round((int(ts) - t0) / 1000), "s:", srv._state["game_over"]["reason"], ")")
    if os.environ.get("TRACE"):
        hs, st = srv._height_st, srv._stand_st
        print(f"  t={(int(ts) - t0) / 1000:6.1f}s  top_hist={[round(x) for x in hs['hist'][-3:]]} ref={round(float(np.median(hs['ref'])), 1) if hs['ref'] else None}  n={srv._stand_dbg.get('n_col')},{srv._stand_dbg.get('n_top')},{srv._stand_dbg.get('n_all')} ref={[round(float(np.median([r[i] for r in st['ref']]))) for i in range(3)] if st['ref'] else None} close={srv._stand_dbg.get('close')} p95={srv._stand_dbg.get('p95')} hand={srv._stand_dbg.get('n_hand')}  stand low={st['low']} back={st['back']}  game_over={bool(srv._state['game_over'])} note={srv._state['depth_dbg']['why'][:40]}")
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
print("standing check:", {k: (v if k != "ref" else v[:3]) for k, v in srv._stand_st.items()}, "| depth note:", srv._state["depth_dbg"])
