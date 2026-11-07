#!/usr/bin/env python3
"""Does the LiDAR tower fit put the overlay on the real tower better than the markers alone?
Metric (independent of the LiDAR): mean edge strength under the tower's visible edges projected into the RGB image (higher = better).
 P0 markers only | P1 markers + LiDAR box-fit correction (median over the run's first frames) | P2 + per-frame edge snapping
    .venv/bin/python benchmark_pose.py"""
import glob, json, os
import numpy as np
import tower_points as tp, tower_fit as tf, tower_edges as te

RUNS = {"20261102_133032": 8, "20261106_092828": 7, "20261106_093055": 7}
FIT_N, EVAL_N = 12, 25

def frames(run):
    mv = [json.loads(l) for l in open(f"dataset/live_{run}/moves.jsonl")]
    t0 = mv[0]["t"]; tp_ = min([m["t"] for m in mv if m["ev"] != "meta"] or [10**15])
    st = sorted(int(os.path.basename(f).split("_")[0]) for f in glob.glob(os.path.join(tp.SAVE_DIR, "*_intrinsics.json")))
    return [s for s in st if t0 - 2000 <= s < tp_]

res = {"P0": [], "P1": [], "P2": []}
for run, layers in RUNS.items():
    tp.reset_correction()
    use = []
    for ts in frames(run):
        fr = tp.load_frame(str(ts))
        if fr is None: continue
        d = tp.detect(fr["rgb"]); s = tp.solve(d, fr["K"], layout=tp.LAYOUTS["A"], corrected=False) if len(d) >= 3 else None
        if s and s[2] <= 10: use.append((fr, s))
    if len(use) < 8: print(run, "too few frames", len(use)); continue
    fits = []
    for fr, s in use[:FIT_N]:
        f = tf.fit_box(tp.depth_to_tower(fr, s[0], s[1]))
        if f and f["inliers"] >= .55 and f["rms"] <= 4: fits.append(f)
    rv = np.median([tf.log(f["R"]) for f in fits], 0); tc = np.median([f["t"] for f in fits], 0)
    Rc = tf.rot(rv); tp.set_correction(Rc, tc)
    print(f"run {run}: {layers} layers, {len(use)} frames; LiDAR correction shift {np.round(tc,1)} mm, tilt {np.degrees(np.linalg.norm(rv)):.2f} deg")
    pick = use[FIT_N:][::max(1, (len(use) - FIT_N) // EVAL_N)][:EVAL_N]
    for fr, (R, t, _) in pick:
        img = fr["rgb"]; K = fr["K"]; edges = te.sample_edges(layers * te.LAYER_MM); gx, gy = te.edge_maps(img, 1.2)
        sc = lambda R_, t_: te.score(np.zeros(6), gx, gy, edges, R_, t_, K, None, "iphone")
        R1, t1 = R @ Rc, t + R @ tc
        x, _, _ = te.refine(img, K, None, R1, t1, layers); R2, t2 = te.apply_delta(R1, t1, x)
        res["P0"].append(sc(R, t)); res["P1"].append(sc(R1, t1)); res["P2"].append(sc(R2, t2))
print("\nedge strength under the projected tower edges (iPhone frames, hand-free)")
for k, name in (("P0", "markers only"), ("P1", "markers + LiDAR fit"), ("P2", "markers + LiDAR + edge snapping")):
    v = np.array(res[k]); print(f"  {name:34s} n={len(v)}  mean {v.mean():.2f}  median {np.median(v):.2f}")
a, b, c = (np.array(res[k]) for k in ("P0", "P1", "P2"))
print(f"  LiDAR fit better than markers on {100*np.mean(b>a):.0f}% of frames; edge snapping better than fit on {100*np.mean(c>b):.0f}%")
