#!/usr/bin/env python3
"""Height-estimation baselines on recorded LiDAR frames (before the first pull, so the true layer count is the typed one)
and on a synthetic hand-in-view stress test.     .venv/bin/python benchmark_height.py"""
import glob, json, os, sys, time
import numpy as np
import tower_points as tp, tower_fit as tf

RUNS = {"20261102_133032": 8, "20261106_092828": 7, "20261106_093055": 7}
EVERY = 2

def naive(P):
    near, _ = tf._split(P, 110.0)
    return float(np.percentile(near[:, 2], 98.0)) if len(near) > 300 else None

def make_ls(sigmas, iters):
    """Same 7-DoF box model as tower_fit.fit_box but with the loss / schedule changed."""
    def fit(P, h0=None):
        near, ring = tf._split(P, 110.0)
        if len(near) < 300: return None
        rng = np.random.default_rng(0)
        if len(near) > 2500: near = near[rng.choice(len(near), 2500, replace=False)]
        if len(ring) > 1250: ring = ring[rng.choice(len(ring), 1250, replace=False)]
        th = np.zeros(7); th[6] = float(h0 if h0 else np.percentile(near[:, 2], 98.0))
        sc = np.array([1, 1, 1, .01, .01, .01, 1.0])
        for s in sigmas:
            for _ in range(iters):
                r0 = tf._resid(th, near, ring)
                w = np.ones_like(r0) if s is None else (s ** 2 / (s ** 2 + r0 ** 2)) ** 2
                J = np.empty((len(r0), 7))
                for k in range(7):
                    d = np.zeros(7); d[k] = sc[k] * .05
                    J[:, k] = (tf._resid(th + d, near, ring) - r0) / d[k]
                A = J.T @ (J * w[:, None]) + 1e-3 * np.eye(7)
                lim = np.array([15, 15, 15, .15, .15, .15, 20.0])
                th = th + np.clip(np.linalg.solve(A, -(J.T @ (w * r0))), -lim, lim)
        return {"H": float(th[6])}
    return fit

METHODS = {
    "A  98th-percentile of z (no fit)":        lambda P: naive(P),
    "B  box fit, plain least squares":         lambda P: (make_ls([None], 30)(P) or {}).get("H"),
    "C  box fit, robust, single scale 30 mm":  lambda P: (make_ls([30.0], 30)(P) or {}).get("H"),
    "D  box fit, robust, 30->5 mm (ours)":     lambda P: (tf.fit_box(P) or {}).get("H"),
}

def stats(name, errs, ms):
    e = np.array([x for x in errs if x is not None]); miss = sum(x is None for x in errs)
    if not len(e): print(f"{name:42s} no estimate"); return
    print(f"{name:42s} n={len(e):3d}  exact {100*np.mean(e==0):4.0f}%  within 1 layer {100*np.mean(np.abs(e)<=1):4.0f}%  "
          f"MAE {np.mean(np.abs(e)):.2f} layers  bias {np.mean(e):+.2f}  no-fit {miss}  {np.mean(ms):.0f} ms")

def real():
    allres = {m: ([], []) for m in METHODS}
    for run, layers in RUNS.items():
        mv = [json.loads(l) for l in open(f"dataset/live_{run}/moves.jsonl")]
        t_first_pull = min([m["t"] for m in mv if m["ev"] != "meta"] or [10**15]); t0 = mv[0]["t"]
        stamps = sorted(int(os.path.basename(f).split("_")[0]) for f in glob.glob(os.path.join(tp.SAVE_DIR, "*_intrinsics.json")))
        stamps = [s for s in stamps if t0 - 2000 <= s < t_first_pull][::EVERY]
        n = 0
        for ts in stamps:
            fr = tp.load_frame(str(ts))
            if fr is None: continue
            d = tp.detect(fr["rgb"])
            s = tp.solve(d, fr["K"], layout=tp.LAYOUTS["A"]) if len(d) >= 3 else None
            if not s or s[2] > 10: continue
            P = tp.depth_to_tower(fr, s[0], s[1])
            n += 1
            for m, f in METHODS.items():
                t = time.time(); H = f(P); ms = (time.time() - t) * 1000
                allres[m][0].append(None if H is None else int(round(H / tf.LAYER_MM)) - layers); allres[m][1].append(ms)
        print(f"  run {run}: {layers} layers, {n} usable frames")
    print("\nREAL FRAMES (hand-free, before the first pull; error = estimated - typed layers)")
    for m, (e, ms) in allres.items(): stats(m, e, ms)

def synth_stress():
    import test_tower_fit as _t   # runs its own asserts; reuses the generator
    rng = np.random.default_rng(7)
    print("\nSYNTHETIC: 12-layer tower, 1 mm noise, a hand blob near the tower (fraction of near points replaced)")
    for frac in (0.0, 0.1, 0.25, 0.4):
        res = {m: [] for m in METHODS}
        for trial in range(15):
            H = 180.0
            P = _t.synth(np.array([6.0, -4.0, 3.0]), np.array([0.03, 0.0, 0.05]), H)
            k = int(frac * len(P)); idx = rng.choice(len(P), k, replace=False)
            c = np.array([rng.uniform(-60, -30), rng.uniform(-60, 60), rng.uniform(60, 200)])
            P[idx] = c + rng.normal(0, 18, (k, 3))
            for m, f in METHODS.items():
                Hh = f(P); res[m].append(None if Hh is None else abs(Hh - H) / tf.LAYER_MM)
        print(f"  hand fraction {int(frac*100):2d}%: " + "  ".join(f"{m[0]} {np.nanmean([x for x in v if x is not None]):.2f}" for m, v in res.items()) + "   (mean |height error| in layers)")

if __name__ == "__main__":
    real(); synth_stress()
