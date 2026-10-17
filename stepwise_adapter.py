#!/usr/bin/env python3
"""
Adapter: our ARKit capture -> Aadit's STEPWISE/ThirdEye Jenga depth reader.

  our RGB + depth + confidence + intrinsics   (iphone_saved/)
  our marker pose (tower_points.py)           -> DepthView(depth_mm, K, R, t)
  their read_view() / merge()                 -> per-slot block / empty / moving

Reports how their thresholds (tol 6 mm, alone 20 mm, occluded 60 mm) behave on
REAL LiDAR frames, which their repo has not yet been tested on.

Run isolated:  .venv/bin/python -I stepwise_adapter.py [--layers 9] [--min-conf 2]
"""
import argparse, collections, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "third_party", "STEPWISE"))
sys.path.insert(0, HERE)

import tower_points as tp                                   # our calibration
from thirdeye.perception.tower import DepthView, read_view, merge, STANDARD   # their reader


def build_view(fr, R, t, min_conf):
    """Depth in mm, intrinsics scaled to the depth map, low-confidence pixels dropped."""
    depth = fr["depth"].astype(np.float64) * 1000.0
    if fr["conf"] is not None:
        depth[fr["conf"] < min_conf] = 0.0
    h, w = depth.shape
    sx, sy = w / fr["rgb"].shape[1], h / fr["rgb"].shape[0]
    K = np.diag([sx, sy, 1.0]) @ fr["K"]
    return DepthView(depth, K, R, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=9, help="layers actually in the tower")
    ap.add_argument("--min-conf", type=int, default=2, help="drop depth below this confidence (0-2)")
    ap.add_argument("--every", type=int, default=3)
    args = ap.parse_args()

    layout = tp.LAYOUTS["A"]
    stamps = sorted(os.path.basename(f).split("_")[0]
                    for f in glob.glob(f"{tp.SAVE_DIR}/*_intrinsics.json"))

    per_state_expected = collections.Counter()      # (expected, verdict) -> n
    end_states = collections.Counter()              # seen / occluded / unseen
    seen_offsets = []                               # offsets of ends that should be flush
    air_offsets = []
    per_layer = collections.defaultdict(collections.Counter)
    used = 0

    for ts in stamps[::args.every]:
        fr = tp.load_frame(ts)
        if fr is None or fr["conf"] is None:
            continue
        dets = tp.detect(fr["rgb"])
        s = tp.solve(dets, fr["K"], layout=layout)
        if not s or s[2] > 6:
            continue
        R, t, _ = s
        view = build_view(fr, R, t, args.min_conf)
        ends = read_view(view)                       # their code, unchanged
        verdict = merge(ends, {})                    # one camera -> second dict empty
        used += 1
        for slot, e in ends.items():
            end_states[e.state] += 1
            if e.state != "seen":
                continue
            (seen_offsets if slot.layer < args.layers else air_offsets).append(e.offset_mm)
        for slot, st in verdict.states.items():
            exp = "block" if slot.layer < args.layers else "air"
            per_state_expected[(exp, st)] += 1
            per_layer[slot.layer][st] += 1

    print(f"frames used: {used}   (min_conf={args.min_conf}, tower layers assumed = {args.layers})")
    print(f"\nend faces their reader could evaluate, summed over frames: {dict(end_states)}")

    if seen_offsets:
        a = np.array(seen_offsets)
        print(f"\nOFFSET on end faces that should be flush (block present), n={len(a)}:")
        print(f"  median {np.median(a):+.1f} mm | std {a.std():.1f} mm | "
              f"within +-6 mm (their tol): {100*np.mean(np.abs(a) <= 6):.0f}% | "
              f"p5..p95 {np.percentile(a,5):+.0f}..{np.percentile(a,95):+.0f} mm")
    if air_offsets:
        b = np.array(air_offsets)
        print(f"OFFSET on slots above the tower (should read far / empty), n={len(b)}: "
              f"median {np.median(b):+.0f} mm, >=20 mm (their 'empty'): {100*np.mean(b >= 20):.0f}%")

    print("\nMERGED VERDICT vs truth (single camera, so merge() uses its 'alone' rules):")
    for exp in ("block", "air"):
        tot = sum(v for (e, _), v in per_state_expected.items() if e == exp)
        if not tot:
            continue
        row = {st: per_state_expected[(exp, st)] for st in ("block", "empty", "moving", "unknown")}
        print(f"  truth={exp:5s} n={tot:4d}  " + "  ".join(f"{k}={100*v/tot:4.0f}%" for k, v in row.items()))

    print("\nper layer (verdicts, all frames):")
    for layer in sorted(per_layer):
        if layer > args.layers + 2:
            break
        c = per_layer[layer]
        n = sum(c.values())
        print(f"  layer {layer:2d}  " + "  ".join(f"{k}={100*c[k]/n:3.0f}%" for k in ("block", "empty", "moving", "unknown")))


if __name__ == "__main__":
    main()
