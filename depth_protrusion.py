#!/usr/bin/env python3
"""
Can the iPhone LiDAR see how far each block sticks out?

For every visible block end face we take the high-confidence depth points that land on that face (inset from
its edges), express them in the tower frame, and measure their signed distance from where an untouched face
would sit (along the face's outward normal).  +mm = sticking out, -mm = set back / hole.

    MARKER_MAP=off MARKER_HALF=91 LIDAR_SAVE_DIR=game_04 .venv/bin/python depth_protrusion.py [stride] [max_frames]

What it prints: how many depth points a face gets, how big the frame-to-frame scatter of a face is (the noise
floor), the face-to-face spread within one frame after removing the shared offset, and what that means for the
smallest push-out we could detect.
"""
import collections
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import face_crops as fc
import tower_points as tp

LAYERS = int(os.environ.get("TOWER_LAYERS", 12))
INSET_A, INSET_Z = 8.0, 4.0           # keep points this far inside the 25 x 15 mm face (half sizes 12.5 / 7.5)
WINDOW = (-100.0, 30.0)               # accept points from 100 mm behind to 30 mm in front of the nominal face
MIN_PTS, MIN_COS = 8, 0.25


def depth_points(depth, conf, K, rgb_w, rgb_h, R, t, min_conf=2, dmax=1.5):
    """Depth map (metres) -> points in the tower frame (mm), high-confidence pixels only."""
    h, w = depth.shape
    fx, fy = K[0, 0] * w / rgb_w, K[1, 1] * h / rgb_h
    cx, cy = K[0, 2] * w / rgb_w, K[1, 2] * h / rgb_h
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    m = (depth > 0.05) & (depth < dmax)
    if conf is not None:
        m &= conf >= min_conf
    d = depth[m] * 1000.0
    X = np.stack([(us[m] - cx) * d / fx, (vs[m] - cy) * d / fy, d], axis=1)
    return (R.T @ (X - t).T).T


def face_offsets(P, cam, layers=None):
    """{(layer, slot, sign): (median offset mm, n points)} for the faces the camera sees well."""
    out = {}
    for layer in range(layers or LAYERS):
        zc = fc.HEIGHT * (layer + 0.5)
        sel = P[np.abs(P[:, 2] - zc) < (fc.HEIGHT / 2 - INSET_Z)]
        if len(sel) < MIN_PTS:
            continue
        for slot in range(3):
            for sign in (1, -1):
                _c, c, n = fc.face_geometry(layer, slot, sign)
                to = cam - c
                if float(n @ to) / (np.linalg.norm(to) + 1e-9) < MIN_COS:
                    continue
                along = np.array([n[1], -n[0], 0.0])                   # in-plane horizontal direction
                m = np.abs((sel - c) @ along) < (fc.WIDTH / 2 - INSET_A)
                d = (sel[m] - c) @ n
                d = d[(d > WINDOW[0]) & (d < WINDOW[1])]
                if len(d) >= MIN_PTS:
                    out[(layer, slot, sign)] = (float(np.median(d)), len(d))
    return out


def main():
    STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    MAX_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    stamps = sorted(os.path.basename(f).split("_")[0] for f in glob.glob(os.path.join(tp.SAVE_DIR, "*_intrinsics.json")))
    print(f"{len(stamps)} frames in {tp.SAVE_DIR}; using every {STRIDE}th, up to {MAX_FRAMES}")
    series = collections.defaultdict(list)          # face -> [(frame index, offset)]
    rel_series = collections.defaultdict(list)      # same, minus the frame's median over faces
    frames_ok, n_pts = 0, []
    for i, ts in enumerate(stamps[::STRIDE]):
        if frames_ok >= MAX_FRAMES:
            break
        fr = tp.load_frame(ts)
        if fr is None:
            continue
        dets = tp.detect(fr["rgb"])
        if len(dets) < 3:
            continue
        sol = tp.solve(dets, fr["K"], layout=tp.LAYOUTS["A"])
        if not sol or sol[2] > 7.0:
            continue
        R, t, _ = sol
        P = tp.depth_to_tower(fr, R, t)
        cam = -R.T @ t
        offs = face_offsets(P, cam)
        if len(offs) < 3:
            continue
        med = float(np.median([v[0] for v in offs.values()]))
        for k, (v, n) in offs.items():
            series[k].append((i, v)); rel_series[k].append((i, v - med)); n_pts.append(n)
        frames_ok += 1
    print(f"{frames_ok} usable frames, {len(series)} different faces seen; depth points per face: "
          f"median {int(np.median(n_pts))}, min {min(n_pts)}")
    if frames_ok < 5:
        return

    def stats(S, label):
        sd = [np.std([v for _, v in s]) for s in S.values() if len(s) >= 8]
        print(f"  {label}: frame-to-frame scatter of one face, median {np.median(sd):.1f} mm, "
              f"90th percentile {np.percentile(sd, 90):.1f} mm  ({len(sd)} faces)")
    print("\nNOISE (scatter of a single frame's reading):")
    stats(series, "raw offset            ")
    stats(rel_series, "minus shared offset   ")

    # averaging: mean of K consecutive readings of one face
    print("\nAVERAGING (does the scatter shrink with more frames?):")
    for K in (1, 4, 8, 16):
        sds = []
        for s in rel_series.values():
            v = np.array([x for _, x in s])
            if len(v) >= 2 * K:
                m = v[:len(v) // K * K].reshape(-1, K).mean(axis=1)
                sds.append(np.std(m))
        if sds:
            print(f"  average of {K:2d} frames: scatter {np.median(sds):.2f} mm")

    # settled offset of each face: where the faces really sit versus the ideal plane (a fixed bias would hide a push-out)
    bias = {k: np.median([v for _, v in s]) for k, s in rel_series.items() if len(s) >= 8}
    b = np.array(list(bias.values()))
    print(f"\nFACE-TO-FACE spread of the settled offsets (blocks are not perfectly aligned, markers not perfect): "
          f"std {b.std():.1f} mm, range {b.min():+.1f} .. {b.max():+.1f} mm")
    worst = sorted(bias.items(), key=lambda kv: -abs(kv[1]))[:8]
    print("  largest: " + ", ".join(f"L{k[0] + 1}S{k[1]}{'+' if k[2] > 0 else '-'} {v:+.1f}" for k, v in worst))


if __name__ == "__main__":
    main()
