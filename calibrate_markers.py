#!/usr/bin/env python3
"""
Measure where the four ArUco markers really are, so the mat does not have to be placed perfectly.

Each camera (iPhone, Android) measures every marker it can see as a 6-DoF pose (marker size is known).
Within one camera we express the markers on the floor plane relative to one reference marker, and take the
median over many frames.  The two cameras see different subsets, so their maps are stitched together through
the markers they both see (a 2-D rigid fit).  The result is turned into the tower frame (origin = middle of the
ring, +x along ID1 -> ID2) and written to marker_map.json, which tower_points.py then uses.

    .venv/bin/python calibrate_markers.py            # live: both phones streaming to the viewer server
    .venv/bin/python calibrate_markers.py --frames 20
"""
import argparse
import itertools
import json
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import face_crops as fc

SERVER = "http://127.0.0.1:8000"
H = 20.0                                                   # half of the 40 mm marker
OBJ = np.array([[-H, H, 0], [H, H, 0], [H, -H, 0], [-H, -H, 0]], np.float64)
P = cv2.aruco.DetectorParameters()
P.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
DET = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), P)


def grab(path):
    try:
        raw = urllib.request.urlopen(SERVER + path, timeout=5).read()
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def iphone_K(w):
    s = w / 1920.0
    return np.array([[1399.0082 * s, 0, 964.9536 * s], [0, 1399.0082 * s, 717.8016 * s], [0, 0, 1.0]]), None


def marker_poses(img, K, dist):
    cs, ids, _ = DET.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    out = {}
    if ids is None:
        return out
    for c, i in zip(cs, ids.flatten()):
        if int(i) > 3:
            continue
        ok, r, t = cv2.solvePnP(OBJ, c.reshape(4, 2).astype(np.float64), K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            out[int(i)] = (cv2.Rodrigues(r)[0], t.reshape(3))
    return out


def floor_coords(poses, ref):
    """Markers on the floor plane, relative to `ref`: {id: (x, y, yaw_deg)} (ref at the origin, yaw 0)."""
    Rr, tr = poses[ref]
    n = np.mean([R[:, 2] for R, _ in poses.values()], axis=0)
    n /= np.linalg.norm(n)
    if n @ (-tr) < 0:
        n = -n                                             # normal points up, towards the camera
    x0 = Rr[:, 0] - (Rr[:, 0] @ n) * n
    x0 /= np.linalg.norm(x0)
    y0 = np.cross(n, x0)
    out = {}
    for i, (R, t) in poses.items():
        d = t - tr
        out[i] = (float(d @ x0), float(d @ y0), float(np.degrees(np.arctan2(R[:, 0] @ y0, R[:, 0] @ x0))))
    return out


def camera_map(frames, name):
    """Median floor map over many frames, in the gauge of the marker seen most often."""
    seen = {}
    for f in frames:
        for i in f:
            seen[i] = seen.get(i, 0) + 1
    ref = max(seen, key=lambda i: (seen[i], -i))
    per = {}
    for f in frames:
        if ref in f and len(f) >= 3:
            for i, v in floor_coords(f, ref).items():
                per.setdefault(i, []).append(v)
    res, spread = {}, {}
    for i, v in per.items():
        a = np.array(v)
        ang = np.radians(a[:, 2])
        res[i] = (float(np.median(a[:, 0])), float(np.median(a[:, 1])),
                  float(np.degrees(np.arctan2(np.median(np.sin(ang)), np.median(np.cos(ang))))))
        spread[i] = float(np.hypot(a[:, 0].std(), a[:, 1].std()))
    print(f"  {name}: reference marker {ref}, {len(frames)} frames, markers {sorted(res)}, "
          f"frame-to-frame scatter {np.round([spread[i] for i in sorted(res)], 1).tolist()} mm")
    return res


def rigid2d(src, dst):
    """Rotation angle + translation mapping src points to dst (least squares, no scale)."""
    a, b = np.array(src), np.array(dst)
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    R = vt.T @ u.T
    if np.linalg.det(R) < 0:
        vt[-1] *= -1
        R = vt.T @ u.T
    return R, cb - R @ ca


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true", help="measure and report, do not write marker_map.json")
    args = ap.parse_args()

    st = json.loads(urllib.request.urlopen(SERVER + "/status.json", timeout=5).read())
    cams = {"iphone": "/preview_raw.jpg", "android": "/android_raw.jpg"}
    frames = {"iphone": [], "android": []}
    print(f"collecting {args.frames} frames from each camera (keep both phones still)...")
    for _ in range(args.frames):
        for name, path in cams.items():
            img = grab(path)
            if img is None:
                continue
            h, w = img.shape[:2]
            if name == "iphone":
                K, dist = iphone_K(w)
            else:
                K, dist = fc.android_K(w, h)
            if K is None:
                continue
            p = marker_poses(img, K, dist)
            if len(p) >= 3:
                frames[name].append(p)
        time.sleep(0.45)
    maps = {n: camera_map(f, n) for n, f in frames.items() if len(f) >= 3}
    if len(maps) < 2:
        sys.exit(f"need usable frames from BOTH cameras; got { {n: len(f) for n, f in frames.items()} }. "
                 "Check that both phones show 3+ markers in the viewer.")

    # stitch: express the iPhone map in the Android gauge through the markers both see
    A, B = maps["android"], maps["iphone"]
    common = sorted(set(A) & set(B))
    if len(common) < 2:
        sys.exit(f"the cameras share only markers {common}; need at least 2 in common")
    R, tr = rigid2d([B[i][:2] for i in common], [A[i][:2] for i in common])
    rot = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    resid = [float(np.linalg.norm(R @ np.array(B[i][:2]) + tr - np.array(A[i][:2]))) for i in common]
    print(f"  stitched through markers {common}; disagreement between the cameras on those: "
          f"{np.round(resid, 1).tolist()} mm")
    G = dict(A)
    for i, (x, y, yaw) in B.items():
        if i not in G:
            p = R @ np.array([x, y]) + tr
            G[i] = (float(p[0]), float(p[1]), float((yaw + rot + 180) % 360 - 180))
    if sorted(G) != [0, 1, 2, 3]:
        sys.exit(f"could only place markers {sorted(G)}; all four are needed")

    # tower frame: origin = middle of the ring, +x along ID1 -> ID2 (and ID0 -> ID3)
    pts = {i: np.array(G[i][:2]) for i in G}
    c = np.mean(list(pts.values()), axis=0)
    xv = (pts[2] - pts[1]) + (pts[3] - pts[0])
    xv /= np.linalg.norm(xv)
    yv = np.array([-xv[1], xv[0]])
    if (pts[0] - pts[1]) @ yv < 0:
        sys.exit("the ring runs the other way round from the one this software assumes (IDs must go "
                 "1 -> 2 -> 3 -> 0 counter-clockwise seen from above); swap two markers or tell me")
    ang = float(np.degrees(np.arctan2(xv[1], xv[0])))
    out = {}
    for i in sorted(G):
        d = pts[i] - c
        out[i] = {"x": round(float(d @ xv), 2), "y": round(float(d @ yv), 2),
                  "yaw": round(float((G[i][2] - ang + 180) % 360 - 180), 2)}
    side = [np.hypot(out[a]["x"] - out[b]["x"], out[a]["y"] - out[b]["y"]) for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))]
    diag = [np.hypot(out[a]["x"] - out[b]["x"], out[a]["y"] - out[b]["y"]) for a, b in ((0, 2), (1, 3))]
    print("\nmeasured marker map (tower frame, mm):")
    for i in sorted(out):
        print(f"  ID{i}: x {out[i]['x']:+7.1f}  y {out[i]['y']:+7.1f}  yaw {out[i]['yaw']:+7.1f} deg")
    print(f"ring sides {np.round(side, 1).tolist()} mm | diagonals {np.round(diag, 1).tolist()} mm")
    path = os.path.join(HERE, "marker_map.json")
    if args.dry_run:
        print("(dry run, not written)")
        return
    json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "marker_mm": 40.0, "markers": {str(i): v for i, v in out.items()}},
              open(path, "w"), indent=2)
    print(f"\nwrote {path}  (restart the server to use it)")


if __name__ == "__main__":
    main()
