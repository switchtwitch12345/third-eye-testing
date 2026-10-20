#!/usr/bin/env python3
"""
Step 4a: put LiDAR depth into the tower's marker-defined coordinate frame and
check calibration. The desk plane must come out at z = 0 mm.

Tower frame (mm): origin = centre of tower footprint, +x right, +y away from
the phone, +z up. Markers (40 mm, DICT_4X4_50) lie flat at z = 0:
    ID0 (-82.5,-82.5)  ID1 (+82.5,-82.5)  ID2 (-82.5,+82.5)  ID3 (+82.5,+82.5)

Usage:  .venv/bin/python tower_points.py [--frame TIMESTAMP]
"""
import argparse, glob, json, os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE_DIR = os.environ.get("LIDAR_SAVE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "iphone_saved"))
MARKER_MM = 40.0
HALF = float(os.environ.get("MARKER_HALF", 80.0))  # half the marker spacing (mm): 80 = current mat (~160 mm); 91 = games 03/04
LAYOUTS = {   # IDs run around the square; A = counter-clockwise from above, B = clockwise
    # Your bottom layer runs along the markers' y direction, so the tower frame is the marker
    # square turned -90 degrees: that way layer 0 runs along x, as the tower geometry assumes.
    "A": {0: (-HALF, HALF), 1: (-HALF, -HALF), 2: (HALF, -HALF), 3: (HALF, HALF)},
    "B": {0: (-HALF, -HALF), 1: (-HALF, HALF), 2: (HALF, HALF), 3: (HALF, -HALF)},
}
MARKER_YAW = {}      # id -> degrees; filled from marker_map.json (measured orientation of each marker)
TOWER_ROT = 0        # the tower frame is turned by -90 degrees this many times (so layer 0 runs along x)
_BASE = None         # (positions, yaws) of the markers before any such turn
_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "marker_map.json")


def _apply_rotation(k):
    """Re-express the marker layout in a tower frame turned by -90 degrees k times. Edits in place so every
    module that already holds LAYOUTS["A"] sees the change."""
    global TOWER_ROT
    pos, yaw = dict(_BASE[0]), dict(_BASE[1])
    for _ in range(k % 4):
        pos = {i: (y, -x) for i, (x, y) in pos.items()}
        yaw = {i: v - 90.0 for i, v in yaw.items()}
    TOWER_ROT = k % 4
    LAYOUTS["A"].clear(); LAYOUTS["A"].update(pos)
    MARKER_YAW.clear(); MARKER_YAW.update(yaw)


def _load_marker_map():
    """marker_map.json (from calibrate_markers.py) overrides the nominal square. MARKER_MAP=off ignores it."""
    global _BASE
    rot = 0
    if os.environ.get("MARKER_MAP") != "off" and os.path.exists(_MAP_PATH):
        j = json.load(open(_MAP_PATH))
        m = j["markers"]
        _BASE = ({int(k): (float(v["x"]), float(v["y"])) for k, v in m.items()},
                 {int(k): float(v["yaw"]) for k, v in m.items()})
        rot = int(j.get("tower_rot90", 0))
    else:
        _BASE = (dict(LAYOUTS["A"]), {i: 0.0 for i in LAYOUTS["A"]})
    _apply_rotation(rot)


def set_tower_rotation(k, persist=True):
    """Turn the tower frame (used when the bottom layer runs the other way). Saved next to the marker map."""
    _apply_rotation(k)
    if persist:
        j = json.load(open(_MAP_PATH)) if os.path.exists(_MAP_PATH) else {
            "markers": {str(i): {"x": x, "y": y, "yaw": 0.0} for i, (x, y) in _BASE[0].items()}, "marker_mm": 40.0}
        j["tower_rot90"] = TOWER_ROT
        json.dump(j, open(_MAP_PATH, "w"), indent=2)


_load_marker_map()
CENTRES = LAYOUTS["A"]
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
DETECTOR = cv2.aruco.ArucoDetector(DICT)


def marker_corners_3d(mid, rot, layout=None):
    cx, cy = (layout or CENTRES)[mid]
    h = MARKER_MM / 2
    base = [(-h, h), (h, h), (h, -h), (-h, -h)]  # TL TR BR BL, marker "up" = +y
    base = base[rot:] + base[:rot]                # rotate marker by 90deg * rot
    yaw = np.radians(MARKER_YAW.get(mid, 0.0))    # measured orientation, if a marker map exists
    c, sn = np.cos(yaw), np.sin(yaw)
    return np.array([[cx + dx * c - dy * sn, cy + dx * sn + dy * c, 0.0] for dx, dy in base], np.float64)


def load_frame(ts):
    meta = json.load(open(f"{SAVE_DIR}/{ts}_intrinsics.json"))
    rgb = cv2.imread(f"{SAVE_DIR}/{ts}_rgb.jpg")
    dfile = glob.glob(f"{SAVE_DIR}/{ts}_depth_*.bin")[0]
    w, h = map(int, dfile.split("_depth_")[1][:-4].split("x"))
    depth = np.fromfile(dfile, np.float32)
    if depth.size != w * h:
        return None
    conf = None
    cf = glob.glob(f"{SAVE_DIR}/{ts}_conf_*.bin")
    if cf:
        c = np.fromfile(cf[0], np.uint8)
        if c.size == w * h:
            conf = c.reshape(h, w)
    K = np.array([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]], np.float64)
    return dict(ts=ts, rgb=rgb, depth=depth.reshape(h, w), conf=conf, K=K, w=w, h=h)


def detect(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = DETECTOR.detectMarkers(gray)
    out = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            if int(i) in CENTRES:
                out[int(i)] = c.reshape(4, 2).astype(np.float64)
    return out


def solve(dets, K, rot=None, layout=None):
    """Pose of tower frame in camera. Stage 1 uses marker centres only (independent of how
    each marker was rotated); stage 2 refines with corners using the best per-marker rotation."""
    layout = layout or CENTRES
    ids = [i for i in dets if i in layout]
    if len(ids) < 3:
        return None
    ctr3 = np.array([[*layout[i], 0.0] for i in ids], np.float64)
    ctr2 = np.array([dets[i].mean(axis=0) for i in ids], np.float64)
    cands = []                                       # stage-1 poses from the marker centres (possibly several: the fit is ambiguous)
    n, rv, tv, _ = cv2.solvePnPGeneric(ctr3, ctr2, K, None, flags=cv2.SOLVEPNP_IPPE if len(ids) >= 4 else cv2.SOLVEPNP_SQPNP)
    for r_, t_ in zip(rv, tv):
        cands.append((r_, t_))
    if len(ids) == 3:
        # Three centres fit several different poses exactly (a low camera cannot tell them apart), and the solver
        # may land on a wrong one.  Start from many viewpoints, keep every pose that fits, and let the corners decide.
        dist = float(np.linalg.norm(cands[0][1])) if cands else 400.0
        up = np.array([0.0, 0.0, 1.0])
        for az in np.arange(0, 360, 45):
            for el in (35.0, 55.0, 75.0):
                a_, e_ = np.radians(az), np.radians(el)
                C = dist * np.array([np.cos(e_) * np.cos(a_), np.cos(e_) * np.sin(a_), np.sin(e_)])
                zc = -C / np.linalg.norm(C)
                yc = -up - (-up @ zc) * zc
                yc /= np.linalg.norm(yc)
                xc = np.cross(yc, zc)
                Rw = np.stack([xc, yc, zc])
                ok_, r_, t_ = cv2.solvePnP(ctr3, ctr2, K, None, cv2.Rodrigues(Rw)[0], (-Rw @ C).reshape(3, 1), True,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
                if ok_:
                    cands.append((r_, t_))

    def centre_err(r_, t_):
        p, _ = cv2.projectPoints(ctr3, r_, t_, K, None)
        return float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - ctr2) ** 2, axis=1))))

    def refine(rvec, tvec):
        """stage 2: per-marker rotation that best explains the detected corners, then refine on corners"""
        obj, img = [], []
        for i in ids:
            cand = []
            for r in range(4):
                c3 = marker_corners_3d(i, r, layout)
                p, _ = cv2.projectPoints(c3, rvec, tvec, K, None)
                cand.append((float(np.mean(np.linalg.norm(p.reshape(-1, 2) - dets[i], axis=1))), c3))
            e, c3 = min(cand, key=lambda x: x[0])
            obj.append(c3); img.append(dets[i])
        obj = np.concatenate(obj); img = np.concatenate(img)
        ok, rv2, tv2 = cv2.solvePnP(obj, img, K, None, rvec.copy(), tvec.copy(), True, flags=cv2.SOLVEPNP_ITERATIVE)
        p, _ = cv2.projectPoints(obj, rv2, tv2, K, None)
        return rv2, tv2, float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - img) ** 2, axis=1))))

    best, seen = None, []
    for r_, t_ in cands:
        R_, _ = cv2.Rodrigues(r_)
        cam = -R_.T @ t_.reshape(3)
        if cam[2] <= 0 or centre_err(r_, t_) > 2.0:
            continue                                 # camera must be above the desk, and the pose must fit the centres
        if any(np.linalg.norm(cam - c) < 5.0 for c in seen):
            continue                                 # same pose again
        seen.append(cam)
        rv2, tv2, err = refine(r_, t_)
        R2, _ = cv2.Rodrigues(rv2)
        if (-R2.T @ tv2.reshape(3))[2] <= 0:
            continue
        if best is None or err < best[2]:
            best = (R2, tv2.reshape(3), err)
    return best


def depth_to_tower(fr, R, t, min_conf=2, dmax=1.5):
    h, w = fr["h"], fr["w"]
    K = fr["K"].copy()
    sx = w / fr["rgb"].shape[1]
    sy = h / fr["rgb"].shape[0]
    fx, fy, cx, cy = K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    d = fr["depth"]
    m = (d > 0.05) & (d < dmax)
    if fr["conf"] is not None:
        m &= fr["conf"] >= min_conf
    d_mm = d[m] * 1000.0
    X = np.stack([(us[m] - cx) * d_mm / fx, (vs[m] - cy) * d_mm / fy, d_mm], axis=1)
    return (R.T @ (X - t).T).T  # camera -> tower frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=str, default=None)
    args = ap.parse_args()

    stamps = sorted(os.path.basename(f).split("_")[0] for f in glob.glob(f"{SAVE_DIR}/*_intrinsics.json"))
    print(f"{len(stamps)} frames")

    # 1) which ring order matches the desk?  (mirror layouts can't both fit with camera above)
    cache = {}
    errs = {k: [] for k in LAYOUTS}
    for ts in stamps[::4]:
        fr = load_frame(ts)
        if fr is None:
            continue
        dets = detect(fr["rgb"])
        if len(dets) < 3:
            continue
        cache[ts] = (fr, dets)
        for k, lay in LAYOUTS.items():
            s_ = solve(dets, fr["K"], layout=lay)
            if s_:
                errs[k].append(s_[2])
    print("reprojection error per layout (px, median):")
    for k in LAYOUTS:
        print(f"  layout {k}: {np.median(errs[k]) if errs[k] else float('nan'):.2f}  (n={len(errs[k])})")
    lk = min(LAYOUTS, key=lambda k: np.median(errs[k]) if errs[k] else 1e9)
    layout = LAYOUTS[lk]
    print(f"-> using layout {lk}")

    # 2) choose a frame
    if args.frame:
        fr = load_frame(args.frame)
        dets = detect(fr["rgb"])
    else:
        best = None
        for ts, (f, dets_) in cache.items():
            s = solve(dets_, f["K"], layout=layout)
            if s and (best is None or (len(dets_), -s[2]) > best[0]):
                best = ((len(dets_), -s[2]), ts)
        fr, dets = cache[best[1]]
    R, t, err = solve(dets, fr["K"], layout=layout)
    print(f"frame {fr['ts']}: markers={sorted(dets)} reproj={err:.2f}px "
          f"camera distance to tower centre={np.linalg.norm(t)/10:.1f} cm")

    # 3) depth -> tower frame, desk plane check
    P = depth_to_tower(fr, R, t)
    print(f"{len(P)} high-confidence points")
    ring = P[(np.abs(P[:, 0]) < 220) & (np.abs(P[:, 1]) < 220)
             & ((np.abs(P[:, 0]) > 70) | (np.abs(P[:, 1]) > 70)) & (np.abs(P[:, 2]) < 40)]
    if len(ring) > 20:
        print(f"DESK PLANE (should be z=0): n={len(ring)} median z = {np.median(ring[:,2]):+.1f} mm, "
              f"std = {np.std(ring[:,2]):.1f} mm")
    tower = P[(np.abs(P[:, 0]) < 70) & (np.abs(P[:, 1]) < 70) & (P[:, 2] > 5) & (P[:, 2] < 300)]
    print(f"TOWER region: n={len(tower)}"
          + (f", z range {tower[:,2].min():.0f}..{tower[:,2].max():.0f} mm" if len(tower) else ""))

    # 4) plots
    win = (np.abs(P[:, 0]) < 200) & (np.abs(P[:, 1]) < 200) & (P[:, 2] > -50) & (P[:, 2] < 320)
    Q = P[win]
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for a, (i, j, ti) in zip(ax, [(0, 1, "top-down  x / y"), (0, 2, "front  x / z"), (1, 2, "side  y / z")]):
        a.scatter(Q[:, i], Q[:, j], c=Q[:, 2], s=3, cmap="viridis")
        a.set_title(ti); a.set_aspect("equal"); a.grid(alpha=.3)
        if (i, j) == (0, 1):
            a.add_patch(plt.Rectangle((-37.5, -37.5), 75, 75, fill=False, ec="r", lw=2))
            for mid, (cx, cy) in layout.items():
                a.add_patch(plt.Rectangle((cx-20, cy-20), 40, 40, fill=False, ec="k"))
                a.text(cx, cy, str(mid), ha="center", va="center")
        else:
            a.axhline(0, c="k", lw=1)
            ax_k = -37.5, 75
            a.add_patch(plt.Rectangle((-37.5, 0), 75, 270, fill=False, ec="r", lw=1.5))
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tower_check.png")
    plt.savefig(out, dpi=90)
    print("saved", out)


if __name__ == "__main__":
    main()
