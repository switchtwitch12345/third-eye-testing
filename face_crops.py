#!/usr/bin/env python3
"""
Cut every visible block end-face out of every frame, rectified straight-on, with its identity.

    crop = the face (25 x 15 mm, +25% context) warped to a 100 x 60 picture, whatever the camera angle
    index.jsonl rows: cam, ts_ms, t_sec, layer, slot, sign, cos (viewing angle), pose_src, reproj, file

This is the dataset the unlabelled / few-shot model learns from.  It is the same code whether run
offline over recorded folders (extract_crops.py) or live while a game is played (server).
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

HALF, WIDTH, HEIGHT = 37.5, 25.0, 15.0
OUT_W, OUT_H = 100, 60
MARGIN = 0.25
MAX_LAYERS = 12
MIN_COS = 0.20            # skip faces seen almost edge-on
DET = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))


def face_geometry(layer, slot, sign):
    """Corners (TL,TR,BR,BL as seen from outside), centre and outward normal, tower frame, mm."""
    z = HEIGHT * (layer + 0.5)
    a = -HALF + WIDTH * (slot + 0.5)
    if layer % 2 == 0:                     # runs along x: ends at x = +-HALF, slots across y
        c, n = np.array([sign * HALF, a, z]), np.array([float(sign), 0.0, 0.0])
    else:                                  # runs along y
        c, n = np.array([a, sign * HALF, z]), np.array([0.0, float(sign), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    r = np.cross(-n, up)                   # viewer's right-hand direction
    w, h = WIDTH * (1 + 2 * MARGIN) / 2, HEIGHT * (1 + 2 * MARGIN) / 2
    corners = np.array([c - r * w + up * h, c + r * w + up * h, c + r * w - up * h, c - r * w - up * h])
    return corners, c, n


DST = np.float32([[0, 0], [OUT_W - 1, 0], [OUT_W - 1, OUT_H - 1], [0, OUT_H - 1]])

# LONG SIDE faces (75 x 15 mm).  Only the two OUTER blocks of a layer show one on the tower's surface (the middle
# block is covered by its neighbours).  A layer's block ENDS sit on one pair of tower faces and its long sides on the
# other pair, so a phone that sees a layer's ends edge-on usually sees its outer blocks' sides squarely.
SIDE_W, SIDE_H = 200, 40
DST_SIDE = np.float32([[0, 0], [SIDE_W - 1, 0], [SIDE_W - 1, SIDE_H - 1], [0, SIDE_H - 1]])


def side_geometry(layer, slot, sign):
    """Long side face of an outer block.  slot 0 faces the negative side, slot 2 the positive side.
    Returns (corners, centre, outward normal) like face_geometry, or None for the middle slot."""
    if slot not in (0, 2):
        return None
    z = HEIGHT * (layer + 0.5)
    side = -1.0 if slot == 0 else 1.0
    if sign != side:
        return None
    if layer % 2 == 0:                       # runs along x: slots across y, sides at y = +-HALF
        c, n = np.array([0.0, side * HALF, z]), np.array([0.0, side, 0.0])
    else:                                    # runs along y: slots across x, sides at x = +-HALF
        c, n = np.array([side * HALF, 0.0, z]), np.array([side, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    r = np.cross(-n, up)
    w, h = 75.0 * (1 + 2 * MARGIN) / 2, HEIGHT * (1 + 2 * MARGIN) / 2
    return np.array([c - r * w + up * h, c + r * w + up * h, c + r * w - up * h, c - r * w - up * h]), c, n


def crops_for_frame(img, K, R, t, layers=MAX_LAYERS):
    """-> list of (layer, slot, sign, cos, crop_bgr) for faces the camera sees properly."""
    h, w = img.shape[:2]
    cam = -R.T @ t
    out = []
    for layer in range(layers):
        for slot in range(3):
            for sign in (1, -1):
                corners, c, n = face_geometry(layer, slot, sign)
                to_cam = cam - c
                cos = float(n @ to_cam) / (np.linalg.norm(to_cam) + 1e-9)
                if cos < MIN_COS:
                    continue
                uvw = (corners @ R.T + t) @ K.T
                if (uvw[:, 2] <= 0).any():
                    continue
                uv = (uvw[:, :2] / uvw[:, 2:3]).astype(np.float32)
                if uv[:, 0].min() < 0 or uv[:, 1].min() < 0 or uv[:, 0].max() >= w or uv[:, 1].max() >= h:
                    continue
                M = cv2.getPerspectiveTransform(uv, DST)
                out.append((layer, slot, sign, cos, cv2.warpPerspective(img, M, (OUT_W, OUT_H), flags=cv2.INTER_AREA)))
    return out


def side_crops_for_frame(img, K, R, t, layers=MAX_LAYERS):
    """-> list of (layer, slot, sign, cos, crop_bgr) for outer blocks' long side faces seen well enough."""
    h, w = img.shape[:2]
    cam = -R.T @ t
    out = []
    for layer in range(layers):
        for slot in (0, 2):
            sign = -1 if slot == 0 else 1
            g = side_geometry(layer, slot, sign)
            if g is None:
                continue
            corners, c, n = g
            to_cam = cam - c
            cos = float(n @ to_cam) / (np.linalg.norm(to_cam) + 1e-9)
            if cos < MIN_COS:
                continue
            uvw = (corners @ R.T + t) @ K.T
            if (uvw[:, 2] <= 0).any():
                continue
            uv = (uvw[:, :2] / uvw[:, 2:3]).astype(np.float32)
            if uv[:, 0].min() < 0 or uv[:, 1].min() < 0 or uv[:, 0].max() >= w or uv[:, 1].max() >= h:
                continue
            M = cv2.getPerspectiveTransform(uv, DST_SIDE)
            out.append((layer, slot, sign, cos, cv2.warpPerspective(img, M, (SIDE_W, SIDE_H), flags=cv2.INTER_AREA)))
    return out


def solve_pose(img, K, dist=None):
    """(R, t, reproj) from the markers, or None.  Needs 3+ markers."""
    import tower_points as tp
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    corners, ids, _ = DET.detectMarkers(g)
    if ids is None:
        return None
    d = {int(i): c.reshape(4, 2).astype(np.float64) for c, i in zip(corners, ids.flatten()) if int(i) < 4}
    if dist is not None and len(dist):
        d = {i: cv2.undistortPoints(c.reshape(-1, 1, 2), K, dist, P=K).reshape(4, 2) for i, c in d.items()}
    s = tp.solve(d, K, layout=tp.LAYOUTS["A"])
    return s if (s and s[2] <= 7.0) else None


def android_K(w, h, calib_path=None):
    """Lens for this Android stream shape, derived from the calibration (handles a 90 degree turn)."""
    p = calib_path or os.path.join(HERE, "camera_android.json")
    if not os.path.exists(p):
        return None, None
    c = json.load(open(p))
    fx, fy, cx, cy, W, H = c["fx"], c["fy"], c["cx"], c["cy"], c["width"], c["height"]
    dist = np.array(c.get("dist") or [], np.float64)
    if abs(w * H - h * W) > 0.01 * w * H and abs(w * W - h * H) < 0.01 * w * H:
        fx, fy, cx, cy, W, H = fy, fx, H - cy, cx, H, W
        if len(dist) >= 5:
            dist = np.array([dist[0], dist[1], 0.0, 0.0, dist[4]])
    s = w / W
    if abs(h - H * s) > 4:
        return None, None
    return np.array([[fx * s, 0, cx * s], [0, fy * s, cy * s], [0, 0, 1.0]]), dist


# iPhone intrinsics ship with every frame (<ts>_intrinsics.json)
def iphone_K(meta, w):
    s = w / meta["rgb_width"]
    return np.array([[meta["fx"] * s, 0, meta["cx"] * s], [0, meta["fy"] * s, meta["cy"] * s], [0, 0, 1.0]])


class DatasetWriter:
    """Incrementally turns a capture folder into crops + index.jsonl.  Safe to call step() repeatedly."""

    def __init__(self, src, out, stride=2, since_ms=0, until_ms=None, jpeg_quality=88):
        self.src, self.out, self.stride = src, out, max(1, stride)
        self.since, self.until, self.q = since_ms, until_ms, jpeg_quality
        self.done = set()
        self.last_pose = {"iphone": None, "android": None}
        self.counts = {"frames": 0, "crops": 0, "skipped_no_pose": 0}
        self.t0 = None
        os.makedirs(out, exist_ok=True)
        self.index = open(os.path.join(out, "index.jsonl"), "a")

    def skip_existing(self):
        """Live mode: only frames that arrive from now on."""
        for f in glob.glob(f"{self.src}/*_rgb.jpg"):
            self.done.add(os.path.basename(f).split("_")[0])
        for f in glob.glob(f"{self.src}/*_android.jpg"):
            self.done.add("a" + os.path.basename(f).split("_")[0])

    def _write(self, cam, ts, img, K, pose, src):
        R, t, err = pose
        for layer, slot, sign, cos, crop in crops_for_frame(img, K, R, t):
            fn = f"{cam}/{ts}_L{layer}_S{slot}_{'p' if sign > 0 else 'm'}.jpg"
            path = os.path.join(self.out, fn)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, self.q])
            self.index.write(json.dumps({"cam": cam, "ts_ms": int(ts), "t_sec": round((int(ts) - self.t0) / 1000, 2),
                                         "layer": layer, "slot": slot, "sign": sign, "cos": round(cos, 3),
                                         "pose_src": src, "reproj": round(err, 2), "file": fn}) + "\n")
            self.counts["crops"] += 1
        for layer, slot, sign, cos, crop in side_crops_for_frame(img, K, R, t):
            fn = f"{cam}/{ts}_L{layer}_S{slot}_{'p' if sign > 0 else 'm'}_side.jpg"
            path = os.path.join(self.out, fn)
            cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, self.q])
            self.index.write(json.dumps({"cam": cam, "ts_ms": int(ts), "t_sec": round((int(ts) - self.t0) / 1000, 2),
                                         "layer": layer, "slot": slot, "sign": sign, "cos": round(cos, 3), "kind": "side",
                                         "pose_src": src, "reproj": round(err, 2), "file": fn}) + "\n")
            self.counts["crops"] += 1
        self.counts["frames"] += 1

    def _pose(self, cam, img, K, dist):
        s = solve_pose(img, K, dist)
        if s:
            self.last_pose[cam] = s
            return s, "marker"
        if self.last_pose[cam] is not None:
            return self.last_pose[cam], "carried"
        return None, None

    def step(self, limit=None):
        """Process new complete frames.  Returns number processed."""
        n = 0
        stamps = sorted(os.path.basename(f).split("_")[0] for f in glob.glob(f"{self.src}/*_rgb.jpg"))
        for i, ts in enumerate(stamps):
            ms = int(ts)
            if ts in self.done or ms < self.since or (self.until and ms > self.until):
                continue
            if self.t0 is None:
                self.t0 = ms
            if i % self.stride:
                self.done.add(ts)
                continue
            mp = f"{self.src}/{ts}_intrinsics.json"
            if not os.path.exists(mp):
                continue
            img = cv2.imread(f"{self.src}/{ts}_rgb.jpg")
            if img is None:
                continue
            try:
                meta = json.load(open(mp))
            except Exception:
                continue
            K = iphone_K(meta, img.shape[1])
            pose, src = self._pose("iphone", img, K, None)
            self.done.add(ts)
            if pose is None:
                self.counts["skipped_no_pose"] += 1
                continue
            self._write("iphone", ts, img, K, pose, src)
            n += 1
            if limit and n >= limit:
                break
        for f in sorted(glob.glob(f"{self.src}/*_android.jpg")):
            ts = os.path.basename(f).split("_")[0]
            ms = int(ts)
            key = "a" + ts
            if key in self.done or ms < self.since or (self.until and ms > self.until):
                continue
            self.done.add(key)
            if self.t0 is None:
                self.t0 = ms
            if len(self.done) % self.stride:
                continue
            img = cv2.imread(f)
            if img is None:
                continue
            K, dist = android_K(img.shape[1], img.shape[0])
            if K is None:
                continue
            pose, src = self._pose("android", img, K, dist)
            if pose is None:
                self.counts["skipped_no_pose"] += 1
                continue
            self._write("android", ts, img, K, pose, src)
            n += 1
        self.index.flush()
        return n

    def close(self):
        self.index.close()
