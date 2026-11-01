#!/usr/bin/env python3
"""
One-time calibration of the (permanently fixed) marker board, from many views of it.

The per-game marker measurement (calibrate_markers.py) places the four markers flat and from a few frames of one setup, and the
pose solver then fits that map to every picture with 3-4 px of leftover error.  Here every marker gets a full 3-D pose (so a
marker that is a little tilted or higher than its neighbours is allowed), solved together with the camera pose of every view
(bundle adjustment): the map that explains all the views from all the places the phones have stood.

    .venv/bin/python calibrate_board.py            # uses the saved frames since --since and the live pictures; writes marker_map.json
    .venv/bin/python calibrate_board.py --dry-run  # only report

Needs: iphone_saved/ frames (iPhone: *_rgb.jpg + *_intrinsics.json, Android: *_android.jpg), and optionally the live server.
"""
import argparse
import glob
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
import tower_points as tp

SAVE_DIR = os.path.join(HERE, "iphone_saved")
H = 20.0
OBJ = np.array([[-H, H, 0], [H, H, 0], [H, -H, 0], [-H, -H, 0]], np.float64)     # TL TR BR BL, marker frame
P = cv2.aruco.DetectorParameters()
P.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
DET = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), P)


def rot(v):
    return cv2.Rodrigues(np.asarray(v, np.float64).reshape(3, 1))[0]


def detect(img, K, dist):
    corners, ids, _ = DET.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    out = {}
    if ids is None:
        return out
    for c, i in zip(corners, ids.flatten()):
        if int(i) > 3:
            continue
        q = c.reshape(4, 2).astype(np.float64)
        if dist is not None and len(dist):
            q = cv2.undistortPoints(q.reshape(-1, 1, 2), K, dist, P=K).reshape(4, 2)
        out[int(i)] = q
    return out


def frames(since_ms, every, live):
    """Yield (name, cam, K, {id: corners}) for sampled saved frames and the live pictures."""
    got = []
    ip = sorted(glob.glob(f"{SAVE_DIR}/*_rgb.jpg"))
    ip = [f for f in ip if int(os.path.basename(f).split("_")[0]) >= since_ms][::every]
    for f in ip:
        ts = os.path.basename(f).split("_")[0]
        try:
            meta = json.load(open(f"{SAVE_DIR}/{ts}_intrinsics.json"))
        except Exception:
            continue
        img = cv2.imread(f)
        if img is None:
            continue
        K = fc.iphone_K(meta, img.shape[1])
        d = detect(img, K, None)
        if len(d) >= 3:
            got.append((f"iphone {ts}", "iphone", K, d))
    an = sorted(glob.glob(f"{SAVE_DIR}/*_android.jpg"))
    an = [f for f in an if int(os.path.basename(f).split("_")[0]) >= since_ms][::max(1, every // 3)]
    for f in an:
        img = cv2.imread(f)
        if img is None:
            continue
        K, dist = fc.android_K(img.shape[1], img.shape[0])
        if K is None:
            continue
        d = detect(img, K, dist)
        if len(d) >= 3:
            got.append((f"android {os.path.basename(f).split('_')[0]}", "android", K, d))
    if live:
        for cam, path in (("iphone", "/preview_raw.jpg"), ("android", "/android_raw.jpg")):
            for k in range(6):
                try:
                    raw = urllib.request.urlopen("http://127.0.0.1:8000" + path, timeout=3).read()
                except Exception:
                    break
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    break
                if cam == "iphone":
                    K, dist = np.array([[1399.0082 * img.shape[1] / 1920, 0, 964.9536 * img.shape[1] / 1920],
                                        [0, 1399.0082 * img.shape[1] / 1920, 717.8016 * img.shape[1] / 1920], [0, 0, 1.0]]), None
                else:
                    K, dist = fc.android_K(img.shape[1], img.shape[0])
                d = detect(img, K, dist)
                if len(d) >= 3:
                    got.append((f"{cam} live{k}", cam, K, d))
                time.sleep(0.2)
    return got


def pick_diverse(views, sep_mm=25.0, sep_deg=3.0, cap=40):
    """Keep views whose camera sits somewhere new (or looks another way): the same view a hundred times adds nothing."""
    keep = {"iphone": [], "android": []}
    for v in views:
        name, cam, K, d, R, t, err = v
        C = -R.T @ t
        fwd = R[2]
        ok = True
        for _, _, _, _, R2, t2, _ in keep[cam]:
            C2 = -R2.T @ t2
            ang = np.degrees(np.arccos(np.clip(fwd @ R2[2], -1, 1)))
            if np.linalg.norm(C - C2) < sep_mm and ang < sep_deg:
                ok = False
                break
        if ok and len(keep[cam]) < cap:
            keep[cam].append(v)
    return keep["iphone"] + keep["android"]


class BA:
    """Marker poses (marker -> tower frame) and view poses (tower frame -> camera), from the corner pictures."""

    def __init__(self, views, markers0):
        self.views = views
        self.nv = len(views)
        self.p = np.zeros(24 + 6 * self.nv)
        for m in range(4):
            r, t = markers0[m]
            self.p[6 * m:6 * m + 3], self.p[6 * m + 3:6 * m + 6] = r, t
        for k, (name, cam, K, d, R, t, err) in enumerate(views):
            self.p[24 + 6 * k:24 + 6 * k + 3] = cv2.Rodrigues(R)[0].ravel()
            self.p[24 + 6 * k + 3:24 + 6 * k + 6] = t

    def _marker_corners(self, p, m):
        return OBJ @ rot(p[6 * m:6 * m + 3]).T + p[6 * m + 3:6 * m + 6]

    def view_res(self, p, k, mc=None):
        name, cam, K, d, *_ = self.views[k]
        R, t = rot(p[24 + 6 * k:24 + 6 * k + 3]), p[24 + 6 * k + 3:24 + 6 * k + 6]
        out = []
        for m, q in d.items():
            X = (mc[m] if mc is not None else self._marker_corners(p, m)) @ R.T + t
            u = X[:, 0] / X[:, 2] * K[0, 0] + K[0, 2]
            v = X[:, 1] / X[:, 2] * K[1, 1] + K[1, 2]
            out.append(np.stack([u - q[:, 0], v - q[:, 1]], axis=1).ravel())
        return np.concatenate(out)

    def residuals(self, p):
        mc = {m: self._marker_corners(p, m) for m in range(4)}
        return np.concatenate([self.view_res(p, k, mc) for k in range(self.nv)])

    def solve(self, iters=40, log=print):
        p = self.p
        lam = 1e-3
        sizes = [len(self.view_res(p, k)) for k in range(self.nv)]
        offs = np.concatenate([[0], np.cumsum(sizes)])
        r = self.residuals(p)
        for it in range(iters):
            w = np.where(np.abs(r) < 2.0, 1.0, 2.0 / np.abs(r))            # Huber: a bad corner does not drag the map
            J = np.zeros((len(r), len(p)))
            for j in range(24):                                              # marker parameters touch every view
                d = np.zeros(len(p)); d[j] = 1e-4 if j % 6 < 3 else 1e-2
                J[:, j] = (self.residuals(p + d) - r) / d[j]
            for k in range(self.nv):                                         # a view's own parameters touch only that view
                for j in range(6):
                    jj = 24 + 6 * k + j
                    d = np.zeros(len(p)); d[jj] = 1e-4 if j < 3 else 1e-2
                    J[offs[k]:offs[k + 1], jj] = (self.view_res(p + d, k) - r[offs[k]:offs[k + 1]]) / d[jj]
            A = J.T @ (J * w[:, None])
            g = J.T @ (w * r)
            for _ in range(8):
                step = np.linalg.solve(A + lam * (np.diag(np.diag(A)) + 1e-6 * np.eye(len(p))), -g)
                r2 = self.residuals(p + step)
                if np.sum(np.minimum(r2 ** 2, 4 * np.abs(r2))) < np.sum(np.minimum(r ** 2, 4 * np.abs(r))):
                    p = p + step; r = r2; lam = max(lam / 3, 1e-9)
                    break
                lam *= 4
            else:
                break
            if np.linalg.norm(step) < 1e-6:
                break
        self.p = p
        return float(np.sqrt(np.mean(r ** 2))), r


def to_tower_frame(ba, views):
    """Re-express everything in the frame calibrate_markers.py uses: origin = ring middle, z up (toward the cameras),
    +x along marker 1 -> 2 (and 0 -> 3)."""
    p = ba.p
    mc = {m: ba._marker_corners(p, m) for m in range(4)}
    ctr = np.array([mc[m].mean(axis=0) for m in range(4)])
    o = ctr.mean(axis=0)
    n = np.linalg.svd(ctr - o)[2][-1]
    cams = [-rot(p[24 + 6 * k:24 + 6 * k + 3]).T @ p[24 + 6 * k + 3:24 + 6 * k + 6] for k in range(ba.nv)]
    if (np.mean(cams, axis=0) - o) @ n < 0:
        n = -n
    xv = (ctr[2] - ctr[1]) + (ctr[3] - ctr[0])
    xv = xv - (xv @ n) * n
    xv /= np.linalg.norm(xv)
    yv = np.cross(n, xv)
    if (ctr[0] - ctr[1]) @ yv < 0:
        sys.exit("the ring runs the other way round from the one this software assumes (IDs 1 -> 2 -> 3 -> 0 counter-clockwise "
                 "seen from above)")
    Rf = np.stack([xv, yv, n])                        # rows: new axes
    corners = {m: (mc[m] - o) @ Rf.T for m in range(4)}
    return corners, Rf, o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-10-31 12:30", help="use saved frames from this local time on (the board has not changed since)")
    ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--no-live", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    since_ms = int(time.mktime(time.strptime(args.since, "%Y-%m-%d %H:%M")) * 1000)

    print("collecting views ...")
    raw = frames(since_ms, args.every, not args.no_live)
    print(f"  {len(raw)} pictures with 3+ markers")
    tp.reset_correction()
    scored = []
    for name, cam, K, d in raw:
        s = tp.solve(d, K, layout=tp.LAYOUTS["A"], corrected=False)
        if s and s[2] < 12:
            scored.append((name, cam, K, d, s[0], s[1], s[2]))
    print(f"  {len(scored)} fitted with the current map; reprojection error with the current map: "
          + ", ".join(f"{c} {np.mean([v[6] for v in scored if v[1] == c]):.2f} px (n={sum(1 for v in scored if v[1] == c)})" for c in ('iphone', 'android') if any(v[1] == c for v in scored)))
    views = pick_diverse(scored)
    print(f"  {len(views)} distinct viewpoints kept ({sum(1 for v in views if v[1] == 'iphone')} iPhone, {sum(1 for v in views if v[1] == 'android')} Android)")
    if len(views) < 8:
        sys.exit("need at least 8 distinct viewpoints; play a few games / move the phones a little, then run again")

    # held-out views: test the map on pictures it was not fitted to
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(views))
    held = [views[i] for i in idx[: max(3, len(views) // 5)]]
    fit = [views[i] for i in idx[max(3, len(views) // 5):]]

    markers0 = []
    for m in range(4):
        c = tp.marker_corners_3d(m, 0, tp.LAYOUTS["A"])
        ok, rv, tv = cv2.solvePnP(OBJ, c.astype(np.float64), np.eye(3), None, flags=cv2.SOLVEPNP_IPPE_SQUARE) if False else (True, None, None)
        # marker -> tower: the marker's own frame is x right, y up, z out; its corners in the tower frame are c
        A = OBJ - OBJ.mean(axis=0); B = c - c.mean(axis=0)
        U, _, Vt = np.linalg.svd(A.T @ B); Rm = (U @ Vt).T
        if np.linalg.det(Rm) < 0:
            Vt[-1] *= -1; Rm = (U @ Vt).T
        markers0.append((cv2.Rodrigues(Rm)[0].ravel(), c.mean(axis=0) - Rm @ OBJ.mean(axis=0)))
    ba = BA(fit, markers0)
    r0 = float(np.sqrt(np.mean(ba.residuals(ba.p) ** 2)))
    rms, res = ba.solve()
    print(f"bundle adjustment: reprojection RMS {r0:.2f} px with the old map -> {rms:.2f} px with the calibrated board")
    corners, Rf, o = to_tower_frame(ba, fit)
    print("marker positions in the tower frame (mm):")
    for m in range(4):
        c = corners[m].mean(axis=0)
        nrm = np.cross(corners[m][1] - corners[m][0], corners[m][3] - corners[m][0]); nrm /= np.linalg.norm(nrm)
        print(f"  ID{m}: x {c[0]:+7.1f}  y {c[1]:+7.1f}  z {c[2]:+5.2f}   tilt {np.degrees(np.arccos(np.clip(abs(nrm[2]), 0, 1))):.2f} deg")

    # the held-out test: solve each unseen view against the OLD map and against the NEW one
    def err_with(corner_map, v):
        name, cam, K, d, R, t, e = v
        obj = np.concatenate([corner_map[m] for m in d]); img = np.concatenate([d[m] for m in d])
        ok, rv, tv = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        pr, _ = cv2.projectPoints(obj, rv, tv, K, None)
        return float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - img) ** 2, axis=1)))), rv, tv
    old = {m: tp.marker_corners_3d(m, 0, tp.LAYOUTS["A"]) for m in range(4)}
    for cam in ("iphone", "android"):
        hv = [v for v in held if v[1] == cam]
        if hv:
            eo = [err_with(old, v)[0] for v in hv]; en = [err_with(corners, v)[0] for v in hv]
            print(f"held-out {cam} views (n={len(hv)}): corner error {np.mean(eo):.2f} px with the old map -> {np.mean(en):.2f} px with the new one")
    if args.dry_run:
        print("(dry run, nothing written)")
        return
    # the map that is kept uses every viewpoint (the held-out ones included)
    mk = [(ba.p[6 * m:6 * m + 3].copy(), ba.p[6 * m + 3:6 * m + 6].copy()) for m in range(4)]
    ba2 = BA(views, mk)
    rms, res = ba2.solve()
    corners, Rf, o = to_tower_frame(ba2, views)
    fit = views
    print(f"final calibration on all {len(views)} viewpoints: reprojection RMS {rms:.2f} px")
    path = os.path.join(HERE, "marker_map.json")
    j = json.load(open(path)) if os.path.exists(path) else {}
    os.replace(path, path.replace(".json", ".before-board-calibration.json")) if os.path.exists(path) else None
    legacy = {}
    for m in range(4):
        c = corners[m].mean(axis=0)
        ax = corners[m][1] - corners[m][0]
        legacy[str(m)] = {"x": round(float(c[0]), 2), "y": round(float(c[1]), 2),
                          "yaw": round(float(np.degrees(np.arctan2(ax[1], ax[0]))), 2)}
    j.update({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "marker_mm": 40.0, "markers": legacy,
              "corners": {str(m): np.round(corners[m], 3).tolist() for m in range(4)},
              "fixed": True, "calibrated_views": len(fit), "reproj_rms_px": round(rms, 3)})
    json.dump(j, open(path, "w"), indent=2)
    print("wrote marker_map.json (fixed board). Restart the server to use it.")


if __name__ == "__main__":
    main()
