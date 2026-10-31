#!/usr/bin/env python3
"""
Work out the tower's height (layer count) and which way its top blocks run, from the camera pictures alone.

HEIGHT: look straight down on the top layer through each camera, as if the top surface were at height z.
        Only at the true height do two cameras on different sides see the *same* picture there (any other z
        shifts the two views apart, in opposite directions).  So the z that makes them agree is the top.
        layers = z / 15 mm.
DIRECTION: in that straight-down picture the seams between blocks (and the wood grain) are edges running along
        the blocks' long side, so the direction with the weaker "across" edges is the blocks' length.

Used by the viewer server (automatically) and runnable alone for a check:
    .venv/bin/python tower_auto.py
"""
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import face_crops as fc

LAYER_MM = 15.0


def camera_K(name, img):
    h, w = img.shape[:2]
    if name == "iphone":
        s = w / 1920.0
        return np.array([[1399.0082 * s, 0, 964.9536 * s], [0, 1399.0082 * s, 717.8016 * s], [0, 0, 1.0]]), None
    return fc.android_K(w, h)


def rectify(img, K, R, t, z, half, px):
    """The plane at height z, seen from straight above, sampled at `px` pixels per mm (img must be undistorted)."""
    n = int(2 * half * px)
    xs = np.linspace(-half, half, n)
    ys = np.linspace(half, -half, n)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, float(z))], 1)
    uvw = (pts @ R.T + t) @ K.T
    return cv2.remap(img, (uvw[:, 0] / uvw[:, 2]).reshape(X.shape).astype(np.float32),
                     (uvw[:, 1] / uvw[:, 2]).reshape(X.shape).astype(np.float32), cv2.INTER_LINEAR)


def _grey_hp(rect):
    # plain brightness (lightly blurred): the bright wood top against the dark floor is a far sharper cue than
    # fine grain, which the two cameras (different sides, lenses, lighting) do not reproduce identically
    return cv2.GaussianBlur(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)


def _ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-6 else 0.0


def prepare(views):
    """views: {'iphone': bgr, 'android': bgr} -> {name: (undistorted img, K, R, t, reproj)} for those with a pose."""
    out = {}
    for name, img in views.items():
        if img is None:
            continue
        K, dist = camera_K(name, img)
        if K is None:
            continue
        sol = fc.solve_pose(img, K, dist)
        if sol is None:
            continue
        und = cv2.undistort(img, K, dist) if (dist is not None and len(dist)) else img
        out[name] = (und, K, sol[0], sol[1], sol[2])
    return out


def height(prep, z_lo=60.0, z_hi=300.0, step=3.0, half=35.0, px=3):
    """Top-surface height by two-camera agreement.  Returns (z_mm or None, peak ncc, curve)."""
    if len(prep) < 2:
        return None, None, []
    (n1, a), (n2, b) = list(prep.items())[:2]
    curve = []
    for z in np.arange(z_lo, z_hi + 0.1, step):
        ra = _grey_hp(rectify(a[0], a[1], a[2], a[3], z, half, px))
        rb = _grey_hp(rectify(b[0], b[1], b[2], b[3], z, half, px))
        curve.append((float(z), _ncc(ra, rb)))
    zs, cs = np.array([c[0] for c in curve]), np.array([c[1] for c in curve])
    i = int(np.argmax(cs))
    # peak must stand clear of the rest of the curve, otherwise the pictures simply do not match
    rest = np.delete(cs, slice(max(0, i - 4), i + 5))
    if cs[i] < 0.30 or (len(rest) and cs[i] - np.median(rest) < 0.25):
        return None, float(cs[i]), curve
    return float(zs[i]), float(cs[i]), curve


def direction(prep, z, half=30.0, px=5, inner=26.0):
    """'x' or 'y': the way the top layer's blocks run in the current tower frame, and the evidence."""
    ratios = []
    for name, (img, K, R, t, _e) in prep.items():
        r = rectify(img, K, R, t, z, half, px)
        g = cv2.GaussianBlur(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.2)
        c, w = g.shape[0] // 2, int(inner * px)
        g = g[c - w:c + w, c - w:c + w]
        across_x = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)).mean()    # edges running along y
        across_y = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)).mean()    # edges running along x
        ratios.append(across_y / max(across_x, 1e-6))
    if not ratios:
        return None, None
    r = float(np.exp(np.mean(np.log(ratios))))               # geometric mean over cameras
    return ("x" if r > 1 else "y"), r


def analyse(views, layers_hint=None):
    prep = prepare(views)
    res = {"cams": sorted(prep), "layers": None, "z": None, "ncc": None, "axis": None, "ratio": None}
    z, ncc, _ = height(prep)
    res["ncc"] = ncc
    if z is not None:
        res["z"] = z
        res["layers"] = int(round(z / LAYER_MM))
    z_dir = z if z is not None else (layers_hint * LAYER_MM if layers_hint else None)
    if z_dir:
        res["axis"], res["ratio"] = direction(prep, z_dir)
    return res


def analyse_burst(grab, n=8, interval=0.3, layers_hint=None):
    """Many frame pairs, averaged.  The phones sit on stands, so each camera's best marker fit is reused for
    the frames where a marker drops out; and the height evidence is summed over the burst instead of judged
    from one frame.  grab() -> {'iphone': bgr|None, 'android': bgr|None}.  Takes about n * interval seconds."""
    best = {}                                  # name -> (img_unused, K, R, t, reproj)
    curves, ratios_at, frames_used = [], [], 0
    zs = None
    it = 0
    while it < n or (frames_used < 4 and it < 2 * n):      # keep going until there are enough good pairs
        it += 1
        views = grab()
        for name, img in views.items():
            if img is None:
                continue
            K, dist = camera_K(name, img)
            if K is None:
                continue
            sol = fc.solve_pose(img, K, dist)
            if sol is not None and (name not in best or sol[2] < best[name][4]):
                best[name] = (None, K, sol[0], sol[1], sol[2])
        prep = {}
        for name, img in views.items():
            if img is None or name not in best:
                continue
            K, dist = camera_K(name, img)
            und = cv2.undistort(img, K, dist) if (dist is not None and len(dist)) else img
            prep[name] = (und, best[name][1], best[name][2], best[name][3], best[name][4])
        if len(prep) >= 2:
            (n1, a), (n2, b) = list(prep.items())[:2]
            grid = np.arange(60.0, 300.1, 3.0)
            cur = []
            for z in grid:
                ra = _grey_hp(rectify(a[0], a[1], a[2], a[3], z, 35.0, 3))
                rb = _grey_hp(rectify(b[0], b[1], b[2], b[3], z, 35.0, 3))
                cur.append(_ncc(ra, rb))
            curves.append(cur)
            zs = grid
            frames_used += 1
        time.sleep(interval)
    res = {"cams": sorted(best), "pairs": frames_used, "layers": None, "z": None, "ncc": None,
           "axis": None, "ratio": None, "ok": False, "reason": None}
    if layers_hint:
        # the height is already known (the iPhone's LiDAR): only which way the top blocks run is left
        z_top = layers_hint * LAYER_MM
        res["layers"], res["z"] = int(layers_hint), z_top
        views = grab()
        prep = {nm: (cv2.undistort(img, best[nm][1], camera_K(nm, img)[1]) if camera_K(nm, img)[1] is not None and len(camera_K(nm, img)[1]) else img,
                     best[nm][1], best[nm][2], best[nm][3], best[nm][4])
                for nm, img in views.items() if img is not None and nm in best}
        res["axis"], res["ratio"] = direction(prep, z_top)
        res["ok"] = res["axis"] is not None
        if not res["ok"]:
            res["reason"] = "no camera found 3+ markers"
        return res
    if frames_used < 3:
        missing = [c for c in ("iphone", "android") if c not in best]
        res["reason"] = (f"{'+'.join(missing)} never found 3+ markers" if missing
                         else f"only {frames_used} usable frame pairs")
        return res
    avg = np.mean(curves, axis=0)
    # The top surface is where agreement falls off a cliff as z rises: above the tower the two cameras see
    # different things.  (Below it there is a broad, slowly falling hump -- just the inside of the tower, where
    # both cameras see bright wood against the dark floor -- so the highest value is NOT the top.)
    drop = avg[:-1] - avg[1:]
    lo = int(np.searchsorted(zs, 45.0))
    cand = [j for j in range(lo, len(drop)) if avg[j] >= 0.10 and drop[j] >= 0.18]
    if not cand:
        res["reason"] = (f"no clear top edge in the two-camera agreement (steepest drop {drop[lo:].max():.2f}); "
                         "on a very tall tower, type the layer count by hand")
        res["ncc"] = float(avg.max())
        return res
    i = max(cand, key=lambda j: drop[j])
    z_top = float(zs[i] + 1.5)
    res["ncc"], res["z"] = float(avg[i]), z_top
    zs = zs.copy(); zs[i] = res["z"]
    res["layers"] = int(round(zs[i] / LAYER_MM))
    # direction, from the last frames at the found height
    views = grab()
    prep = {nm: (cv2.undistort(img, best[nm][1], camera_K(nm, img)[1]) if camera_K(nm, img)[1] is not None and len(camera_K(nm, img)[1]) else img,
                 best[nm][1], best[nm][2], best[nm][3], best[nm][4])
            for nm, img in views.items() if img is not None and nm in best}
    res["axis"], res["ratio"] = direction(prep, float(zs[i]))
    res["ok"] = res["axis"] is not None
    return res


def _grab(path):
    try:
        raw = urllib.request.urlopen("http://127.0.0.1:8000" + path, timeout=5).read()
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


if __name__ == "__main__":
    for k in range(5):
        r = analyse({"iphone": _grab("/preview_raw.jpg"), "android": _grab("/android_raw.jpg")}, layers_hint=10)
        print(f"run {k + 1}: cameras {r['cams']}  top surface z = {None if r['z'] is None else round(r['z'])} mm "
              f"(agreement {None if r['ncc'] is None else round(r['ncc'], 2)})  -> layers {r['layers']}  | "
              f"top blocks run along {r['axis']} (evidence x{None if r['ratio'] is None else round(r['ratio'], 2)})")
        time.sleep(0.6)
