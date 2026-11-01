#!/usr/bin/env python3
"""
Snap the drawn tower onto the real one in a camera's own picture.

The marker pose says where the tower SHOULD be.  The tower itself is in the picture: its outline (the vertical edges, the
top and the base) is a strong edge against the table.  So we start from the marker pose and move the tower frame a little
(a few mm, a couple of degrees) until the drawn outline lies on the strongest image edges.  The move is kept small by a
penalty, so a hand, a shadow or the seams between blocks cannot drag it far.

Used once per camera (at Re-detect and when the overlay is locked), not on every frame.
    delta = (R_d, t_d): the tower frame as this camera sees it:  X_marker = R_d X_tower + t_d
"""
import cv2
import numpy as np

import tower_points as tp

HALF = 37.5
LAYER_MM = 15.0
SIGMA_MM = 8.0       # how far the tower may move from where the markers put it before the penalty bites
SIGMA_DEG = 2.5


def rot(v):
    return cv2.Rodrigues(np.asarray(v, np.float64).reshape(3, 1))[0]


def box_edges(H):
    """The 12 edges of the tower box as (3-D start, 3-D end, normals of the two faces they join)."""
    c = [(-HALF, -HALF), (HALF, -HALF), (HALF, HALF), (-HALF, HALF)]
    e = []
    for i in range(4):                                         # vertical edges: between the two side faces that meet there
        (x, y), (x2, y2) = c[i], c[(i + 1) % 4]
        a = np.array([x, y, 0.0]); b = np.array([x, y, H])
        n1 = np.array([np.sign(x), 0.0, 0.0]); n2 = np.array([0.0, np.sign(y), 0.0])
        e.append((a, b, n1, n2))
    for z, nz in ((H, 1.0), (0.0, -1.0)):                      # top and base rectangles: between a side face and the top / bottom
        for i in range(4):
            (x, y), (x2, y2) = c[i], c[(i + 1) % 4]
            side = np.array([(x + x2) / (2 * HALF) if x == x2 else 0.0, (y + y2) / (2 * HALF) if y == y2 else 0.0, 0.0])
            e.append((np.array([x, y, z]), np.array([x2, y2, z]), side, np.array([0.0, 0.0, nz])))
    return e


def sample_edges(H, step=3.0):
    """Points along every edge, with the edge's two face normals: [(P (n,3), n1, n2, direction)]."""
    out = []
    for a, b, n1, n2 in box_edges(H):
        L = float(np.linalg.norm(b - a))
        k = max(2, int(L / step))
        P = a[None, :] + (b - a)[None, :] * np.linspace(0.04, 0.96, k)[:, None]
        out.append((P, n1, n2, (b - a) / L))
    return out


def edge_maps(img, sigma):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), sigma)
    return cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0, cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0


def _sample(m, uv):
    h, w = m.shape
    x = np.clip(uv[:, 0], 0, w - 1.001); y = np.clip(uv[:, 1], 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    return (m[y0, x0] * (1 - fx) * (1 - fy) + m[y0, x0 + 1] * fx * (1 - fy) + m[y0 + 1, x0] * (1 - fx) * fy + m[y0 + 1, x0 + 1] * fx * fy)


def score(params, gx, gy, edges, R, t, K, dist, cam, cap=40.0):
    """Mean edge strength (across the edge) under the visible edges of the tower moved by `params` = (dx, dy, dz, rx, ry, rz in degrees)."""
    Rd = rot(np.radians(params[3:6])); td = np.asarray(params[:3], float)
    Rm = R @ Rd
    tm = t + R @ td
    camc = -Rm.T @ tm
    tot, cnt = 0.0, 0
    h, w = gx.shape
    for P, n1, n2, d in edges:
        mid = P[len(P) // 2]
        # the edge is visible when at least one of its two faces looks at the camera
        if not ((n1 @ (camc - mid)) > 0 or (n2 @ (camc - mid)) > 0):
            continue
        X = P @ Rm.T + tm
        if (X[:, 2] <= 10).any():
            continue
        uv = np.stack([X[:, 0] / X[:, 2] * K[0, 0] + K[0, 2], X[:, 1] / X[:, 2] * K[1, 1] + K[1, 2]], axis=1)
        uv = tp.distort_px(uv, K, dist)
        if (uv[:, 0] < 2).any() or (uv[:, 1] < 2).any() or (uv[:, 0] > w - 3).any() or (uv[:, 1] > h - 3).any():
            continue
        dd = uv[-1] - uv[0]
        nrm = np.array([-dd[1], dd[0]]) / (np.linalg.norm(dd) + 1e-9)       # across the edge, in the picture
        s = np.abs(_sample(gx, uv) * nrm[0] + _sample(gy, uv) * nrm[1])
        tot += float(np.minimum(s, cap).sum()); cnt += len(s)
    return tot / cnt if cnt else 0.0


def refine(img, K, dist, R, t, layers, passes=(4.0, 2.0), sigma_mm=SIGMA_MM, sigma_deg=SIGMA_DEG, log=None):
    """(delta params, score before, score after).  `R, t` is the marker pose of the tower frame in this camera."""
    H = layers * LAYER_MM
    edges = sample_edges(H)
    x = np.zeros(6)
    pen = lambda p: 0.5 * ((np.sum(p[:3] ** 2) / sigma_mm ** 2) + (np.sum(p[3:] ** 2) / sigma_deg ** 2)) * 1.0
    best = None
    before = None
    for sig in (4.0, 2.0, 1.2):                                  # coarse to fine: wide, forgiving edges first
        gx, gy = edge_maps(img, sig)
        f = lambda p: score(p, gx, gy, edges, R, t, K, dist, "cam") - 1.5 * pen(p)
        if before is None:
            before = score(x, gx, gy, edges, R, t, K, dist, "cam")
        cur = f(x)
        steps = np.array([4.0, 4.0, 4.0, 1.2, 1.2, 1.2]) * (sig / 4.0 + 0.5)
        for _ in range(60):
            improved = False
            for j in range(6):
                for s in (+1, -1):
                    y = x.copy(); y[j] += s * steps[j]
                    v = f(y)
                    if v > cur + 1e-6:
                        x, cur, improved = y, v, True
            if not improved:
                steps *= 0.5
                if steps[0] < 0.15:
                    break
    gx, gy = edge_maps(img, 1.2)
    return x, before, score(x, gx, gy, edges, R, t, K, dist, "cam")


def apply_delta(R, t, params):
    """The pose of the moved tower frame in the camera."""
    Rd = rot(np.radians(params[3:6]))
    return R @ Rd, t + R @ np.asarray(params[:3], float)
