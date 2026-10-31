#!/usr/bin/env python3
"""
Where is the tower REALLY, and how tall?  Measured from the LiDAR points, not assumed from the marker layout.

The overlay used to be drawn around the middle of the marker square, with a fixed footprint and a layer count typed
by hand.  A different board, a tower standing a few centimetres off-centre, or a marker layout that does not quite
match the printed sheet all made it drift away from the real tower (on one board it was 38 mm off, with the whole
frame tilted).  The depth points know better: we line the model of the tower -- a 75 x 75 mm box of height H, on a
flat board -- up with them.

Seven numbers are fitted: a shift (x, y, z), a tilt/twist (rx, ry, rz) and the height H.  The residual of a point is
its distance to the box surface (the tower's sides and top) or to the board plane (the ring of board around it).  A
robust loss, tightened step by step, lets the hand or a fallen block be ignored.

Everything is in the tower frame in mm (x right, y away, z up), for points already put there by the marker pose; the
result is a CORRECTION to that frame (see tower_points.set_correction): x_current = R x_fitted + t.
"""
import numpy as np

HALF = 37.5                      # the tower is 75 x 75 mm
LAYER_MM = 15.0
TOP_OFFSET = 0.0                 # checked on five recorded towers (7, 9, 9, 16, 16 layers): the fitted height is n * 15 to within 0.4 layers


def rot(v):
    """Rotation matrix for the rotation vector v (Rodrigues)."""
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(v) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def log(R):
    """Rotation vector of a rotation matrix."""
    c = (np.trace(R) - 1.0) / 2.0
    th = float(np.arccos(np.clip(c, -1.0, 1.0)))
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


def box_sdf(Q, H):
    """Signed distance of points to the box [-HALF, HALF]^2 x [0, H] (negative inside)."""
    d = np.abs(Q - np.array([0.0, 0.0, H / 2.0])) - np.array([HALF, HALF, H / 2.0])
    return np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0)


def _split(P, window):
    r = np.maximum(np.abs(P[:, 0]), np.abs(P[:, 1]))
    near = P[(r < window) & (P[:, 2] > 12.0) & (P[:, 2] < 330.0)]
    ring = P[(r > window - 40.0) & (r < window + 50.0) & (np.abs(P[:, 2]) < 30.0)]
    return near, ring


def _resid(theta, near, ring):
    R, t, H = rot(theta[3:6]), theta[:3], theta[6]
    qn = (near - t) @ R                      # = R^T (x - t), row-wise
    qr = (ring - t) @ R
    return np.concatenate([box_sdf(qn, H), qr[:, 2]])


def fit_box(P, window=110.0, h0=None, max_pts=2500, seed=0):
    """Line the tower model up with the points P.  Returns dict(R, t, H, inliers, rms, n) or None.
    R, t move the current frame onto the fitted one: x_current = R x_fitted + t."""
    near, ring = _split(P, window)
    if len(near) < 300:
        return None
    rng = np.random.default_rng(seed)
    if len(near) > max_pts:
        near = near[rng.choice(len(near), max_pts, replace=False)]
    if len(ring) > max_pts // 2:
        ring = ring[rng.choice(len(ring), max_pts // 2, replace=False)]
    theta = np.zeros(7)
    theta[6] = float(h0 if h0 else np.percentile(near[:, 2], 98.0))
    scale = np.array([1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 1.0])      # step sizes for the numerical Jacobian
    for sigma in (30.0, 20.0, 12.0, 8.0, 5.0):                     # coarse to fine: far points count less and less
        for _ in range(6):
            r0 = _resid(theta, near, ring)
            w = (sigma ** 2 / (sigma ** 2 + r0 ** 2)) ** 2           # Geman-McClure weights
            J = np.empty((len(r0), 7))
            for k in range(7):
                d = np.zeros(7); d[k] = scale[k] * 0.05
                J[:, k] = (_resid(theta + d, near, ring) - r0) / d[k]
            A = J.T @ (J * w[:, None]) + 1e-3 * np.eye(7)
            step = np.linalg.solve(A, -(J.T @ (w * r0)))
            theta = theta + np.clip(step, -np.array([15, 15, 15, .15, .15, .15, 20.0]), np.array([15, 15, 15, .15, .15, .15, 20.0]))
    r = _resid(theta, near, ring)
    rn = r[:len(near)]
    inl = np.abs(rn) < 4.0
    return {"R": rot(theta[3:6]), "t": theta[:3].copy(), "H": float(theta[6]),
            "inliers": float(inl.mean()), "rms": float(np.sqrt(np.mean(rn[inl] ** 2))) if inl.any() else 99.0,
            "n": int(len(near)), "yaw_deg": float(np.degrees(theta[5])), "tilt_deg": float(np.degrees(np.hypot(theta[3], theta[4])))}


def layers_from_height(H):
    return int(round((H - TOP_OFFSET) / LAYER_MM))

