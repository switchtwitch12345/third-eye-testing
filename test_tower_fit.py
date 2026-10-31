#!/usr/bin/env python3
"""tower_fit must recover a known tower position, tilt and height from synthetic LiDAR points.
    .venv/bin/python test_tower_fit.py"""
import numpy as np

import tower_fit as tf

rng = np.random.default_rng(1)


def synth(shift, rv, H, noise=1.0):
    """Points on the visible sides and top of a 75 x 75 x H tower plus the board, as seen from one corner,
    in a frame that is shifted / tilted by (shift, rv) from the true tower frame."""
    pts = []
    n = 1800
    # near sides (x = -37.5 and y = +37.5), the top, and the board ring
    z = rng.uniform(0, H, n); y = rng.uniform(-37.5, 37.5, n); pts.append(np.c_[np.full(n, -37.5), y, z])
    z = rng.uniform(0, H, n); x = rng.uniform(-37.5, 37.5, n); pts.append(np.c_[x, np.full(n, 37.5), z])
    x = rng.uniform(-37.5, 37.5, n); y = rng.uniform(-37.5, 37.5, n); pts.append(np.c_[x, y, np.full(n, H)])
    r = rng.uniform(75, 150, 2 * n); a = rng.uniform(0, 2 * np.pi, 2 * n)
    pts.append(np.c_[r * np.cos(a), r * np.sin(a), np.zeros(2 * n)])
    P = np.concatenate(pts) + rng.normal(0, noise, (len(np.concatenate(pts)), 3))
    # what the marker frame sees: the true frame moved by the error  x_marker = R x_true + t
    return P @ tf.rot(rv).T + shift


def check(shift, rv_deg, H):
    rv = np.radians(rv_deg)
    P = synth(np.array(shift, float), rv, H)
    f = tf.fit_box(P)
    assert f is not None and f["inliers"] > 0.8, f
    # the fit must undo the error: the fitted frame sits where the true frame is
    err_t = np.linalg.norm(f["t"] - np.array(shift))
    assert err_t < 2.5, (shift, f["t"], err_t)
    assert abs(f["H"] - H) < 3.0, (H, f["H"])


check([0, 0, 0], [0, 0, 0], 120.0)               # already right: stays put
check([20, -25, 0], [0, 0, 0], 120.0)            # a tower standing off-centre, like a different board
check([3, -4, 7], [0.047, 0, 0.04], 236.0)       # what the new board showed: ~2.7 deg tilt, 7 mm too high, tall tower
check([-15, 10, 5], [0, 0.05, 0.2], 90.0)        # twisted 11 degrees as well
assert tf.layers_from_height(236.0) == 16 and tf.layers_from_height(137.0) == 9
print("tower_fit: all checks passed")
