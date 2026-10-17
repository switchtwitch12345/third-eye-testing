"""RGB appearance cue per slot end face, projected with the marker pose.

The depth map is 256x192; the colour image is 1920x1440, so a block end that is a
handful of depth pixels is tens of colour pixels.  We sample the same 4x4 points
their depth reader uses, in a lightly blurred grey image, and compare against the
same points in the reference scan.
"""
import numpy as np
import cv2
from thirdeye.events import Slot
from thirdeye.perception.tower import STANDARD


def grey_blur(bgr, k=9):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.blur(g, (k, k))


def rgb_samples(grey, K_rgb, R, t, view, geom=STANDARD):
    """Slot -> 16 grey values on the end face the camera sees (None if off-image)."""
    h, w = grey.shape
    out = {}
    for layer in range(geom.layers):
        for slot in range(geom.slots):
            for face, (centre, normal) in geom.ends(layer, slot).items():
                if not view.faces(normal, centre):
                    continue
                pts = geom.samples(layer, slot, face)
                uvw = (pts @ R.T + t) @ K_rgb.T
                if (uvw[:, 2] <= 0).any():
                    continue
                u = np.round(uvw[:, 0] / uvw[:, 2]).astype(int)
                v = np.round(uvw[:, 1] / uvw[:, 2]).astype(int)
                if not ((u >= 0) & (u < w) & (v >= 0) & (v < h)).all():
                    continue
                out[Slot(layer, slot)] = grey[v, u]
    return out
