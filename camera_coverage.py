#!/usr/bin/env python3
"""
Can each phone see BOTH kinds of layer?  Layers alternate: odd layers (1,3,5..) have their block ENDS on the
tower's x faces, even layers (2,4,6..) on its y faces.  A phone sees an end face squarely (cos near 1) or edge-on
(cos near 0).  To spot a pulled block in every layer a phone must see both x ends and y ends at a decent angle,
which means standing at a corner (about 45 degrees to both faces), not square-on to one face.

    .venv/bin/python camera_coverage.py        (both phones streaming, markers visible)
"""
import os, sys, urllib.request
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import face_crops as fc
import tower_auto as ta

def best(view_pos, axis):
    """best (largest) cos over the two faces on this axis, seen from the camera, at mid-height of the tower"""
    out = 0.0
    for sign in (1, -1):
        n = np.zeros(3); n[axis] = sign
        c = np.zeros(3); c[axis] = sign * 37.5; c[2] = 70
        to = view_pos - c
        out = max(out, float(n @ to) / np.linalg.norm(to))
    return out

import time
poses = {}
for _ in range(20):                      # a single frame can lose a marker: try a few
    for name, path in (("iphone", "/preview_raw.jpg"), ("android", "/android_raw.jpg")):
        if name in poses:
            continue
        img = ta._grab(path)
        if img is None:
            continue
        K, dist = ta.camera_K(name, img)
        sol = fc.solve_pose(img, K, dist)
        if sol is not None:
            poses[name] = sol
    if len(poses) == 2:
        break
    time.sleep(0.3)
frames = {"iphone": True, "android": True}
print("camera | position in tower frame (mm)          | sees ODD-layer ends (x faces) | sees EVEN-layer ends (y faces)")
ok_all = True
for name in frames:
    if name not in poses:
        print(f"{name:7s}| markers not found (is it streaming and seeing 3+ markers?)"); ok_all = None; continue
    R, t, err = poses[name]
    pos = -R.T @ t
    cx, cy = best(pos, 0), best(pos, 1)
    def word(c): return "squarely" if c >= 0.55 else ("ok" if c >= 0.40 else "GRAZING" if c >= 0.20 else "NOT VISIBLE")
    print(f"{name:7s}| x {pos[0]:+6.0f}  y {pos[1]:+6.0f}  z {pos[2]:5.0f}  | cos {cx:.2f} ({word(cx):11s})         | cos {cy:.2f} ({word(cy)})")
    ok_all = (ok_all is not None) and ok_all and min(cx, cy) >= 0.40
    horiz = float(np.hypot(pos[0], pos[1])); dist3 = float(np.linalg.norm(pos - np.array([0, 0, 70])))
    el = float(np.degrees(np.arctan2(pos[2] - 70, horiz))); az = float(np.degrees(np.arctan2(abs(pos[1]), abs(pos[0]))))
    print(f"        looking down at {el:.0f} deg; {az:.0f} deg round from the x face (45 is ideal: equal view of both face types)")
    tgt = np.radians(35)
    z_new = 70 + dist3 * np.sin(tgt); h_new = dist3 * np.cos(tgt)
    cos_t = np.cos(tgt) * np.cos(np.radians(45))
    if el > 42:
        print(f"        -> TOO STEEP. Same distance, lower: move it ~{(pos[2] - z_new) / 10:.0f} cm DOWN and ~{(h_new - horiz) / 10:.0f} cm back, "
              f"so it looks down at about 35 deg. Both face types would then read about {cos_t:.2f} instead of {min(cx, cy):.2f}.")
    elif az < 25 or az > 65:
        print(f"        -> square-on to one face. Slide it round toward the corner (azimuth about 45 deg).")
    if False and min(cx, cy) < 0.40:
        weak = "y faces (even layers)" if cy < cx else "x faces (odd layers)"
        print(f"        -> weak on the {weak}. Move this phone toward a CORNER: aim for about equal |x| and |y| ({max(abs(pos[0]), abs(pos[1])):.0f} mm out on the larger axis).")
print("\nverdict:", "could not measure both phones" if ok_all is None else ("both phones see both layer types well" if ok_all else "at least one phone is square-on to a face: layers of the other type are seen at a glancing angle"))
