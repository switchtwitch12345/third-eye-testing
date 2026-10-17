#!/usr/bin/env python3
"""
One-time lens calibration for the Android camera (the iPhone reports its own).

1. Show a checkerboard on the Mac screen:
       .venv/bin/python calibrate_camera.py --show-board
   No measuring needed: a lens's intrinsics do not depend on the square size.

2. Point the phone at the screen from different distances and angles:
       .venv/bin/python calibrate_camera.py --url http://127.0.0.1:8000/android_raw.jpg
   It grabs ~20 views, then writes camera_android.json (fx, fy, cx, cy, distortion).
"""
import argparse, json, os, sys, time, urllib.request
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
INNER = (9, 6)          # inner corners: 10 x 7 squares


def make_board(square_px=90):
    h, w = (INNER[1] + 1) * square_px, (INNER[0] + 1) * square_px
    img = np.full((h + 2 * square_px, w + 2 * square_px), 255, np.uint8)
    for r in range(INNER[1] + 1):
        for c in range(INNER[0] + 1):
            if (r + c) % 2 == 0:
                y, x = square_px + r * square_px, square_px + c * square_px
                img[y:y + square_px, x:x + square_px] = 0
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-board", action="store_true")
    ap.add_argument("--url")
    ap.add_argument("--square-mm", type=float, default=20.0,
                    help="any value: the lens (focal length, centre, distortion) does not depend on it")
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(HERE, "camera_android.json"))
    args = ap.parse_args()

    if args.show_board:
        path = os.path.join(HERE, "checkerboard.png")
        cv2.imwrite(path, make_board())
        os.system(f'open "{path}"')
        print("Opened checkerboard.png. Make it large on the screen, then point the Android at it.")
        return

    if not args.url:
        sys.exit("need --url (or use --show-board)")

    objp = np.zeros((INNER[0] * INNER[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:INNER[0], 0:INNER[1]].T.reshape(-1, 2) * args.square_mm
    obj_pts, img_pts, size, last = [], [], None, 0.0
    print(f"move the phone around the board; need {args.views} distinct views. Ctrl+C to stop early.\n")
    try:
        while len(obj_pts) < args.views:
            try:
                raw = urllib.request.urlopen(args.url, timeout=3).read()
            except Exception as e:
                print(f"\rcannot reach phone: {e}", end="")
                time.sleep(1)
                continue
            gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
            size = gray.shape[::-1]
            ok, corners = cv2.findChessboardCornersSB(gray, INNER, cv2.CALIB_CB_EXHAUSTIVE)
            if ok and time.time() - last > 1.2:
                # skip near-duplicate views
                c = corners.reshape(-1, 2)
                if not img_pts or np.abs(c - img_pts[-1].reshape(-1, 2)).mean() > 12:
                    obj_pts.append(objp)
                    img_pts.append(corners)
                    last = time.time()
                    print(f"\rviews {len(obj_pts)}/{args.views}   ", end="", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    print()
    if len(obj_pts) < 8:
        sys.exit(f"only {len(obj_pts)} views; need at least 8 for a stable result")

    rms, K, dist, _r, _t = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    out = {"width": size[0], "height": size[1], "fx": float(K[0, 0]), "fy": float(K[1, 1]),
           "cx": float(K[0, 2]), "cy": float(K[1, 2]), "dist": dist.ravel().tolist(),
           "rms_px": float(rms), "views": len(obj_pts)}
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nsaved {args.out}")
    print("RMS reprojection error should be < 1.0 px." + ("" if rms < 1.0 else "  Too high: redo with more varied angles."))


if __name__ == "__main__":
    main()
