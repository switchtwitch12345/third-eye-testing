#!/usr/bin/env python3
"""
Second camera (no LiDAR): pull frames from an Android phone running "IP Webcam" and
save them next to the iPhone capture, only while the iPhone is recording.

    .venv/bin/python android_cam.py --url http://192.168.0.77:8080/shot.jpg

Files:  iphone_saved/<ms>_android.jpg      (ms = this Mac's clock; the phones are on network time)
        iphone_saved/android_info.json     (image size, and intrinsics once calibrated)
"""
import argparse, json, os, sys, time, urllib.request
import numpy as np
import cv2

DETECTOR = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))


def get(url, timeout=3.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="IP Webcam still URL, e.g. http://192.168.0.77:8080/shot.jpg")
    ap.add_argument("--server", default="http://127.0.0.1:8000", help="the iPhone viewer server")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "iphone_saved"))
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--always", action="store_true", help="save even when the iPhone is not recording")
    ap.add_argument("--calib", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_android.json"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    period = 1.0 / args.fps
    saved, last_print, info_written, warned = 0, 0.0, False, False
    print(f"pulling {args.url} at {args.fps:g} fps -> {args.out}")
    print("saving only while the iPhone is recording (use --always to override)\n")

    while True:
        t_start = time.time()
        try:
            raw = get(args.url)
        except Exception as e:
            if not warned:
                print(f"cannot reach the phone ({e}). Is IP Webcam running and on the same Wi-Fi?")
                warned = True
            time.sleep(1.0)
            continue
        warned = False
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            time.sleep(0.5)
            continue
        _c, ids, _ = DETECTOR.detectMarkers(img)
        found = sorted(int(i) for i in ids.flatten() if int(i) < 4) if ids is not None else []

        recording = args.always
        if not recording:
            try:
                recording = bool(json.loads(get(args.server + "/status.json", 1.0)).get("capturing"))
            except Exception:
                recording = False

        if recording:
            ms = int(time.time() * 1000)
            with open(os.path.join(args.out, f"{ms}_android.jpg"), "wb") as f:
                f.write(raw)
            saved += 1
            if not info_written:
                info = {"width": int(img.shape[1]), "height": int(img.shape[0])}
                if os.path.exists(args.calib):
                    info["calibration"] = json.load(open(args.calib))
                json.dump(info, open(os.path.join(args.out, "android_info.json"), "w"))
                info_written = True

        if time.time() - last_print > 1.0:
            last_print = time.time()
            print(f"\r{img.shape[1]}x{img.shape[0]}  markers {found or '-'}  "
                  f"{'● RECORDING' if recording else 'idle       '}  saved {saved}   ", end="", flush=True)

        dt = time.time() - t_start
        if dt < period:
            time.sleep(period - dt)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped")
