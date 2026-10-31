#!/usr/bin/env python3
"""
Wired mode for the iPhone app: carries its requests over the USB cable to the game server.

The phone cannot call the Mac over a cable, so the app queues its requests on port 8765 of the phone and this
bridge (through `iproxy`, which forwards a Mac port into the phone over USB) collects them, plays each one to the
game server on this Mac, and hands the answer back.  The app notices the bridge is collecting and switches to the
cable by itself; stop the bridge (or unplug) and it falls back to Wi-Fi within a few seconds.

    brew install libimobiledevice      # one time (gives `iproxy`)
    .venv/bin/python scripts/usb_bridge.py
"""
import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--phone", default="http://127.0.0.1:8765", help="where the phone's port shows up on this Mac")
ap.add_argument("--server", default="http://127.0.0.1:8000", help="the game server")
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--no-iproxy", action="store_true", help="do not start iproxy (it is already running, or --phone is a test server)")
args = ap.parse_args()

stats = {"relayed": 0, "failed": 0, "last_ok": 0.0}
stop = threading.Event()


def call(method, url, body=None, headers=None, timeout=10):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def worker():
    while not stop.is_set():
        try:
            status, data = call("GET", args.phone + "/next", timeout=6)
        except Exception:
            time.sleep(1.0)                       # phone not reachable yet (cable, trust, app not open)
            continue
        if status != 200 or not data:
            continue
        nl = data.index(b"\n")
        meta, body = json.loads(data[:nl]), data[nl + 1:]
        hdr = {k: v for k, v in meta["headers"].items() if k.lower() not in ("host", "content-length", "connection")}
        try:
            st, ans = call(meta["method"], args.server + meta["path"], body or None, hdr, timeout=10)
            stats["relayed"] += 1
            stats["last_ok"] = time.time()
        except Exception:
            st, ans = 502, b""
            stats["failed"] += 1
        try:
            call("POST", f"{args.phone}/reply/{meta['id']}", ans, {"X-Status": str(st)}, timeout=6)
        except Exception:
            pass


def main():
    proc = None
    if not args.no_iproxy:
        if not shutil.which("iproxy"):
            sys.exit("iproxy is missing. One time:  brew install libimobiledevice")
        proc = subprocess.Popen(["iproxy", "8765:8765"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("iproxy started (port 8765 of the iPhone -> this Mac)")
    for _ in range(args.workers):
        threading.Thread(target=worker, daemon=True).start()
    print("bridge running. Phone app: the address box turns green when the cable is in use. Ctrl-C to stop.")
    try:
        last = 0
        while True:
            time.sleep(5)
            if stats["relayed"] != last:
                print(f"relayed {stats['relayed']} requests ({stats['failed']} failed)")
                last = stats["relayed"]
            elif time.time() - stats["last_ok"] > 15:
                print("nothing coming from the phone: cable plugged in? iPhone unlocked and trusted? app open?")
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
