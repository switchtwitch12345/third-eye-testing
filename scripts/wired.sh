#!/usr/bin/env bash
# Optional: run the two phones over USB cables instead of Wi-Fi.  Run it after scripts/start_all.sh.
#   Android : USB debugging + `adb reverse`, so the phone reaches this Mac as "localhost" (no network at all);
#             the android-capture app notices by itself and switches to the cable
#   iPhone  : the app queues its requests and scripts/usb_bridge.py collects them over the cable (needs the rebuilt app)
set -uo pipefail
export PATH="/opt/homebrew/bin:$PATH:$HOME/Android/Sdk/platform-tools:$HOME/Library/Android/sdk/platform-tools"

echo "== Android =="
if ! command -v adb >/dev/null; then
  echo "adb is not installed.  One time:  brew install android-platform-tools"
else
  adb start-server >/dev/null 2>&1
  if adb devices | awk 'NR>1 && $2=="device"' | grep -q .; then
    adb reverse tcp:8000 tcp:8000 >/dev/null && echo "ok. Open the Third Eye camera app: the IP box turns green (USB). (Or Chrome at http://localhost:8000/cam)"
  else
    echo "no Android phone found. Plug it in, turn on Developer options > USB debugging, tap Allow on the phone, run again."
  fi
fi

echo "== iPhone =="
if ! command -v iproxy >/dev/null; then
  echo "iproxy is not installed.  One time:  brew install libimobiledevice"
else
  echo "Plug the iPhone in (unlocked, tap Trust), open the app, then in a second terminal run:"
  echo "    .venv/bin/python scripts/usb_bridge.py"
  echo "The IP box in the app turns green when the cable is in use; if the cable goes away it falls back to Wi-Fi."
fi
