#!/usr/bin/env bash
# Both phones on cables, hands-off: keep it running next to scripts/start_all.sh.
#   Android : whenever a phone with USB debugging is plugged in, `adb reverse` is (re)applied, so the camera app
#             finds the Mac on localhost and switches to the cable; unplug it and the app falls back to Wi-Fi
#   iPhone  : runs scripts/usb_bridge.py (its app switches to the cable while the bridge is collecting)
# Ctrl-C stops both.  Nothing here is needed for Wi-Fi.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH:$HOME/Android/Sdk/platform-tools:$HOME/Library/Android/sdk/platform-tools"

.venv/bin/python scripts/usb_bridge.py &
BRIDGE=$!
trap 'kill $BRIDGE 2>/dev/null; exit 0' INT TERM EXIT

if ! command -v adb >/dev/null; then
  echo "adb not found (brew install android-platform-tools): Android stays on Wi-Fi"
fi
last=""
while true; do
  if command -v adb >/dev/null; then
    now=$(adb devices 2>/dev/null | awk 'NR>1 && $2=="device" {print $1}' | tr '\n' ' ')
    if [ "$now" != "$last" ]; then
      if [ -n "$now" ]; then
        adb reverse tcp:8000 tcp:8000 >/dev/null 2>&1 && echo "Android on the cable: $now"
      else
        echo "Android cable removed (the app falls back to Wi-Fi)"
      fi
      last="$now"
    fi
  fi
  sleep 3
done
