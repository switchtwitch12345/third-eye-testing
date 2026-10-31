#!/usr/bin/env bash
# One command for the demo: our game server (phones connect here, port 8000) and Third Eye's UI (port 8001).
set -euo pipefail
cd "$(dirname "$0")/.."
pkill -f "iphone_saved.py" 2>/dev/null || true
pkill -f "thirdeye.server.app" 2>/dev/null || true
sleep 1
nohup .venv/bin/python iphone_saved.py > /tmp/game_server.log 2>&1 &
nohup scripts/run_thirdeye.sh > /tmp/thirdeye.log 2>&1 &
sleep 5
IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "<this Mac's IP>")
echo "Game server  : http://localhost:8000      (type $IP in the phone apps)"
echo "Third Eye UI : http://localhost:8001/#/game   <- show this one"
echo "Logs: /tmp/game_server.log  /tmp/thirdeye.log     Stop: pkill -f iphone_saved.py; pkill -f thirdeye.server.app"
