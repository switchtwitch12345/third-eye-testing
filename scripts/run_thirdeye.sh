#!/usr/bin/env bash
# Start Aadit's Third Eye server (rules, 3D tower, referee panel) on port 8001, next to our game server on 8000.
# In its Live page, "follow" http://127.0.0.1:8000 to judge the game that our server is tracking.
set -euo pipefail
cd "$(dirname "$0")/../third_party/STEPWISE"
export PATH="/opt/homebrew/bin:$PATH"
exec .venv/bin/python -m thirdeye.server.app --host 127.0.0.1 --port "${THIRDEYE_PORT:-8001}"
