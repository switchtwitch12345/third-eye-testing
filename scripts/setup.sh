#!/usr/bin/env bash
# One-time setup: Python environment + Aadit Pani's STEPWISE/ThirdEye repo (depth reader and tower geometry).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
# Aadit Pani's Third Eye repo (formerly STEPWISE): depth reader, tower geometry, rule checker, web UI, LLM referee.
# Kept as its own checkout (not vendored); we follow his real-data branch.
if [ ! -d third_party/STEPWISE ]; then
  mkdir -p third_party
  git clone https://github.com/AaditPani-RVU/ThirdEye.git third_party/STEPWISE
fi
(cd third_party/STEPWISE && git remote set-branches origin '*' && git fetch -q --all --prune \
  && git checkout -q jenga-real-data-eval 2>/dev/null || git checkout -q --track origin/jenga-real-data-eval; git pull -q --ff-only || true)
# our additions to his UI/server: the Game tab and the /game proxy (patches/thirdeye-game-console.patch)
(cd third_party/STEPWISE && git apply --check ../../patches/thirdeye-game-console.patch 2>/dev/null \
  && git apply ../../patches/thirdeye-game-console.patch && echo "applied the Game-console patch" || true)
# his environment (Python 3.11+) and web UI
PY=$(command -v python3.12 || command -v python3.11 || command -v python3.13 || true)
if [ -n "$PY" ] && [ ! -x third_party/STEPWISE/.venv/bin/python ]; then
  "$PY" -m venv third_party/STEPWISE/.venv
  V=third_party/STEPWISE/.venv
  # explicit packages: pip's resolver backtracks for ages on "-e .[server]" with the newest pydantic
  $V/bin/pip install -q --default-timeout 60 numpy pyyaml "pydantic==2.13.5" fastapi "uvicorn[standard]" pytest httpx
  SP=$($V/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  if [ -d .venv/lib/python3.*/site-packages/cv2 ]; then          # our OpenCV is an abi3 build: it works in his Python too
    cp -R .venv/lib/python3.*/site-packages/cv2 .venv/lib/python3.*/site-packages/opencv_contrib_python_headless-*.dist-info "$SP"/
  else
    $V/bin/pip install -q --default-timeout 60 opencv-contrib-python-headless
  fi
  $V/bin/pip install -q --no-deps -e third_party/STEPWISE
fi
if command -v npm >/dev/null 2>&1; then
  (cd third_party/STEPWISE/web && npm ci --ignore-scripts && npm run build)
fi
echo "Done. Start our game server:  .venv/bin/python iphone_saved.py"
echo "      Start Third Eye on top of it:  scripts/run_thirdeye.sh   (UI on http://127.0.0.1:8001)"
