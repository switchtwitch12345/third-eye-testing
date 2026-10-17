#!/usr/bin/env bash
# One-time setup: Python environment + Aadit Pani's STEPWISE/ThirdEye repo (depth reader and tower geometry).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
if [ ! -d third_party/STEPWISE ]; then
  mkdir -p third_party
  git clone https://github.com/AaditPani-RVU/STEPWISE.git third_party/STEPWISE
fi
echo "Done. Start the server with:  .venv/bin/python iphone_saved.py"
