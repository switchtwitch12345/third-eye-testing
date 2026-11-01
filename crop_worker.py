#!/usr/bin/env python3
"""A worker for face_crops.CropPool: reads one JSON job per line on stdin, cuts that frame's face crops, answers with one
JSON line.  Started by the game server; not meant to be run by hand."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import face_crops as fc

for line in sys.stdin:
    try:
        out = fc.process_frame(json.loads(line))
    except Exception as e:                      # never let one bad frame stop the worker
        out = {"skip": f"{type(e).__name__}: {e}"}
    print(json.dumps(out), flush=True)
