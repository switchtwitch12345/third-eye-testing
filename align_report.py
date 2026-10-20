#!/usr/bin/env python3
"""
Registration-based change detection on the face crops.

Problem found earlier: when the whole tower (or a camera) shifts a few millimetres, every face slides inside its
crop, and any "compare to the reference" method calls that a change.  Fix: register each crop to its own
reference first (phase correlation), then measure two separate things:

    shift      how far the face slid (dx, dy in px).  Subtract the median shift of all faces in the same frame
               (the whole tower moving) and what is left is a block that moved RELATIVE to the others: a
               pushed-out / loosened block.
    residual   what is still different after alignment (lighting-normalised): a block that is missing or
               different, not just displaced.

    .venv/bin/python align_report.py game05_every_frame
"""
import collections
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset")
RUN = sys.argv[1]
REF_SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
MIN_COS, MAX_LAYER = 0.30, 11
FY0, FY1, FX0, FX1 = 6, 54, 10, 90             # region used for registration (the face plus a little context)
CY0, CY1, CX0, CX1 = 13, 47, 20, 80            # core of the face, for the residual after alignment
HANN = cv2.createHanningWindow((FX1 - FX0, FY1 - FY0), cv2.CV_32F)


def norm(g):
    g = g.astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-3)


rows = []
for l in open(os.path.join(DATA, RUN, "index.jsonl")):
    r = json.loads(l)
    if r["cos"] >= MIN_COS and r["layer"] < MAX_LAYER:
        r["path"] = os.path.join(DATA, RUN, r["file"])
        rows.append(r)
print(f"loading {len(rows)} crops ...", flush=True)
IM = np.stack([norm(cv2.GaussianBlur(cv2.cvtColor(cv2.imread(r["path"]), cv2.COLOR_BGR2GRAY), (3, 3), 0)) for r in rows])

marks = json.load(open(os.path.join(DATA, RUN, "marks.json")))
t0 = rows[0]["ts_ms"] - rows[0]["t_sec"] * 1000
taps = [(m - t0) / 1000.0 for m in marks["marks_ms"]]
key = lambda r: (r["cam"], r["layer"], r["slot"], r["sign"])

groups = collections.defaultdict(list)
for i, r in enumerate(rows):
    if r["t_sec"] <= REF_SECONDS:
        groups[key(r)].append(i)
ref = {k: IM[v].mean(axis=0) for k, v in groups.items() if len(v) >= 3}
print(f"faces with a reference: {len(ref)}\n")

# register every crop to its face's reference
shift, resid = np.zeros((len(rows), 2), np.float32), np.zeros(len(rows), np.float32)
for i, r in enumerate(rows):
    k = key(r)
    if k not in ref:
        continue
    a, b = ref[k][FY0:FY1, FX0:FX1], IM[i][FY0:FY1, FX0:FX1]
    (dx, dy), _ = cv2.phaseCorrelate(a, b, HANN)
    dx, dy = float(np.clip(dx, -12, 12)), float(np.clip(dy, -8, 8))
    shift[i] = (dx, dy)
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    al = cv2.warpAffine(IM[i], M, (100, 60), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    resid[i] = float(np.abs(ref[k][CY0:CY1, CX0:CX1] - al[CY0:CY1, CX0:CX1]).mean())
raw_resid = np.array([np.abs(ref[key(r)][CY0:CY1, CX0:CX1] - IM[i][CY0:CY1, CX0:CX1]).mean() if key(r) in ref else 0
                      for i, r in enumerate(rows)], np.float32)

late = np.array([r["t_sec"] > REF_SECONDS and key(r) in ref for r in rows])
print(f"how much of the 'change' was just sliding:  median residual {np.median(raw_resid[late]):.3f} before aligning, "
      f"{np.median(resid[late]):.3f} after  ({100 * (1 - np.median(resid[late]) / np.median(raw_resid[late])):.0f}% explained by a shift)")
print(f"typical face shift after the game starts: median |dx| {np.median(np.abs(shift[late, 0])):.1f} px, "
      f"|dy| {np.median(np.abs(shift[late, 1])):.1f} px\n")


def per_tap(value_of):
    """per-tap {(layer, slot): score}: change of (value - frame median), best face of each block"""
    frames = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if late[i]:
            frames[(r["cam"], r["ts_ms"])].append((i, key(r), r["t_sec"]))
    rel = collections.defaultdict(list)
    for lst in frames.values():
        if len(lst) < 6:
            continue
        vals = {i: value_of(i, lst) for i, _, _ in lst}
        med = np.median(list(vals.values()))
        for i, k, t in lst:
            rel[k].append((t, vals[i] - med))
    win = lambda k, lo, hi: (lambda v: np.median(v) if len(v) >= 2 else None)([x for t, x in rel[k] if lo <= t < hi])
    bounds = [REF_SECONDS] + taps + [max(t for v in rel.values() for t, _ in v) + 1]
    out = []
    for n, tp in enumerate(taps, 1):
        pb = collections.defaultdict(list)
        for k in rel:
            b, a = win(k, bounds[n - 1] + 1.5, tp - 0.5), win(k, tp + 1.5, bounds[n + 1] - 0.5)
            if b is not None and a is not None:
                pb[(k[1], k[2])].append(a - b)
        out.append({blk: max(v) for blk, v in pb.items()})
    return out


def rel_shift(i, lst):                       # slide relative to the other faces in the same frame
    mdx = np.median([shift[j][0] for j, _, _ in lst]); mdy = np.median([shift[j][1] for j, _, _ in lst])
    return float(np.hypot(shift[i][0] - mdx, shift[i][1] - mdy))


R = per_tap(lambda i, lst: float(resid[i]))
S = per_tap(rel_shift)
top = lambda d, n=3: sorted(d.items(), key=lambda kv: -kv[1])[:n]
fmt = lambda d, p: ", ".join(f"L{l + 1}S{s} ({v:+.{p}f})" for (l, s), v in top(d))
print("block that changed most at each tap:\n")
print("  tap  time | CONTENT changed after alignment                    | block SLID relative to the others (px)")
for n, tp in enumerate(taps, 1):
    print(f"  {n:3d}  {tp:4.0f}s | {fmt(R[n - 1], 2):50s} | {fmt(S[n - 1], 1)}")


def snr(score):
    v = np.array(sorted(score.values(), reverse=True))
    return v[0] / max(np.median(np.abs(v)), 1e-6) if len(v) > 3 else 0.0


print(f"\nclarity (top block vs the typical change, median over taps): content x{np.median([snr(s) for s in R if s]):.1f}, "
      f"relative slide x{np.median([snr(s) for s in S if s]):.1f}")
l1 = [sum(1 for (l, _), v in top(d) if l == 0) for d in R if d]
print(f"bottom-layer (L1) blocks in the content top-3 at each tap: {l1}   (before alignment they led taps 2, 4 and 6)")
