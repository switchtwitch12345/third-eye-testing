#!/usr/bin/env python3
"""
Classical-CV view of the face crops: edges + Hough lines at the places where a block's boundaries should be.

Every crop is the end face of one block, rectified (100 x 60 px) with 25% context, so the face itself is the
central rectangle x 17..83, y 10..50.  For a present block we expect straight boundaries at the four sides of
that rectangle and a plain interior.  A missing block shows a dark void and different boundaries; a pushed-out
block shifts or tilts the boundary lines.  Features per crop (all explainable):

   dark        fraction of the face rectangle that is much darker than the crop's own median (a void)
   int_edge    edge density inside the rectangle (a plain face has few)
   bnd_top/bottom/left/right   edge density in a thin strip on each expected boundary
   off_top/bottom/left/right   where the strongest edge actually is, relative to where it should be (px)
   h_len, v_len, h_n, v_n      Hough line support: total length and number of near-horizontal / near-vertical lines
   skew        mean tilt of the long lines (degrees)

Then the same per-tap analysis as tap_report.py (global drift removed), and a comparison with the learned
fingerprint method (needs its embedding cache from tap_report.py).

    .venv/bin/python geo_report.py game05_every_frame
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
X0, X1, Y0, Y1 = 17, 83, 10, 50            # the face inside the 100 x 60 crop (25% context on every side)


def features(path):
    img = cv2.imread(path)
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    face = g[Y0:Y1, X0:X1].astype(np.float32)
    med = float(np.median(g))
    dark = float((face < 0.55 * med).mean())
    edges = cv2.Canny(g, 35, 100)
    inner = edges[Y0 + 4:Y1 - 4, X0 + 4:X1 - 4]
    int_edge = float(inner.mean() / 255.0)
    # edge density profiles across / along the face, to find boundaries and how far they sit from expected
    row = edges[:, X0:X1].mean(axis=1) / 255.0            # y profile (horizontal boundaries)
    col = edges[Y0:Y1, :].mean(axis=0) / 255.0            # x profile (vertical boundaries)

    def strip(profile, centre, half=3):
        return float(profile[max(0, centre - half):centre + half + 1].mean())

    def offset(profile, centre, span=9):
        lo, hi = max(0, centre - span), min(len(profile), centre + span + 1)
        return float(np.argmax(profile[lo:hi]) + lo - centre)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 14, minLineLength=18, maxLineGap=4)
    h_len = v_len = h_n = v_n = 0.0
    tilts = []
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            ang = (ang + 90) % 180 - 90                    # -90..90
            ln = float(np.hypot(x2 - x1, y2 - y1))
            if abs(ang) < 20:
                h_len += ln; h_n += 1; tilts.append(ang)
            elif abs(ang) > 70:
                v_len += ln; v_n += 1; tilts.append(ang - np.sign(ang) * 90)
    return np.array([dark, int_edge,
                     strip(row, Y0), strip(row, Y1), strip(col, X0), strip(col, X1),
                     offset(row, Y0), offset(row, Y1), offset(col, X0), offset(col, X1),
                     h_len / 60.0, v_len / 40.0, h_n, v_n, float(np.mean(tilts)) if tilts else 0.0], np.float32)


NAMES = ["dark", "int_edge", "bnd_top", "bnd_bottom", "bnd_left", "bnd_right",
         "off_top", "off_bottom", "off_left", "off_right", "h_len", "v_len", "h_n", "v_n", "skew"]

rows = []
for l in open(os.path.join(DATA, RUN, "index.jsonl")):
    r = json.loads(l)
    if r["cos"] >= MIN_COS and r["layer"] < MAX_LAYER:
        r["path"] = os.path.join(DATA, RUN, r["file"])
        rows.append(r)
cache = os.path.join(DATA, f"geo_{RUN}.npz")
if os.path.exists(cache) and len(np.load(cache)["f"]) == len(rows):
    F = np.load(cache)["f"]
else:
    print(f"computing geometric features for {len(rows)} crops ...", flush=True)
    F = np.stack([features(r["path"]) for r in rows])
    np.savez(cache, f=F)
print(f"{len(rows)} crops, {F.shape[1]} features each: {dict(collections.Counter(r['cam'] for r in rows))}")

marks = json.load(open(os.path.join(DATA, RUN, "marks.json")))
t0 = rows[0]["ts_ms"] - rows[0]["t_sec"] * 1000
taps = [(m - t0) / 1000.0 for m in marks["marks_ms"]]
key = lambda r: (r["cam"], r["layer"], r["slot"], r["sign"])


def block_scores(drift_of, label):
    """per-tap {block: score}: change in (face drift minus the frame-wide median drift), best face per block"""
    frames = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r["t_sec"] > REF_SECONDS and key(r) in ref:
            frames[(r["cam"], r["ts_ms"])].append((key(r), drift_of(i, key(r)), r["t_sec"]))
    rel = collections.defaultdict(list)
    for lst in frames.values():
        if len(lst) < 6:
            continue
        med = np.median([d for _, d, _ in lst])
        for k, d, t in lst:
            rel[k].append((t, d - med))
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


# ---------------- geometric drift: standardised distance from each face's own reference ----------------
ref, scale = {}, {}
groups = collections.defaultdict(list)
for i, r in enumerate(rows):
    if r["t_sec"] <= REF_SECONDS:
        groups[key(r)].append(i)
gstd = F[[i for v in groups.values() for i in v]].std(axis=0) + 1e-3          # typical natural variation
for k, idx in groups.items():
    if len(idx) >= 3:
        ref[k] = F[idx].mean(axis=0)
        scale[k] = np.maximum(F[idx].std(axis=0), 0.5 * gstd)
geo = lambda i, k: float(np.sqrt(np.mean(np.clip((F[i] - ref[k]) / scale[k], -6, 6) ** 2)))
G = block_scores(geo, "geo")
print(f"faces with a reference: {len(ref)}\n")


def top(score, n=3):
    return sorted(score.items(), key=lambda kv: -kv[1])[:n]


def snr(score):
    v = np.array(sorted(score.values(), reverse=True))
    return v[0] / max(np.median(np.abs(v)), 1e-6) if len(v) > 3 else 0.0


# ---------------- optional: the learned-fingerprint scores, from the cached embeddings ----------------
E = None
for name in (f"emb2_{RUN}_resnet18.npz", f"emb2_{RUN}.npz"):
    p = os.path.join(DATA, name)
    if os.path.exists(p) and len(np.load(p)["e"]) == len(rows):
        E = np.load(p)["e"]
        break
L = None
if E is not None:
    eref = {k: (lambda m: m / np.linalg.norm(m))(E[v].mean(0)) for k, v in groups.items() if len(v) >= 3}
    ref_backup = ref
    ref = eref
    L = block_scores(lambda i, k: 1 - float(E[i] @ eref[k]), "emb")
    ref = ref_backup

print("block that changed most at each tap (layer from the bottom, slot digit):\n")
print("  tap   time | HOUGH / edge geometry                              | learned fingerprint (ResNet18)")
for n, tp in enumerate(taps, 1):
    g = ", ".join(f"L{l + 1}S{s} ({v:+.1f})" for (l, s), v in top(G[n - 1]))
    e = ", ".join(f"L{l + 1}S{s} ({v:+.2f})" for (l, s), v in top(L[n - 1])) if L else "-"
    print(f"  {n:3d}  {tp:4.0f}s | {g:50s} | {e}")

print("\nHOW CLEARLY EACH METHOD SEES A CHANGE (top block vs the typical change, median over taps):")
print(f"  Hough / geometry : x{np.median([snr(s) for s in G if s]):.1f}")
if L:
    print(f"  fingerprint      : x{np.median([snr(s) for s in L if s]):.1f}")
    agree, rho = [], []
    for g, e in zip(G, L):
        common = sorted(set(g) & set(e))
        if len(common) < 8:
            continue
        a = np.argsort(np.argsort([g[b] for b in common])); b_ = np.argsort(np.argsort([e[b] for b in common]))
        rho.append(float(np.corrcoef(a, b_)[0, 1]))
        agree.append(len({b for b, _ in top(g)} & {b for b, _ in top(e)}))
    print(f"\nAGREEMENT between the two independent methods (taps compared: {len(agree)}):")
    print(f"  same block in both top-3 lists: {sum(1 for a in agree if a > 0)}/{len(agree)} taps "
          f"(by chance alone about {1 - (1 - 3 / 32) ** 3:.0%} would)")
    print(f"  rank correlation of all block scores per tap: median {np.median(rho):+.2f}")
print("\nwhich geometric features moved most after the taps (standardised, all faces):")
late = [i for i, r in enumerate(rows) if r["t_sec"] > taps[1] + 2 and key(r) in ref]
z = np.mean([np.abs((F[i] - ref[key(rows[i])]) / scale[key(rows[i])]) for i in late], axis=0)
print("  " + ", ".join(f"{NAMES[j]} {z[j]:.1f}" for j in np.argsort(-z)[:6]))
