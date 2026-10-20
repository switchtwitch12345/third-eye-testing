#!/usr/bin/env python3
"""
Which block changed at each tap?  Unlabelled, from the face-crop dataset (both cameras).

For every visible block face we measure how far its picture has drifted from ITS OWN look at the start of the
game (pretrained vision-model fingerprint, cosine distance).  Most of that drift is shared by the whole tower
(light, small shifts), so in every frame we subtract the median drift of all faces.  A tap's score for a face is
how much that remaining drift grew between "before the tap" and "after the tap".  A block's score is the best of
its faces, from either camera.

    /Users/mitul/.venvs/jenga-ml/bin/python tap_report.py live_20261020_142720
"""
import collections
import json
import os
import sys

import numpy as np
import torch
import torchvision
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset")
RUN = sys.argv[1]
REF_SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
MIN_COS, MAX_LAYER = 0.30, 11

MODEL = os.environ.get("FP_MODEL", "resnet18")          # resnet18 | dinov2
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
if MODEL == "dinov2":
    from transformers import AutoModel
    net = AutoModel.from_pretrained("facebook/dinov2-small").eval().to(dev)
    SIZE = (196, 112)                                     # multiples of the 14 px patch
    run = lambda x: net(pixel_values=x).pooler_output
else:
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net = net.eval().to(dev)
    SIZE = (200, 120)
    run = net


def embed(paths, bs=128):
    out = []
    for i in range(0, len(paths), bs):
        ims = [np.asarray(Image.open(p).convert("RGB").resize(SIZE, Image.BILINEAR)) for p in paths[i:i + bs]]
        x = torch.from_numpy(np.stack(ims)).to(dev).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            f = run((x - mean) / std)
        out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
    return np.concatenate(out)


rows = []
for l in open(os.path.join(DATA, RUN, "index.jsonl")):
    r = json.loads(l)
    if r["cos"] >= MIN_COS and r["layer"] < MAX_LAYER:
        r["path"] = os.path.join(DATA, RUN, r["file"])
        rows.append(r)
cache = os.path.join(DATA, f"emb2_{RUN}_{MODEL}.npz")
if os.path.exists(cache) and len(np.load(cache)["e"]) == len(rows):
    E = np.load(cache)["e"]
else:
    print(f"embedding {len(rows)} crops on {dev} ...", flush=True)
    E = embed([r["path"] for r in rows])
    np.savez(cache, e=E)
print(f"model: {MODEL} | {len(rows)} crops: {dict(collections.Counter(r['cam'] for r in rows))}")

marks = json.load(open(os.path.join(DATA, RUN, "marks.json")))
t0 = rows[0]["ts_ms"] - rows[0]["t_sec"] * 1000
taps = [(m - t0) / 1000.0 for m in marks["marks_ms"]]
print(f"taps at {[round(t) for t in taps]} s; reference = first {REF_SECONDS:.0f} s\n")

key = lambda r: (r["cam"], r["layer"], r["slot"], r["sign"])
ref = collections.defaultdict(list)
for i, r in enumerate(rows):
    if r["t_sec"] <= REF_SECONDS:
        ref[key(r)].append(E[i])
ref = {k: (lambda m: m / np.linalg.norm(m))(np.mean(v, 0)) for k, v in ref.items() if len(v) >= 3}
print(f"faces with a reference: {len(ref)}")

frames = collections.defaultdict(list)
for i, r in enumerate(rows):
    k = key(r)
    if k in ref and r["t_sec"] > REF_SECONDS:
        frames[(r["cam"], r["ts_ms"])].append((k, 1 - float(E[i] @ ref[k]), r["t_sec"]))
rel = collections.defaultdict(list)
for (cam, ts), lst in frames.items():
    if len(lst) < 6:
        continue
    med = np.median([d for _, d, _ in lst])
    for k, d, t in lst:
        rel[k].append((t, d - med))


def window(k, lo, hi):
    v = [x for t, x in rel[k] if lo <= t < hi]
    return np.median(v) if len(v) >= 2 else None


bounds = [REF_SECONDS] + taps + [max(t for v in rel.values() for t, _ in v) + 1]
summary = []
print("block that changed most at each tap  (layer from the bottom, slot digit, evidence)\n")
for n, tp in enumerate(taps, 1):
    lo_b, hi_b = bounds[n - 1] + 1.5, tp - 0.5
    lo_a, hi_a = tp + 1.5, bounds[n + 1] - 0.5
    per_block = collections.defaultdict(list)
    for k in rel:
        b, a = window(k, lo_b, hi_b), window(k, lo_a, hi_a)
        if b is not None and a is not None:
            per_block[(k[1], k[2])].append((a - b, k))
    scored = sorted(((max(c for c, _ in v), key_, v) for key_, v in per_block.items()), reverse=True)
    noise = np.median([abs(c) for _, _, v in scored for c, _ in v]) if scored else 0
    top = ", ".join(f"L{l + 1}S{s} ({sc:+.2f}, {len(v)} faces, {len({k[0] for _, k in v})} cam)" for sc, (l, s), v in scored[:3])
    print(f"  tap {n} at {tp:4.0f}s: {top}    [{len(scored)} blocks compared, typical |change| {noise:.2f}]")
    if scored and noise > 0:
        two_cam = sum(1 for _, _, v in scored[:3] if len({k[0] for _, k in v}) == 2)
        summary.append((scored[0][0] / noise, (scored[0][0] - scored[1][0]) / noise, two_cam))
if summary:
    a = np.array(summary)
    print(f"\nSUMMARY [{MODEL}]: top block stands out by x{np.median(a[:, 0]):.1f} the typical change (median over taps); "
          f"leads the runner-up by x{np.median(a[:, 1]):.1f}; taps where the top-3 blocks all have both cameras: "
          f"{int((a[:, 2] == 3).sum())}/{len(a)}")
