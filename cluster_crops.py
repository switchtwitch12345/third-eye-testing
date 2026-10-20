#!/usr/bin/env python3
"""
Unlabelled first look at the face-crop dataset.

  1. embed every crop with a pretrained vision model (ResNet18, ImageNet weights)
  2. anomaly view: how far does each face drift from ITS OWN intact-tower look over time?
  3. cluster view: do the crops fall into natural groups, and what are they?

Run with an environment that has torch (not the repo .venv, whose Python 3.15 has no torch wheels):
    /Users/mitul/.venvs/jenga-ml/bin/python cluster_crops.py
"""
import collections
import json
import os
import sys

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset")
SOURCES = sys.argv[1:] or ["game03", "game04"]
MIN_COS, BODY_LAYERS, REF_SECONDS = 0.35, 10, 8.0

dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
net = torchvision.models.resnet18(weights=weights)
net.fc = torch.nn.Identity()
net = net.eval().to(dev)
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def embed(paths, bs=256):
    out = []
    for i in range(0, len(paths), bs):
        ims = [np.asarray(Image.open(p).convert("RGB").resize((200, 120), Image.BILINEAR)) for p in paths[i:i + bs]]
        x = torch.from_numpy(np.stack(ims)).to(dev).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            f = net((x - mean) / std)
        out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
    return np.concatenate(out)


def load(src):
    rows = []
    for l in open(os.path.join(DATA, src, "index.jsonl")):
        r = json.loads(l)
        if r["cam"] == "iphone" and r["cos"] >= MIN_COS and r["layer"] < BODY_LAYERS:
            r["path"] = os.path.join(DATA, src, r["file"])
            r["src"] = src
            rows.append(r)
    return rows


all_rows, all_emb = [], []
for src in SOURCES:
    rows = load(src)
    cache = os.path.join(DATA, f"emb_{src}.npz")
    if os.path.exists(cache) and len(np.load(cache)["e"]) == len(rows):
        emb = np.load(cache)["e"]
    else:
        print(f"embedding {src}: {len(rows)} crops on {dev} ...", flush=True)
        emb = embed([r["path"] for r in rows])
        np.savez(cache, e=emb)
    all_rows += rows
    all_emb.append(emb)
E = np.concatenate(all_emb)
print(f"{len(E)} crops embedded ({E.shape[1]}-d)\n")

# ---- 2. drift from each face's own intact-tower look ------------------------------------------
print("=== how far does a face drift from its OWN intact-tower look? (cosine distance, median) ===")
for src in SOURCES:
    idx = [i for i, r in enumerate(all_rows) if r["src"] == src]
    ref = collections.defaultdict(list)
    for i in idx:
        r = all_rows[i]
        if r["t_sec"] <= REF_SECONDS:
            ref[(r["layer"], r["slot"], r["sign"])].append(E[i])
    ref = {k: (lambda m: m / np.linalg.norm(m))(np.mean(v, axis=0)) for k, v in ref.items()}
    d, t = [], []
    for i in idx:
        r = all_rows[i]
        k = (r["layer"], r["slot"], r["sign"])
        if r["t_sec"] > REF_SECONDS and k in ref:
            d.append(1 - float(E[i] @ ref[k])); t.append(r["t_sec"])
    d, t = np.array(d), np.array(t)
    marks = json.load(open(os.path.join(DATA, src, "marks.json")))
    mk = [round((m - marks["t0_ms"]) / 1000) for m in marks["marks_ms"]]
    print(f"\n{src}  (taps at {mk} s)")
    print("   time    median drift   p90 drift   crops")
    for lo in range(int(REF_SECONDS), int(t.max()) + 10, 20):
        sel = (t >= lo) & (t < lo + 20)
        if sel.sum() > 50:
            print(f"  {lo:3d}-{lo+20:3d}s   {np.median(d[sel]):.3f}        {np.percentile(d[sel], 90):.3f}      {sel.sum()}")

# ---- 3. clusters --------------------------------------------------------------------------------
K = 10
km = KMeans(n_clusters=K, n_init=4, random_state=0).fit(E)
lab = km.labels_
print(f"\n=== {K} clusters of the crops ===")
tiles, W, H = 12, 100, 60
canvas = Image.new("RGB", (tiles * (W + 2) + 150, K * (H + 4)), (30, 30, 30))
dr = ImageDraw.Draw(canvas)
rng = np.random.default_rng(0)
for c in range(K):
    members = np.where(lab == c)[0]
    late = np.mean([all_rows[i]["t_sec"] > 150 for i in members])
    layer = np.median([all_rows[i]["layer"] for i in members])
    print(f"  cluster {c}: {len(members):5d} crops | median layer {layer + 1:.0f} | share from late in the game (>150 s): {100 * late:3.0f}%")
    dr.text((4, c * (H + 4) + 4), f"#{c}  n={len(members)}", fill=(255, 255, 255))
    dr.text((4, c * (H + 4) + 20), f"layer~{layer + 1:.0f}", fill=(200, 200, 200))
    dr.text((4, c * (H + 4) + 36), f"late {100 * late:.0f}%", fill=(200, 200, 200))
    for j, i in enumerate(rng.choice(members, min(tiles, len(members)), replace=False)):
        im = Image.open(all_rows[i]["path"]).convert("RGB").resize((W, H))
        canvas.paste(im, (150 + j * (W + 2), c * (H + 4)))
out = os.path.join(HERE, "cluster_montage.png")
canvas.save(out)
print(f"\nmontage saved: {out}")
