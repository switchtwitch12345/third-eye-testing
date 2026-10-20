#!/usr/bin/env python3
"""Score the layer-finder against the player's typed corrections.   .venv/bin/python label_eval.py [run_dir]"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2, numpy as np
import face_crops as fc, move_infer as mi

HERE = os.path.dirname(os.path.abspath(__file__))
run = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(f"{HERE}/dataset/live_*"), key=os.path.getmtime)[-1]
recs = [json.loads(l) for l in open(f"{run}/labels.jsonl")]
marks = {r["seq"]: r for r in recs if "mark_ms" in r}
corr = {r["seq"]: r["correction"] for r in recs if "correction" in r}
rows = [json.loads(l) for l in open(f"{run}/index.jsonl")]
start_ms = rows[0]["ts_ms"]
# iPhone pose of this game, to map left/middle/right to slots
ts = sorted(int(os.path.basename(f).split("_")[0]) for f in glob.glob(f"{HERE}/iphone_saved/*_rgb.jpg"))
t0 = min(t for t in ts if t >= start_ms - 2000)
img = cv2.imread(f"{HERE}/iphone_saved/{t0}_rgb.jpg"); meta = json.load(open(f"{HERE}/iphone_saved/{t0}_intrinsics.json"))
K = fc.iphone_K(meta, img.shape[1]); R, t, err = fc.solve_pose(img, K)
LAYERS = 9
slots = mi.slots_for_positions(R, t, K, LAYERS)
print(f"run {os.path.basename(run)}: {len(marks)} moves, {len(corr)} corrected by you (moves {sorted(corr)})\n")

seqs = sorted(marks)
results = {}
for s in seqs:
    prev = marks[seqs[seqs.index(s) - 1]]["mark_ms"] if seqs.index(s) else None
    results[s] = mi.infer(run, marks[s]["mark_ms"], prev, start_ms, marks[s]["pos"], slots, list(range(LAYERS)))
truth = {s: corr.get(s) for s in seqs}          # None = not corrected: assumed right (see below)

def pick(res, cands, how):
    det = {L: v for L, v in res["detail"].items() if (L - 1) in cands}
    def sc(v):
        z = list(v.values())
        return {"max": max(z), "mean": float(np.mean(z)), "min": min(z) if len(z) > 1 else min(z) - 3}[how]
    return sorted(det, key=lambda L: -sc(det[L]))

variants = [("current: every layer, strongest face", range(LAYERS), "max"),
            ("exclude the top layer (can't be pulled)", range(LAYERS - 1), "max"),
            ("exclude top layer + both cameras must agree (mean)", range(LAYERS - 1), "mean"),
            ("exclude top layer + weakest camera (min)", range(LAYERS - 1), "min")]
print("move | typed  | model said | you said | model's ranking")
for s in seqs:
    r = results[s]
    rk = pick(r, range(LAYERS), "max")[:4] if r["ok"] else r["reason"]
    pred = rk[0] if r["ok"] else None
    print(f"  {s}  | {marks[s]['pos']:6s} | {('L%d' % pred) if pred else '-':10s} | {('L%d' % truth[s]) if truth[s] else '(no correction)':16s} | {rk}")

print("\naccuracy, counting a move with no correction as 'model was right':")
for name, cands, how in variants:
    ok = tot = top2 = 0
    for s in seqs:
        r = results[s]
        if not r["ok"]:
            continue
        rk = pick(r, cands, how)
        ref = truth[s] or rk[0] if truth[s] is None else truth[s]
        if truth[s] is None:                      # assumed right under the CURRENT model's answer
            ref = pick(r, range(LAYERS), "max")[0]
        tot += 1; ok += rk[0] == ref; top2 += ref in rk[:2]
    print(f"  {name:55s} exact {ok}/{tot}   truth in top-2 {top2}/{tot}")
strict = [s for s in seqs if truth[s] and results[s]["ok"]]
print(f"\nSTRICT (only the {len(strict)} moves you explicitly corrected; the model was wrong on all of these by definition):")
for name, cands, how in variants:
    ok = sum(pick(results[s], cands, how)[0] == truth[s] for s in strict)
    t2 = sum(truth[s] in pick(results[s], cands, how)[:2] for s in strict)
    t3 = sum(truth[s] in pick(results[s], cands, how)[:3] for s in strict)
    print(f"  {name:55s} now right {ok}/{len(strict)}   truth in top-2 {t2}/{len(strict)}   top-3 {t3}/{len(strict)}")
