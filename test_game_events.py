#!/usr/bin/env python3
"""Synthetic-game test of game_events.Analyzer using their ray-cast tower renderer.
Run:  .venv/bin/python -I test_game_events.py
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "third_party", "STEPWISE"))
sys.path.insert(0, os.path.join(HERE, "third_party", "STEPWISE", "tests"))
sys.path.insert(0, HERE)

from synthetic_tower import render, CORNER_A, K          # their test helper
from thirdeye.perception.tower import DepthView
import game_events as ge

rng = np.random.default_rng(0)
R, T = CORNER_A


def frame(occ, pushed=None, noise=3.0, hand=False):
    d = render(occ, R, T, pushed) + rng.normal(0, noise, (192, 256))
    if hand:
        d[30:150, 50:210] = 200.0           # a hand ~20 cm from the camera
    return ge.read_ends(DepthView(d, K, R, T))


def run(seq, noise_label, ref_n=6):
    an = ge.Analyzer(seq[:ref_n])
    ev_all = []
    for i in range(ref_n, len(seq)):
        ev, _ = an.step(i, i / 3.0, seq[i])
        ev_all += ev
    return ev_all


full = [[1, 1, 1] for _ in range(9)]
rm = [r[:] for r in full]; rm[4][1] = 0
top = [r[:] for r in rm] + [[1, 0, 0]]
seq = ([frame(full) for _ in range(9)] +
       [frame(full, hand=True) for _ in range(3)] +
       [frame(rm) for _ in range(4)] +
       [frame(rm, hand=True) for _ in range(2)] +
       [frame(rm, pushed={(6, 0): 12.0}) for _ in range(4)] +
       [frame(rm) for _ in range(4)] +
       [frame(top) for _ in range(4)])

ok = True
print("== scripted game: remove (L5,s1) ; push (L7,s0) 12 mm ; place on top (L10,s0) ==")
for e in run(seq, "3mm"):
    print(f"   {e.kind:6s} layer {e.layer + 1:2d} slot {e.slot}  {e.note}")
kinds = [(e.kind, e.layer, e.slot) for e in run(seq, "3mm")]
want = [("REMOVE", 4, 1), ("PUSH", 6, 0), ("ADD", 9, 0)]
for w in want:
    hit = w in kinds
    ok &= hit
    print(f"   expect {w}: {'OK' if hit else 'MISSED'}")
extra = [k for k in kinds if k not in want]
print(f"   unexpected events: {extra if extra else 'none'}")
ok &= not extra

for noise in (3.0, 5.0, 8.0):
    quiet = [frame(full, noise=noise) for _ in range(60)]
    ev = run(quiet, "")
    print(f"== 60 intact frames, depth noise {noise:.0f} mm: {len(ev)} false events {[(e.kind, e.layer, e.slot) for e in ev][:4]}")
    if noise <= 5.0:
        ok &= len(ev) == 0

# how deep does a hole look from one corner? (informs ALONE)
an = ge.Analyzer([frame(full, noise=1.0) for _ in range(6)])
_, rel, _ = an.classify(frame(rm, noise=1.0))
hole = [(s, v) for s, v in rel.items() if (s.layer, s.slot) == (4, 1)]
print(f"== recess of the removed block seen from one corner: {hole[0][1]:+.0f} mm (ALONE={ge.ALONE})"
      if hole else "== removed slot not visible from this corner")
print("\nRESULT:", "PASS" if ok else "FAIL")
