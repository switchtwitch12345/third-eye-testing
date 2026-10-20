#!/usr/bin/env python3
"""
Which blocks may be pulled?  (green = yes, red = no)

Everything follows from two things: the official rule, and balance.

GAME RULE
  1. A block can only come from BELOW the highest complete layer.  The top complete layer, and anything above
     it, stays put.  House rule of this game: the bottom layer is never pulled from either.

BALANCE (statics, all blocks weigh the same)
  Each layer rests on the layer below it.  Whatever sits above a layer behaves as one lump of weight at its
  centre of mass (COM).  The tower stands only while that COM is over the "footprint" of the blocks still in the
  layer below it -- the smallest rectangle around them.  A block is 25 mm wide, 75 mm long, and a layer is 75 mm
  across, so for a layer with blocks in slots S (0 = one edge, 1 = middle, 2 = other edge):

      footprint across the layer  = from the outer edge of the lowest slot to the outer edge of the highest slot
      footprint along the blocks  = the full 75 mm

  2. After a pull, for EVERY layer of the tower the COM of everything above that layer must lie inside that
     layer's footprint, with a safety margin (MARGIN mm: real blocks wobble, and a COM right on the edge is a
     tower about to tip).  Layers below or at the pulled block can be unbalanced BY the pull; layers above it were
     already unbalanced -- the tower is standing on friction and luck -- and any pull underneath would jolt it,
     so those count too.  This is what makes a heavily played tower end in 'no moves left'.
  3. A layer must never be emptied (the layers above would drop).

What this gives on a straight tower (and why):
  * Full layer: any block may go (what remains is {M,R}, {L,R} or {L,M}: the COM is still over them).
  * {L,M} or {M,R} left: the EDGE block may go (only the middle remains, COM still over it) but NOT the middle
    (only an edge block would remain, and the COM hangs 25 mm off the side: the tower falls).
  * {L,R} left (middle gone): neither may go (one edge block alone cannot hold the tower).
  * {M} alone: nothing may go (it is the last block).
  And because the check uses the real COM of everything above, a lopsided tower is handled too: if the upper
  layers already lean to one side, a pull that adds to the lean is red even where the layer alone would be fine.
"""
import numpy as np

W, LEN = 25.0, 75.0          # block width and length, mm (a layer is LEN across)
BOTTOM_LAYER_LEGAL = False   # house rule of this game (official Jenga allows it)
MARGIN = 4.0                 # mm of safety between the COM and the edge of the footprint


def _centre(layer, slot):
    a = -LEN / 2 + W * (slot + 0.5)
    return (0.0, a) if layer % 2 == 0 else (a, 0.0)        # even layers run along x, slots across y


def _footprint(layer, slots):
    lo, hi = -LEN / 2 + W * min(slots), -LEN / 2 + W * (max(slots) + 1)
    return (-LEN / 2, LEN / 2, lo, hi) if layer % 2 == 0 else (lo, hi, -LEN / 2, LEN / 2)


def failing_layers(layers, present, margin=MARGIN):
    """Layers (0-based) whose blocks cannot carry what is above them, for the set `present` of (layer, slot)."""
    bad = []
    for j in range(layers):
        slots = [s for s in range(3) if (j, s) in present]
        above = [_centre(l, s) for (l, s) in present if l > j]
        if not above:
            continue
        if not slots:
            bad.append(j)
            continue
        cx, cy = np.mean(above, axis=0)
        x0, x1, y0, y1 = _footprint(j, slots)
        if not (x0 + margin <= cx <= x1 - margin and y0 + margin <= cy <= y1 - margin):
            bad.append(j)
    return bad


def explain(layers, removed, margin=MARGIN):
    """{(layer, slot): (may_pull, reason)} for every block still standing."""
    gone = {tuple(b) for b in removed}
    present = {(l, s) for l in range(layers) for s in range(3)} - gone
    complete = [l for l in range(layers) if all((l, s) in present for s in range(3))]
    top_complete = max(complete, default=-1)
    out = {}
    for (l, s) in sorted(present):
        if l == 0 and not BOTTOM_LAYER_LEGAL:
            out[(l, s)] = (False, "house rule: the bottom layer is never pulled from")
            continue
        if l >= top_complete:
            out[(l, s)] = (False, "top complete layer (or above it): the rules keep these in place")
            continue
        bad = failing_layers(layers, present - {(l, s)}, margin)
        if not bad:
            out[(l, s)] = (True, "tower stays balanced")
        elif bad[0] > l:
            out[(l, s)] = (False, f"layer {bad[0] + 1} above is already balancing on a thread; any pull below would jolt it")
        elif bad[0] == l:
            left = sorted(u for u in range(3) if u != s and (l, u) in present)
            why = ("it is the last block of its layer" if not left else
                   "the remaining block(s) of its layer would not sit under the weight above")
            out[(l, s)] = (False, why)
        else:
            out[(l, s)] = (False, f"it would shift weight so that layer {bad[0] + 1} could not carry it")
    return out


def removable(layers, removed, margin=MARGIN):
    """(layer, slot) -> True if that block may be pulled."""
    return {k: v[0] for k, v in explain(layers, removed, margin).items()}
