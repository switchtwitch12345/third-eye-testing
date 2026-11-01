#!/usr/bin/env python3
"""Checks the pull rules against cases worked out by hand.   .venv/bin/python test_jenga_rules.py"""
from jenga_rules import removable
import jenga_rules as jr

L, M, R = 0, 1, 2
N = 9


def ok(removed, block):
    return removable(N, set(removed))[block]


# fresh tower: everything below the top layer is fine, the top layer is not
r = removable(N, set())
assert all(v for (l, s), v in r.items() if 0 < l < N - 1)
assert not any(v for (l, s), v in r.items() if l == 0)       # house rule: never the bottom layer
assert not any(v for (l, s), v in r.items() if l == N - 1)

# the user's examples
assert not ok([(4, L)], (4, M))              # left gone from layer 5 -> pulling the middle drops it
assert ok([(4, L)], (4, R))                  # ...but the right one may go (middle stays)
assert not ok([(3, L), (3, R)], (3, M))      # both edges gone from layer 4 -> last block
# middle gone: the two edges hold the tower together, neither can leave
assert not ok([(2, M)], (2, L)) and not ok([(2, M)], (2, R))
# {L,M} / {M,R}: the edge may go, the middle may not
assert ok([(5, R)], (5, L))                  # {L,M} left: the edge may go, only the middle remains
assert not ok([(5, R)], (5, M))

# lone middle block: last block of its layer
assert not ok([(3, L), (3, R)], (3, M))

# balance across layers: layer 2 (index 1, runs along y) has only its left slot... never reachable by a legal pull,
# but the weight check must still see it.  Build a lean: layer index 2 keeps only {L,M} (right gone), and layer
# index 4 also keeps only {L,M}: the stack above layer 1 leans towards slot L.  Taking the layer-2 middle would
# leave one edge block under the lean -> red.
assert not ok([(2, R), (4, R)], (2, M))
# a lean that makes an otherwise fine pull unsafe: lower layer {M,R}, upper layers lean towards R-side's opposite
lean = [(2, R), (4, R), (6, R)]
assert ok(lean, (1, L))                      # near the bottom, balanced enough: still fine

# never empties a layer, never pulls from the top complete layer
assert not ok([(1, L), (1, M)], (1, R))
assert not ok([], (N - 1, M))
print("all rule checks passed")


# --- blocks put back on top ----------------------------------------------------------------------------------
n, rem = jr.effective_tower(8, [(3, 0)])
assert n == 9 and (8, 0) not in rem and (8, 1) in rem and (8, 2) in rem and (3, 0) in rem, (n, rem)   # one pulled block starts layer 9
n, rem = jr.effective_tower(8, [(3, 0), (4, 2), (2, 1)])
assert n == 9 and {(l, s) for l in range(9) for s in range(3)} - set(rem) >= {(8, 0), (8, 1), (8, 2)}, (n, rem)   # three pulls fill it
n, rem = jr.effective_tower(8, [(3, 0)], placed=False)
assert n == 8 and rem == [(3, 0)], (n, rem)
pullable = jr.removable(*jr.effective_tower(8, [(3, 0)]))
assert all(not pullable.get((8, s), False) for s in range(3)), "the new partial top layer must never be pullable"
print("blocks-put-back-on-top checks passed")
