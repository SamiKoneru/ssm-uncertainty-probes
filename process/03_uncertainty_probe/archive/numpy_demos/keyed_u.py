"""Per-ITEM uncertainty: run the tape bank on a key-value memory, query it."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np, torch

D, L, K = 8, 1200, 12
taus = np.logspace(np.log10(0.3), np.log10(256), K)
alphas = np.exp(-1.0 / taus)

kA, kB, kC = np.eye(D)[0], np.eye(D)[1], np.eye(D)[2]      # three orthogonal keys
vA, vC = np.eye(D)[4], np.eye(D)[5]                        # stable values
vB1, vB2 = np.eye(D)[6], np.eye(D)[7]                      # B keeps flip-flopping

S = np.zeros((K, D, D))                                    # K tapes of the WHOLE table
Z = np.zeros((K, D))                                       # per-key accumulated weight
writes = 0
for t in range(L):
    which = t % 3
    if which == 0:   k, v = kA, vA                         # A: always the same value
    elif which == 1: k, v = kB, (vB1 if (t // 3) % 2 == 0 else vB2)   # B: contested
    else:            k, v = kC, vC                         # C: always the same value
    S = alphas[:, None, None] * S + np.outer(v, k)[None]
    Z = alphas[:, None] * Z + k[None]
    writes += 1

def u_at(q):
    # divide by the key's OWN accumulated weight, so recall does not depend
    # on how long ago this particular key was last written
    r = (S @ q) / (Z @ q + 1e-9)[:, None]                  # [K,D]
    bands = np.linalg.norm(r[:-1] - r[1:], axis=-1)        # [K-1]
    return np.median(bands), r[0], r[-1]

print("query   u (median over bands)   fast-tape recall      slow-tape recall")
for name, q, in (("A", kA), ("B", kB), ("C", kC)):
    u, rf, rs = u_at(q)
    print(f"  {name}   {u:>18.4f}   {np.round(rf,2)}  {np.round(rs,2)}")
print("\nA and C were written the same value every time; B alternated between two.")
