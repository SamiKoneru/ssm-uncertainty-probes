"""Delta rule for the value + accumulated residual for the uncertainty."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
D, L, a, beta = 8, 900, 0.99, 1.0
kA, kB, kC = np.eye(D)[0], np.eye(D)[1], np.eye(D)[2]
vA, vC, vB1, vB2 = np.eye(D)[4], np.eye(D)[5], np.eye(D)[6], np.eye(D)[7]

S = np.zeros((D, D)); R = np.zeros(D); Z = np.zeros(D)
for t in range(L):
    w = t % 3
    if w == 0:   k, v = kA, vA
    elif w == 1: k, v = kB, (vB1 if (t // 3) % 2 == 0 else vB2)
    else:        k, v = kC, vC
    resid = np.linalg.norm(S @ k - v)                     # what delta already computes
    S = S @ (np.eye(D) - beta * np.outer(k, k)) + beta * np.outer(v, k)   # delta write
    R = a * R + resid * k                                 # accumulate surprise AT THE KEY
    Z = a * Z + k

print("key   delta-rule recall (the value)      u = accumulated surprise at that key")
for n, q, in (("A", kA), ("B", kB), ("C", kC)):
    print(f"  {n}   {np.round(S @ q, 2)}   {(R @ q) / (Z @ q):.4f}")
print("\nA and C: same value every write.  B: alternates between two values.")
print(f"extra state: R and Z are {D}-vectors, vs {D*D} per extra matrix tape")
