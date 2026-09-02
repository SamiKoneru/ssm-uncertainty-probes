"""Why phi kills selectivity, and a formulation that avoids needing phi at all."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
np.seterr(all="ignore")
nrm = lambda M: M / np.linalg.norm(M, axis=-1, keepdims=True)

print("selectivity of the readout kernel: self-match vs cross-match")
print(f"{'dims':>6}{'raw k.k self/cross':>22}{'relu^2 self/cross':>21}")
for dk in (32, 128, 512, 2048):
    K = nrm(np.random.default_rng(0).normal(size=(16, dk)))
    P = np.maximum(K, 0) ** 2
    off = ~np.eye(16, dtype=bool)
    raw = (K @ K.T); ph = (P @ P.T)
    print(f"{dk:>6}{np.diag(raw).mean()/np.abs(raw[off]).mean():>21.1f}x"
          f"{np.diag(ph).mean()/np.abs(ph[off]).mean():>20.1f}x")

def run(K, mode, dv=32, rounds=80, a=0.99, seed=0):
    rng = np.random.default_rng(seed); nk, dk = K.shape
    V1, V2 = nrm(rng.normal(size=(nk, dv))), nrm(rng.normal(size=(nk, dv)))
    con = np.arange(nk) % 2 == 1
    S, R, Z, E, Ez = np.zeros((dv, dk)), np.zeros(dk), np.zeros(dk), np.zeros((dv, dk)), np.zeros(dk)
    phi = lambda x: np.maximum(x, 0) ** 2
    for rd in range(rounds):
        for i in range(nk):
            k = K[i]; v = V2[i] if (con[i] and rd % 2) else V1[i]
            e = S @ k - v; S = S - np.outer(e, k)
            R, Z = a * R + np.linalg.norm(e) * phi(k), a * Z + phi(k)
            E, Ez = a * E + np.outer(e, k), a * Ez + k          # residual VECTORS, raw keys
    if mode == "R/Z ratio (phi)":
        u = np.array([phi(K[i]) @ R for i in range(nk)]) / np.array([phi(K[i]) @ Z for i in range(nk)])
    else:                                                       # ||E q||, raw keys, no division
        u = np.array([np.linalg.norm(E @ K[i]) for i in range(nk)])
    return u[~con].mean(), u[con].mean()

print("\nseparation of contested from stable keys (16 random keys)")
print(f"{'dims':>6}{'R/Z ratio (phi)':>20}{'||E q||  (raw k)':>20}")
for dk in (32, 128, 512):
    K = nrm(np.random.default_rng(1).normal(size=(16, dk)))
    a1 = run(K, "R/Z ratio (phi)"); a2 = run(K, "E")
    print(f"{dk:>6}{a1[1]/a1[0]:>19.2f}x{a2[1]/a2[0]:>19.2f}x")
