"""
Decouple the feature maps: RAW keys for the value memory S (sharp addressing,
no division), phi(k) for R and Z (needs a well-behaved ratio).

Question: does any phi give BOTH a valid normalizer AND enough localization to
tell a contested key from a stable one?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

PHIS = {
    "identity (raw)": lambda x: x,
    "relu":           lambda x: np.maximum(x, 0),
    "elu+1":          lambda x: np.where(x > 0, x + 1, np.exp(x)),
    "relu^2":         lambda x: np.maximum(x, 0) ** 2,
    "exp(4x)":        lambda x: np.exp(4 * x),
}

def run(phi, n_keys=16, dk=32, dv=32, rounds=80, a=0.99, seed=0):
    rng = np.random.default_rng(seed)
    nrm = lambda M: M / np.linalg.norm(M, axis=-1, keepdims=True)
    K, V1, V2 = nrm(rng.normal(size=(n_keys, dk))), nrm(rng.normal(size=(n_keys, dv))), \
                nrm(rng.normal(size=(n_keys, dv)))
    contested = np.arange(n_keys) % 2 == 1
    S, R, Z = np.zeros((dv, dk)), np.zeros(dk), np.zeros(dk)
    last = V1.copy()
    for rd in range(rounds):
        for i in range(n_keys):
            k = K[i]
            v = V2[i] if (contested[i] and rd % 2 == 1) else V1[i]
            e = S @ k - v
            S = S - np.outer(e, k)                    # delta rule, RAW key
            pk = phi(k)
            R, Z = a * R + np.linalg.norm(e) * pk, a * Z + pk
            last[i] = v
    zq = np.array([phi(K[i]) @ Z for i in range(n_keys)])
    u  = np.array([phi(K[i]) @ R for i in range(n_keys)]) / zq
    rec = np.mean([np.linalg.norm(S @ K[i] - last[i]) for i in range(n_keys)])
    return u[~contested], u[contested], zq, rec

print(f"{'phi':16s}{'u(stable)':>11}{'u(contested)':>14}{'separation':>12}"
      f"{'min Z.q':>10}{'recall err':>12}")
for name, phi in PHIS.items():
    us, uc, zq, rec = run(phi)
    sep = uc.mean() / us.mean() if us.mean() > 0 else float("nan")
    print(f"{name:16s}{us.mean():>11.4f}{uc.mean():>14.4f}{sep:>11.2f}x"
          f"{zq.min():>10.3f}{rec:>12.4f}")

print("\nkeys per dimension (crowding), phi=relu^2:")
for nk in (8, 16, 32, 64):
    us, uc, zq, rec = run(PHIS["relu^2"], n_keys=nk)
    print(f"  {nk:3d} keys / 32 dims   separation {uc.mean()/us.mean():.2f}x"
          f"   recall err {rec:.3f}")
