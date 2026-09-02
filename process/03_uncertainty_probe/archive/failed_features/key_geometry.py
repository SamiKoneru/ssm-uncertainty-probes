"""
Is the cosine-0.68 key coherence real, or an artifact of a shared mean direction?

If k_i = mu + delta_i, then cos ~ ||mu||^2/(||mu||^2+||delta||^2) and the keys
cluster in a cone even when the delta_i are well separated. Centering would then
recover the capacity for free.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np, torch, torch.nn.functional as F
import mixed_task as T
from gated_model import GatedDeltaLayer

torch.manual_seed(0); rng = np.random.default_rng(0)
m = GatedDeltaLayer(T.N_SYM, gate="mlp")
opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
for s in range(800):
    x, _, _ = T.sample(rng, 24)
    lg, _ = m(x[:, :-1])
    loss = F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
print(f"trained, loss {loss.item():.4f}\n")

with torch.no_grad():
    # the key for symbol s is W_k emb(s), since k comes from a hard shift
    K = F.normalize(m.to_k(m.embed(torch.arange(T.N_SYM))), dim=-1)   # [40,128]
    G = K @ K.T
    off = ~torch.eye(T.N_SYM, dtype=bool)
    raw = G[off].mean().item()

    mu = K.mean(0)
    Kc = F.normalize(K - mu, dim=-1)                                   # centred
    Gc = Kc @ Kc.T
    cen = Gc[off].mean().item()

    # how much of each key is the common direction?
    frac = ((K @ F.normalize(mu, dim=0)) ** 2).mean().item()

    # whitened: decorrelate fully
    U, S_, V = torch.linalg.svd(K - mu, full_matrices=False)
    Kw = F.normalize(U @ V, dim=-1)
    Gw = Kw @ Kw.T; wht = Gw[off].mean().item()

d = 128
print(f"{'configuration':28s}{'mean cross-cos':>16}{'|cos|':>9}{'capacity ~1/c^2':>18}")
for nm, c in (("raw learned keys", raw), ("centred (mu removed)", cen),
              ("whitened", wht), ("random unit vectors", 1/np.sqrt(d))):
    ac = abs(c) if abs(c) > 1e-6 else 1/np.sqrt(d)
    print(f"  {nm:26s}{c:>16.4f}{abs(c):>9.4f}{1/ac**2:>18.1f}")
print(f"\nfraction of key energy in the SHARED MEAN direction: {frac:.4f}")
print(f"({T.N_SYM} keys in {d} dims; random-vector baseline cos = {1/np.sqrt(d):.4f})")
