"""
Can a NON-overwriting value memory see RECURRENCE, which the delta rule erases?

M_t = a M_{t-1} + v_t k_t^T     ACCUMULATES values (never overwrites)
Z_t = a Z_{t-1} + k_t           per-key write count

  membership = <v_t, M k_t> / (||v_t|| ||M k_t||)
  coherence  = ||M k_t|| / (Z.k_t)     repeats add COHERENTLY (n), distinct
                                        values add INCOHERENTLY (sqrt n)

Measured on a trained model WITHOUT wiring anything into the gate. The decisive
number is AUROC restricted to CHANGE writes: among writes that changed the
value, is the new value one the key has SEEN BEFORE (NOISY) or fresh (REVISED)?
Every earlier statistic scored ~0.5 there.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
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
print(f"trained, loss {loss.item():.4f}\n", flush=True)

A = 0.99
cols = {n: [] for n in ("memb", "coh", "s1", "s2", "clw", "sn", "chg")}
ev = np.random.default_rng(12345)
with torch.no_grad():
    for _ in range(4):
        x, cl, sn = T.sample(ev, 32); B, L = x.shape[0], T.L
        emb = m.embed(x[:, :-1])
        xc = F.silu(m.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))
        sh = F.pad(emb, (0, 0, 1, 0))[:, :L]
        k = F.normalize(m.to_k(sh), dim=-1); v = m.to_v(emb)
        S = emb.new_zeros(B, 128, 128)
        M = emb.new_zeros(B, 128, 128); Z = emb.new_zeros(B, 128)
        c = {n: [] for n in ("memb", "coh", "s1", "s2")}
        for t in range(L):
            kt, vt = k[:, t], v[:, t]
            stored = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1)
            e = stored - vt
            mk = torch.bmm(M, kt.unsqueeze(-1)).squeeze(-1)          # accumulated values
            zk = (Z * kt).sum(-1).clamp_min(1e-3)                    # write count
            c["memb"].append((vt * mk).sum(-1) / (vt.norm(dim=-1) * mk.norm(dim=-1) + 1e-6))
            c["coh"].append(mk.norm(dim=-1) / (zk * vt.norm(dim=-1) + 1e-6))
            c["s1"].append(e.norm(dim=-1))
            c["s2"].append((vt * stored).sum(-1) / (vt.norm(dim=-1) ** 2 + 1e-6))
            # gate beta, same as training
            g = m.gmlp(torch.stack([m.to_b(xc)[..., 0][:, t],
                                    torch.log1p(e.norm(dim=-1)),
                                    c["s2"][-1]], -1))[..., 0]
            b = torch.sigmoid(g)
            S = S - b[:, None, None] * e.unsqueeze(-1) * kt.unsqueeze(1)
            M = A * M + torch.bmm(vt.unsqueeze(-1), kt.unsqueeze(1))
            Z = A * Z + kt
        for n in c: cols[n].append(torch.stack(c[n], 1))
        cols["clw"].append(F.pad(cl, (1, 0))[:, :L]); cols["sn"].append(sn)
        xs = x.numpy(); chg = np.zeros((B, L), bool)
        for b_ in range(B):
            last = {}
            for t in range(1, L):
                kk = int(xs[b_, t - 1]); val = int(xs[b_, t])
                chg[b_, t] = (kk in last) and last[kk] != val
                last[kk] = val
        cols["chg"].append(torch.from_numpy(chg))
D = {n: torch.cat(v_).reshape(-1).numpy() for n, v_ in cols.items()}
clw, sn, chg = D["clw"], D["sn"].astype(bool), D["chg"].astype(bool)

print("mean by class of the WRITE key (seen keys only)")
print(f"{'statistic':14s}{'STABLE':>10}{'NOISY':>10}{'REVISED':>10}")
for st in ("s1", "s2", "memb", "coh"):
    v_ = [D[st][(clw == i) & sn].mean() for i in range(3)]
    print(f"  {st:12s}{v_[0]:>10.4f}{v_[1]:>10.4f}{v_[2]:>10.4f}")

print("\nAUROC, NOISY vs REVISED   (every earlier statistic sat at ~0.5)")
for scope, msk in (("all seen writes", sn), ("CHANGE writes only", sn & chg)):
    sub = msk & np.isin(clw, [1, 2]); lab = (clw[sub] == 2).astype(int)
    print(f"  [{scope}, n={sub.sum()}]")
    for st in ("s1", "s2", "memb", "coh"):
        au = roc_auc_score(lab, D[st][sub])
        star = "  <-- SEPARATES" if abs(au - 0.5) > 0.10 else ""
        print(f"     {st:10s} {au:.4f}{star}")
