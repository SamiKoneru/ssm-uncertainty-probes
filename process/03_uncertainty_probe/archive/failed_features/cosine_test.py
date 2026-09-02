"""
Does OVERLAP separate recurrence from novelty where MAGNITUDE cannot?

||e_t|| = ||S_{t-1}k_t - v_t||        deviation from the stored average.
   large for NOISY (v is one of 4 options, S k their mean)
   large for REVISED (v is fresh, S k the stale value)      -> SAME, no separation

overlap = how much of v_t is ALREADY stored at that key:
   cos = <v, Sk> / (||v|| ||Sk||)
   prj = <v, Sk> / ||v||^2            projection coefficient
   NOISY   -> v is a component of the stored mixture -> overlap > 0
   REVISED -> v is a symbol never written there     -> overlap ~ 0

Also tests whether accumulating overlap into a state helps, keeping the
instantaneous and historical terms SEPARATE (the lesson from address_test).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
import mixed_task as T
from gated_model import GatedDeltaLayer

torch.manual_seed(0); rng = np.random.default_rng(0)
m = GatedDeltaLayer(T.N_SYM, gate="none")
opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
for s in range(800):
    x, _, _ = T.sample(rng, 24)
    lg, _ = m(x[:, :-1])
    loss = F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
print(f"model trained, loss {loss.item():.4f}\n", flush=True)

names = ("ne", "cos", "prj", "uk", "ck", "en", "mx", "y", "clw", "sn")
D = {n: [] for n in names}
ev = np.random.default_rng(777)
with torch.no_grad():
    for _ in range(4):
        x, cl, sn = T.sample(ev, 32); B, L = x.shape[0], T.L
        emb = m.embed(x[:, :-1])
        xc = F.silu(m.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))
        sh = F.pad(emb, (0, 0, 1, 0))[:, :L]
        q = F.normalize(m.to_q(emb), dim=-1); k = F.normalize(m.to_k(sh), dim=-1)
        v = m.to_v(emb); beta = torch.sigmoid(m.to_b(xc))[..., 0]
        S = emb.new_zeros(B, 128, 128)
        R = emb.new_zeros(B, 128); C = emb.new_zeros(B, 128)     # ||e|| and overlap memories
        c = {n: [] for n in ("ne", "cos", "prj", "uk", "ck")}; o_l = []
        for t in range(L):
            kt, qt, vt = k[:, t], q[:, t], v[:, t]
            c["uk"].append((R * kt).sum(-1).abs()); c["ck"].append((C * kt).sum(-1))
            stored = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1)   # S_{t-1} k_t
            e = stored - vt; n_ = e.norm(dim=-1)
            ip = (vt * stored).sum(-1)
            c["ne"].append(n_)
            c["cos"].append(ip / (vt.norm(dim=-1) * stored.norm(dim=-1) + 1e-6))
            c["prj"].append(ip / (vt.norm(dim=-1) ** 2 + 1e-6))
            S = S - beta[:, t, None, None] * e.unsqueeze(-1) * kt.unsqueeze(1)
            R = 0.99 * R + n_.unsqueeze(-1) * kt
            C = 0.99 * C + c["prj"][-1].unsqueeze(-1) * kt
            o_l.append(torch.bmm(S, qt.unsqueeze(-1)).squeeze(-1))
        lg = m.head(m.norm(torch.stack(o_l, 1))); pr = F.softmax(lg, -1)
        for n_ in c: D[n_].append(torch.stack(c[n_], 1))
        D["en"].append(-(pr * torch.log(pr + 1e-9)).sum(-1)); D["mx"].append(pr.max(-1).values)
        D["y"].append((lg.argmax(-1) != x[:, 1:]).float())
        D["clw"].append(F.pad(cl, (1, 0))[:, :L])                # class of the WRITE key = cl[t-1]
        D["sn"].append(sn)
Dn = {n: torch.cat(v_).numpy() for n, v_ in D.items()}

clw, sn = Dn["clw"], Dn["sn"].astype(bool)
print("mean of each statistic, by the CLASS OF THE KEY BEING WRITTEN")
print(f"{'statistic':12s}{'STABLE':>10}{'NOISY':>10}{'REVISED':>10}{'NOISY vs REVISED':>19}")
for st in ("ne", "cos", "prj"):
    a = Dn[st]
    v_ = [a[(clw == i) & sn].mean() for i in range(3)]
    print(f"  {st:10s}{v_[0]:>10.4f}{v_[1]:>10.4f}{v_[2]:>10.4f}{v_[1]-v_[2]:>+19.4f}")

ns = Dn["y"].shape[0]; tr = np.arange(ns) < ns // 2
fl = lambda a, s: a[s].reshape(-1)
ytr, yte = fl(Dn["y"], tr), fl(Dn["y"], ~tr)
def auc(sel):
    Xtr = np.stack([fl(Dn[n], tr) for n in sel], 1)
    Xte = np.stack([fl(Dn[n], ~tr) for n in sel], 1)
    g = HistGradientBoostingClassifier(max_iter=300, early_stopping=True,
                                       random_state=0).fit(Xtr, ytr)
    return roc_auc_score(yte, g.predict_proba(Xte)[:, 1])
print(f"\n{'feature set':44s}{'test AUROC':>12}")
for nm, sel in (("||e|| alone", ["ne"]), ("overlap (cos) alone", ["cos"]),
                ("overlap (prj) alone", ["prj"]), ("||e|| + cos + prj", ["ne","cos","prj"]),
                ("||e|| + |R.k| (magnitude only)", ["ne","uk"]),
                ("||e|| + cos + prj + |R.k| + |C.k|", ["ne","cos","prj","uk","ck"]),
                ("entropy + maxprob (reference)", ["en","mx"]),
                ("everything", ["ne","cos","prj","uk","ck","en","mx"])):
    print(f"  {nm:42s}{auc(sel):>12.4f}", flush=True)
