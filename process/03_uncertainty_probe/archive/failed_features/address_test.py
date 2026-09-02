"""
Does the accumulator FLOOR at the instantaneous surprise, as it should?

u read at the READ QUERY   |R.q_t|  does NOT contain ||e_t||  (k_t . q_t ~ 0)
u read at the WRITE KEY    |R.k_t|  DOES contain ||e_t||      (k_t . k_t = 1)

If the accumulator is built correctly, |R.k_t| must be at least as informative
as ||e_t|| alone, with a possibly higher ceiling from the history.
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
print(f"model trained, loss {loss.item():.4f}", flush=True)

F_ = {n: [] for n in ("ne", "uq", "uk_prev", "uk_post", "en", "mx", "y")}
ev = np.random.default_rng(777)
with torch.no_grad():
    for _ in range(4):
        x, cl, sn = T.sample(ev, 32); B, L = x.shape[0], T.L
        emb = m.embed(x[:, :-1])
        xc = F.silu(m.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))
        sh = F.pad(emb, (0, 0, 1, 0))[:, :L]
        q = F.normalize(m.to_q(emb), dim=-1); k = F.normalize(m.to_k(sh), dim=-1)
        v = m.to_v(emb); beta = torch.sigmoid(m.to_b(xc))[..., 0]
        S = emb.new_zeros(B, 128, 128); R = emb.new_zeros(B, 128)
        cols = {n: [] for n in ("ne", "uq", "uk_prev", "uk_post")}
        o_l = []
        for t in range(L):
            kt, qt = k[:, t], q[:, t]
            cols["uk_prev"].append((R * kt).sum(-1).abs())      # BEFORE this write
            e = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1) - v[:, t]
            n = e.norm(dim=-1)
            S = S - beta[:, t, None, None] * e.unsqueeze(-1) * kt.unsqueeze(1)
            R = 0.99 * R + n.unsqueeze(-1) * kt                 # includes ||e_t||*k_t
            cols["ne"].append(n)
            cols["uk_post"].append((R * kt).sum(-1).abs())      # AFTER, contains ||e_t||
            cols["uq"].append((R * qt).sum(-1).abs())           # what I had been measuring
            o_l.append(torch.bmm(S, qt.unsqueeze(-1)).squeeze(-1))
        lg = m.head(m.norm(torch.stack(o_l, 1))); pr = F.softmax(lg, -1)
        for n_ in cols: F_[n_].append(torch.stack(cols[n_], 1))
        F_["en"].append(-(pr * torch.log(pr + 1e-9)).sum(-1))
        F_["mx"].append(pr.max(-1).values)
        F_["y"].append((lg.argmax(-1) != x[:, 1:]).float())
D = {n: torch.cat(v_).numpy() for n, v_ in F_.items()}
ns = D["y"].shape[0]; tr = np.arange(ns) < ns // 2
fl = lambda a, s: a[s].reshape(-1)
ytr, yte = fl(D["y"], tr), fl(D["y"], ~tr)

def auc(sel):
    Xtr = np.stack([fl(D[n], tr) for n in sel], 1)
    Xte = np.stack([fl(D[n], ~tr) for n in sel], 1)
    g = HistGradientBoostingClassifier(max_iter=300, early_stopping=True,
                                       random_state=0).fit(Xtr, ytr)
    return roc_auc_score(yte, g.predict_proba(Xte)[:, 1])

print(f"\n{'feature set':46s}{'test AUROC':>12}")
for nm, sel in (
    ("||e_t||  instantaneous surprise, alone",      ["ne"]),
    ("|R.q_t|  read at QUERY  (what I measured)",   ["uq"]),
    ("|R.k_t|  read at WRITE KEY, before write",    ["uk_prev"]),
    ("|R.k_t|  read at WRITE KEY, after write",     ["uk_post"]),
    ("||e_t|| + |R.k_t| after   (floor + history)", ["ne", "uk_post"]),
    ("||e_t|| + |R.k_t| before  (no double count)", ["ne", "uk_prev"]),
    ("all surprise features",                       ["ne", "uq", "uk_prev", "uk_post"]),
    ("entropy + maxprob  (reference)",              ["en", "mx"]),
    ("all surprise + entropy + maxprob",            ["ne","uq","uk_prev","uk_post","en","mx"]),
):
    print(f"  {nm:44s}{auc(sel):>12.4f}", flush=True)
