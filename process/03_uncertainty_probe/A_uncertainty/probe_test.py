"""
UPPER BOUND on every state-derived uncertainty signal.

Entropy is itself a deterministic function of the pre-head state o_t:
    entropy = f(head(LayerNorm(o_t)))
so a sufficiently expressive probe on o_t can REPRODUCE entropy exactly.
Anything the probe gets ABOVE entropy is information in the state that entropy
does not capture.  If a trained probe can only match entropy, then no
hand-designed statistic (u, ||e||, cosine overlap, two-timescale disagreement,
...) can ever beat it, and the whole family is exhausted.

Split is BY SEQUENCE, not by position, since positions within a sequence are
correlated and a position-level split would leak.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
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

# ---- extract per-position features -----------------------------------------
O, EN, MX, NE, UU, Y = [], [], [], [], [], []
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
        o_l, ne_l, u_l = [], [], []
        for t in range(L):
            kt = k[:, t]
            u_l.append((R * kt).sum(-1).abs())
            e = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1) - v[:, t]
            S = S - beta[:, t, None, None] * e.unsqueeze(-1) * kt.unsqueeze(1)
            o_l.append(torch.bmm(S, q[:, t].unsqueeze(-1)).squeeze(-1))
            ne_l.append(e.norm(dim=-1)); R = 0.99 * R + e.norm(dim=-1, keepdim=True) * kt
        o = torch.stack(o_l, 1)                                   # [B,L,128] pre-head state
        lg = m.head(m.norm(o)); pr = F.softmax(lg, -1)
        O.append(o); EN.append(-(pr * torch.log(pr + 1e-9)).sum(-1))
        MX.append(pr.max(-1).values); NE.append(torch.stack(ne_l, 1))
        UU.append(torch.stack(u_l, 1)); Y.append((lg.argmax(-1) != x[:, 1:]).float())

o = torch.cat(O).numpy(); en = torch.cat(EN).numpy(); mx = torch.cat(MX).numpy()
ne = torch.cat(NE).numpy(); uu = torch.cat(UU).numpy(); y = torch.cat(Y).numpy()
nseq = o.shape[0]; tr_s = np.arange(nseq) < nseq // 2          # SEQUENCE-level split
flat = lambda a, s: a[s].reshape(-1, a.shape[-1]) if a.ndim == 3 else a[s].reshape(-1)
ytr, yte = flat(y, tr_s), flat(y, ~tr_s)
print(f"probe train {len(ytr)} positions, test {len(yte)}, error rate {y.mean():.3f}\n", flush=True)

def feats(sel):
    parts = []
    for nm in sel:
        a = {"o": o, "en": en, "mx": mx, "ne": ne, "u": uu}[nm]
        parts.append(flat(a, tr_s) if a.ndim == 3 else flat(a, tr_s)[:, None])
    tr = np.concatenate(parts, 1)
    parts = []
    for nm in sel:
        a = {"o": o, "en": en, "mx": mx, "ne": ne, "u": uu}[nm]
        parts.append(flat(a, ~tr_s) if a.ndim == 3 else flat(a, ~tr_s)[:, None])
    te = np.concatenate(parts, 1)
    mu, sd = tr.mean(0), tr.std(0) + 1e-8
    return (tr - mu) / sd, (te - mu) / sd

def mlp_auc(sel, hidden=64, epochs=30, seed=0):
    Xtr, Xte = feats(sel)
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    op = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)
    for ep in range(epochs):
        perm = torch.randperm(len(yt))
        for i in range(0, len(yt), 4096):
            b = perm[i:i+4096]
            l = F.binary_cross_entropy_with_logits(net(Xt[b])[:, 0], yt[b])
            op.zero_grad(); l.backward(); op.step()
    with torch.no_grad():
        p = net(torch.tensor(Xte, dtype=torch.float32))[:, 0].numpy()
    return roc_auc_score(yte, p)

def lin_auc(sel):
    Xtr, Xte = feats(sel)
    lr = LogisticRegression(max_iter=3000).fit(Xtr, ytr)
    return roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])

np.savez("probe_feats.npz", o=o, en=en, mx=mx, ne=ne, uu=uu, y=y)

from sklearn.ensemble import HistGradientBoostingClassifier
def gbm_auc(sel):
    Xtr, Xte = feats(sel)
    g = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                       early_stopping=True, random_state=0).fit(Xtr, ytr)
    return roc_auc_score(yte, g.predict_proba(Xte)[:, 1])

print(f"{'probe (gradient-boosted trees)':44s}{'test AUROC':>12}")
for nm, sel in (("entropy alone", ['en']), ("entropy + maxprob", ['en','mx']),
                ("u alone", ['u']), ("||e|| + u (surprise only)", ['ne','u']),
                ("pre-head state o (128d)", ['o']),
                ("o + ||e|| + u", ['o','ne','u']),
                ("o + ||e|| + u + entropy + maxprob", ['o','ne','u','en','mx'])):
    print(f"  {nm:42s}{gbm_auc(sel):>12.4f}", flush=True)
print()
print(f"{'probe (MLP, for comparison)':44s}{'test AUROC':>12}")
print(f"{'entropy alone (logistic)':44s}{lin_auc(['en']):>12.4f}")
print(f"{'entropy + maxprob (logistic)':44s}{lin_auc(['en','mx']):>12.4f}")
print(f"{'entropy + maxprob (MLP)':44s}{mlp_auc(['en','mx']):>12.4f}")
print(f"{'u alone (logistic)':44s}{lin_auc(['u']):>12.4f}")
print(f"{'MLP on pre-head state o (128d)':44s}{mlp_auc(['o']):>12.4f}")
print(f"{'MLP on o + ||e|| + u':44s}{mlp_auc(['o','ne','u']):>12.4f}")
print(f"{'MLP on o + ||e|| + u + entropy + maxprob':44s}{mlp_auc(['o','ne','u','en','mx']):>12.4f}")
