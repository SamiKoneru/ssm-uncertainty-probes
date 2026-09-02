"""
Three questions about the wrong-sign gate:
  1. does a NEGATIVE init on a1 (and positive on a2, both of which match the
     oracle's ordering) reach a better solution?
  2. after training, is the gradient on a1 still pushing consistently, i.e. did
     it converge or just run out of steps?
  3. can (s1,s2) even PREDICT the class per token? if not, the gap to the oracle
     is INFORMATION-limited and no optimisation fixes it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np, torch, torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import mixed_task as T
from gated_model import GatedDeltaLayer

def train(a_init, seed=0, steps=800, batch=24):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = GatedDeltaLayer(T.N_SYM, gate="both", a_init=a_init)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for s in range(steps):
        x, _, _ = T.sample(rng, batch)
        lg, _ = m(x[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m, rng

def evaluate(m):
    L_, A, B_, CW, SN = [], [], [], [], []
    ev = np.random.default_rng(12345)
    with torch.no_grad():
        for _ in range(6):
            x, cl, sn = T.sample(ev, 32)
            lg, beta = m(x[:, :-1])
            L_.append(F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1)).item())
            A.append((lg.argmax(-1) == x[:, 1:]).float()); B_.append(beta)
            CW.append(F.pad(cl, (1, 0))[:, :T.L]); SN.append(sn)
    f = lambda z: torch.cat(z).reshape(-1).numpy()
    return np.mean(L_), f(A), f(B_), f(CW), f(SN).astype(bool)

def grad_probe(m, n=25, batch=16):
    """Is a1 still being pushed? Report mean/std/t over n independent batches."""
    rng = np.random.default_rng(4242); g1, g2 = [], []
    for _ in range(n):
        x, _, _ = T.sample(rng, batch)
        lg, _ = m(x[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1))
        m.zero_grad(); loss.backward()
        g1.append(m.ga.grad.item()); g2.append(m.ga2.grad.item())
    out = []
    for nm, g in (("a1", np.array(g1)), ("a2", np.array(g2))):
        t = g.mean() / (g.std(ddof=1) / np.sqrt(len(g)) + 1e-12)
        out.append((nm, g.mean(), g.std(ddof=1), t))
    return out

if __name__ == "__main__":
    print(f"{'a_init (a1,a2)':>18}{'loss':>9}{'acc':>8}{'a1_final':>10}{'a2_final':>10}"
          f"{'bSTAB':>8}{'bNOIS':>8}{'bREV':>8}")
    models = {}
    for a_init, lab in (((0.0, 0.0), "(0, 0) default"), ((-1.0, 1.0), "(-1, +1) oracle-ish")):
        m, _ = train(a_init); models[lab] = m
        loss, acc, bt, cw, sn = evaluate(m)
        b = [bt[(cw == i) & sn].mean() for i in range(3)]
        print(f"{lab:>18}{loss:>9.4f}{acc.mean():>8.4f}{m.ga.item():>10.4f}"
              f"{m.ga2.item():>10.4f}{b[0]:>8.4f}{b[1]:>8.4f}{b[2]:>8.4f}", flush=True)

    print("\nGRADIENT PROBE after training (25 independent batches):")
    print(f"{'model':>22}{'param':>7}{'mean grad':>13}{'std':>11}{'t-stat':>10}")
    for lab, m in models.items():
        for nm, mu, sd, t in grad_probe(m):
            note = "  <- still pushing" if abs(t) > 3 else "  <- converged"
            print(f"{lab:>22}{nm:>7}{mu:>13.3e}{sd:>11.3e}{t:>10.2f}{note}")

    print("\nCLASS PREDICTABILITY from (s1,s2) per token:")
    m = models["(0, 0) default"]
    S1, S2, CW, SN = [], [], [], []
    ev = np.random.default_rng(999)
    with torch.no_grad():
        for _ in range(3):
            x, cl, sn = T.sample(ev, 32); B, L = x.shape[0], T.L
            emb = m.embed(x[:, :-1])
            xc = F.silu(m.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))
            sh = F.pad(emb, (0, 0, 1, 0))[:, :L]
            k = F.normalize(m.to_k(sh), dim=-1); v = m.to_v(emb)
            beta = torch.sigmoid(m.to_b(xc))[..., 0]
            S = emb.new_zeros(B, 128, 128); c1, c2 = [], []
            for t in range(L):
                kt, vt = k[:, t], v[:, t]
                stored = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1); e = stored - vt
                c1.append(torch.log1p(e.norm(dim=-1)))
                c2.append((vt * stored).sum(-1) / (vt.norm(dim=-1) ** 2 + 1e-6))
                S = S - beta[:, t, None, None] * e.unsqueeze(-1) * kt.unsqueeze(1)
            S1.append(torch.stack(c1, 1)); S2.append(torch.stack(c2, 1))
            CW.append(F.pad(cl, (1, 0))[:, :L]); SN.append(sn)
    f = lambda z: torch.cat(z).reshape(-1).numpy()
    s1, s2, cw, sn = f(S1), f(S2), f(CW), f(SN).astype(bool)
    X = np.stack([s1, s2], 1)[sn]; y = cw[sn].astype(int)
    n = len(y); tr = np.arange(n) < n // 2
    g = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[tr], y[tr])
    pred = g.predict(X[~tr]); pp = g.predict_proba(X[~tr])
    print(f"  3-way accuracy from (s1,s2):  {accuracy_score(y[~tr], pred):.4f}"
          f"   (chance {max(np.bincount(y)/len(y)):.4f})")
    for i, nm in enumerate(("STABLE", "NOISY", "REVISED")):
        print(f"    AUROC {nm:8s} vs rest: {roc_auc_score((y[~tr]==i).astype(int), pp[:, i]):.4f}")
    nr = np.isin(y[~tr], [1, 2])
    print(f"    AUROC NOISY vs REVISED only: "
          f"{roc_auc_score((y[~tr][nr]==2).astype(int), pp[nr, 2]):.4f}   <- the axis that matters")
