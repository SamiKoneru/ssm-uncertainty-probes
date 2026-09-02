"""
DISTRIBUTION SHIFT.  Train at one volatility, evaluate at another.

The model learns "SLOW symbols revise ~15% of the time" and hedges by that
amount.  At eval we raise the SLOW rate without retraining, so the learned
hedge is systematically too small: the model becomes OVERCONFIDENT.

entropy reports the stale training-time prior.
u measures the volatility actually observed in THIS sequence.

If u has any structural advantage anywhere, it is here.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")

import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import revision_task as T
from delta_model import DeltaLayer

TRAIN_RATE = 0.15

def train(seed=0, steps=800, batch=24):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = DeltaLayer(T.N_SYM, d_k=128, d_v=128, v_from_embed=True, kq_from_embed=True)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for s in range(steps):
        x, _, _, _ = T.sample(rng, batch, p_rev=np.array([0.0, TRAIN_RATE, 1.0]))
        lg, _ = m(x[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    torch.save(m.state_dict(), os.path.join(_CK, "shift_model.pt")); return m, rng, loss.item()

@torch.no_grad()
def evaluate(m, rng, rate, n=8, batch=32):
    Y, U, E, X, CL, SN = [], [], [], [], [], []
    for _ in range(n):
        x, cl, st, sn = T.sample(rng, batch, p_rev=np.array([0.0, rate, 1.0]))
        lg, u = m(x[:, :-1]); pr = F.softmax(lg, -1)
        Y.append((pr.argmax(-1) != x[:, 1:]).float()); U.append(u)
        E.append(-(pr * torch.log(pr + 1e-9)).sum(-1)); X.append(pr.max(-1).values)
        CL.append(cl); SN.append(sn)
    f = lambda z: torch.cat(z).reshape(-1).numpy()
    return f(Y), f(U), f(E), f(X), f(CL), f(SN).astype(bool)

def logistic_gain(y, e, u, seed=0):
    z = lambda a: (a - a.mean()) / (a.std() + 1e-9)
    n = len(y); idx = np.random.default_rng(seed).permutation(n); tr, te = idx[:n//2], idx[n//2:]
    out = []
    for cols in ([z(e)], [z(e), z(u)]):
        X = np.stack(cols, 1)
        lr = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        out.append(roc_auc_score(y[te], lr.predict_proba(X[te])[:, 1]))
    return out

if __name__ == "__main__":
    m, rng, loss = train()
    print(f"trained at SLOW rate {TRAIN_RATE}, final loss {loss:.3f}\n")
    print(f"{'eval':>6}{'acc':>8}{'conf':>8}{'calib':>8}{'entropy':>9}"
          f"{'u->err':>9}{'E->err':>9}{'E only':>9}{'E + u':>9}{'delta':>9}")
    print(f"{'rate':>6}{'':>8}{'':>8}{'gap':>8}{'':>9}{'':>9}{'':>9}{'(logit)':>9}{'(logit)':>9}{'':>9}")
    for rate in (0.15, 0.30, 0.45, 0.60, 0.75):
        y, u, e, mx, cl, sn = evaluate(m, rng, rate)
        s_ = (cl == 1) & sn                                  # SLOW symbols, seen before
        acc = 1 - y[s_].mean(); conf = mx[s_].mean()
        a1, a2 = logistic_gain(y[s_], e[s_], u[s_])
        print(f"{rate:>6.2f}{acc:>8.3f}{conf:>8.3f}{conf-acc:>+8.3f}{e[s_].mean():>9.3f}"
              f"{roc_auc_score(y[s_],u[s_]):>9.3f}{roc_auc_score(y[s_],e[s_]):>9.3f}"
              f"{a1:>9.4f}{a2:>9.4f}{a2-a1:>+9.4f}", flush=True)
    print("\ncalib gap = mean confidence - accuracy.  positive = OVERCONFIDENT.")
    print("delta = what u adds on top of entropy (held-out AUROC).")
