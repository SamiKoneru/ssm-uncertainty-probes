import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import induction_task as T
from delta_model import DeltaLayer

def go(seed, steps=700, batch=24, d_k=128):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = DeltaLayer(T.N_SYM, d_k=d_k, d_v=d_k, v_from_embed=True, kq_from_embed=True)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for s in range(steps):
        x, _, _ = T.sample(rng, batch)
        lg, _ = m(x[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % 200 == 0: print(f"   seed{seed} step {s+1} loss {loss.item():.3f}", flush=True)

    U, Y, C, E, X, P, S = [], [], [], [], [], [], []
    with torch.no_grad():
        for _ in range(6):
            x, con, seen = T.sample(rng, 32)
            lg, u = m(x[:, :-1]); pr = F.softmax(lg, -1)
            Y.append((pr.argmax(-1) != x[:, 1:]).float()); U.append(u); C.append(con.float())
            E.append(-(pr * torch.log(pr + 1e-9)).sum(-1)); X.append(1 - pr.max(-1).values)
            P.append(torch.arange(T.L).float().expand(32, -1)); S.append(seen)
    f = lambda z: torch.cat(z).reshape(-1).numpy()
    return dict(u=f(U), y=f(Y), c=f(C), e=f(E), x=f(X), p=f(P), s=f(S).astype(bool))

if __name__ == "__main__":
    for seed in (0, 1):
        r = go(seed)
        u, y, c, e, xx, p, s = (r[k] for k in "uycexps")
        print(f"\n===== seed {seed} =====")
        print(f"accuracy  all {1-y.mean():.3f}   stable {1-y[c==0].mean():.3f}"
              f"   contested {1-y[c==1].mean():.3f}  (ceiling {1/T.N_OPT:.2f})")
        print(f"{'signal':22s}{'AUROC->ERROR':>14}{'AUROC->CONTESTED':>19}")
        for nm, sig in (("u (OURS)", u), ("output entropy", e), ("1 - max softmax", xx),
                        ("position (control)", p)):
            print(f"  {nm:20s}{roc_auc_score(y, sig):>14.3f}{roc_auc_score(c, sig):>19.3f}")
        m2 = s                                     # repeat symbols only (drop first occurrences)
        print(f"  [repeat symbols only, n={m2.sum()}]")
        for nm, sig in (("u (OURS)", u), ("output entropy", e)):
            print(f"  {nm:20s}{roc_auc_score(y[m2], sig[m2]):>14.3f}"
                  f"{roc_auc_score(c[m2], sig[m2]):>19.3f}")
