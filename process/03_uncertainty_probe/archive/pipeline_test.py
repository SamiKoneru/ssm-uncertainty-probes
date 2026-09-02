"""
End-to-end pipeline test with LEARNED projections.

The delta rule solves this task perfectly (100% accuracy), which is convenient:
with no model errors, any signal in u is purely about contestation rather than
about model failure.  So the question here is the mechanism question:

    does |R.q| separate queries to CONTESTED keys from queries to STABLE keys,
    when k and q are learned rather than hand-picked orthogonal vectors?

u is a pure observer: detached, never in the loss, never in the output.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch, torch.nn.functional as F, kv_task
from sklearn.metrics import roc_auc_score
from delta_model import DeltaLayer

def go(d_k, seed, steps=500, batch=24):
    V, L = kv_task.configure(32, 32, 100)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = DeltaLayer(V, d_k=d_k, d_v=d_k)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for s in range(steps):
        seq, msk, _ = kv_task.sample(rng, batch)
        lg, _ = m(seq); sl = msk[:, :-1]
        loss = F.cross_entropy(lg[:, :-1][sl], seq[:, 1:][sl])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    U, C, W, E, P = [], [], [], [], []
    with torch.no_grad():
        for _ in range(6):
            seq, msk, con = kv_task.sample(rng, 32)
            lg, u = m(seq); sl = msk[:, :-1]
            pr = F.softmax(lg[:, :-1], -1)
            U.append(u[:, :-1][sl]); C.append(con[:, :-1][sl].float())
            W.append((pr.argmax(-1) != seq[:, 1:]).float()[sl])
            E.append((-(pr * torch.log(pr + 1e-9)).sum(-1))[sl])
            P.append(torch.arange(L - 1).float().expand(32, -1)[sl])
    f = lambda z: torch.cat(z).numpy()
    u, c, w, e, p = f(U), f(C), f(W), f(E), f(P)
    return dict(acc=1 - w.mean(), u_stable=u[c == 0].mean(), u_con=u[c == 1].mean(),
                auc_u=roc_auc_score(c, u), auc_ent=roc_auc_score(c, e),
                auc_pos=roc_auc_score(c, p), n=len(c))

if __name__ == "__main__":
    print("target: AUROC of each signal at identifying a query to a CONTESTED key")
    print(f"{'d_k':>5}{'seed':>6}{'acc':>7}{'u(stable)':>11}{'u(contested)':>14}"
          f"{'AUROC u':>10}{'AUROC ent':>11}{'AUROC pos':>11}")
    for d_k in (32, 64, 128):
        for seed in (0, 1):
            r = go(d_k, seed)
            print(f"{d_k:>5}{seed:>6}{r['acc']:>7.3f}{r['u_stable']:>11.3f}"
                  f"{r['u_con']:>14.3f}{r['auc_u']:>10.3f}{r['auc_ent']:>11.3f}"
                  f"{r['auc_pos']:>11.3f}")
