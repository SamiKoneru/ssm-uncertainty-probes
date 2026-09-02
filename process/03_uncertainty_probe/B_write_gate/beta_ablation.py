"""
Does the model use PARTIAL writes to encode uncertainty?

beta=1 is an exact overwrite: S q returns only the latest value, so a contested
key's retrieval looks identical to a stable key's.  If partial writes are the
uncertainty mechanism, beta=1 should keep accuracy (the last value is still the
best point estimate) but WRECK calibration.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import induction_task as T
from delta_model import DeltaLayer

def run(fixed_beta, seed=0, steps=700, batch=24):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = DeltaLayer(T.N_SYM, d_k=128, d_v=128, v_from_embed=True,
                   kq_from_embed=True, fixed_beta=fixed_beta)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for s in range(steps):
        x, _, _ = T.sample(rng, batch)
        lg, _ = m(x[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    U, Y, C, E = [], [], [], []
    with torch.no_grad():
        bl = []
        for _ in range(6):
            x, con, seen = T.sample(rng, 32)
            lg, u = m(x[:, :-1]); pr = F.softmax(lg, -1)
            Y.append((pr.argmax(-1) != x[:, 1:]).float()); U.append(u); C.append(con.float())
            E.append(-(pr * torch.log(pr + 1e-9)).sum(-1))
            if fixed_beta is None:
                emb = m.embed(x[:, :-1])
                xc = F.silu(m.conv(emb.transpose(1,2))[..., :T.L].transpose(1,2))
                bl.append(torch.sigmoid(m.to_b(xc)).mean().item())
    f = lambda z: torch.cat(z).reshape(-1).numpy()
    y, u, c, e = f(Y), f(U), f(C), f(E)
    return dict(loss=loss.item(), acc=1-y.mean(), acc_s=1-y[c==0].mean(), acc_c=1-y[c==1].mean(),
                ent_s=e[c==0].mean(), ent_c=e[c==1].mean(),
                auc_e_err=roc_auc_score(y, e), auc_e_con=roc_auc_score(c, e),
                auc_u_err=roc_auc_score(y, u), auc_u_con=roc_auc_score(c, u),
                beta=(np.mean(bl) if bl else fixed_beta))

if __name__ == "__main__":
    print(f"{'beta':>10}{'loss':>8}{'acc':>7}{'acc_s':>7}{'acc_c':>7}"
          f"{'ent_s':>8}{'ent_c':>8}{'E->err':>8}{'E->con':>8}{'u->err':>8}{'u->con':>8}")
    for fb, nm in ((None, "learned"), (1.0, "1.0"), (0.5, "0.5"), (0.2, "0.2")):
        r = run(fb)
        print(f"{nm:>10}{r['loss']:>8.3f}{r['acc']:>7.3f}{r['acc_s']:>7.3f}{r['acc_c']:>7.3f}"
              f"{r['ent_s']:>8.3f}{r['ent_c']:>8.3f}{r['auc_e_err']:>8.3f}{r['auc_e_con']:>8.3f}"
              f"{r['auc_u_err']:>8.3f}{r['auc_u_con']:>8.3f}", flush=True)
        if fb is None: print(f"{'':>10}(learned beta mean = {r['beta']:.4f})", flush=True)
