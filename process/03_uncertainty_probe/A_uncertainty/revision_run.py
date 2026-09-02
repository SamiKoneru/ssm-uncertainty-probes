import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import revision_task as T
from delta_model import DeltaLayer

def run(fixed_beta=None, seed=0, steps=800, batch=24):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = DeltaLayer(T.N_SYM, d_k=128, d_v=128, v_from_embed=True,
                   kq_from_embed=True, fixed_beta=fixed_beta)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for s in range(steps):
        x, _, _, _ = T.sample(rng, batch)
        lg, _ = m(x[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()

    U, Y, E, CL, ST, SN, bl = [], [], [], [], [], [], []
    with torch.no_grad():
        for _ in range(8):
            x, cl, st, sn = T.sample(rng, 32)
            lg, u = m(x[:, :-1]); pr = F.softmax(lg, -1)
            Y.append((pr.argmax(-1) != x[:, 1:]).float()); U.append(u)
            E.append(-(pr * torch.log(pr + 1e-9)).sum(-1))
            CL.append(cl); ST.append(st); SN.append(sn)
            if fixed_beta is None:
                emb = m.embed(x[:, :-1])
                xc = F.silu(m.conv(emb.transpose(1,2))[..., :T.L].transpose(1,2))
                bl.append(torch.sigmoid(m.to_b(xc)).mean().item())
    f = lambda z: torch.cat(z).reshape(-1).numpy()
    return (f(Y), f(U), f(E), f(CL), f(ST).astype(bool), f(SN).astype(bool),
            (np.mean(bl) if bl else fixed_beta), loss.item())

if __name__ == "__main__":
    y, u, e, cl, st, sn, beta, loss = run()
    print(f"learned beta = {beta:.4f}   (was 0.0022 on the STATIONARY induction task)")
    print(f"final loss {loss:.3f}\n")
    print(f"{'class':10s}{'n':>8}{'accuracy':>10}{'mean entropy':>14}")
    for i, nm in enumerate(("stable", "slow", "volatile")):
        m_ = (cl == i) & sn
        print(f"{nm:10s}{m_.sum():>8}{1-y[m_].mean():>10.3f}{e[m_].mean():>14.3f}")

    slow = (cl == 1) & sn
    print(f"\nSLOW symbols split by whether memory is STALE:")
    for nm, m_ in (("fresh (memory ok)", slow & ~st), ("STALE (memory wrong)", slow & st)):
        print(f"  {nm:22s} n={m_.sum():>6}  acc {1-y[m_].mean():.3f}  entropy {e[m_].mean():.3f}")

    thr = np.percentile(e, 33)
    conf = e < thr
    cw, cr = conf & (y > .5), conf & (y <= .5)
    print(f"\nCONFIDENTLY WRONG steps (bottom-33% entropy AND wrong): {cw.sum()}")
    print(f"confidently right: {cr.sum()}   -> error rate among confident steps "
          f"{cw.sum()/(cw.sum()+cr.sum()):.3f}")
    if cw.sum() > 30:
        lab = np.r_[np.ones(cw.sum()), np.zeros(cr.sum())]
        print("\nAmong CONFIDENT steps only, can each signal tell wrong from right?")
        for nm, sig in (("u (OURS)", u), ("entropy", e)):
            print(f"  {nm:12s} AUROC {roc_auc_score(lab, np.r_[sig[cw], sig[cr]]):.3f}")
    print("\nOverall AUROC -> ERROR:")
    for nm, sig in (("u (OURS)", u), ("entropy", e)):
        print(f"  {nm:12s} {roc_auc_score(y, sig):.3f}")
