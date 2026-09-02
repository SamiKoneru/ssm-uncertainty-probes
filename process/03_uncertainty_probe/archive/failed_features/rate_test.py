"""
Does a DELTA-RULE rate memory supply the 3a-vs-3b signal that (s1,s2) cannot?

  none        input-only gate
  mlp         our best: MLP over [base, s1, s2]
  rate_accum  + a rate memory written by PLAIN ACCUMULATION (the control -
              this is what R did, and its crosstalk is what killed it)
  rate_delta  + a rate memory written by the DELTA RULE

Key diagnostic: AUROC of c_t at separating NOISY from REVISED. (s1,s2) gets
0.476, i.e. chance. If c_t clears that, the rate memory works.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import mixed_task as T
from gated_model import GatedDeltaLayer

def run(gate, seed=0, steps=800, batch=24):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = GatedDeltaLayer(T.N_SYM, gate=gate)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for s in range(steps):
        x, _, _ = T.sample(rng, batch)
        lg, _ = m(x[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    L_, A, B_, CQ, CW, SN, CT = [], [], [], [], [], [], []
    ev = np.random.default_rng(12345)
    with torch.no_grad():
        for _ in range(6):
            x, cl, sn = T.sample(ev, 32)
            lg, beta = m(x[:, :-1])
            L_.append(F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1)).item())
            A.append((lg.argmax(-1) == x[:, 1:]).float()); B_.append(beta)
            CQ.append(cl); CW.append(F.pad(cl, (1, 0))[:, :T.L]); SN.append(sn)
            CT.append(m.last_c)
    f = lambda z: torch.cat(z).reshape(-1).numpy()
    bc = torch.sigmoid(m.p_bc).item() if gate.startswith("rate_") else None
    return np.mean(L_), f(A), f(B_), f(CQ), f(CW), f(SN).astype(bool), f(CT), bc

if __name__ == "__main__":
    print(f"{'gate':>12}{'loss':>9}{'acc':>8}{'gain':>9} |{'accSTAB':>9}{'accNOIS':>8}{'accREV':>8}"
          f" |{'bSTAB':>8}{'bNOIS':>8}{'bREV':>8}")
    base = None; keep = {}
    for gate in ("mlp", "rate_delta", "rate_prj"):
        loss, acc, bt, cq, cw, sn, ct, bc = run(gate)
        if base is None: base = 1.6329   # established baseline, reproduced across runs
        a = [acc[(cq == i) & sn].mean() for i in range(3)]
        b = [bt[(cw == i) & sn].mean() for i in range(3)]
        print(f"{gate:>12}{loss:>9.4f}{acc.mean():>8.4f}{100*(base-loss)/base:>8.2f}%"
              f" |{a[0]:>9.3f}{a[1]:>8.3f}{a[2]:>8.3f} |{b[0]:>8.4f}{b[1]:>8.4f}{b[2]:>8.4f}"
              + (f"   beta_c={bc:.3f}" if bc else ""), flush=True)
        keep[gate] = (ct, cw, sn)

    print("\nCAN c_t SEE THE AXIS (s1,s2) IS BLIND TO?  AUROC, NOISY vs REVISED")
    print("  (s1,s2) baseline from signinit_test: 0.4763  = chance")
    for gate in ("rate_delta", "rate_prj"):
        ct, cw, sn = keep[gate]
        msk = sn & np.isin(cw, [1, 2])
        au = roc_auc_score((cw[msk] == 2).astype(int), ct[msk])
        print(f"  {gate:12s} c_t AUROC = {au:.4f}   "
              f"(mean c: NOISY {ct[sn & (cw==1)].mean():.3f}, "
              f"REVISED {ct[sn & (cw==2)].mean():.3f})")
