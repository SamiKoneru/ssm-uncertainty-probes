"""
CEILING: how much is the NOISY-vs-REVISED distinction actually worth?

  none          input-only gate (cannot differentiate; class is per-sequence)
  mlp           our best LEARNED gate, infers everything from s1,s2
  oracle_class  beta = sigmoid(learned scalar per TRUE class). Perfect class
                knowledge, nothing else. Isolates the value of the distinction.
  oracle_mlp    MLP over [base, s1, s2, one-hot TRUE class]. Class knowledge
                PLUS the features.

If oracle_class barely beats 'none', the axis is not worth engineering for and
the MLP gate is already near the achievable limit.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch, torch.nn.functional as F
import mixed_task as T
from gated_model import GatedDeltaLayer

def run(gate, seed=0, steps=800, batch=24):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = GatedDeltaLayer(T.N_SYM, gate=gate)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for s in range(steps):
        x, cl, _ = T.sample(rng, batch)
        cw = F.pad(cl, (1, 0))[:, :T.L]                       # class of the WRITE key
        lg, _ = m(x[:, :-1], cls_w=cw)
        loss = F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    L_, A, B_, CQ, CW, SN = [], [], [], [], [], []
    ev = np.random.default_rng(12345)
    with torch.no_grad():
        for _ in range(6):
            x, cl, sn = T.sample(ev, 32)
            cw = F.pad(cl, (1, 0))[:, :T.L]
            lg, beta = m(x[:, :-1], cls_w=cw)
            L_.append(F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1)).item())
            A.append((lg.argmax(-1) == x[:, 1:]).float()); B_.append(beta)
            CQ.append(cl); CW.append(cw); SN.append(sn)
    f = lambda z: torch.cat(z).reshape(-1).numpy()
    return np.mean(L_), f(A), f(B_), f(CQ), f(CW), f(SN).astype(bool)

if __name__ == "__main__":
    print("acc by class(x_t) | beta by class(x_{t-1}, the WRITE key)\n")
    print(f"{'gate':>13}{'loss':>9}{'acc':>8} |{'accSTAB':>9}{'accNOIS':>8}{'accREV':>8}"
          f" |{'bSTAB':>8}{'bNOIS':>8}{'bREV':>8}")
    base = None
    for gate in ("none", "mlp", "oracle_class", "oracle_mlp"):
        loss, acc, bt, cq, cw, sn = run(gate)
        if base is None: base = loss
        a = [acc[(cq == i) & sn].mean() for i in range(3)]
        b = [bt[(cw == i) & sn].mean() for i in range(3)]
        print(f"{gate:>13}{loss:>9.4f}{acc.mean():>8.4f} |{a[0]:>9.3f}{a[1]:>8.3f}{a[2]:>8.3f}"
              f" |{b[0]:>8.4f}{b[1]:>8.4f}{b[2]:>8.4f}"
              f"   ({100*(base-loss)/base:+.2f}% loss)", flush=True)
