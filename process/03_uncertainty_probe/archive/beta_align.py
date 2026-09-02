"""Is 'mean beta by class' aligned to the right token?

beta_t gates the write to key k_t = W_k emb(x_{t-1}), so it should be labelled
with the class of x_{t-1}, NOT x_t.  mixed_run.py used cl = class(x_t).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch, torch.nn.functional as F
import mixed_task as T
from gated_model import GatedDeltaLayer

torch.manual_seed(0); rng = np.random.default_rng(0)
m = GatedDeltaLayer(T.N_SYM, gate="both")
opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
for s in range(800):
    x, _, _ = T.sample(rng, 24)
    lg, _ = m(x[:, :-1])
    loss = F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
print(f"gate='both' trained, loss {loss.item():.4f}", flush=True)

B_, CQ, CW, SN = [], [], [], []
ev = np.random.default_rng(12345)
with torch.no_grad():
    for _ in range(6):
        x, cl, sn = T.sample(ev, 32)
        lg, beta = m(x[:, :-1])
        B_.append(beta); SN.append(sn)
        CQ.append(cl)                                    # class of x_t   (what I reported)
        CW.append(F.pad(cl, (1, 0))[:, :T.L])            # class of x_{t-1} (the WRITE key)
f = lambda z: torch.cat(z).reshape(-1).numpy()
bt, cq, cw, sn = f(B_), f(CQ), f(CW), f(SN).astype(bool)

print(f"\n{'labelling':38s}{'STABLE':>10}{'NOISY':>10}{'REVISED':>10}{'N vs R':>10}")
for nm, c in (("by class(x_t)   <- what I reported", cq),
              ("by class(x_t-1) <- the WRITE key", cw)):
    v = [bt[(c == i) & sn].mean() for i in range(3)]
    print(f"  {nm:36s}{v[0]:>10.4f}{v[1]:>10.4f}{v[2]:>10.4f}{v[1]-v[2]:>+10.4f}")
