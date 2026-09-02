"""
Is the learned gate doing something per-token that the class mean hides?

Claim: with a1>0 it gives HIGH beta to the FIRST write to a key (||e|| large,
memory empty) and LOW beta to repeats (||e||~0, update is a no-op). The class
mean is dominated by repeats, so it reads low - which looked like "the gate got
STABLE backwards" but would actually be correct per-token behaviour the
per-class oracle cannot express.
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
print(f"trained, a1={m.ga.item():+.3f} a2={m.ga2.item():+.3f}\n", flush=True)

BT, CW, FW, RP = [], [], [], []
ev = np.random.default_rng(12345)
with torch.no_grad():
    for _ in range(4):
        x, cl, sn = T.sample(ev, 32); B, L = x.shape[0], T.L
        lg, beta = m(x[:, :-1])
        cw = F.pad(cl, (1, 0))[:, :L]                       # class of the WRITE key
        # first write to this key?  and: is this write a REPEAT of the stored value?
        first = np.zeros((B, L), bool); rep = np.zeros((B, L), bool)
        xs = x.numpy()
        for b in range(B):
            seen = set(); last = {}
            for t in range(1, L):
                kkey = int(xs[b, t - 1]); val = int(xs[b, t])
                first[b, t] = kkey not in seen
                rep[b, t] = last.get(kkey) == val
                seen.add(kkey); last[kkey] = val
        BT.append(beta); CW.append(cw); FW.append(torch.from_numpy(first))
        RP.append(torch.from_numpy(rep))
f = lambda z: torch.cat(z).reshape(-1).numpy()
bt, cw, fw, rp = f(BT), f(CW), f(FW).astype(bool), f(RP).astype(bool)

print("mean beta, split by whether the write is NEW information")
print(f"{'':10s}{'FIRST write':>14}{'repeat of stored':>19}{'changed value':>16}{'class mean':>13}")
for i, nm in enumerate(("STABLE", "NOISY", "REVISED")):
    c = cw == i
    a = bt[c & fw].mean()
    b = bt[c & ~fw & rp].mean()
    d = bt[c & ~fw & ~rp].mean()
    print(f"  {nm:8s}{a:>14.4f}{b:>19.4f}{d:>16.4f}{bt[c].mean():>13.4f}")
print(f"\noracle's per-class constants for reference:  STABLE 0.752  NOISY 0.231  REVISED 0.617")
print("a per-class gate must use ONE number per row; the learned gate varies within it.")
