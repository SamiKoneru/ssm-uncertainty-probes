"""
Three ways to build the uncertainty memory, on the SAME trained model.

  accum   R = a*R + ||e||*k                       plain superposition (vanilla LA)
  delta   R = R - bR*(R.k - ||e||)*k              erase-before-write, EMA when bR<1
  white   u = |R^T C^-1 q|, C = E[k k^T]          undo key correlation at readout
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_CK = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "checkpoints")

import numpy as np, torch, torch.nn.functional as F, kv_task
from sklearn.metrics import roc_auc_score
from delta_model import DeltaLayer

V, L = kv_task.configure(32, 32, 100)
m = DeltaLayer(V, d_k=128, d_v=128); m.load_state_dict(torch.load(os.path.join(_CK, "kv_model.pt"))); m.eval()
rng = np.random.default_rng(11)
BR = 0.3                                   # write rate for the delta-R variant
res = {n: [] for n in ("accum", "delta", "accum+white", "delta+white")}
C = []

with torch.no_grad():
    for _ in range(8):
        seq, msk, con = kv_task.sample(rng, 32); B = seq.shape[0]
        x = F.silu(m.conv(m.embed(seq).transpose(1, 2))[..., :L].transpose(1, 2))
        q = F.normalize(m.to_q(x), dim=-1); k = F.normalize(m.to_k(x), dim=-1)
        v = m.to_v(x); beta = torch.sigmoid(m.to_b(x))
        S = x.new_zeros(B, 128, 128)
        Ra = x.new_zeros(B, 128); Rd = x.new_zeros(B, 128)
        Cov = torch.eye(128).expand(B, -1, -1).clone() * 1e-3
        out = {n: [] for n in res}
        wq = q[:, 0]
        for t in range(L):
            kt, bt = k[:, t], beta[:, t]
            e = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1) - v[:, t]
            S = S - bt.unsqueeze(-1) * e.unsqueeze(-1) * kt.unsqueeze(1)
            n = e.norm(dim=-1, keepdim=True)
            Ra = 0.99 * Ra + n * kt                                       # accumulate
            Rd = Rd - BR * ((Rd * kt).sum(-1, keepdim=True) - n) * kt     # delta rule
            Cov = 0.99 * Cov + torch.bmm(kt.unsqueeze(-1), kt.unsqueeze(1))
            qt = q[:, t]
            out["accum"].append((Ra * qt).sum(-1).abs())
            out["delta"].append((Rd * qt).sum(-1).abs())
            if t % 3 == 1:                                                # whiten only where read
                reg = Cov + 1e-2 * Cov.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None] * torch.eye(128)
                wq = torch.linalg.solve(reg, qt.unsqueeze(-1)).squeeze(-1)
                wq = F.normalize(wq, dim=-1)
            out["accum+white"].append((Ra * wq).sum(-1).abs())
            out["delta+white"].append((Rd * wq).sum(-1).abs())
        sl = msk[:, :-1]
        for nme in res: res[nme].append(torch.stack(out[nme], 1)[:, :-1][sl])
        C.append(con[:, :-1][sl].float())

c = torch.cat(C).numpy()
print("AUROC at identifying a query to a CONTESTED key")
for nme, val in res.items():
    u = torch.cat(val).numpy()
    print(f"  {nme:14s} {roc_auc_score(c, u):.3f}    stable {u[c==0].mean():8.3f}"
          f"   contested {u[c==1].mean():8.3f}")
