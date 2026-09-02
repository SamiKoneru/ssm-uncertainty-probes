"""Is u a per-location field, and does it localize to the churning channels?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch
from stability_ssm import multiscale_u

L, N = 2000, 8
taus = list(np.logspace(np.log10(0.2), np.log10(512), 12))
la = torch.log(torch.tensor(np.exp(-1.0 / np.array(taus))).float())[:, None].repeat(1, N)
t = np.arange(L)

def u_field(sig):
    x = torch.tensor(sig[None], dtype=torch.float32)
    return multiscale_u(x, la, n_groups=N, agg="median")[0, -500:].mean(0).numpy()

# --- A: channels 0-3 steady, channels 4-7 churning at different rates ------
A = np.stack([np.ones(L)] * 4 + [np.where((t // p) % 2 == 0, 1.0, 2.0)
                                 for p in (4, 16, 64, 256)], -1)
print("A: channels 0-3 held constant, 4-7 flip-flopping")
print("   channel   " + "".join(f"{i:>9d}" for i in range(N)))
print("   u         " + "".join(f"{v:>9.4f}" for v in u_field(A)))

# --- B: the SAME channel carries a steady fact AND a churning one ----------
steady, churn = np.ones(L), np.where((t // 16) % 2 == 0, 0.0, 1.0)
B = np.stack([steady, churn, steady + churn, 0.5 * steady + 0.5 * churn,
              steady + 0.2 * churn, steady + 0.05 * churn,
              np.ones(L), np.ones(L)], -1)
lbl = ["steady", "churn", "sum", "half+half", "steady+0.2c", "steady+0.05c", "-", "-"]
print("\nB: superposing a steady value and a churning one in ONE channel")
print("   channel   " + "".join(f"{s:>13s}" for s in lbl[:6]))
print("   u         " + "".join(f"{v:>13.4f}" for v in u_field(B)[:6]))
