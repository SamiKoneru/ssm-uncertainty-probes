"""Does the K-tape bank fix the bandpass artifact? Same sweep as mechanism_test."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch
from stability_ssm import multiscale_u

L, periods = 2000, [2, 4, 8, 16, 32, 64, 128, 256, 512]
sq = np.stack([np.where((np.arange(L) // (p // 2)) % 2 == 0, 1.0, -1.0) for p in periods], -1)
x = torch.tensor(sq[None], dtype=torch.float32)          # [1,L,C] one channel per period

def sweep(taus, agg="max"):
    la = torch.log(torch.exp(-1.0 / torch.tensor(taus)))[:, None].repeat(1, x.shape[-1])
    u = multiscale_u(x, la, n_groups=x.shape[-1], agg=agg)[0]     # [L,C]
    return u[-500:].mean(0).numpy()

cfgs = {
    "K=2  (spec v1: tau 2,100)": [2.0, 100.0],
    "K=4  (tau 1..128)":         list(np.logspace(0, np.log10(128), 4)),
    "K=8  (tau 1..128)":         list(np.logspace(0, np.log10(128), 8)),
    "K=8  (tau 1..512)":         list(np.logspace(0, np.log10(512), 8)),
    "K=12 (tau 1..512)":         list(np.logspace(0, np.log10(512), 12)),
}
print("mean u per square-wave period. every column is MAXIMALLY unstable,")
print("so a scale-free detector should be FLAT across the row.\n")
print(f"{'config':28s}" + "".join(f"P={p:<6d}" for p in periods) + "  spread(max/min)")
for name, taus in cfgs.items():
    v = sweep(taus)
    print(f"{name:28s}" + "".join(f"{q:<8.3f}" for q in v) + f"  {v.max()/v.min():.2f}x")
