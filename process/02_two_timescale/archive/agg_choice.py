"""Which aggregator over the K-1 bands? Metric: separation of 1 shift vs many."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch
from stability_ssm import ema_bank, bank_gains
from shift_count import make, L, HALF, taus12

def band_responses(sig, taus):
    x = torch.tensor(sig[None, :, None], dtype=torch.float32)
    la = torch.log(torch.tensor(np.exp(-1.0 / np.array(taus))).float())[:, None]
    H = ema_bank(x, la) * bank_gains(la, L, x.device)[None]
    return (H[:, :-1] - H[:, 1:]).abs()[0, :, -HALF:, 0]          # [K-1, HALF]

AGGS = {
    "max":            lambda b: b.amax(0),
    "mean":           lambda b: b.mean(0),
    "median":         lambda b: b.median(0).values,
    "q25":            lambda b: b.quantile(0.25, dim=0),
    "q75":            lambda b: b.quantile(0.75, dim=0),
    "trimmed(top 3)": lambda b: b.sort(0).values[:-3].mean(0),
}
shifts = [1, 2, 4, 8, 16, 32, 64]
B = {n: band_responses(make(n), taus12) for n in shifts}

print(f"{'agg':16s}" + "".join(f"{n:>9d}" for n in shifts) + f"{'64/1 ratio':>13}{'grad to':>10}")
for name, f in AGGS.items():
    v = [f(B[n]).mean().item() for n in shifts]
    ng = {"max": "1 band", "mean": "all 11", "median": "1 band",
          "q25": "1 band", "q75": "1 band", "trimmed(top 3)": "8 bands"}[name]
    print(f"{name:16s}" + "".join(f"{q:>9.4f}" for q in v)
          + f"{v[-1]/max(v[0],1e-12):>12.1f}x{ng:>10}")
