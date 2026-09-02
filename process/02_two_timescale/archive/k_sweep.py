"""
How high should K go?  No training needed: flatness and contrast are properties
of the tape spectrum alone.

  flatness = max/min of u across churn timescales   (want -> 1.0)
  floor    = u on a STABLE-but-noisy signal          (want low)
  contrast = u(unstable) / u(stable)                 (want high)

max-aggregation over K-1 bands should lift the floor as K grows, so contrast
may peak and then decay even while flatness keeps improving.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch
from stability_ssm import multiscale_u

L = 4000
rng = np.random.default_rng(0)
periods = [2, 4, 8, 16, 32, 64, 128, 256, 512]

unstable = np.stack([np.where((np.arange(L) // (p // 2)) % 2 == 0, 1.0, -1.0)
                     for p in periods], -1)
stable = np.stack([1.0 + 0.05 * rng.normal(size=L) for _ in periods], -1)  # steady + 5% noise

def u_of(sig, taus, agg):
    x = torch.tensor(sig[None], dtype=torch.float32)
    la = torch.log(torch.exp(-1.0 / torch.tensor(taus)))[:, None].repeat(1, x.shape[-1])
    return multiscale_u(x, la, n_groups=x.shape[-1], agg=agg)[0, -1000:].mean(0).numpy()

for agg in ("max", "mean"):
    print(f"\nagg = {agg}   (tapes log-spaced over tau 0.2..1024 in every row)")
    print(f"{'K':>4}{'flatness':>11}{'u(unstable)':>14}{'u(stable)':>12}{'contrast':>11}")
    for K in (2, 4, 8, 12, 16, 24, 32, 48):
        taus = list(np.logspace(np.log10(0.2), np.log10(1024), K))
        un, st = u_of(unstable, taus, agg), u_of(stable, taus, agg)
        print(f"{K:>4}{un.max()/un.min():>10.2f}x{un.mean():>14.4f}{st.mean():>12.4f}"
              f"{un.mean()/st.mean():>11.1f}x")
