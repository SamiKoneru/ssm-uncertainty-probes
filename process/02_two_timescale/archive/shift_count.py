"""Does u grow with the NUMBER of shifts, as intuition says it should?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch
from stability_ssm import ema_bank, bank_gains, _agg

L, HALF = 4000, 2000
taus12 = list(np.logspace(np.log10(0.2), np.log10(512), 12))

def u_of(sig, taus, agg):
    x = torch.tensor(sig[None, :, None], dtype=torch.float32)
    la = torch.log(torch.tensor(np.exp(-1.0 / np.array(taus))).float())[:, None]
    H = ema_bank(x, la) * bank_gains(la, L, x.device)[None]
    b = (H[:, :-1] - H[:, 1:]).abs()                 # [1,K-1,L,1]
    return _agg(b, agg)[0, -HALF:, 0].mean().item()

def make(n):
    """n evenly spaced value changes across the measured window."""
    s = np.ones(L)
    if n:
        for i, t in enumerate(np.linspace(HALF, L, n, endpoint=False).astype(int)):
            s[t:] = 1.0 + (i + 1) % 2
    return s

print(f"{'shifts in window':>18}{'gap between':>13}{'K=12 median':>14}{'K=12 max':>11}{'K=2':>9}")
for n in (0, 1, 2, 4, 8, 16, 32, 64):
    s = make(n)
    gap = f"{HALF//n}" if n else "-"
    print(f"{n:>18}{gap:>13}{u_of(s,taus12,'median'):>14.4f}"
          f"{u_of(s,taus12,'max'):>11.4f}{u_of(s,[0.2,512.0],'max'):>9.4f}")
