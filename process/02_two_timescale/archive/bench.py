"""Cost of the K-tape bank, on a realistic layer shape. Forward + backward, CPU."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time, torch, torch.nn as nn
from stability_ssm import ema_bank, bank_gains

B, L, D, N, G = 4, 1024, 512, 1024, 8      # batch, len, d_model, state (2x expand), groups

def timeit(fn, reps=5):
    fn(); fn()                              # warmup
    t = time.perf_counter()
    for _ in range(reps): fn()
    return (time.perf_counter() - t) / reps * 1e3

w_in, w_out = nn.Linear(D, N), nn.Linear(2 * N, D)
w_u = nn.Linear(N, G)                       # narrow projection for the u-bank
h = torch.randn(B, L, D, requires_grad=True)

def block(K, narrow):
    """One layer: in-proj, scans, u, gated out-proj."""
    x = w_in(h)
    la_c = torch.log(torch.tensor([0.5, 0.99]))[:, None].repeat(1, N)
    content = ema_bank(x, la_c) * bank_gains(la_c, L, x.device)[None]     # 2 tapes, full width
    if K > 0:
        z = w_u(x) if narrow else x                                       # [B,L,G] or [B,L,N]
        C = z.shape[-1]
        la = torch.log(torch.exp(-1.0 / torch.logspace(-0.7, 3, K)))[:, None].repeat(1, C)
        H = ema_bank(z, la) * bank_gains(la, L, z.device)[None]
        d = (H[:, :-1] - H[:, 1:]).view(B, K - 1, L, G, C // G)
        u = d.pow(2).mean(-1).sqrt().amax(1)
        gate = (1.0 / (1.0 + u)).repeat_interleave(N // G, -1)
    else:
        gate = 1.0
    y = w_out(torch.cat([content[:, 0] * gate, content[:, 1] * gate], -1))
    y.sum().backward()

def mem(K, C):                              # bytes of bank intermediates
    return B * K * C * (L + 1) * 8 + B * K * C * 2 * L * 4

base = timeit(lambda: block(0, False))
print(f"shape: B={B} L={L} d_model={D} n_state={N} groups={G}   (fwd+bwd, ms)\n")
print(f"{'variant':38s}{'ms':>9}{'vs no-bank':>12}{'bank intermediates':>21}")
print(f"{'no bank (2 content tapes only)':38s}{base:9.1f}{1.0:11.2f}x{'-':>21}")
for K in (4, 8, 12, 16):
    t = timeit(lambda: block(K, False))
    print(f"{f'K={K} bank at FULL width (N={N})':38s}{t:9.1f}{t/base:11.2f}x{mem(K,N)/2**20:18.0f} MB")
print()
for K in (4, 8, 12, 16):
    t = timeit(lambda: block(K, True))
    print(f"{f'K={K} bank on G={G} projection':38s}{t:9.1f}{t/base:11.2f}x{mem(K,G)/2**20:18.1f} MB")
