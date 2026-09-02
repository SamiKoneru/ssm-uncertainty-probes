"""
Unit test of the mechanism itself: no training, no task, no readout.
Feed streams whose stability we control and ask what u actually measures.
"""
import numpy as np, torch
from stability_ssm import ema_parallel

A_F, A_S = 0.5, 0.99
rng = np.random.default_rng(0)

def instability(x, L):
    """x: [1,L,C] -> u: [L,C], bias-corrected on both tapes."""
    C = x.shape[-1]
    out = []
    for a in (A_F, A_S):
        la = torch.full((C,), float(np.log(a)))
        t = torch.arange(L, dtype=torch.float32)[:, None]
        g = (1 - torch.exp(la)[None]) / (1 - torch.exp(la[None] * (t + 1))).clamp_min(1e-6)
        out.append(ema_parallel(x, la) * g)
    return (out[0] - out[1])[0].abs().numpy()

# ---- 1. controlled stability streams -------------------------------------
L = 400
sig = {
    "stable (const 1.0)":        np.ones(L),
    "ONE switch @t=100":         np.where(np.arange(L) < 100, 1.0, 2.0),
    "churn (flip every step)":   np.where(np.arange(L) % 2 == 0, 1.0, -1.0),
    "contested (redraw ea. 20)": np.repeat(rng.normal(size=L // 20), 20),
    "random every step":         rng.normal(size=L),
}
u = instability(torch.tensor(np.stack(list(sig.values()), -1)[None], dtype=torch.float32), L)

print("u at selected steps")
print(f"{'signal':28s}" + "".join(f"t={t:<7d}" for t in (5, 50, 99, 101, 110, 150, 250, 399)))
for i, n in enumerate(sig):
    print(f"{n:28s}" + "".join(f"{u[t,i]:<9.3f}" for t in (5, 50, 99, 101, 110, 150, 250, 399)))

print("\nmean u over last 100 steps  (intended order: stable < one-switch < churn < contested)")
for i, n in sorted(enumerate(sig), key=lambda kv: u[-100:, kv[1] == kv[1]].mean() if False else u[-100:, kv[0]].mean()):
    print(f"  {u[-100:, i].mean():.4f}   {n}")

col = u[:, 1]; pk = 100 + int(col[100:].argmax()); peak = col[pk]
back = next((t for t in range(pk, L) if col[t] < 0.1 * peak), None)
print(f"\nsingle legitimate update: u peaks {peak:.3f} at t={pk}; "
      + (f"falls under 10% of peak at t={back} ({back-100} steps later)"
         if back else f"still {col[-1]/peak:.0%} of peak at t={L-1}, {L-100} steps later"))

# ---- 2. is u a stability detector, or a bandpass filter? -----------------
L2 = 2000
periods = [2, 4, 8, 16, 32, 64, 128, 256, 512]
sq = np.stack([np.where((np.arange(L2) // (p // 2)) % 2 == 0, 1.0, -1.0) for p in periods], -1)
u2 = instability(torch.tensor(sq[None], dtype=torch.float32), L2)
print("\nsquare wave of period P, amplitude 1 (every one is maximally unstable):")
for i, p in enumerate(periods):
    m = u2[-500:, i].mean()
    print(f"  period {p:4d}   mean u = {m:.3f}  {'#' * int(m * 40)}")
