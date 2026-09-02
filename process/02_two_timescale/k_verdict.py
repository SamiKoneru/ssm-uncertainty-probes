"""Is K the knob, or is tape PLACEMENT the knob? And what does K actually buy?"""
import numpy as np, torch
from stability_ssm import ema_bank, bank_gains

L, periods = 4000, [2, 4, 8, 16, 32, 64, 128, 256, 512]
sq = np.stack([np.where((np.arange(L) // (p // 2)) % 2 == 0, 1.0, -1.0) for p in periods], -1)

def bands(sig, taus):
    x = torch.tensor(sig[None], dtype=torch.float32); C = x.shape[-1]
    la = torch.log(torch.exp(-1.0 / torch.tensor(taus)))[:, None].repeat(1, C)
    H = ema_bank(x, la) * bank_gains(la, L, x.device)[None]
    return (H[:, :-1] - H[:, 1:])[0].abs()                     # [K-1,L,C]

print("per-period u, K=2 only, changing ONLY where the two tapes sit:")
print(f"{'tapes':22s}" + "".join(f"P={p:<6d}" for p in periods) + " flatness")
for name, taus in [("tau 2, 100  (spec)", [2.0, 100.0]),
                   ("tau 0.2, 1024", [0.2, 1024.0])]:
    v = bands(sq, taus).amax(0)[-1000:].mean(0).numpy()
    print(f"{name:22s}" + "".join(f"{q:<8.3f}" for q in v) + f" {v.max()/v.min():.2f}x")

# --- what a slow tape costs: false alarm after ONE legitimate update ---------
sw = np.where(np.arange(L) < 1000, 1.0, 2.0)[:, None]
churn = np.where((np.arange(L) // 16) % 2 == 0, 1.0, 2.0)[:, None]   # persistent, same amplitude
print("\ncost of pushing tau_max out (K=2, tau_min=0.2):")
for tmax in (100, 256, 1024):
    c = bands(sw, [0.2, float(tmax)]).amax(0)[:, 0]
    pk = 1000 + int(c[1000:].argmax()); peak = c[pk]
    back = next((t for t in range(pk, L) if c[t] < 0.1 * peak), None)
    print(f"  tau_max={tmax:<5d} one-time update stays flagged {back-1000 if back else '>'+str(L-1000)} steps")

# --- what K DOES buy: transient vs persistent, via how MANY bands fire -------
taus = list(np.logspace(np.log10(0.2), np.log10(1024), 12))
b_sw, b_ch = bands(sw, taus)[:, 2000:, 0], bands(churn, taus)[:, 2000:, 0]   # long after switch
print("\nK=12: fraction of bands firing above 20% of that signal's own peak band")
fire = lambda b: (b.mean(1) > 0.2 * b.mean(1).max()).float().mean().item()
print(f"  one-time update (transient): {fire(b_sw):.2f}")
print(f"  persistent churn:            {fire(b_ch):.2f}")
print("\nseparation of persistent churn from a settled one-time update:")
for agg, f in (("max", lambda b: b.mean(1).max()),
               ("mean", lambda b: b.mean(1).mean()),
               ("median", lambda b: b.mean(1).median())):
    print(f"  agg={agg:7s} u(churn)/u(one-time update) = {f(b_ch)/f(b_sw):6.2f}x")
