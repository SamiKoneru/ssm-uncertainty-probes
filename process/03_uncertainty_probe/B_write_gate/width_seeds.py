"""
Width sweep AND seed variance together. Everything so far is n=1, and the
16 -> 64 gain (+0.74pp) has never been checked across seeds.

Capped at width 128: beyond that the gate has more parameters than the model
(60,844), so a gain could be "more params" rather than "better gating".
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch, torch.nn.functional as F
import mixed_task as T
from gated_model import GatedDeltaLayer

def run(gate, width, seed, steps=800, batch=24):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = GatedDeltaLayer(T.N_SYM, gate=gate, width=width)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for s in range(steps):
        x, _, _ = T.sample(rng, batch)
        lg, _ = m(x[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    L_, A = [], []
    ev = np.random.default_rng(12345)                    # SAME eval stream always
    with torch.no_grad():
        for _ in range(6):
            x, cl, sn = T.sample(ev, 32)
            lg, _ = m(x[:, :-1])
            L_.append(F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1)).item())
            A.append((lg.argmax(-1) == x[:, 1:]).float().mean().item())
    ng = sum(p.numel() for n, p in m.named_parameters() if n.startswith("gmlp"))
    return np.mean(L_), np.mean(A), ng

if __name__ == "__main__":
    SEEDS = (0, 1, 2)
    cfgs = [("none", 16, "input-only baseline"), ("mlp", 16, "gate 3->16->1"),
            ("mlp_wide", 32, "gate 3->32->32->1"), ("mlp_wide", 64, "gate 3->64->64->1"),
            ("mlp_wide", 128, "gate 3->128->128->1")]
    base = {}
    print(f"{'config':22s}{'gate par':>9}{'loss mean':>11}{'sd':>8}{'acc mean':>10}{'sd':>8}{'vs base':>10}")
    for gate, w, lab in cfgs:
        L_, A = [], []
        for sd in SEEDS:
            l, a, ng = run(gate, w, sd); L_.append(l); A.append(a)
        L_, A = np.array(L_), np.array(A)
        if gate == "none": base = L_.copy()
        gain = 100 * (base.mean() - L_.mean()) / base.mean()
        print(f"{lab:22s}{ng:>9,}{L_.mean():>11.4f}{L_.std(ddof=1):>8.4f}"
              f"{A.mean():>10.4f}{A.std(ddof=1):>8.4f}{gain:>9.2f}%", flush=True)
    print(f"\nseeds {SEEDS}, identical eval stream; sd is across seeds")
