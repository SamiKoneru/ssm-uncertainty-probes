"""
Reproduce the headline result: memory-gated beta vs the input-only baseline.

    python train.py            # 3 seeds each, ~55 min on CPU
    python train.py --quick    # 1 seed, 300 steps, ~4 min, sanity check only

Expected (3 seeds, identical eval stream):

    config              loss mean      sd   acc mean      sd    vs base
    input-only baseline    1.6322  0.0018     0.6342  0.0018      0.00%
    memory-gated (w=32)    1.5428  0.0039     0.6458  0.0046      5.47%

The task is built so an input-only gate PROVABLY cannot differentiate: each
symbol's class is randomised per sequence, so token identity carries zero
information about which write policy a key needs.
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import task as T
from model import GatedDeltaSSM


def run(memory_gate, seed=0, steps=800, batch=24, gate_width=32):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = GatedDeltaSSM(T.N_SYM, memory_gate=memory_gate, gate_width=gate_width)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for _ in range(steps):
        x, _, _ = T.sample(rng, batch)
        loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, T.N_SYM),
                               x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()

    losses, accs, betas, cls_w, seen = [], [], [], [], []
    ev = np.random.default_rng(12345)            # identical eval stream for all configs
    with torch.no_grad():
        for _ in range(6):
            x, cl, sn = T.sample(ev, 32)
            lg, b = m(x[:, :-1], return_beta=True)
            losses.append(F.cross_entropy(lg.reshape(-1, T.N_SYM),
                                          x[:, 1:].reshape(-1)).item())
            accs.append((lg.argmax(-1) == x[:, 1:]).float().mean().item())
            betas.append(b)
            # beta_t writes to the key from token t-1, so label it with that
            cls_w.append(F.pad(cl, (1, 0))[:, :T.L]); seen.append(sn)
    f = lambda z: torch.cat(z).reshape(-1).numpy()
    return np.mean(losses), np.mean(accs), f(betas), f(cls_w), f(seen).astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    seeds = (0,) if a.quick else (0, 1, 2)
    steps = 300 if a.quick else 800

    print(f"{'config':22s}{'loss mean':>11}{'sd':>8}{'acc mean':>10}{'sd':>8}{'vs base':>10}")
    base = None
    for mg, lab in ((False, "input-only baseline"), (True, "memory-gated (w=32)")):
        L, A, B, C, S = [], [], None, None, None
        for s in seeds:
            l, ac, B, C, S = run(mg, seed=s, steps=steps)
            L.append(l); A.append(ac)
        L, A = np.array(L), np.array(A)
        if base is None: base = L.mean()
        sdL = L.std(ddof=1) if len(L) > 1 else 0.0
        sdA = A.std(ddof=1) if len(A) > 1 else 0.0
        print(f"{lab:22s}{L.mean():>11.4f}{sdL:>8.4f}{A.mean():>10.4f}{sdA:>8.4f}"
              f"{100*(base-L.mean())/base:>9.2f}%", flush=True)
        if mg:
            b = [B[(C == i) & S].mean() for i in range(3)]
            print(f"{'':22s}mean beta by key class: STABLE {b[0]:.3f}  "
                  f"NOISY {b[1]:.3f}  REVISED {b[2]:.3f}")
            print(f"{'':22s}(NOISY/REVISED invert when pooled over repeat writes - "
                  f"see process/README.md)")


if __name__ == "__main__":
    main()
