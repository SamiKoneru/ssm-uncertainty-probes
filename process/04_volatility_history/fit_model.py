"""Train the SHIPPED gated model once and save it, so every measurement in this
directory reads the same model instead of retraining.

Published protocol exactly (gated_delta_ssm/train.py): 800 steps, batch 24,
AdamW lr 2e-3, grad clip 1.0, GatedDeltaSSM(memory_gate=True, gate_width=32).

    python fit_model.py --seed 0

Writes checkpoints/gated_seed<N>.pt. Capped at 2 threads: the machine is shared.
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "gated_delta_ssm"))

import numpy as np
import torch
import torch.nn.functional as F

import task as T
from model import GatedDeltaSSM

HERE = os.path.dirname(os.path.abspath(__file__))


def fit(seed, steps=800, batch=24, gate_width=32, memory_gate=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = GatedDeltaSSM(T.N_SYM, memory_gate=memory_gate, gate_width=gate_width)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    t0 = time.time()
    for s in range(steps):
        x, _, _ = T.sample(rng, batch)
        loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, T.N_SYM),
                               x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s % 100 == 0 or s == steps - 1:
            print(f"  step {s:4d}  loss {loss.item():.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-gate", action="store_true",
                    help="train the input-only baseline instead")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    tag = f"base_seed{a.seed}" if a.no_gate else f"gated_seed{a.seed}"
    print(f"training {tag}, {a.steps} steps, {a.threads} threads", flush=True)
    m = fit(a.seed, steps=a.steps, memory_gate=not a.no_gate)
    os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)
    out = os.path.join(HERE, "checkpoints", f"{tag}.pt")
    torch.save({"state_dict": m.state_dict(), "seed": a.seed, "steps": a.steps,
                "memory_gate": not a.no_gate, "gate_width": 32}, out)
    print(f"saved {out}", flush=True)
