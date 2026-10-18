"""What the exact per-address bookkeeping costs per training step.

The machine is shared and heavily loaded, so a between-process timing is
worthless: the same arm measured twice ten minutes apart differed by 2.5x.
This times both configurations INSIDE ONE PROCESS, alternating, and reports
the MEDIAN over repetitions, so load hits both equally.

The comparison that isolates the bookkeeping is n_hist=5 (bookkeeping runs)
against n_hist=0 (it does not), both with memory_gate=True. Masking the
columns does NOT skip the bookkeeping, so `ours` pays the same cost as
`ours+hist`: it is the price of the MECHANISM, not of the information.

    python cost.py
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "gated_delta_ssm"))

import task as T                                                  # noqa: E402
import arms as A                                                  # noqa: E402


def step(m, opt, x):
    loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, T.N_SYM),
                           x[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    cfg = {"none": ("none", 0), "gate, no bookkeeping (n_hist=0)": ("ours_pub", 0),
           "gate + bookkeeping (n_hist=5)": ("ours+hist", 5)}
    models, opts = {}, {}
    for lab, (arm, _) in cfg.items():
        torch.manual_seed(0)
        models[lab] = A.build(arm)
        opts[lab] = torch.optim.AdamW(models[lab].parameters(), lr=2e-3)

    rng = np.random.default_rng(321)
    xs = [T.sample(rng, a.batch)[0] for _ in range(a.steps)]
    for lab in cfg:                                   # warm up all three
        step(models[lab], opts[lab], xs[0])

    t = {lab: [] for lab in cfg}
    for _ in range(a.reps):
        for lab in cfg:                               # alternate every rep
            t0 = time.perf_counter()
            for x in xs:
                step(models[lab], opts[lab], x)
            t[lab].append((time.perf_counter() - t0) / a.steps)

    print(f"{a.reps} reps x {a.steps} steps, batch {a.batch}, "
          f"{a.threads} threads, alternating inside one process\n")
    print(f"{'configuration':34s}{'median s/step':>15}{'min':>9}{'max':>9}"
          f"{'vs no-bookkeeping':>20}")
    ref = float(np.median(t["gate, no bookkeeping (n_hist=0)"]))
    for lab in cfg:
        v = np.array(t[lab])
        print(f"{lab:34s}{np.median(v):>15.3f}{v.min():>9.3f}{v.max():>9.3f}"
              f"{100 * (np.median(v) - ref) / ref:>+19.1f}%")
    print("\nload average now: " + os.popen("uptime").read().split("load averages:")[-1].strip())


if __name__ == "__main__":
    main()
