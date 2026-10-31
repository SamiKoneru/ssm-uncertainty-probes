"""Train ONE arm at ONE seed with the published protocol and save it.

    python run_arm.py --arm ours+hist --seed 0

Published protocol exactly (gated_delta_ssm/train.py): 800 steps, batch 24,
AdamW lr 2e-3, grad clip 1.0, gate_width 32. For a given seed every arm sees
the IDENTICAL training stream, because the data comes from
np.random.default_rng(seed) and T.sample consumes the same draws whatever the
gate is.

No evaluation happens here. Scoring lives in score.py so the held-out stream
stays untouched until the verdict is written. Capped at 2 threads: the machine
is shared.
"""
import argparse
import json
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

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "checkpoints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(A.ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    tag = f"{a.arm.replace('+', '_')}_seed{a.seed}"
    print(f"arm {a.arm}  seed {a.seed}  {a.steps} steps  batch {a.batch}  "
          f"{a.threads} threads", flush=True)

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    m = A.build(a.arm)
    n_par = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)

    t0 = time.time()
    for s in range(a.steps):
        x, _, _ = T.sample(rng, a.batch)
        loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, T.N_SYM),
                               x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s % 100 == 0 or s == a.steps - 1:
            print(f"  step {s:4d}  loss {loss.item():.4f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    secs = time.time() - t0

    clamp = (m.n_clamped / m.n_hist_rows) if m.n_hist_rows else 0.0
    os.makedirs(CKPT, exist_ok=True)
    out = os.path.join(CKPT, f"{tag}.pt")
    torch.save({"state_dict": m.state_dict(), "arm": a.arm, "seed": a.seed,
                "steps": a.steps, "batch": a.batch, "gate_width": 32,
                "params": n_par, "train_secs": secs,
                "h_ratio_clamp_rate": clamp,
                "final_train_loss": float(loss.item())}, out)
    print(f"saved {out}   {secs:.0f}s   params {n_par:,}   "
          f"h_ratio clamp rate {100 * clamp:.3f}%", flush=True)
    print("RESULT " + json.dumps({"arm": a.arm, "seed": a.seed,
                                  "secs": round(secs, 1), "params": n_par,
                                  "clamp": round(clamp, 6)}), flush=True)


if __name__ == "__main__":
    main()
