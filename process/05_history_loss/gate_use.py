"""Did the gate actually USE the history, or learn to ignore it?

A null loss result has two very different explanations and this file
separates them:

    (a) the gate put real weight on the history columns and beta moved with
        them, and it still bought no loss  ->  the INFORMATION is not worth
        anything to this task
    (b) the gate left the history columns near their initialisation and beta
        barely responds                    ->  an OPTIMISATION failure, and
        the information question is untouched

Two readings, because either alone can mislead:

  column weight    ||W1[:, j]||, the first-layer weight norm on input j.
                   Cheap, but a large norm on a feature with tiny variance
                   moves nothing.
  beta sensitivity mean |d beta / d x_j| x sd(x_j), i.e. how far beta actually
                   moves per one standard deviation of feature j, measured by
                   autograd through the real gate on real collected inputs.
                   This is the one that matters.

    python gate_use.py --stream 5150
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "gated_delta_ssm"))

import task as T                                                  # noqa: E402
import arms as A                                                  # noqa: E402
import score as SC                                                # noqa: E402

COLS = ["tok", "s1", "s2"] + A.HIST


@torch.no_grad()
def collect_inputs(m, stream, batches=3, batch=32):
    ev = np.random.default_rng(stream)
    out = []
    for _ in range(batches):
        x, _, _ = T.sample(ev, batch)
        _, _, f = m(x[:, :-1], return_beta=True, return_feats=True)
        out.append(f.reshape(-1, f.shape[-1]))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=int, default=5150)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+",
                    default=["ours", "ours+hist", "ours+rate"])
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    print(f"stream {a.stream}   seeds {a.seeds}\n")

    for arm in a.arms:
        W, Sens, Sd, init_W = [], [], [], []
        for s in a.seeds:
            m, ck = SC.load(arm, s)
            if m is None:
                continue
            gin = collect_inputs(m, a.stream)
            n_in = gin.shape[-1]
            W.append(m.gate[0].weight.detach().norm(dim=0).numpy())
            # what the SAME column norm looks like at initialisation, so a
            # "did not move" column is recognisable
            torch.manual_seed(s)
            init_W.append(A.build(arm).gate[0].weight.detach()
                          .norm(dim=0).numpy())
            g = gin.clone().requires_grad_(True)
            torch.sigmoid(m.gate(g)[..., 0]).sum().backward()
            sd = gin.std(0).numpy()
            Sens.append(g.grad.abs().mean(0).numpy() * sd)
            Sd.append(sd)
        if not W:
            print(f"{arm}: no checkpoints, skipped\n")
            continue
        W, Sens, Sd = np.mean(W, 0), np.mean(Sens, 0), np.mean(Sd, 0)
        iW = np.mean(init_W, 0)
        print(f"=== {arm} " + "=" * (58 - len(arm)))
        print(f"  {'feature':11s}{'sd(x)':>9}{'||W1|| init':>13}"
              f"{'||W1|| trained':>16}{'moved':>9}"
              f"{'d beta per 1 sd':>18}")
        for j in range(len(W)):
            print(f"  {COLS[j]:11s}{Sd[j]:>9.3f}{iW[j]:>13.3f}{W[j]:>16.3f}"
                  f"{W[j] / max(iW[j], 1e-9):>8.2f}x{Sens[j]:>18.5f}")
        inst = Sens[:3].sum()
        hist = Sens[3:].sum() if len(Sens) > 3 else 0.0
        print(f"  total beta sensitivity: instantaneous {inst:.5f}   "
              f"history {hist:.5f}   history share "
              f"{100 * hist / max(inst + hist, 1e-12):.1f}%\n")


if __name__ == "__main__":
    main()
