"""A1b's missing control: is a fitted policy's counterfactual gain structural,
or just its MEAN beta?

A1b substitutes beta into the frozen model, so any policy whose beta
distribution sits far from the one the readout was trained with pays an
off-policy penalty. The CE ordering in A1b.1 partly tracks |mean_beta - 0.3123|,
which is a competing explanation for every comparison in it. The control is a
CONSTANT-beta sweep: if a constant at the same mean does as well as a fitted
policy, that policy's structure bought nothing.

Reported on both CE and ACCURACY. Accuracy is an argmax, so it is far less
sensitive to the readout's scale being miscalibrated for a shifted beta
distribution, which is exactly what the off-policy penalty does.

    python audit_cf_controls.py --tag gated_seed0 --stream 505
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import _paths                                                  # noqa: F401
import audit_counterfactual as CF
import features as FE
import task as T

# the policies measured in A1b.1, with their observed mean beta
OBSERVED = [("the gate's own beta (shipped)", 1.4648, 0.6596, 0.3123),
            ("best constant beta", 1.6095, 0.6673, 0.5325),
            ("3 feats, linear", 2.1067, 0.6036, 0.6683),
            ("3 feats, GBT", 1.5927, 0.6596, 0.7811),
            ("5 zero-state, GBT", 1.6176, 0.6690, 0.8047),
            ("5 zero-state + history, GBT", 1.5418, 0.6690, 0.5259),
            ("TRUE CLASS LABEL (oracle)", 1.4806, 0.6754, 0.7103),
            ("per-write beta* (non-causal)", 1.4695, 0.7009, 0.7130)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gated_seed0")
    ap.add_argument("--stream", type=int, default=505)
    ap.add_argument("--fit-batches", type=int, default=6)
    ap.add_argument("--eval-batches", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(2)
    m, ck = FE.load(a.tag)

    # the SAME eval batches A1b used: same seed, same draw order
    ev = np.random.default_rng(a.stream)
    xs = [T.sample(ev, 32) for _ in range(a.fit_batches + a.eval_batches)]
    ev_x = xs[a.fit_batches:]

    grid = [0.05, 0.10, 0.20, 0.3123, 0.40, 0.5259, 0.60, 0.70, 0.7130,
            0.80, 0.90, 1.00]
    print(f"A1b-control  model {a.tag}  stream {a.stream}  "
          f"{a.eval_batches} held-out batches (the same ones A1b used)")
    print(f"\n  CONSTANT-beta sweep on the frozen model:")
    print(f"  {'beta':>8}{'CE':>10}{'acc':>9}")
    res = {}
    for b in grid:
        ce = np.mean([CF.replay_policy(m, x, "const", policy=b)["ce"]
                      for x, _, _ in ev_x])
        ac = np.mean([CF.replay_policy(m, x, "const", policy=b)["acc"]
                      for x, _, _ in ev_x])
        res[b] = (ce, ac)
        print(f"  {b:>8.4f}{ce:>10.4f}{ac:>9.4f}", flush=True)

    bb = min(res, key=lambda z: res[z][0])
    ba = max(res, key=lambda z: res[z][1])
    print(f"\n  best constant on CE  beta={bb:.4f}  CE {res[bb][0]:.4f}")
    print(f"  best constant on acc beta={ba:.4f}  acc {res[ba][1]:.4f}")

    print(f"\n  each A1b policy against a constant AT ITS OWN MEAN beta:")
    print(f"  {'policy':32s}{'mean b':>8}{'CE':>9}{'const CE':>10}"
          f"{'CE edge':>9}{'acc':>8}{'const acc':>11}{'acc edge':>10}")
    for nm, ce, ac, mb in OBSERVED:
        near = min(grid, key=lambda z: abs(z - mb))
        cce, cac = res[near]
        print(f"  {nm:32s}{mb:>8.4f}{ce:>9.4f}{cce:>10.4f}"
              f"{cce - ce:>+9.4f}{ac:>8.4f}{cac:>11.4f}{ac - cac:>+10.4f}")
    print("\n  'edge' is the policy minus the matched constant, positive =\n"
          "  the policy's STRUCTURE helped beyond its mean beta. The matched\n"
          "  constant is the nearest grid point, which is exact for the gate\n"
          "  (0.3123), the history policy (0.5259) and beta* (0.7130).")


if __name__ == "__main__":
    main()
