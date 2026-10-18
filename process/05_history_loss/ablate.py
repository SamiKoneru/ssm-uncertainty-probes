"""Which history column is the TRAINED gate actually leaning on?

Zeroes one history column at a time in an already-trained model and rescores.
Free, no retraining. It is a complement to the dedicated single-feature arms,
not a substitute: a trained gate has co-adapted to all its columns, so
knocking one out at eval time overstates that column's importance relative to
training without it. Read the two together.

Reported as the loss INCREASE from removing each column, so larger means the
trained gate depended on it more.

    python ablate.py --arm ours+hist --seed 0 --stream 5150

EXPLORATORY: motivated by seed 0's visit-count profile.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "gated_delta_ssm"))

import arms as A                                                  # noqa: E402
import score as SC                                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=int, default=5150)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--arms", nargs="+",
                    default=["ours+hist", "ours+oracle", "ours+bayes"])
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    print(f"stream {a.stream}   seeds {a.seeds}   "
          f"loss INCREASE when one column is zeroed at eval time\n")
    for arm in a.arms:
        rows, base = {}, []
        n_hist = A.ARMS[arm][0]
        keep = A.ARMS[arm][1]
        for sd in a.seeds:
            m, ck = SC.load(arm, sd)
            if m is None:
                continue
            base.append(SC.evaluate(m, a.stream)["loss"])
            for j in keep:
                saved = float(m.hist_mask[j])
                m.hist_mask[j] = 0.0
                rows.setdefault(j, []).append(
                    SC.evaluate(m, a.stream)["loss"])
                m.hist_mask[j] = saved
            # and all of them at once. This does NOT land near `ours`:
            # these weights co-adapted to the columns being removed, so the
            # model is off-distribution. It measures co-adaptation, not the
            # value of history.
            saved = m.hist_mask.clone()
            m.hist_mask.zero_()
            rows.setdefault("ALL", []).append(SC.evaluate(m, a.stream)["loss"])
            m.hist_mask.copy_(saved)
        if not base:
            print(f"{arm}: no checkpoint, skipped\n")
            continue
        b = float(np.mean(base))
        print(f"=== {arm}   as trained {b:.4f}")
        order = sorted((k for k in rows if k != "ALL"),
                       key=lambda k: -np.mean(rows[k]))
        for j in order:
            v = float(np.mean(rows[j]))
            print(f"    drop {A.HIST[j]:11s} {v:>8.4f}   {v - b:>+8.4f}")
        v = float(np.mean(rows["ALL"]))
        print(f"    drop {'ALL history':11s} {v:>8.4f}   {v - b:>+8.4f}"
              f"   (co-adaptation, NOT comparable to `ours`)")
        print()


if __name__ == "__main__":
    main()
