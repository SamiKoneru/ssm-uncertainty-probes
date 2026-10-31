"""ANOMALY noticed while measuring, reported because the experiment rules
require it: the shipped width-32 gate's learned policy does not look like the
one every README in this repo describes.

The stated policy (gated_delta_ssm/README.md:23, model.py:21, process/03:45) is
"overwrite decisively when the incoming value is far from and unlike what is
stored; write softly when it is already there", i.e. beta RISES with ||e||.

PART 2 of separability.py measured the opposite in aggregate: mean beta is
0.3096 on repeat writes (mean ||e|| 4.24) and 0.1951 on change writes (mean
||e|| 8.57). This file checks whether that is a real inversion or a Simpson's
paradox of its own, by splitting beta on class AND write type and by plotting
the beta-vs-s1 response inside each stratum.

It matters beyond bookkeeping: process/03/README.md:220-229 calls the direction
a learned continuous beta picks "THE ONE THING STILL OPEN", with the modern LM
literature damping large residuals and classical adaptive filtering amplifying
them, and lists C4 "policy direction: AMPLIFY on large residual" as TAKEN.

    python beta_policy.py --stream 777
"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features as FE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=int, default=777)
    ap.add_argument("--tag", default="gated_seed0")
    ap.add_argument("--batches", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(2)

    m, ck = FE.load(a.tag)
    d = FE.collect(m, a.stream, batches=a.batches)
    clw, wt, beta, s1 = d["cls_w"], d["wtype"], d["beta"], d["f_s1"]
    pos_ok = d["pos"] > 0
    seen = (wt != FE.UNSEEN)

    print(f"model {a.tag}   stream {a.stream}   loss {d['_loss']:.4f}")

    print("\nmean beta by CLASS of the write address x WRITE TYPE")
    print(f"{'':9s}{'unseen':>22}{'repeat':>22}{'change':>22}")
    print(f"{'':9s}" + "".join(f"{'beta  ||e||       n':>22}" for _ in range(3)))
    for i, nm in enumerate(("STABLE", "NOISY", "REVISED")):
        row = ""
        for code in (FE.UNSEEN, FE.REPEAT, FE.CHANGE):
            mk = (clw == i) & (wt == code) & pos_ok
            if mk.sum() < 5:
                row += f"{'-':>22}"
            else:
                row += (f"{beta[mk].mean():>8.3f}{d['ne'][mk].mean():>6.2f}"
                        f"{mk.sum():>8d}")
        print(f"{nm:9s}{row}")

    print("\nis beta MONOTONE in s1 = log(1+||e||)?  mean beta by s1 decile")
    for nm, mk in (("all writes t>0", pos_ok), ("repeat writes", wt == FE.REPEAT),
                   ("change writes", wt == FE.CHANGE)):
        q = np.quantile(s1[mk], np.linspace(0, 1, 11))
        cells = []
        for j in range(10):
            sel = mk & (s1 >= q[j]) & (s1 <= q[j + 1])
            cells.append(beta[sel].mean() if sel.sum() else np.nan)
        rho = spearmanr(s1[mk], beta[mk]).statistic
        print(f"  {nm:15s} " + " ".join(f"{c:.3f}" for c in cells)
              + f"   spearman {rho:+.3f}")

    print("\n  s1 decile EDGES for all writes t>0 (so the deciles above are "
          "readable):")
    q = np.quantile(s1[pos_ok], np.linspace(0, 1, 11))
    print("   " + " ".join(f"{z:.2f}" for z in q))

    print("\nwithin-stratum spearman(beta, s1), by class and write type:")
    for i, nm in enumerate(("STABLE", "NOISY", "REVISED")):
        for code, cn in ((FE.REPEAT, "repeat"), (FE.CHANGE, "change")):
            mk = (clw == i) & (wt == code) & pos_ok
            if mk.sum() < 50:
                print(f"  {nm:8s} {cn:7s} n={mk.sum():<6d} -")
                continue
            print(f"  {nm:8s} {cn:7s} n={mk.sum():<6d} "
                  f"spearman {spearmanr(s1[mk], beta[mk]).statistic:+.3f}   "
                  f"mean beta {beta[mk].mean():.3f}")

    print("\nthe aggregate fact, restated: beta on seen writes")
    for code, cn in ((FE.REPEAT, "repeat"), (FE.CHANGE, "change")):
        mk = (wt == code)
        print(f"  {cn:7s} mean beta {beta[mk].mean():.4f}   "
              f"mean ||e|| {d['ne'][mk].mean():.3f}   "
              f"mean beta*||e|| {(beta[mk]*d['ne'][mk]).mean():.4f}")
    mk = seen
    print(f"  pooled spearman(beta, ||e||) on seen writes: "
          f"{spearmanr(d['ne'][mk], beta[mk]).statistic:+.3f}")


if __name__ == "__main__":
    main()
