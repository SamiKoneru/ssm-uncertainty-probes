"""TASK 2. Is the long-term history signal THERE at all?

Hypothesis: successive residual DIRECTIONS at the same address separate the
classes. A NOISY address redraws from a fixed set so its residuals scatter; a
REVISED address carries one persistent error being slowly corrected under the
low beta the gate actually learns, so its residuals stay aligned.

Everything here is measured with EXACT out-of-model bookkeeping, which is an
upper bound on any mechanism: no superposition crosstalk, and crosstalk is what
killed every per-key store in this repo (process/03/README.md:95-110).

Pre-registered primary metric and bars (PREREGISTRATION.md):

    delta = AUROC(instantaneous + history) - AUROC(instantaneous)
    PASS   delta >= +0.05 AND absolute AUROC >= 0.70
    KILL   delta <  +0.02  or absolute AUROC < 0.60

Also killed if the history features are only a proxy for repeat-vs-change: a
feature at chance inside BOTH strata adds nothing the instantaneous features do
not already carry.

    python history_probe.py --stream 777
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import auroc as A
import features as FE

INST = FE.INST
HIST = FE.HIST
ORACLE = FE.ORACLE


def hdr(s):
    print(f"\n{'='*74}\n{s}\n{'='*74}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=int, default=777)
    ap.add_argument("--tag", default="gated_seed0")
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    m, ck = FE.load(a.tag)
    d = FE.collect(m, a.stream, batches=a.batches)
    d["pos"] = d["pos"].astype(np.float64)
    seqs = d["seq"]
    clw = d["cls_w"]
    wt = d["wtype"]
    seen = wt != FE.UNSEEN
    change = wt == FE.CHANGE
    repeat = wt == FE.REPEAT
    lab = (clw == 2).astype(int)
    nr = np.isin(clw, [1, 2])

    print(f"model {a.tag} (seed {ck['seed']})   eval stream {a.stream}, "
          f"{a.batches} batches of 32")
    print(f"loss {d['_loss']:.4f}   acc {d['_acc']:.4f}")
    print("history features are defined only on writes to an address written\n"
          "before, so every row below is restricted to those.")

    # ---------------------------------------------------------------- part 1
    hdr("PART 1  predicted direction: is cos(e_t, e_prev) higher on REVISED?")
    print("pre-registered prediction: YES. The other direction gets reported as\n"
          "a direction error, not as a success.")
    print(f"\n{'feature':12s}{'NOISY mean':>12}{'REVISED mean':>14}"
          f"{'direction':>26}")
    for f in HIST + ORACLE:
        mn = np.nanmean(d[f][seen & (clw == 1)])
        mr = np.nanmean(d[f][seen & (clw == 2)])
        tag = "REVISED higher (predicted)" if mr > mn else "NOISY higher"
        print(f"{f:12s}{mn:>12.4f}{mr:>14.4f}{tag:>26}")

    # ---------------------------------------------------------------- part 2
    hdr("PART 2  single-feature AUROC, NOISY vs REVISED, by stratum")
    print("The KILL for 'it is only a proxy for write type': a feature at chance\n"
          "inside BOTH change and repeat strata carries nothing new.")
    conds = (("all seen writes", seen), ("CHANGE only", change),
             ("REPEAT only", repeat))
    names = HIST + ORACLE + ["f_s1", "f_s2", "f_nstored", "pos"]
    print(f"\n{'feature':12s}" + "".join(f"{nm:>20}" for nm, _ in conds))
    print(f"{'':12s}" + "".join(f"{'AUROC      n':>20}" for _ in conds))
    for f in names:
        row = ""
        for _, extra in conds:
            mk = nr & extra
            au, n = A.single(d[f][mk], lab[mk])
            row += f"{au:>13.4f}{n:>7d}"
        print(f"{f:12s}{row}")

    # ---------------------------------------------------------------- part 3
    hdr("PART 3  THE PRE-REGISTERED PRIMARY METRIC")
    print("paired on identical rows and identical by-sequence split; bootstrap SD\n"
          "is of the DELTA, resampling sequences.")
    mk = nr & seen
    print(f"\n{'comparison':44s}{'base':>8}{'with':>8}{'delta':>9}{'sd':>8}"
          f"{'n_test':>8}")
    rows = [
        ("inst -> inst + history", INST, INST + HIST),
        ("inst -> inst + h_cos only", INST, INST + ["h_cos"]),
        ("inst -> inst + h_meane only (rate memory)", INST, INST + ["h_meane"]),
        ("inst + hist -> + ORACLE membership", INST + HIST,
         INST + HIST + ORACLE),
        ("inst -> inst + ORACLE membership", INST, INST + ORACLE),
        ("inst -> inst + position (control)", INST, INST + ["pos"]),
    ]
    for nm, fa, fb in rows:
        au_a, au_b, dl, sd, nte = A.joint_pair(d, mk, fa, fb, lab, seqs)
        print(f"{nm:44s}{au_a:>8.4f}{au_b:>8.4f}{dl:>+9.4f}{sd:>8.4f}{nte:>8d}")

    print("\n  bars:  PASS  delta >= +0.05 AND absolute >= 0.70")
    print("         KILL  delta <  +0.02  OR absolute < 0.60")

    hdr("PART 4  the same, restricted to CHANGE writes (where beta matters most)")
    mk = nr & change
    print(f"\n{'comparison':44s}{'base':>8}{'with':>8}{'delta':>9}{'sd':>8}"
          f"{'n_test':>8}")
    for nm, fa, fb in rows:
        au_a, au_b, dl, sd, nte = A.joint_pair(d, mk, fa, fb, lab, seqs)
        print(f"{nm:44s}{au_a:>8.4f}{au_b:>8.4f}{dl:>+9.4f}{sd:>8.4f}{nte:>8d}")

    hdr("PART 5  is the class label even recoverable from per-address history?")
    print("If the near-label oracle features cannot separate the classes at\n"
          "L=256 (about 6 visits per address), then 'the class label IS each\n"
          "key's long-term behaviour' is false at this length, and NO history\n"
          "mechanism can reach the per-class oracle.")
    for nm, extra in conds:
        mk = nr & extra
        for fs, lbl in ((ORACLE, "oracle alone"),
                        (ORACLE + HIST, "oracle + history"),
                        (INST + HIST + ORACLE, "everything")):
            au, sd, ntot, nte = A.joint(d, mk, fs, lab, seqs)
            print(f"  {nm:18s}{lbl:20s} AUROC {au:.4f} +/- {sd:.4f}"
                  f"   n_test {nte}")
    print("\n  visits per address, by class (seen writes):")
    for i, cn in enumerate(("STABLE", "NOISY", "REVISED")):
        v = d["h_nvis"][seen & (clw == i)]
        o = d["o_ndist"][seen & (clw == i)]
        print(f"    {cn:8s} mean prior visits {np.nanmean(v):.2f}   "
              f"mean distinct values seen {np.nanmean(o):.2f}")


if __name__ == "__main__":
    main()
