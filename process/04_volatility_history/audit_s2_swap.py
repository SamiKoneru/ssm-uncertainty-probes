"""AUDIT. The 0.61 -> 0.81 headline overstates the model effect, and chasing why
answers an open question the repo has been carrying.

audit_trend.py showed the legacy gate's pooled separability is 0.791 with all
five zero-extra-state features but only 0.616 with the three the shipped gate
uses. So on the LEGACY model the two extra features are worth +0.175, while on
the shipped model they are worth +0.005. My "the gate manufactures its own
signal, 0.61 -> 0.81" framing compared both models on only three features and so
credited the gate with a gap that mostly belongs to the FEATURE SET.

That points straight at model.py:144-146 and process/03/README.md:217, which
recommend, UNTESTED, replacing the algebraically redundant s2 with ||S_{t-1}k_t||:

    s2 = ( ||stored||^2 + ||v||^2 - ||e||^2 ) / ( 2 ||v||^2 )

so conditioned on ||v||, (s1, s2) is a bijection of (||e||, ||stored||) and s2
adds nothing beyond s1 and the two norms. The recommendation has never been
measured. This measures its separability on both models.

    python audit_s2_swap.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import auroc as A
import features as FE
from audit_trend import collect_legacy

SETS = [
    ("shipped          [tok, s1, s2]", ["f_tok", "f_s1", "f_s2"]),
    ("recommended      [tok, s1, |Sk|]", ["f_tok", "f_s1", "f_nstored"]),
    ("s1 + both norms  [tok, s1, |Sk|, |v|]",
     ["f_tok", "f_s1", "f_nstored", "f_nv"]),
    ("all five zero-extra-state", FE.INST),
]

if __name__ == "__main__":
    torch.set_num_threads(2)
    print("AUROC NOISY vs REVISED, pooled over all seen writes, by-sequence")
    print("split, gradient-boosted tree. Stream 777.\n")
    print(f"{'feature set':40s}{'legacy':>10}{'seed 0':>10}{'seed 1':>10}")
    got = {}
    for nm, getter in (("legacy", lambda: collect_legacy(777)),
                       ("seed 0", lambda: FE.collect(FE.load("gated_seed0")[0], 777)),
                       ("seed 1", lambda: FE.collect(FE.load("gated_seed1")[0], 777))):
        d = getter()
        clw, wt, seqs = d["cls_w"], d["wtype"], d["seq"]
        mk = np.isin(clw, [1, 2]) & (wt != FE.UNSEEN)
        lab = (clw == 2).astype(int)
        got[nm] = {}
        for snm, fs in SETS:
            au, sd, _, _ = A.joint(d, mk, fs, lab, seqs)
            got[nm][snm] = (au, sd)
    for snm, _ in SETS:
        row = "".join(f"{got[m][snm][0]:>10.4f}" for m in ("legacy", "seed 0", "seed 1"))
        print(f"{snm:40s}{row}")
    print(f"\n{'':40s}" + "".join(f"{'+/-'+format(got[m][SETS[0][0]][1], '.4f'):>10}"
                                  for m in ("legacy", "seed 0", "seed 1")))
    print("\nswapping s2 for ||Sk||, which is the repo's untested recommendation:")
    for m in ("legacy", "seed 0", "seed 1"):
        a = got[m]["shipped          [tok, s1, s2]"][0]
        b = got[m]["recommended      [tok, s1, |Sk|]"][0]
        print(f"  {m:8s} {a:.4f} -> {b:.4f}   {b-a:+.4f}")
