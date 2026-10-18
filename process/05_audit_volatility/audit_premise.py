"""A7 and A2.5. Was "the 0.80 premise bar is indecisive" honest, and how big
would an off-by-one have been?

A7. 04 measured the pooled AUROC of the gate's own three features at 0.8095 on
stream 777 and 0.7983 on 888, against a pre-registered bar of 0.80, and recorded
the result as INDECISIVE because the two straddle the bar by 0.0017.

Checked by reading git history, before measuring: 777 was named PRIMARY in the
pre-registration commit (a90679d), two commits before the harness existed, so
the designation was not chosen after seeing numbers. Both streams are the same
size and both fresh. So neither is "sounder" on design grounds, and the straddle
is a question about EVAL SAMPLING noise, which more eval data answers.

    "indecisive" was HONEST if a pooled 95% CI contains 0.80
    "indecisive" was WRONG  if it excludes 0.80, in either direction

A2.5. The size of a hypothetical off-by-one, for the record. A2 established
that the shipped addressing is correct, but also that the assert cited for it
gives zero coverage, so the magnitude is worth having on file.

    python audit_premise.py --tag gated_seed0
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


import _paths                                                  # noqa: F401
import auroc as A                                              # noqa: E402
import features as FE                                          # noqa: E402

THREE = ["f_tok", "f_s1", "f_s2"]


def hdr(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}", flush=True)


def pooled(d_list):
    """Concatenate several streams, keeping sequence ids globally unique."""
    out, off = {}, 0
    for d in d_list:
        dd = dict(d)
        dd["seq"] = dd["seq"] + off
        off = dd["seq"].max() + 1
        for k, v in dd.items():
            if k.startswith("_"):
                continue
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gated_seed0")
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--streams", type=int, nargs="+",
                    default=[777, 888, 505, 606])
    a = ap.parse_args()
    torch.set_num_threads(2)

    m, ck = FE.load(a.tag)
    print(f"A7  model {a.tag} (seed {ck['seed']})")
    print(f"  NOTE: 606 is my held-out stream. It appears here because A7 is a\n"
          f"  question about 04's bar, not about my own conclusion, and the\n"
          f"  pre-registration licenses pooling all four. My own bars are not\n"
          f"  scored on it.")

    ds = {}
    for s in a.streams:
        ds[s] = FE.collect(m, s, batches=a.batches)
        print(f"  stream {s}: loss {ds[s]['_loss']:.4f}  "
              f"rows {ds[s]['seq'].size}")

    hdr("A7  the premise bar, per stream and pooled")
    print("  pooled AUROC of the gate's own [tok, s1, s2], NOISY vs REVISED,\n"
          "  writes to an address written before, GBT, by-sequence split.")

    def measure(d, seed=0, nboot=1000):
        seen = d["wtype"] != FE.UNSEEN
        nr = np.isin(d["cls_w"], [1, 2])
        mk = seen & nr
        lab = (d["cls_w"] == 2).astype(int)
        X = np.stack([d[f][mk] for f in THREE], 1)
        sc, yte, ste = A.fit_scores(X, lab[mk], d["seq"][mk], kind="gbm",
                                    seed=seed)
        au = float(roc_auc_score(yte, sc))
        b = A.boot_auroc(sc, yte, ste, n=nboot, seed=seed)
        return au, b, int(mk.sum())

    print(f"\n  {'stream':>10}{'AUROC':>10}{'boot sd':>10}"
          f"{'95% CI':>22}{'n rows':>10}{'vs 0.80':>10}")
    for s in a.streams:
        au, b, n = measure(ds[s])
        lo, hi = np.percentile(b, [2.5, 97.5])
        side = "above" if lo > 0.80 else "below" if hi < 0.80 else "straddles"
        print(f"  {s:>10}{au:>10.4f}{b.std(ddof=1):>10.4f}"
              f"   [{lo:.4f}, {hi:.4f}]{n:>10}{side:>10}")

    for label, sel in (("777+888 (04's two)", [777, 888]),
                       ("all four", a.streams)):
        dp = pooled([ds[s] for s in sel])
        au, b, n = measure(dp)
        lo, hi = np.percentile(b, [2.5, 97.5])
        verdict = ("HONEST: the pooled CI contains 0.80"
                   if lo <= 0.80 <= hi else
                   "WRONG: the pooled CI EXCLUDES 0.80 "
                   f"({'above' if lo > 0.80 else 'below'})")
        print(f"\n  pooled {label}: AUROC {au:.4f} +/- {b.std(ddof=1):.4f}"
              f"   95% CI [{lo:.4f}, {hi:.4f}]   n {n}")
        print(f"    -> 04's 'indecisive' was {verdict}")

    hdr("A7b  how much of the 777-vs-888 gap is eval sampling at all?")
    a7, b7, _ = measure(ds[777])
    a8, b8, _ = measure(ds[888])
    print(f"  my retrained seed-0 model: 777 {a7:.4f}   888 {a8:.4f}   "
          f"gap {a7 - a8:+.4f}")
    print(f"  04 reported:               777 0.8095   888 0.7983   "
          f"gap +0.0112")
    print(f"  bootstrap sd of a single-stream estimate ~{b7.std(ddof=1):.4f},\n"
          f"  so a gap of this size is about "
          f"{abs(a7 - a8) / b7.std(ddof=1):.1f} single-stream SDs.")
    print("\n  Across SPLIT seeds on one stream, for comparison:")
    sp = [measure(ds[777], seed=i, nboot=50)[0] for i in range(10)]
    print(f"    777 across 10 splits: mean {np.mean(sp):.4f}  "
          f"sd {np.std(sp, ddof=1):.4f}  min {min(sp):.4f}  max {max(sp):.4f}")

    # ---------------------------------------------------------------- A2.5
    hdr("A2.5  the size of a hypothetical off-by-one (A2 showed there is none)")
    d = ds[505]
    seen = d["wtype"] != FE.UNSEEN
    print("  s1 alone, NOISY vs REVISED, on CHANGE writes unless stated.\n"
          "  Each row changes exactly ONE indexing choice.")
    change = d["wtype"] == FE.CHANGE
    variants = [
        ("as shipped: label class(x_{t-1}), mask sn_addr",
         d["cls_w"], change),
        ("label class(x_t) instead (the known trap)", d["cls_q"], change),
        ("mask sn_task instead (describes the VALUE token)",
         d["cls_w"], d["sn_task"].astype(bool) & change),
        ("pooled over all seen writes, as shipped", d["cls_w"], seen),
        ("pooled, label class(x_t)", d["cls_q"], seen),
    ]
    print(f"\n  {'variant':54s}{'AUROC':>9}{'n':>9}")
    for nm, cls, mkx in variants:
        nr = np.isin(cls, [1, 2])
        mk = nr & mkx
        au, n = A.single(d["f_s1"][mk], (cls[mk] == 2).astype(int))
        print(f"  {nm:54s}{au:>9.4f}{n:>9d}")
    print("\n  04 reported the label trap as 0.8073 -> 0.5201 on 777 and the\n"
          "  mask slip as worth 0.010 of AUROC. Compare above.")


if __name__ == "__main__":
    main()
