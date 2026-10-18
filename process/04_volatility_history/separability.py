"""TASK 1. How well do the INSTANTANEOUS gate features separate NOISY from
REVISED, under correct labelling and correct conditioning?

The repo records three disagreeing values:

    0.834   process/README.md:109                   "s1 ... on CHANGE writes"
    0.353   process/README.md:110                   "s1 ... POOLED"
    0.476   archive/failed_features/rate_test.py:11  "(s1,s2) gets 0.476 = chance"

Bars are in PREREGISTRATION.md. The premise test is the pooled joint-classifier
AUROC over every zero-extra-state feature: >= 0.80 kills the session's idea,
<= 0.70 supports it.

    python separability.py --stream 777
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import auroc as A
import features as FE

INST_SHIPPED = ["f_tok", "f_s1", "f_s2"]
INST_ALL = FE.INST            # + ||S k|| and ||v||, both zero extra state


def nr_mask(cls, extra):
    """NOISY or REVISED only, plus whatever conditioning `extra` adds."""
    return np.isin(cls, [1, 2]) & extra


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
    seqs = d["seq"]
    clw, clq = d["cls_w"], d["cls_q"]
    wt = d["wtype"]
    seen = wt != FE.UNSEEN                    # address written before: exact
    change = wt == FE.CHANGE
    repeat = wt == FE.REPEAT
    pos_ok = d["pos"] > 0                     # t=0 writes to the pad address

    print(f"model {a.tag} (seed {ck['seed']}, {ck['steps']} steps)   "
          f"eval stream {a.stream}, {a.batches} batches of 32")
    print(f"loss {d['_loss']:.4f}   acc {d['_acc']:.4f}   "
          f"rows {seqs.size}   published on 12345: 1.5428 / 0.6458")

    # ---------------------------------------------------------------- part 1
    hdr("PART 1  write-type composition by class of the WRITE ADDRESS")
    print("repo claim (process/03/README.md:73-75): REVISED = 65% repeats + 35% "
          "changes,\n            NOISY = '~all changes'")
    print(f"\n{'class':10s}{'n writes':>10}{'unseen':>9}{'repeat':>9}{'change':>9}"
          f"{'repeat|seen':>13}")
    for i, nm in enumerate(("STABLE", "NOISY", "REVISED")):
        mk = (clw == i) & pos_ok
        n = mk.sum()
        u, r, c = [np.mean(wt[mk] == z) for z in (FE.UNSEEN, FE.REPEAT, FE.CHANGE)]
        rs = np.mean(wt[mk & seen] == FE.REPEAT)
        print(f"{nm:10s}{n:>10d}{u:>9.3f}{r:>9.3f}{c:>9.3f}{rs:>13.3f}")

    # ---------------------------------------------------------------- part 2
    hdr("PART 2  does beta matter on repeat writes?  ||e|| by write type")
    print("if ||e|| on repeats were ~0, beta would multiply zero there and the\n"
          "pooled conditioning would be harmless dilution rather than a trap")
    ne = d["ne"]
    print(f"\n{'write type':12s}{'n':>9}{'mean ||e||':>12}{'median':>10}"
          f"{'mean beta':>11}{'mean beta*||e||':>17}")
    for nm, mk in (("unseen", (wt == FE.UNSEEN) & pos_ok),
                   ("repeat", repeat), ("change", change)):
        print(f"{nm:12s}{mk.sum():>9d}{ne[mk].mean():>12.4f}"
              f"{np.median(ne[mk]):>10.4f}{d['beta'][mk].mean():>11.4f}"
              f"{(d['beta'][mk]*ne[mk]).mean():>17.4f}")

    # ---------------------------------------------------------------- part 3
    hdr("PART 3  the factorial.  AUROC NOISY vs REVISED, single raw features")
    print("AUROC > 0.5 means the feature reads HIGHER on REVISED.")
    conds = [
        ("all seen writes", seen),
        ("CHANGE writes only", change),
        ("REPEAT writes only", repeat),
        ("every position t>0", pos_ok),
    ]
    feats = ["f_s1", "f_s2", "f_nstored", "f_nv", "f_tok", "beta"]
    print(f"\n{'conditioning':22s}{'n':>8}" + "".join(f"{f:>12}" for f in feats))
    for nm, extra in conds:
        mk = nr_mask(clw, extra)
        lab = (clw[mk] == 2).astype(int)
        row = []
        for f in feats:
            au, n = A.single(d[f][mk], lab)
            row.append(f"{au:>12.4f}")
        print(f"{nm:22s}{mk.sum():>8d}" + "".join(row))

    hdr("PART 3b  the labelling trap: same thing labelled by class(x_t)")
    print("class(x_t) labels a write by the class of the VALUE token, not the\n"
          "address. This is the known wrong labelling.")
    print(f"\n{'conditioning':22s}{'n':>8}" + "".join(f"{f:>12}" for f in feats))
    for nm, extra in conds:
        mk = nr_mask(clq, extra)
        lab = (clq[mk] == 2).astype(int)
        row = []
        for f in feats:
            au, n = A.single(d[f][mk], lab)
            row.append(f"{au:>12.4f}")
        print(f"{nm:22s}{mk.sum():>8d}" + "".join(row))

    hdr("PART 3c  the 'seen' mask: the task's own array describes x_t, not the "
        "address")
    print("every earlier script masked with sn (about x_t) while labelling with\n"
          "class(x_{t-1}). These are the three masks side by side, s1 only.")
    for nm, extra in (("sn_task (as used)", d["sn_task"].astype(bool) & pos_ok),
                      ("sn_task shifted", d["sn_addr"].astype(bool) & pos_ok),
                      ("exact bookkeeping", seen)):
        mk = nr_mask(clw, extra)
        au, n = A.single(d["f_s1"][mk], (clw[mk] == 2).astype(int))
        print(f"  {nm:22s} n={mk.sum():>6d}   s1 AUROC {au:.4f}")

    # ---------------------------------------------------------------- part 4
    hdr("PART 4  replicating signinit_test's procedure (the 0.476)")
    print("3-way gradient-boosted classifier on (s1,s2), scored by P(REVISED),\n"
          "token-level 50/50 split, masked by the task's sn array. Exactly as\n"
          "archive/failed_features/signinit_test.py:100-110 does it.")
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score
    sn = d["sn_task"].astype(bool)
    X = np.stack([d["f_s1"][sn], d["f_s2"][sn]], 1)
    y = clw[sn].astype(int)
    n = len(y); tr = np.arange(n) < n // 2
    g = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[tr], y[tr])
    pp = g.predict_proba(X[~tr])
    print(f"\n  3-way accuracy {accuracy_score(y[~tr], g.predict(X[~tr])):.4f}"
          f"   (chance {max(np.bincount(y)/len(y)):.4f})")
    for i, nm in enumerate(("STABLE", "NOISY", "REVISED")):
        print(f"    AUROC {nm:8s} vs rest: "
              f"{roc_auc_score((y[~tr]==i).astype(int), pp[:, i]):.4f}")
    rr = np.isin(y[~tr], [1, 2])
    print(f"    AUROC NOISY vs REVISED only: "
          f"{roc_auc_score((y[~tr][rr]==2).astype(int), pp[rr, 2]):.4f}"
          f"   <- signinit reported 0.4763")
    print("\n  and the SAME features scored as a 2-class problem trained on the\n"
          "  NOISY-vs-REVISED contrast directly, by-sequence split:")
    for nm, extra in conds[:3]:
        mk = nr_mask(clw, extra)
        au, sd, ntot, nte = A.joint(d, mk, ["f_s1", "f_s2"],
                                    (clw == 2).astype(int), seqs)
        print(f"    {nm:22s} AUROC {au:.4f} +/- {sd:.4f}   (n_test {nte})")

    # ---------------------------------------------------------------- part 5
    hdr("PART 5  THE PREMISE TEST (pre-registered)")
    print("Pooled over all seen writes, which is where the gate must act without\n"
          "being told whether the write is a repeat or a change.")
    print("  >= 0.80  premise DIES, instantaneous info is close to sufficient")
    print("  <= 0.70  premise SUPPORTED, there is an information gap")
    mk = nr_mask(clw, seen)
    lab = (clw == 2).astype(int)
    print(f"\n{'feature set':38s}{'kind':>6}{'AUROC':>9}{'boot sd':>10}{'n_test':>9}")
    for nm, fs in (("shipped gate [tok, s1, s2]", INST_SHIPPED),
                   ("all zero-state [+ ||Sk||, ||v||]", INST_ALL)):
        for kind in ("logit", "gbm"):
            au, sd, ntot, nte = A.joint(d, mk, fs, lab, seqs, kind=kind)
            print(f"{nm:38s}{kind:>6}{au:>9.4f}{sd:>10.4f}{nte:>9d}")
    print("\n  for reference, the same classifier on CHANGE writes only:")
    mk2 = nr_mask(clw, change)
    for nm, fs in (("shipped gate [tok, s1, s2]", INST_SHIPPED),
                   ("all zero-state [+ ||Sk||, ||v||]", INST_ALL)):
        au, sd, ntot, nte = A.joint(d, mk2, fs, lab, seqs, kind="gbm")
        print(f"{nm:38s}{'gbm':>6}{au:>9.4f}{sd:>10.4f}{nte:>9d}")


if __name__ == "__main__":
    main()
