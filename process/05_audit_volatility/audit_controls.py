"""A3 and A4. Is the +0.081 history gain capacity, and are the error bars honest?

A3 - capacity. 04's only control on the history gain was raw position (+0.003),
one low-entropy feature against five history features. Five junk features are
added here instead, two of them sharing the history features' GROUPED structure:

    j_gauss      iid N(0,1) per row
    j_unif       iid U(0,1) per row
    j_heavy      iid Student-t(2) per row, matching h_ratio's tail
    j_seqconst   one N(0,1) draw per SEQUENCE
    j_addrconst  one N(0,1) draw per (SEQUENCE, ADDRESS)

`j_addrconst` is the sharpest control available: exactly the per-address grouped
structure a volatility memory would have, carrying zero class information. If
the tree can turn that into AUROC, part of the +0.081 is capacity rather than
information. `j_seqconst` doubles as a check that the by-sequence split works.

A4.1 - error bars. `auroc.py`'s bootstrap resamples only TEST sequences with the
classifier and the split held FIXED, so it cannot see train/test-split or
fitting variability. The delta is recomputed across independent split seeds and
the two spreads compared.

A4.2 - leakage. The real label is replaced by a class drawn independently per
(sequence, address). Nothing can predict an independent label, so any AUROC away
from 0.5 is leakage.

Bars in PREREGISTRATION.md sections A3 and A4.

    python audit_controls.py --stream 505 --tag gated_seed0
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import auroc as A                                              # noqa: E402
import beta_optimal as BO                                      # noqa: E402
import features as FE                                          # noqa: E402
import task as T                                               # noqa: E402

INST, HIST = FE.INST, FE.HIST
JUNK = ["j_gauss", "j_unif", "j_heavy", "j_seqconst", "j_addrconst"]


def hdr(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=int, default=505)
    ap.add_argument("--tag", default="gated_seed0")
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--splits", type=int, default=20)
    a = ap.parse_args()
    torch.set_num_threads(2)

    m, ck = FE.load(a.tag)
    d = BO.collect(m, a.stream, batches=a.batches, junk_rng=a.stream + 1)
    seqs = d["seq"]
    seen = d["wtype"] != FE.UNSEEN
    nr = np.isin(d["cls_w"], [1, 2])
    mk = seen & nr
    lab = (d["cls_w"] == 2).astype(int)
    print(f"A3/A4  model {a.tag} (seed {ck['seed']})   stream {a.stream}   "
          f"loss {d['_loss']:.4f}")
    print(f"  rows in 04's primary set: {mk.sum()}")

    # ------------------------------------------------------------------- A3
    hdr("A3  is the history gain capacity? five junk features vs five real ones")
    print("  all paired on identical rows and the identical by-sequence split.")
    rows = [
        ("5 zero-state -> + 5 HISTORY (04's claim)", INST, INST + HIST),
        ("5 zero-state -> + 5 JUNK (capacity control)", INST, INST + JUNK),
        ("5 zero-state -> + position only (04's control)", INST,
         INST + ["pos"]),
        ("5 zero-state -> + j_addrconst only", INST, INST + ["j_addrconst"]),
        ("5 zero-state -> + j_seqconst only", INST, INST + ["j_seqconst"]),
        ("+ 5 JUNK -> + 5 JUNK + 5 HISTORY", INST + JUNK,
         INST + JUNK + HIST),
    ]
    print(f"\n  {'comparison':46s}{'base':>8}{'with':>8}{'delta':>9}"
          f"{'sd':>8}{'n_test':>8}")
    got = {}
    for nm, fa, fb in rows:
        au_a, au_b, dl, sd, nte = A.joint_pair(d, mk, fa, fb, lab, seqs)
        got[nm] = dl
        print(f"  {nm:46s}{au_a:>8.4f}{au_b:>8.4f}{dl:>+9.4f}{sd:>8.4f}"
              f"{nte:>8d}")
    gj = got["5 zero-state -> + 5 JUNK (capacity control)"]
    print(f"\n  BARS: gain_junk >= +0.02 -> (b) WEAKENED, part of the history\n"
          f"        gain is capacity;  <= +0.01 -> capacity REJECTED.")
    print(f"  gain_junk = {gj:+.4f}  ->  "
          f"{'(b) WEAKENED' if gj >= 0.02 else 'capacity REJECTED' if gj <= 0.01 else 'AMBIGUOUS'}")

    print("\n  single-feature AUROC of each junk control (must be ~0.5):")
    for f in JUNK + ["pos"]:
        au, n = A.single(d[f][mk], lab[mk])
        print(f"    {f:14s}{au:>8.4f}   n {n}")

    # ----------------------------------------------------------------- A4.1
    hdr("A4.1  is the reported bootstrap SD of the delta honest?")
    print("  auroc.py resamples TEST sequences only, with the fitted\n"
          "  classifier and the split held fixed. Below: the same delta\n"
          "  recomputed across independent by-sequence SPLITS.")
    ds, sds = [], []
    for sd_ in range(a.splits):
        au_a, au_b, dl, sdb, _ = A.joint_pair(d, mk, INST, INST + HIST, lab,
                                              seqs, seed=sd_)
        ds.append(dl)
        sds.append(sdb)
    ds, sds = np.array(ds), np.array(sds)
    print(f"\n  {a.splits} splits: delta mean {ds.mean():+.4f}   "
          f"sd across splits {ds.std(ddof=1):.4f}")
    print(f"  mean reported bootstrap sd (within one split) "
          f"{sds.mean():.4f}")
    ratio = ds.std(ddof=1) / sds.mean()
    print(f"  ratio_sd = {ratio:.2f}")
    print(f"  BARS: >= 2.0 UNDERSTATED, <= 1.5 honest")
    print(f"  -> {'UNDERSTATED' if ratio >= 2.0 else 'HONEST' if ratio <= 1.5 else 'AMBIGUOUS'}")
    print(f"\n  min/max delta across splits: {ds.min():+.4f} / {ds.max():+.4f}")

    # ----------------------------------------------------------------- A4.2
    hdr("A4.2  leakage probe: a label drawn independently per (seq, address)")
    rng = np.random.default_rng(a.stream + 7)
    nseq = int(seqs.max()) + 1
    fake_tab = rng.integers(0, 2, size=(nseq, T.N_SYM))
    print("  addresses are recovered from the token stream, not from features;\n"
          "  the stream is redrawn from the same seed, so it is the same rows.")
    ev = np.random.default_rng(a.stream)
    addrs = []
    for b in range(a.batches):
        x, _, _ = T.sample(ev, 32)
        xx = x.numpy()
        addrs.append(np.concatenate(
            [np.zeros((32, 1), np.int64), xx[:, :T.L - 1]], 1))
    addr = np.concatenate(addrs).reshape(-1)
    fake = fake_tab[seqs, addr]
    print(f"  fake label base rate {fake[mk].mean():.4f}")
    for nm, fs in (("5 zero-state", INST), ("5 zero-state + history",
                                            INST + HIST),
                   ("history only", HIST)):
        au, sd, _, nte = A.joint(d, mk, fs, fake, seqs)
        off = abs(au - 0.5)
        print(f"  {nm:26s} AUROC {au:.4f} +/- {sd:.4f}   |AUROC-0.5| "
              f"{off:.4f}   {'PASS' if off <= 0.02 else 'FAIL'}")
    print("  bar: |AUROC - 0.5| <= 0.02")

    # sanity: the REAL label on the same rows, same call path
    au, sd, _, _ = A.joint(d, mk, INST + HIST, lab, seqs)
    print(f"\n  for scale, the REAL label through the same code path: "
          f"{au:.4f} +/- {sd:.4f}")


if __name__ == "__main__":
    main()
