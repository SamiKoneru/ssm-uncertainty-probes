"""AUDIT. Checks I should have run during the session and did not.

1. Does the model this directory measures actually REPRODUCE the published
   result? Every number here describes `gated_seed0`. If that training run
   differs from the published one, every conclusion describes a different model.
   The published figure is loss 1.5428 / acc 0.6458 at gate_width=32 on eval
   stream 12345, and the input-only baseline is 1.6322. Neither was ever checked
   in this directory: all measurement ran on streams 777 and 888, so an absolute
   comparison was never possible.
2. Is the +0.08 history gain CAPACITY rather than information? The session's only
   control was raw position. Add high-entropy uninformative features.
3. Can the gradient-boosted tree exploit NaN as a label-correlated branch? On the
   `seen` mask the history features should never be NaN. Verified, not assumed.
4. Does the by-sequence split actually separate sequences?

    python audit_sanity.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                "gated_delta_ssm"))

import auroc as A
import features as FE
import task as T
from model import GatedDeltaSSM


@torch.no_grad()
def score(m, stream, batches=6, batch=32):
    ev = np.random.default_rng(stream)
    L_, A_ = [], []
    for _ in range(batches):
        x, _, _ = T.sample(ev, batch)
        lg = m(x[:, :-1])
        L_.append(F.cross_entropy(lg.reshape(-1, T.N_SYM),
                                  x[:, 1:].reshape(-1)).item())
        A_.append((lg.argmax(-1) == x[:, 1:]).float().mean().item())
    return float(np.mean(L_)), float(np.mean(A_))


def hdr(s):
    print(f"\n{'='*74}\n{s}\n{'='*74}", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)

    hdr("CHECK 1  does gated_seed0 reproduce the published result?")
    print("published, 3 seeds, stream 12345:  gated 1.5428 / 0.6458,"
          "  baseline 1.6322")
    print(f"\n{'model':22s}{'stream':>8}{'loss':>9}{'acc':>9}")
    for tag in ("gated_seed0", "gated_seed1"):
        m, ck = FE.load(tag)
        for stream in (12345, 777, 888):
            l, a = score(m, stream)
            print(f"{tag:22s}{stream:>8}{l:>9.4f}{a:>9.4f}")
    print("\nNOTE: no input-only baseline was ever trained in this directory,"
          "\nso the GAIN cannot be computed here, only the absolute loss"
          "\ncompared against the published absolute loss.")

    hdr("CHECK 2  is the +0.08 history gain capacity rather than information?")
    m, _ = FE.load("gated_seed0")
    d = FE.collect(m, 777, batches=6)
    d["pos"] = d["pos"].astype(np.float64)
    clw, wt, seqs = d["cls_w"], d["wtype"], d["seq"]
    mk = np.isin(clw, [1, 2]) & (wt != FE.UNSEEN)
    lab = (clw == 2).astype(int)

    # five uninformative features with the same marginal shapes as the history
    # block: continuous, bounded, heavy-tailed, and a count
    rng = np.random.default_rng(31337)
    n = clw.size
    d["junk1"] = rng.uniform(-1, 1, n)                      # like a cosine
    d["junk2"] = rng.normal(0, 1, n)
    d["junk3"] = rng.standard_cauchy(n)                     # heavy tailed
    d["junk4"] = rng.integers(0, 60, n).astype(np.float64)  # like a visit count
    d["junk5"] = rng.exponential(8, n)                      # like a mean norm
    JUNK = ["junk1", "junk2", "junk3", "junk4", "junk5"]
    # and a per-SEQUENCE constant, the shape most able to fake signal under a
    # by-sequence split if the split ever leaked
    sj = rng.normal(0, 1, seqs.max() + 1)
    d["junk_seq"] = sj[seqs]

    print(f"\n{'comparison':46s}{'base':>8}{'with':>8}{'delta':>9}{'sd':>8}")
    for nm, fa, fb in (
            ("inst -> inst + 5 EXACT history features", FE.INST,
             FE.INST + FE.HIST),
            ("inst -> inst + 5 RANDOM features", FE.INST, FE.INST + JUNK),
            ("inst -> inst + a per-sequence constant", FE.INST,
             FE.INST + ["junk_seq"]),
            ("inst -> inst + position", FE.INST, FE.INST + ["pos"])):
        a_, b_, dl, sd, nte = A.joint_pair(d, mk, fa, fb, lab, seqs)
        print(f"{nm:46s}{a_:>8.4f}{b_:>8.4f}{dl:>+9.4f}{sd:>8.4f}")

    hdr("CHECK 3  NaN as a label-correlated tree branch")
    for f in FE.HIST + FE.ORACLE:
        v = d[f][mk]
        bad = ~np.isfinite(v)
        print(f"  {f:12s} non-finite on the analysed rows: {bad.sum():6d}"
              f"   ({100*bad.mean():.3f}%)")
    print("  any non-finite at all on these rows means the tree could branch on"
          "\n  missingness; zero means it cannot.")

    hdr("CHECK 4  does the by-sequence split separate sequences?")
    s = seqs[mk]
    tr, te = A.seq_split(s, 0)
    ov = np.intersect1d(np.unique(s[tr]), np.unique(s[te]))
    print(f"  train sequences {len(np.unique(s[tr]))}, "
          f"test {len(np.unique(s[te]))}, OVERLAP {len(ov)}")
    print(f"  train rows {tr.sum()}, test rows {te.sum()}, "
          f"total {mk.sum()}, disjoint: {tr.sum()+te.sum() == mk.sum()}")
    print(f"  class balance train {lab[mk][tr].mean():.4f} "
          f"test {lab[mk][te].mean():.4f}")
