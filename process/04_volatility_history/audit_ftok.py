"""AUDIT. Why does f_tok, whose marginal AUROC is 0.50, add +0.17 in company?

audit_s2_swap.py put the legacy gate at 0.7859 on [tok, s1, s2] where
model_vs_task.py had it at 0.6161 on [s1, s2]. A feature that is at chance on
its own adding 0.17 in combination is either a real interaction or a leak, and
it has to be settled, because f_tok is in the PREMISE TEST's feature set.

Two candidate explanations:
  INTERACTION  f_tok is a learned projection of a width-4 causal conv, so it
               sees x_{t-3..t}. The local token window itself contains
               repeat-vs-change information, which is exactly the conditioning
               variable s1 needs. That is legitimate: it is part of the shipped
               architecture and available to the real gate at no cost.
  LEAK         f_tok identifies the sequence or the address well enough for the
               tree to recover a per-sequence symbol->class map. The class is
               re-randomised per sequence and the split is by sequence, so this
               should be impossible, but it is testable.

The leak test: replace f_tok with the information it could be leaking and see
whether that reproduces the gain. If the window's repeat/change content
reproduces it, it is an interaction.

    python audit_ftok.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import auroc as A
import features as FE
import task as T
from audit_trend import collect_legacy


def add_window_feats(d, stream, batches=6, batch=32):
    """Rebuild the token stream and derive, from the width-4 conv window ALONE,
    the repeat/change structure the conv could be seeing. No memory, no state."""
    ev = np.random.default_rng(stream)
    w_eq1, w_eq2, w_eq3, w_ndist = [], [], [], []
    for _ in range(batches):
        x, _, _ = T.sample(ev, batch)
        tk = x[:, :T.L].numpy()
        B, L = tk.shape
        e1 = np.zeros((B, L)); e2 = np.zeros((B, L)); e3 = np.zeros((B, L))
        nd = np.zeros((B, L))
        for t in range(L):
            # the conv at t sees x_{t-3..t}; these are functions of that window
            e1[:, t] = (tk[:, t] == tk[:, max(t - 1, 0)]) if t >= 1 else 0
            e2[:, t] = (tk[:, t] == tk[:, max(t - 2, 0)]) if t >= 2 else 0
            e3[:, t] = (tk[:, t] == tk[:, max(t - 3, 0)]) if t >= 3 else 0
            lo = max(t - 3, 0)
            nd[:, t] = [len(set(tk[b, lo:t + 1])) for b in range(B)]
        w_eq1.append(e1); w_eq2.append(e2); w_eq3.append(e3); w_ndist.append(nd)
    f = lambda z: np.concatenate(z).reshape(-1)
    d["w_eq1"], d["w_eq2"] = f(w_eq1), f(w_eq2)
    d["w_eq3"], d["w_ndist"] = f(w_eq3), f(w_ndist)
    return d


WIN = ["w_eq1", "w_eq2", "w_eq3", "w_ndist"]

if __name__ == "__main__":
    torch.set_num_threads(2)
    for mnm, getter in (("legacy linear", lambda: collect_legacy(777)),
                        ("shipped seed 0",
                         lambda: FE.collect(FE.load("gated_seed0")[0], 777))):
        d = add_window_feats(getter(), 777)
        clw, wt, seqs = d["cls_w"], d["wtype"], d["seq"]
        mk = np.isin(clw, [1, 2]) & (wt != FE.UNSEEN)
        lab = (clw == 2).astype(int)
        print(f"\n=== {mnm} ===")
        au, n = A.single(d["f_tok"][mk], lab[mk])
        print(f"  f_tok marginal AUROC {au:.4f}  (n {n})")
        print(f"\n  {'feature set':44s}{'AUROC':>9}{'sd':>8}")
        for snm, fs in (
                ("[s1, s2]", ["f_s1", "f_s2"]),
                ("[tok, s1, s2]              the gate's own", ["f_tok", "f_s1", "f_s2"]),
                ("[tok] alone", ["f_tok"]),
                ("[s1, s2] + conv-window repeat structure",
                 ["f_s1", "f_s2"] + WIN),
                ("[tok, s1, s2] + window structure",
                 ["f_tok", "f_s1", "f_s2"] + WIN),
                ("window structure alone", WIN),
                ("[s1, s2, |Sk|, |v|]  (no tok)",
                 ["f_s1", "f_s2", "f_nstored", "f_nv"]),
                ("all five zero-extra-state", FE.INST)):
            a_, sd, _, _ = A.joint(d, mk, fs, lab, seqs)
            print(f"  {snm:44s}{a_:>9.4f}{sd:>8.4f}")
