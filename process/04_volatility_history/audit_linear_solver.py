"""AUDIT. Is the 'linear 0.676' sound, given the solver warnings around it?

The session auditing this directory noted (its A9) that PART 5 of
`logs/separability_777.log` interleaves divide-by-zero, overflow and
invalid-value RuntimeWarnings from the logistic solver into the very rows that
print `linear 0.676`. That is a fair catch: those warnings are in the committed
log.

I did check this during the session and found the figure solver-independent, but
I ran the check inline and reported it in conversation only. **A verification
that lives in chat scrollback is not a verification**, so it is recorded here.

The warnings come from inside LBFGS's line search, not from the features: every
input column is finite (`audit_sanity.py` CHECK 3 and the range check below).
The test is whether the fitted AUROC depends on the solver, and what a
better-conditioned fit of the same model class gives.

    python audit_linear_solver.py
"""
import os
import sys
import warnings

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import auroc as A
import features as FE

if __name__ == "__main__":
    torch.set_num_threads(2)
    m, _ = FE.load("gated_seed0")
    d = FE.collect(m, 777, batches=6)
    clw, wt, seqs = d["cls_w"], d["wtype"], d["seq"]
    mk = np.isin(clw, [1, 2]) & (wt != FE.UNSEEN)
    lab = (clw == 2).astype(int)
    F3 = ["f_tok", "f_s1", "f_s2"]
    X = np.stack([d[f][mk] for f in F3], 1)
    y, s = lab[mk], seqs[mk]
    tr, te = A.seq_split(s, 0)

    print("input columns on the analysed rows:")
    for i, f in enumerate(F3):
        print(f"  {f:8s} finite {np.isfinite(X[:, i]).all()}   "
              f"range [{X[:, i].min():.4g}, {X[:, i].max():.4g}]")

    print("\nthe warnings arise inside the solver, not from the inputs. Does the")
    print("fitted AUROC depend on the solver?\n")
    print(f"  {'solver':14s}{'AUROC':>9}{'warnings':>11}")
    for sol in ("lbfgs", "newton-cg", "liblinear", "saga"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, solver=sol)).fit(X[tr], y[tr])
            au = roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])
        print(f"  {sol:14s}{au:>9.4f}{len(w):>11d}")

    print("\nand the nonlinear reads of the SAME three features, for the")
    print("linear-vs-nonlinear comparison the 0.676 is one half of:")
    print(f"  {'model':34s}{'AUROC':>9}")
    for nm, pipe in (
            ("spline basis, 6 knots",
             make_pipeline(StandardScaler(),
                           SplineTransformer(n_knots=6, degree=3),
                           LogisticRegression(max_iter=5000))),
            ("polynomial degree 2",
             make_pipeline(StandardScaler(),
                           PolynomialFeatures(2, include_bias=False),
                           LogisticRegression(max_iter=5000)))):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X[tr], y[tr])
        print(f"  {nm:34s}"
              f"{roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1]):>9.4f}")
    au_t, _, _, _ = A.joint(d, mk, F3, lab, seqs, kind="gbm")
    print(f"  {'gradient-boosted tree':34s}{au_t:>9.4f}")
    print("\nNOTE: the +0.134 'free nonlinear headroom' ARGUMENT built on this")
    print("gap is retracted (AUDIT.md RETRACTED 2) because the shipped gate is")
    print("already nonlinear. The MEASUREMENT is what this file checks.")
