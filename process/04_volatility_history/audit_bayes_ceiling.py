"""AUDIT, independent check. What is the EXACT Bayes ceiling for NOISY vs
REVISED given an address's value history?

AUDIT.md listed this as unchecked: "whether the ORACLE feature set is a
sufficient statistic for the per-address history. It is close by construction,
but the true Bayes ceiling given the full value multiset was not computed, so
0.92 is a measured ceiling for these features, not a proof."

This computes it from task.py's generative process ALONE. No model, no trained
weights, no features: the exact log-likelihood ratio for the observed value
sequence at an address. It is therefore an upper bound on ANY per-address
history mechanism, since every such mechanism sees a lossy function of that
sequence.

The two likelihoods, read off task.py:20-41.

REVISED. `nxt = succ[s]` is read BEFORE the possible revision, so the value at
visit i+1 equals the value at visit i unless a revision fired at visit i:
    P(v_1)         = 1/40
    P(v_i = v_{i-1}) = (1 - P_REV) + P_REV/N_SYM
    P(v_i = other)   = P_REV/N_SYM        for each specific other symbol

NOISY. The option set O is N_OPT slots, each an independent uniform draw over
N_SYM (so O may contain duplicates), and each visit picks a slot uniformly:
    P(v) = N_SYM^-N_OPT * SUM over (m_1..m_k, r), m_j >= 1, sum m_j + r = N_OPT,
           of  multinomial(N_OPT; m_1..m_k, r) * (N_SYM-k)^r
                * prod_j (m_j / N_OPT)^{c_j}
where u_1..u_k are the distinct observed values with counts c_1..c_k. Note
P(v|NOISY) = 0 exactly once k > N_OPT, which is a hard discriminator.

Self-check: the formula must give exactly 1/N_SYM for a single observation.

    python audit_bayes_ceiling.py
"""
import os
import sys
from functools import lru_cache
from math import factorial, log

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                "gated_delta_ssm"))

from sklearn.metrics import roc_auc_score

import task as T

NS, NO, PR = T.N_SYM, T.N_OPT, T.P_REV


def _compositions(k, total):
    """All (m_1..m_k, r) with m_j >= 1 and sum m_j + r = total."""
    if k == 0:
        yield (), total
        return
    for first in range(1, total - (k - 1) + 1):
        for rest, r in _compositions(k - 1, total - first):
            yield (first,) + rest, r


@lru_cache(maxsize=None)
def log_p_noisy(counts):
    """counts: sorted tuple of per-distinct-value observation counts."""
    k = len(counts)
    if k > NO:
        return -np.inf
    n_fact = factorial(NO)
    tot = 0.0
    for ms, r in _compositions(k, NO):
        coef = n_fact
        for m in ms:
            coef //= factorial(m)
        coef //= factorial(r)
        term = coef * ((NS - k) ** r)
        for m, c in zip(ms, counts):
            term *= (m / NO) ** c
        tot += term
    return log(tot) - NO * log(NS)


@lru_cache(maxsize=None)
def log_p_revised(n, n_same):
    """n observations, n_same of the n-1 transitions repeated the value."""
    p_same = (1 - PR) + PR / NS
    p_diff = PR / NS
    return (-log(NS) + n_same * log(p_same)
            + (n - 1 - n_same) * log(p_diff))


def selfcheck():
    """A single observation must have probability exactly 1/N_SYM either way."""
    a = np.exp(log_p_noisy((1,)))
    b = np.exp(log_p_revised(1, 0))
    print(f"  single observation: P(v|NOISY) = {a:.10f}, "
          f"P(v|REVISED) = {b:.10f}, 1/N_SYM = {1/NS:.10f}")
    ok = abs(a - 1 / NS) < 1e-12 and abs(b - 1 / NS) < 1e-12
    print(f"  normalisation self-check: {'PASS' if ok else 'FAIL'}")
    return ok


def run(stream, batches=6, batch=32):
    """Score every write to a previously-written address, exactly the pooled
    conditioning used throughout this directory."""
    ev = np.random.default_rng(stream)
    llr, lab, nvis = [], [], []
    for _ in range(batches):
        x, cl, _ = T.sample(ev, batch)
        seq = x.numpy()
        cls = cl.numpy()                      # cls[b, t] = class of x[b, t]
        for b in range(batch):
            hist = {}
            for t in range(1, T.L):
                a, val = int(seq[b, t - 1]), int(seq[b, t])
                prior = hist.get(a)
                if prior is not None:         # address written before: a "seen" write
                    vs = prior + [val]
                    c = {}
                    for z in vs:
                        c[z] = c.get(z, 0) + 1
                    counts = tuple(sorted(c.values()))
                    n_same = sum(1 for i in range(1, len(vs))
                                 if vs[i] == vs[i - 1])
                    ln = log_p_noisy(counts)
                    lr = log_p_revised(len(vs), n_same)
                    # class of the WRITE ADDRESS = class(x_{t-1})
                    k = int(cls[b, t - 1])
                    if k in (1, 2):
                        llr.append(lr - ln)
                        lab.append(1 if k == 2 else 0)
                        nvis.append(len(prior))
                hist.setdefault(a, []).append(val)
    return np.array(llr), np.array(lab), np.array(nvis)


if __name__ == "__main__":
    print("exact Bayes ceiling from the GENERATOR only, no model involved")
    print(f"N_SYM {NS}, N_OPT {NO}, P_REV {PR}, L {T.L}\n")
    if not selfcheck():
        sys.exit(1)

    for stream in (777, 888):
        llr, lab, nvis = run(stream)
        fin = np.isfinite(llr)
        au = roc_auc_score(lab, np.where(fin, llr, 1e9))
        print(f"\nstream {stream}:  n = {len(lab)}  "
              f"(NOISY {int((lab==0).sum())}, REVISED {int(lab.sum())})")
        print(f"  EXACT BAYES AUROC, pooled over all seen writes: {au:.4f}")
        print(f"  writes where P(v|NOISY) = 0 exactly "
              f"(> {NO} distinct values seen): {int((~fin).sum())}"
              f"  ({100*(~fin).mean():.2f}%), all REVISED: "
              f"{bool(lab[~fin].all()) if (~fin).any() else 'n/a'}")
        print(f"\n  by number of PRIOR visits to the address:")
        print(f"    {'prior visits':>14}{'n':>8}{'Bayes AUROC':>14}")
        for lo, hi in ((1, 1), (2, 2), (3, 4), (5, 8), (9, 1000)):
            m = (nvis >= lo) & (nvis <= hi)
            if m.sum() < 30 or len(np.unique(lab[m])) < 2:
                continue
            a = roc_auc_score(lab[m], np.where(fin[m], llr[m], 1e9))
            tag = f"{lo}" if lo == hi else f"{lo}-{hi if hi < 1000 else '+'}"
            print(f"    {tag:>14}{int(m.sum()):>8}{a:>14.4f}")

    print("\nfor comparison, measured in this directory on stream 777:")
    print("  shipped gate's own three features          0.8095")
    print("  all five zero-extra-state features         0.8122")
    print("  + all five EXACT history features          0.8929")
    print("  + the 'near-label oracle' features         0.9206")
