"""The EXACT Bayes log-likelihood ratio for NOISY vs REVISED, per write, from
an address's value history.

Derived from `gated_delta_ssm/task.py`'s generative process alone: no model,
no trained weights, no features. It is a SUFFICIENT STATISTIC for the value
history at an address, so it upper-bounds every per-address history mechanism,
all of which see some lossy function of that sequence.

The math is 04's `audit_bayes_ceiling.py` (branch
claude/optimistic-easley-09c6ab), reimplemented here INCREMENTALLY so it can
run inside a training loop: that script rebuilds the count multiset from
scratch at every write, which is O(visits) and the task's absorbing states push
visits past 250 on some addresses. `selfcheck()` reproduces its normalisation
test and `test_parity.py` checks this file against a from-scratch
recomputation.

REVISED. `nxt = succ[s]` is read BEFORE the possible revision, so the value at
visit i+1 equals the value at visit i unless a revision fired at visit i:

    P(v_1)           = 1/N_SYM
    P(v_i = v_{i-1}) = (1 - P_REV) + P_REV/N_SYM
    P(v_i = other)   = P_REV/N_SYM            for each specific other symbol

NOISY. The option set is N_OPT slots, each an independent uniform draw over
N_SYM (so it may contain duplicates), and each visit picks a slot uniformly:

    P(v) = N_SYM^-N_OPT * SUM over (m_1..m_k, r), m_j >= 1, sum m_j + r = N_OPT,
           of  multinomial(N_OPT; m_1..m_k, r) * (N_SYM-k)^r
                * prod_j (m_j / N_OPT)^{c_j}

P(v|NOISY) is exactly 0 once more than N_OPT distinct values have been seen,
which makes such a write PROVABLY REVISED. That is an infinite LLR, so it is
clamped; the clamp rate is reported rather than hidden.

A write to a FRESH address has one observation, and both likelihoods are
exactly 1/N_SYM, so its LLR is exactly 0.

The GATE is fed `sigmoid(LLR)`, the posterior P(REVISED | the address's value
history) under a flat prior over the two classes. Sigmoid is monotone, so this
is the same statistic and scores the IDENTICAL AUROC (0.9449 both ways,
verified), but it lands in [0, 1] alongside `o_memb` instead of spanning
[-20, 20], which would otherwise dominate the gate's first layer: every other
column sits between -1 and 5. A fresh address gets exactly 0.5, the flat
no-evidence posterior, which plays the role 0 plays for the other columns.

VERIFIED against 04's audit_bayes_ceiling.py, which recomputes from scratch:
unclamped AUROC 0.9449 on stream 777 (theirs 0.9449), the P(v|NOISY) = 0 rate
9.27% (theirs "9 to 10%"), and AUROC by prior writes to the address
0.6868 / 0.7858 / 0.8896 / 0.9773 / 0.9997 at 1 / 2 / 3-4 / 5-8 / 9+
(theirs 0.687 / 0.786 / 0.890 / 0.977 / 0.9997).

THIS IS NOT A BUILDABLE MECHANISM. It reads N_OPT and P_REV off the task
generator. It exists to answer one question: if even a sufficient statistic
for the value history buys no loss, the question is closed for every
per-address history mechanism at once.
"""
import os
import sys
from functools import lru_cache
from math import factorial, log

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "gated_delta_ssm"))

import task as T                                                  # noqa: E402

NS, NO, PR = T.N_SYM, T.N_OPT, T.P_REV
LLR_CLAMP = 20.0
P_SAME = (1 - PR) + PR / NS
P_DIFF = PR / NS


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
    """n observations at the address, n_same of the n-1 transitions repeated."""
    return (-log(NS) + n_same * log(P_SAME)
            + (n - 1 - n_same) * log(P_DIFF))


def selfcheck():
    """A single observation must have probability exactly 1/N_SYM either way."""
    a, b = np.exp(log_p_noisy((1,))), np.exp(log_p_revised(1, 0))
    ok = abs(a - 1 / NS) < 1e-12 and abs(b - 1 / NS) < 1e-12
    return ok, a, b, 1 / NS


def llr_batch(tokens):
    """tokens: LongTensor [B, L] -> (llr [B, L] clamped, seen mask [B, L],
    stats dict).

    Incremental: each address carries its value->count dict, its observation
    count, its repeated-transition count and its last value, so a write costs
    O(k log k) in the number of DISTINCT values rather than O(visits).
    Short-circuits as soon as k > N_OPT, where P(v|NOISY) = 0 exactly.
    """
    tk = tokens.detach().cpu().numpy()
    B, L = tk.shape
    out = np.zeros((B, L), np.float32)
    seen = np.zeros((B, L), bool)
    n_zero_noisy, n_sat, n_seen = 0, 0, 0
    for b in range(B):
        counts, n_obs, n_same, last = {}, {}, {}, {}
        row = tk[b]
        for t in range(1, L):
            a, val = int(row[t - 1]), int(row[t])
            c = counts.get(a)
            if c is None:                 # fresh address: one observation, LLR 0
                counts[a] = {val: 1}
                n_obs[a], n_same[a], last[a] = 1, 0, val
                continue
            c[val] = c.get(val, 0) + 1
            n_obs[a] += 1
            if last[a] == val:
                n_same[a] += 1
            last[a] = val
            seen[b, t] = True
            n_seen += 1
            k = len(c)
            lr = log_p_revised(n_obs[a], n_same[a])
            if k > NO:                    # P(v|NOISY) = 0: provably REVISED
                out[b, t] = LLR_CLAMP
                n_zero_noisy += 1
            else:
                v = lr - log_p_noisy(tuple(sorted(c.values())))
                if v > LLR_CLAMP or v < -LLR_CLAMP:
                    n_sat += 1
                out[b, t] = min(max(v, -LLR_CLAMP), LLR_CLAMP)
    stats = {"seen": n_seen, "zero_noisy": n_zero_noisy, "saturated": n_sat}
    return torch.from_numpy(out), torch.from_numpy(seen), stats


def posterior_batch(tokens):
    """tokens: LongTensor [B, L] -> FloatTensor [B, L] in [0, 1].

    P(REVISED | the address's value history) under a flat prior. This is what
    the gate reads. Fresh addresses and t=0 get exactly 0.5: no evidence.
    Clamping the LLR at +/-20 first is immaterial, since sigmoid(20) rounds to
    1.0 in float32 anyway, and it does not change AUROC.
    """
    llr, _, stats = llr_batch(tokens)
    return torch.sigmoid(llr), stats


if __name__ == "__main__":
    ok, a, b, ref = selfcheck()
    print(f"single observation: P(v|NOISY) = {a:.10f}, "
          f"P(v|REVISED) = {b:.10f}, 1/N_SYM = {ref:.10f}")
    print(f"normalisation self-check: {'PASS' if ok else 'FAIL'}")
    from sklearn.metrics import roc_auc_score
    ev = np.random.default_rng(777)
    V, M, C = [], [], []
    tot = {"seen": 0, "zero_noisy": 0, "saturated": 0}
    for _ in range(6):
        x, cl, _ = T.sample(ev, 32)
        llr, seen, st = llr_batch(x[:, :-1])
        for kk in tot:
            tot[kk] += st[kk]
        V.append(llr.numpy().ravel()); M.append(seen.numpy().ravel())
        C.append(torch.nn.functional.pad(cl, (1, 0))[:, :T.L].numpy().ravel())
    v, seen, cw = np.concatenate(V), np.concatenate(M), np.concatenate(C)
    mk = seen & ((cw == 1) | (cw == 2))
    lab = (cw[mk] == 2).astype(int)
    print(f"\nstream 777, 6 batches. Writes to a previously written address: "
          f"{tot['seen']}, of which NOISY or REVISED: {mk.sum()}")
    print(f"  P(v|NOISY) = 0 exactly (k > N_OPT, provably REVISED): "
          f"{100 * tot['zero_noisy'] / tot['seen']:.2f}% of seen writes")
    print(f"  finite but |LLR| > {LLR_CLAMP:.0f}: "
          f"{100 * tot['saturated'] / tot['seen']:.2f}% "
          f"(mostly STABLE absorbing addresses, at the NOISY end)")
    print(f"\nBayes AUROC NOISY vs REVISED: {roc_auc_score(lab, v[mk]):.4f}"
          f"   [04's audit_bayes_ceiling.py: 0.9449]")
    print(f"  as the posterior the gate reads: "
          f"{roc_auc_score(lab, 1 / (1 + np.exp(-v[mk]))):.4f}"
          f"   (monotone, so identical by construction)")
