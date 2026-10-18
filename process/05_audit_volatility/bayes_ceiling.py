"""A10. Is 04's ~0.92 "near-label oracle ceiling" a real ceiling?

04's AUDIT.md lists this as unchecked: "whether the ORACLE feature set is a
sufficient statistic for the per-address history ... the true Bayes ceiling
given the full value multiset was not computed, so 0.92 is a measured ceiling
for these features, not a proof."

It matters because 04 used ~0.92 to argue part of the per-class oracle gap is
unreachable in principle: "At L=256 the class label is only partly recoverable
from per-address history at all."

Nothing here needs a trained model. At address `s` the values written there are
exactly the successors `s` emits, and the generative process at `s` advances
only when `s` is visited, so that value sequence is a SUFFICIENT STATISTIC for
`class(s)` and the Bayes-optimal score is computable in closed form.

where, for one address, with values v_1..v_n written there up to and including
the current write (the same information set o_memb and o_ndist use):
    N = N_SYM = 40     vocabulary
    K = N_OPT = 4      slots in a NOISY address's option multiset
    p = P_REV = 0.35   per-visit revision probability for REVISED
    k                  number of DISTINCT values among v_1..v_n
    c_j                count of the j-th distinct value, sum c_j = n
    m_j                multiplicity of the j-th distinct value in the
                       (unobserved) option multiset, r = K - sum m_j

REVISED is a two-state Markov chain on "did the value change":
    P(v_i = v_{i-1}) = (1-p) + p/N          both "no revision" and
                                            "revised back to the same symbol"
    P(v_i = w != v_{i-1}) = p/N
    L_REV = (1/N) * prod_{i=2..n} [(1-p) + p/N] or [p/N]

NOISY emits iid uniform draws from a K-slot multiset of iid Uniform(N) symbols.
Marginalising over that unobserved multiset:
    L_NOISY = sum_{m_j >= 1, sum m_j <= K}
                K!/(prod m_j! r!) * (N-k)^r / N^K * prod_j (m_j/K)^{c_j}
and L_NOISY = 0 when k > K: more than K distinct values cannot come out of a
K-slot option set, so those rows are logically certain REVISED.

Score: LLR = log L_REV - log L_NOISY.

Bars in PREREGISTRATION.md section A10: Bayes AUROC >= 0.95 means ~0.92 is NOT
a real ceiling; <= 0.94 means it is about right.

    python bayes_ceiling.py --stream 505
"""
import argparse
import itertools
import math
from functools import lru_cache

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import _paths                                                  # noqa: F401
import auroc as A                                              # noqa: E402
import task as T                                               # noqa: E402

N, K, P = T.N_SYM, T.N_OPT, T.P_REV
A_SAME = (1.0 - P) + P / N          # consecutive values agree
A_DIFF = P / N                      # consecutive values differ, to one symbol


def hdr(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}", flush=True)


@lru_cache(maxsize=None)
def log_l_noisy(counts):
    """log P(value sequence | NOISY), as a function of the sorted count vector.

    Depends only on the multiset of per-value counts: the sum below runs over
    ordered multiplicity assignments, so permuting (c_j, m_j) jointly leaves it
    unchanged.
    """
    k = len(counts)
    if k > K:
        return -math.inf
    tot = 0.0
    for m in itertools.product(range(1, K + 1), repeat=k):
        sm = sum(m)
        if sm > K:
            continue
        r = K - sm
        coef = math.factorial(K)
        for mj in m:
            coef //= math.factorial(mj)
        coef //= math.factorial(r)
        term = coef * ((N - k) ** r) / (N ** K)
        for mj, cj in zip(m, counts):
            term *= (mj / K) ** cj
        tot += term
    return math.log(tot) if tot > 0 else -math.inf


def log_l_revised(n, n_same):
    """log P(value sequence | REVISED). Only the number of consecutive
    agreements matters."""
    return (-math.log(N) + n_same * math.log(A_SAME)
            + (n - 1 - n_same) * math.log(A_DIFF))


# --------------------------------------------------------------------------
# self-tests: a wrong likelihood would silently produce a confident AUROC
# --------------------------------------------------------------------------
def _seq_stats(seq):
    cnt = {}
    for v in seq:
        cnt[v] = cnt.get(v, 0) + 1
    same = sum(1 for i in range(1, len(seq)) if seq[i] == seq[i - 1])
    return tuple(sorted(cnt.values())), same


def self_test(mc=400000, seed=0):
    print("  normalisation: sum of L over every value sequence must be 1")
    ok = True
    for n in (1, 2):
        sr = sn_ = 0.0
        for seq in itertools.product(range(N), repeat=n):
            counts, same = _seq_stats(seq)
            sr += math.exp(log_l_revised(n, same))
            lw = log_l_noisy(counts)
            sn_ += math.exp(lw) if lw > -math.inf else 0.0
        print(f"    n={n}  REVISED {sr:.12f}   NOISY {sn_:.12f}")
        ok &= abs(sr - 1) < 1e-9 and abs(sn_ - 1) < 1e-9
    print(f"    {'PASS' if ok else 'FAIL'}")
    assert ok, "likelihood normalisation FAILED"

    print("\n  Monte Carlo: simulate one address directly, compare to L")
    rng = np.random.default_rng(seed)
    for n in (2, 3):
        emp = {1: {}, 2: {}}
        for cls in (1, 2):
            for _ in range(mc):
                if cls == 1:
                    opts = rng.integers(N, size=K)
                    s = tuple(int(opts[j]) for j in rng.integers(K, size=n))
                else:
                    cur = int(rng.integers(N))
                    out = []
                    for _ in range(n):
                        out.append(cur)
                        if rng.random() < P:
                            cur = int(rng.integers(N))
                    s = tuple(out)
                key = _seq_stats(s)
                emp[cls][key] = emp[cls].get(key, 0) + 1
        print(f"    n={n}: {'pattern':>18}{'emp NOISY':>12}{'L NOISY':>12}"
              f"{'emp REV':>12}{'L REV':>12}")
        pats = sorted(set(emp[1]) | set(emp[2]),
                      key=lambda z: -emp[1].get(z, 0))[:4]
        for key in pats:
            counts, same = key
            # how many distinct ordered sequences share this (counts, same)?
            nseq = sum(1 for s in itertools.product(range(N), repeat=n)
                       if _seq_stats(s) == key)
            lw = log_l_noisy(counts)
            pn = (math.exp(lw) if lw > -math.inf else 0.0) * nseq
            pr = math.exp(log_l_revised(n, same)) * nseq
            print(f"    {str(key):>23}{emp[1].get(key,0)/mc:>12.5f}"
                  f"{pn:>12.5f}{emp[2].get(key,0)/mc:>12.5f}{pr:>12.5f}")
            assert abs(emp[1].get(key, 0) / mc - pn) < 0.01, "MC NOISY mismatch"
            assert abs(emp[2].get(key, 0) / mc - pr) < 0.01, "MC REV mismatch"
        print("    PASS")


# --------------------------------------------------------------------------
def collect(stream, batches=6, batch=32):
    """Per-write Bayes LLR and the model-independent oracle features.

    Every column here is a function of the token stream only, so no trained
    model is involved and 04's row set is reproduced exactly: a write at step t
    addresses x_{t-1} with value x_t, and "written before" means that address
    already had a write.

    All per-address state is incremental. A naive version that rebuilds the
    value history each step is cubic on this task, because 21% of sequences
    send more than 120 writes to one absorbing STABLE address.
    """
    ev = np.random.default_rng(stream)
    names = ("llr", "o_memb", "o_ndist", "h_nvis", "h_nret", "h_nsame",
             "h_maxmult", "cls_w", "wtype", "seq", "kgtK")
    acc = {n: [] for n in names}
    off = 0
    for _ in range(batches):
        x, cl, _ = T.sample(ev, batch)
        xx, cc = x.numpy(), cl.numpy()
        L = T.L
        out = {n: np.zeros((batch, L)) for n in names}
        for b in range(batch):
            st = {}      # address -> incremental state
            for t in range(1, L):
                a, v = int(xx[b, t - 1]), int(xx[b, t])
                e = st.get(a)
                if e is None:
                    # first write to this address: UNSEEN, no history columns
                    counts_v = (1,)
                    same_v = 0
                    st[a] = {"last": v, "n": 1, "cnt": {v: 1},
                             "ret": 0, "same": 0}
                else:
                    cnt, n_prev = e["cnt"], e["n"]
                    out["wtype"][b, t] = 1 if e["last"] == v else 2
                    out["o_memb"][b, t] = float(v in cnt)
                    out["o_ndist"][b, t] = float(len(cnt))
                    out["h_nvis"][b, t] = float(n_prev)
                    den = max(n_prev - 1, 1)
                    out["h_nret"][b, t] = e["ret"] / den
                    out["h_nsame"][b, t] = e["same"] / den
                    # the sequence INCLUDING this write
                    was = v in cnt
                    cnt[v] = cnt.get(v, 0) + 1
                    same_v = e["same"] + (1 if e["last"] == v else 0)
                    counts_v = tuple(sorted(cnt.values()))
                    e["ret"] += 1 if was else 0
                    e["same"] = same_v
                    e["last"] = v
                    e["n"] = n_prev + 1
                n_v = e["n"] if e is not None else 1
                lw = log_l_noisy(counts_v)
                lr = log_l_revised(n_v, same_v)
                out["llr"][b, t] = 50.0 if lw == -math.inf else lr - lw
                out["kgtK"][b, t] = float(lw == -math.inf)
                out["h_maxmult"][b, t] = float(max(counts_v))
                out["cls_w"][b, t] = float(cc[b, t - 1])
            out["seq"][b, :] = off + b
        for n in names:
            acc[n].append(out[n])
        off += batch
    return {n: np.concatenate(v).reshape(-1) for n, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=int, default=505)
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--mc", type=int, default=400000)
    a = ap.parse_args()
    torch.set_num_threads(2)

    hdr("A10.0  self-tests on the likelihoods")
    self_test(mc=a.mc)

    hdr(f"A10.1  the exact Bayes ceiling, stream {a.stream}")
    d = collect(a.stream, batches=a.batches)
    seen = d["wtype"] != 0
    nr = np.isin(d["cls_w"].astype(int), [1, 2])
    mk = seen & nr
    lab = (d["cls_w"].astype(int) == 2).astype(int)
    print(f"  rows in 04's primary row set: {mk.sum()}   "
          f"base rate REVISED {lab[mk].mean():.4f}")
    print("  (no trained model is involved: every column is a function of the\n"
          "  token stream, which is how 04's row set and labels are defined.)")

    au, n = A.single(d["llr"][mk], lab[mk])
    print(f"\n  {'score':44s}{'AUROC':>9}{'n':>9}")
    print(f"  {'EXACT BAYES LLR (sufficient statistic)':44s}{au:>9.4f}{n:>9}")
    for f, nm in (("o_memb", "o_memb alone"), ("o_ndist", "o_ndist alone"),
                  ("h_nret", "return rate (richer summary)"),
                  ("h_nsame", "consecutive-repeat rate"),
                  ("h_nvis", "prior visit count")):
        a1, n1 = A.single(d[f][mk], lab[mk])
        print(f"  {nm:44s}{a1:>9.4f}{n1:>9}")

    print("\n  classifier over feature sets, by-sequence split, GBT:")
    for nm, fs in (("04's oracle set [o_memb, o_ndist]",
                    ["o_memb", "o_ndist"]),
                   ("04's oracle set + visit count",
                    ["o_memb", "o_ndist", "h_nvis"]),
                   ("richer history summaries",
                    ["o_memb", "o_ndist", "h_nvis", "h_nret", "h_nsame",
                     "h_maxmult"]),
                   ("the Bayes LLR alone", ["llr"]),
                   ("the Bayes LLR + everything above",
                    ["llr", "o_memb", "o_ndist", "h_nvis", "h_nret",
                     "h_nsame", "h_maxmult"])):
        auj, sd, _, nte = A.joint(d, mk, fs, lab, d["seq"], kind="gbm")
        print(f"    {nm:44s}{auj:>9.4f} +/- {sd:.4f}   n_test {nte}")

    hdr("A10.2  where the Bayes score gets its certainty")
    kg = d["kgtK"][mk] > 0.5
    print(f"  rows with more distinct values than the {K} option slots: "
          f"{kg.mean():.4f}")
    print(f"    those are LOGICALLY certain REVISED. Among them, actual "
          f"REVISED fraction {lab[mk][kg].mean():.4f}")
    print(f"    (it must be 1.0; anything else means the likelihood is wrong)")
    rest = ~kg
    au2, n2 = A.single(d["llr"][mk][rest], lab[mk][rest])
    print(f"\n  excluding them, Bayes AUROC on the remaining {n2} rows "
          f"{au2:.4f}")
    print("  so the ceiling is not carried only by the logically-certain rows.")

    hdr("A10.3  the pre-registered bar")
    print(f"  04 reports its near-label ceiling as 0.9206 (everything) /\n"
          f"  0.9236 (oracle + history) on stream 777, and its ORACLE FEATURES\n"
          f"  ALONE as 0.7847.")
    print(f"\n  exact Bayes ceiling here: {au:.4f}")
    print(f"  bars: >= 0.95 -> ~0.92 is NOT a real ceiling")
    print(f"        <= 0.94 -> ~0.92 is about right")
    print(f"  -> {'~0.92 IS NOT A REAL CEILING' if au >= 0.95 else '~0.92 is about right' if au <= 0.94 else 'AMBIGUOUS'}")


if __name__ == "__main__":
    main()
