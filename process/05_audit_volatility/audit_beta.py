"""A1. Does per-address history predict the OPTIMAL BETA better than it
predicts the CLASS LABEL?

Both metrics, on IDENTICAL rows and the IDENTICAL by-sequence split, so the
comparison is between the two metrics and nothing else. See beta_optimal.py for
the closed form and PREREGISTRATION.md section A1 for the bars.

    gain_nonlin = (3 feats, GBT) - (3 feats, linear)
    gain_hist   = (5 zero-state + history, GBT) - (5 zero-state, GBT)
    R           = gain_hist / gain_nonlin

04's class-AUROC answer is R_class = 0.081/0.134 = 0.60.
Bars: R_beta >= 1.00 -> (a)+(b) collapse.  <= 0.75 -> they survive.

    python audit_beta.py --stream 505 --tag gated_seed0
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


import _paths                                                  # noqa: F401
import auroc as A                                              # noqa: E402
import beta_optimal as BO                                      # noqa: E402
import features as FE                                          # noqa: E402
import task as T                                               # noqa: E402
import bayes_ceiling as BC                                     # noqa: E402

THREE = ["f_tok", "f_s1", "f_s2"]
GATEKEY = "the trained gate's own beta"
INST, HIST = FE.INST, FE.HIST


def hdr(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}", flush=True)


def leverage_table(d, mk):
    """How much of the beta leverage sits on rows class-AUROC weights equally."""
    Aw = d["A"][mk]
    ne = np.sqrt(Aw)
    order = np.argsort(-Aw)
    cum = np.cumsum(Aw[order]) / Aw.sum()
    print(f"  rows {mk.sum()}   ||e_t|| median {np.median(ne):.4f}"
          f"   mean {ne.mean():.4f}")
    print(f"\n  {'||e_t|| <':>12}{'share of rows':>16}{'share of total'
          ' leverage A':>28}")
    for thr in (1e-4, 1e-2, 0.1, 0.5, 1.0):
        s = ne < thr
        print(f"  {thr:>12.0e}{s.mean():>16.4f}"
              f"{Aw[s].sum() / Aw.sum():>28.6f}")
    for frac in (0.1, 0.25, 0.5):
        kk = int(frac * len(Aw))
        print(f"  top {frac:.0%} of rows by leverage carry "
              f"{cum[kk - 1]:.4f} of it")


def fit_eval(d, mk, seqs, split_seed, policies):
    """Fit every policy on train sequences, score on test. Returns per-policy
    (rer, cer, beta_test) plus the row index of the test half."""
    tr, te = A.seq_split(seqs[mk], split_seed)
    Aa, Bb, Cc = d["A"][mk], d["B"][mk], d["C"][mk]
    cls = d["cls_w"][mk]
    b_const = BO.const_beta(Aa[tr], Bb[tr])
    bstar_te = BO.beta_star(Aa[te], Bb[te])
    out = {}

    def rec(name, beta_te):
        r, c, _ = BO.score(Aa[te], Bb[te], Cc[te], beta_te, b_const, bstar_te)
        out[name] = (r, c, beta_te)

    rec("best constant beta", np.full(te.sum(), b_const))
    for name, feats, kind in policies:
        X = np.stack([d[f][mk] for f in feats], 1)
        p = BO.Policy(kind, feats, seed=split_seed).fit(X[tr], Aa[tr], Bb[tr])
        rec(name, p.predict(X[te]))
    # the true class label, as the best constant beta inside each class
    pc = BO.per_class_beta(Aa[tr], Bb[tr], cls[tr])
    gl = float(np.clip(Bb[tr].sum() / Aa[tr].sum(), 0, 1))
    rec("TRUE CLASS LABEL (oracle)",
        np.array([pc.get(int(c), gl) for c in cls[te]]))
    rec("per-write beta* (cheating)", bstar_te)
    rec(GATEKEY, d["beta"][mk][te])
    return out, tr, te, b_const


def boot_ratio(d, mk, seqs, out, te, b_const, n=400, seed=0):
    """Sequence-level bootstrap of the two gains and their ratio."""
    Aa, Bb, Cc = d["A"][mk], d["B"][mk], d["C"][mk]
    At, Bt, Ct = Aa[te], Bb[te], Cc[te]
    st = seqs[mk][te]
    uniq, inv = np.unique(st, return_inverse=True)
    rows = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    keys = ["3 feats, linear", "3 feats, GBT", "5 zero-state, GBT",
            "5 zero-state + history, GBT"]
    gn, gh, rr = [], [], []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows[p] for p in pick])
        ec = BO.err(At[idx], Bt[idx], Ct[idx], b_const).mean()
        e = {k: BO.err(At[idx], Bt[idx], Ct[idx], out[k][2][idx]).mean()
             for k in keys}
        a = (e["3 feats, linear"] - e["3 feats, GBT"]) / ec
        b = (e["5 zero-state, GBT"] - e["5 zero-state + history, GBT"]) / ec
        gn.append(a); gh.append(b)
        if abs(a) > 1e-12:
            rr.append(b / a)
    return (np.array(gn), np.array(gh), np.array(rr))


def class_auroc_same_rows(d, mk, seqs, split_seed):
    """04's metric, on exactly the rows and split used above."""
    lab = (d["cls_w"][mk] == 2).astype(int)
    s = seqs[mk]
    res = {}
    for name, feats, kind in (("3 feats, linear", THREE, "logit"),
                              ("3 feats, GBT", THREE, "gbm"),
                              ("5 zero-state, GBT", INST, "gbm"),
                              ("5 zero-state + history, GBT", INST + HIST,
                               "gbm")):
        X = np.stack([d[f][mk] for f in feats], 1)
        sc, yte, _ = A.fit_scores(X, lab, s, kind=kind, seed=split_seed)
        res[name] = float(roc_auc_score(yte, sc))
    return res


BUCKETS = ((1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 8, "5-8"),
           (9, 10 ** 9, "9+"))


def visit_crosstab(d, mk, seqs, out, te, b_const, llr, lab):
    """Where beta's LEVERAGE sits, against where history can identify the class.

    04's session found Bayes identifiability rises steeply with prior visits
    (0.69 at one prior write to 0.9997 at nine or more). That only matters for
    beta if the writes where beta has LEVERAGE live at the same visit counts.
    A_t = ||e_t||^2 is that leverage, so the two can be crossed directly.
    """
    Aa, Bb, Cc = d["A"][mk], d["B"][mk], d["C"][mk]
    nv = d["h_nvis"][mk]
    wt = d["wtype"][mk]
    At, Bt, Ct, nvt, wtt = Aa[te], Bb[te], Cc[te], nv[te], wt[te]
    ltr = llr[mk][te]
    ytr = lab[mk][te]
    tot_lev = Aa.sum()
    base_all = BO.err(At, Bt, Ct, b_const).mean()

    print(f"\n  {'prior':>7}{'rows':>8}{'share of':>10}{'mean':>9}"
          f"{'Bayes':>8}{'hist gain':>11}{'hist gain':>11}")
    print(f"  {'visits':>7}{'':>8}{'leverage':>10}{'||e||':>9}"
          f"{'AUROC':>8}{'beta RER':>11}{'x leverage':>11}")
    rows = []
    for lo, hi, nm in BUCKETS:
        sel_all = (nv >= lo) & (nv <= hi)
        sel = (nvt >= lo) & (nvt <= hi)
        if sel.sum() < 30:
            continue
        lev = Aa[sel_all].sum() / tot_lev
        au = (roc_auc_score(ytr[sel], ltr[sel])
              if len(np.unique(ytr[sel])) > 1 else float("nan"))
        # the same globally fitted policies, scored inside this bucket only
        e5 = BO.err(At[sel], Bt[sel], Ct[sel],
                    out["5 zero-state, GBT"][2][sel]).mean()
        eh = BO.err(At[sel], Bt[sel], Ct[sel],
                    out["5 zero-state + history, GBT"][2][sel]).mean()
        gh = (e5 - eh) / base_all
        print(f"  {nm:>7}{int(sel.sum()):>8}{lev:>10.4f}"
              f"{np.sqrt(At[sel]).mean():>9.4f}{au:>8.4f}{gh:>+11.5f}"
              f"{gh * lev:>+11.5f}")
        rows.append((nm, lev, au, gh))
    print("\n  'hist gain beta RER' is history's beta-error reduction inside\n"
          "  that bucket, as a fraction of the SAME global best-constant\n"
          "  baseline, so the column is additive across buckets.")

    print(f"\n  and the same leverage split by write type:")
    for nm, code in (("repeat", FE.REPEAT), ("change", FE.CHANGE)):
        s_ = wt == code
        print(f"    {nm:8s} rows {int(s_.sum()):>7}  "
              f"share of leverage {Aa[s_].sum() / tot_lev:>7.4f}  "
              f"mean ||e|| {np.sqrt(Aa[s_]).mean():>8.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=int, default=505)
    ap.add_argument("--tag", default="gated_seed0")
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--splits", type=int, default=10)
    a = ap.parse_args()
    torch.set_num_threads(2)

    m, ck = FE.load(a.tag)
    print(f"A1  model {a.tag} (seed {ck['seed']})   stream {a.stream}   "
          f"{a.batches} batches of 32")
    BO._self_test(m, seed=a.stream)
    d = BO.collect(m, a.stream, batches=a.batches)
    print(f"  model loss on this stream {d['_loss']:.4f}")
    seqs = d["seq"]

    # the Bayes LLR, on the same rows. Both collectors draw from
    # default_rng(stream) with the same batch loop, so the flattened row order
    # is identical; asserted rather than assumed.
    bc = BC.collect(a.stream, batches=a.batches)
    assert np.array_equal(bc["cls_w"].astype(int), d["cls_w"]), "row mismatch"
    assert np.array_equal(bc["wtype"].astype(int), d["wtype"].astype(int)), \
        "wtype mismatch"
    llr = bc["llr"]
    lab_all = (d["cls_w"] == 2).astype(int)

    seen = d["wtype"] != FE.UNSEEN
    nr = np.isin(d["cls_w"], [1, 2])
    mk = seen & nr & d["valid"]
    lab = lab_all

    hdr("A1.4  how good is the optimal-beta proxy itself?")
    print("  The target v_next assumes the content written at address\n"
          "  k = normalize(W_k emb(s)) is what gets read back by\n"
          "  q = normalize(W_q emb(s)). Those are different learned\n"
          "  projections of the same token, so the proxy is only as tight as\n"
          "  their alignment. Per symbol, over the whole vocabulary:")
    with torch.no_grad():
        ee = m.embed(torch.arange(T.N_SYM))
        kk = torch.nn.functional.normalize(m.to_k(ee), dim=-1)
        qq = torch.nn.functional.normalize(m.to_q(ee), dim=-1)
        same = (kk * qq).sum(-1)
        cross = kk @ qq.T
        offdiag = cross[~torch.eye(T.N_SYM, dtype=bool)]
    print(f"    cos(k_s, q_s)  same symbol : mean {same.mean():+.4f}  "
          f"min {same.min():+.4f}  max {same.max():+.4f}")
    print(f"    cos(k_s, q_r)  different   : mean {offdiag.mean():+.4f}  "
          f"sd {offdiag.std():.4f}")
    print(f"    margin (same - mean other) {same.mean() - offdiag.mean():+.4f}")
    print("  A1b measures actual cross-entropy and so needs none of this; it\n"
          "  is the validity check on A1's ranking, not on A1b's.")

    hdr("A1.0  the leverage that pooled class-AUROC throws away")
    print("  A_t = ||e_t||^2 is how much beta can change the stored content.\n"
          "  Pooled class-AUROC weights every row equally regardless of A_t.")
    print("\n  04's primary row set (NOISY|REVISED, address written before):")
    leverage_table(d, mk)

    policies = [("3 feats, linear", THREE, "linear"),
                ("3 feats, GBT", THREE, "gbm"),
                ("5 zero-state, GBT", INST, "gbm"),
                ("5 zero-state + history, GBT", INST + HIST, "gbm")]

    for label, rows in (("04's primary row set (NOISY|REVISED, seen)", mk),
                        ("declared secondary: ALL classes, seen",
                         seen & d["valid"]),
                        ("declared tertiary: ALL writes with a target",
                         d["valid"])):
        hdr(f"A1.1  beta quality on {label}")
        out, tr, te, b_const = fit_eval(d, rows, seqs, 0, policies)
        print(f"  n_train {tr.sum()}  n_test {te.sum()}   "
              f"best constant beta {b_const:.4f}")
        print(f"\n  {'policy':34s}{'RER':>10}{'CER':>10}{'mean beta':>12}")
        for k, (r, c, bt) in out.items():
            print(f"  {k:34s}{r:>+10.4f}{c:>+10.4f}{bt.mean():>12.4f}")

        gn = out["3 feats, GBT"][0] - out["3 feats, linear"][0]
        gh = (out["5 zero-state + history, GBT"][0]
              - out["5 zero-state, GBT"][0])
        print(f"\n  gain_nonlin (linear -> GBT on the 3 features) {gn:>+9.4f}")
        print(f"  gain_hist   (zero-state -> + history)          {gh:>+9.4f}")
        if abs(gn) > 1e-12:
            print(f"  R_beta = gain_hist / gain_nonlin               "
                  f"{gh / gn:>+9.4f}")
        bgn, bgh, brr = boot_ratio(d, rows, seqs, out, te, b_const)
        print(f"  bootstrap sd: gain_nonlin {bgn.std(ddof=1):.4f}   "
              f"gain_hist {bgh.std(ddof=1):.4f}   "
              f"R_beta {brr.std(ddof=1):.4f}")
        print(f"  R_beta 95% bootstrap interval "
              f"[{np.percentile(brr, 2.5):+.3f}, "
              f"{np.percentile(brr, 97.5):+.3f}]")

        # (c) says the remaining oracle gap is credit assignment. 04 INFERS
        # that residually (not information, not capacity). Here it is a
        # measurement: the gap between the best policy fittable on the gate's
        # OWN three features and what the trained gate actually does with them.
        best3 = out["3 feats, GBT"][0]
        actual = out[GATEKEY][0]
        print(f"\n  CREDIT-ASSIGNMENT GAP, measured not inferred:")
        print(f"    best policy on the gate's own 3 features {best3:>+10.4f} RER")
        print(f"    the trained gate's actual beta           {actual:>+10.4f} RER")
        print(f"    gap the gate leaves on its own features  "
              f"{best3 - actual:>+10.4f}")

        if rows is mk:
            hdr("A1.5  does history identify the class WHERE beta has leverage?")
            visit_crosstab(d, rows, seqs, out, te, b_const, llr, lab)

        co = out["TRUE CLASS LABEL (oracle)"][0]
        den = co - out["5 zero-state, GBT"][0]
        print(f"\n  share_hist denominator RER(class) - RER(5 feats) "
              f"{den:>+9.4f}")
        if den <= 0:
            print("  DEGENERATE BRANCH (pre-declared): the true class label is\n"
                  "  worth nothing beyond the instantaneous features on the\n"
                  "  beta target. This INDEPENDENTLY vindicates (a). Ratio not\n"
                  "  computed.")
        else:
            print(f"  share_hist = gain_hist / that             "
                  f"{gh / den:>+9.4f}")

        if rows is mk:
            hdr("A1.2  04's OWN metric on exactly these rows and this split")
            ca = class_auroc_same_rows(d, rows, seqs, 0)
            print(f"\n  {'feature set':34s}{'class AUROC':>14}"
                  f"{'beta RER':>12}")
            for k in ca:
                print(f"  {k:34s}{ca[k]:>14.4f}{out[k][0]:>+12.4f}")
            cgn = ca["3 feats, GBT"] - ca["3 feats, linear"]
            cgh = (ca["5 zero-state + history, GBT"]
                   - ca["5 zero-state, GBT"])
            print(f"\n  {'':34s}{'class AUROC':>14}{'beta RER':>12}")
            print(f"  {'gain_nonlin':34s}{cgn:>+14.4f}{gn:>+12.4f}")
            print(f"  {'gain_hist':34s}{cgh:>+14.4f}{gh:>+12.4f}")
            print(f"  {'R = hist / nonlin':34s}"
                  f"{cgh / cgn if abs(cgn) > 1e-9 else float('nan'):>+14.4f}"
                  f"{gh / gn if abs(gn) > 1e-9 else float('nan'):>+12.4f}")
            print("\n  (04 reports R_class = 0.081/0.134 = 0.60 on streams "
                  "777/888.)")

            hdr("A1.3  R_beta across independent by-sequence splits")
            rs = []
            for sd in range(a.splits):
                o2, _, _, _ = fit_eval(d, rows, seqs, sd, policies)
                g1 = o2["3 feats, GBT"][0] - o2["3 feats, linear"][0]
                g2 = (o2["5 zero-state + history, GBT"][0]
                      - o2["5 zero-state, GBT"][0])
                rs.append((g1, g2, g2 / g1 if abs(g1) > 1e-12 else np.nan))
            rs = np.array(rs)
            print(f"\n  {'split':>7}{'gain_nonlin':>14}{'gain_hist':>12}"
                  f"{'R_beta':>10}")
            for i, r in enumerate(rs):
                print(f"  {i:>7}{r[0]:>+14.4f}{r[1]:>+12.4f}{r[2]:>+10.4f}")
            print(f"\n  mean  {rs[:, 0].mean():>+14.4f}{rs[:, 1].mean():>+12.4f}"
                  f"{np.nanmean(rs[:, 2]):>+10.4f}")
            print(f"  sd    {rs[:, 0].std(ddof=1):>+14.4f}"
                  f"{rs[:, 1].std(ddof=1):>+12.4f}"
                  f"{np.nanstd(rs[:, 2], ddof=1):>+10.4f}")
            print("\n  BARS: R_beta >= 1.00 -> (a)+(b) COLLAPSE;  <= 0.75 -> "
                  "they SURVIVE;\n        between -> AMBIGUOUS.")


if __name__ == "__main__":
    main()
