"""A9. Is claim (b)'s linear baseline numerically valid?

04/logs/separability_777.log PART 5 interleaves divide-by-zero, overflow and
invalid-value RuntimeWarnings from the logistic solver into the rows that
produce 'linear 0.676' - one of the two numbers in (b)'s +0.134. See
PREREGISTRATION.md section A9 for the bar.

    python audit_linear.py --stream 505 --tag gated_seed0
"""
import argparse, os, sys, warnings
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auroc as A, features as FE

THREE = ["f_tok", "f_s1", "f_s2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=int, default=505)
    ap.add_argument("--tag", default="gated_seed0")
    ap.add_argument("--batches", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(2)
    m, ck = FE.load(a.tag)
    d = FE.collect(m, a.stream, batches=a.batches)
    seen = d["wtype"] != FE.UNSEEN
    nr = np.isin(d["cls_w"], [1, 2])
    mk = seen & nr
    lab = (d["cls_w"] == 2).astype(int)
    y, s = lab[mk], d["seq"][mk]
    tr, te = A.seq_split(s, 0)

    print(f"A9  model {a.tag} (seed {ck['seed']})  stream {a.stream}  "
          f"rows {mk.sum()}")
    print("\n  census of every zero-state feature column on these rows:")
    print(f"  {'column':12s}{'n':>8}{'n_inf':>7}{'n_nan':>7}"
          f"{'min':>14}{'max':>14}{'max|.|':>14}")
    for f in FE.INST:
        v = d[f][mk]
        fin = np.isfinite(v)
        print(f"  {f:12s}{v.size:>8}{int(np.isinf(v).sum()):>7}"
              f"{int(np.isnan(v).sum()):>7}{v[fin].min():>14.4g}"
              f"{v[fin].max():>14.4g}{np.abs(v[fin]).max():>14.4g}")

    for fs, nm in ((THREE, "shipped gate [tok, s1, s2]"),
                   (FE.INST, "all zero-state")):
        X = np.stack([d[f][mk] for f in fs], 1)
        print(f"\n  --- {nm} ---")
        # 1. exactly 04's path, with warnings surfaced
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Xtr, Xte = A._clean(X[tr], X[te])
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000))
            clf.fit(Xtr, y[tr])
            au0 = roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1])
            msgs = sorted({str(x.message)[:60] for x in w})
        print(f"  04's _clean path                 AUROC {au0:.4f}"
              f"   warnings {len(msgs)}")
        for mm in msgs:
            print(f"      {mm}")
        print(f"      non-finite after _clean: train "
              f"{int((~np.isfinite(Xtr)).sum())}  test "
              f"{int((~np.isfinite(Xte)).sum())}")
        # 2. rank/quantile transform: immune to tails entirely
        qt = make_pipeline(QuantileTransformer(output_distribution="normal",
                                               n_quantiles=1000,
                                               random_state=0),
                           LogisticRegression(max_iter=5000))
        Xtr2 = np.nan_to_num(X[tr], nan=0.0, posinf=0.0, neginf=0.0)
        Xte2 = np.nan_to_num(X[te], nan=0.0, posinf=0.0, neginf=0.0)
        qt.fit(Xtr2, y[tr])
        au1 = roc_auc_score(y[te], qt.predict_proba(Xte2)[:, 1])
        print(f"  rank/quantile + logistic         AUROC {au1:.4f}")
        # 3. _clean then hard assert finite
        Xtr3 = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
        Xte3 = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)
        clf3 = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000))
        clf3.fit(Xtr3, y[tr])
        au2 = roc_auc_score(y[te], clf3.predict_proba(Xte3)[:, 1])
        print(f"  _clean + forced finite           AUROC {au2:.4f}")
        best = max(au0, au1, au2)
        if fs is THREE:
            print(f"\n  04 reported linear 0.6758/0.6760 for this set.")
            print(f"  best clean variant here {best:.4f}, "
                  f"delta vs 0.676 {best - 0.676:+.4f}")
            print(f"  bar: >= +0.01 -> (b)'s +0.134 is INFLATED; "
                  f"within 0.01 -> SOUND")
            print(f"  -> {'INFLATED' if best - 0.676 >= 0.01 else 'SOUND'}")


if __name__ == "__main__":
    main()
