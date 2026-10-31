"""AUROC helpers with a BY-SEQUENCE split and paired sequence-level bootstrap.

A per-token split leaks: writes inside one sequence share that sequence's option
sets and revision history, so a token-level split lets the model memorise the
sequence rather than learn the statistic. Every split here is by sequence.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _clean(Xtr, Xte):
    """Logistic regression cannot take NaN or inf. Impute NaN to the TRAIN
    median and winsorise to the TRAIN 0.1/99.9 percentiles, both fitted on the
    training half only so nothing leaks from the test half. A tree needs none
    of this, which is why `kind="gbm"` skips it.
    """
    med = np.nanmedian(Xtr, 0)
    lo = np.nanpercentile(np.where(np.isfinite(Xtr), Xtr, np.nan), 0.1, axis=0)
    hi = np.nanpercentile(np.where(np.isfinite(Xtr), Xtr, np.nan), 99.9, axis=0)
    out = []
    for X in (Xtr, Xte):
        Z = np.where(np.isfinite(X), X, med)
        out.append(np.clip(Z, lo, hi))
    return out


def single(score, label):
    """AUROC of one raw feature. NaN-safe: NaN rows are dropped."""
    ok = np.isfinite(score)
    if ok.sum() < 20 or len(np.unique(label[ok])) < 2:
        return float("nan"), int(ok.sum())
    return float(roc_auc_score(label[ok], score[ok])), int(ok.sum())


def seq_split(seqs, seed=0):
    uniq = np.unique(seqs)
    perm = np.random.default_rng(seed).permutation(uniq)
    tr = np.isin(seqs, perm[: len(uniq) // 2])
    return tr, ~tr


def fit_scores(X, y, seqs, kind="gbm", seed=0):
    """Train on half the SEQUENCES, return (test scores, test labels, test seqs)."""
    tr, te = seq_split(seqs, seed)
    if kind == "gbm":
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
        Xtr, Xte = X[tr], X[te]
    else:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000))
        Xtr, Xte = _clean(X[tr], X[te])
    clf.fit(Xtr, y[tr])
    return clf.predict_proba(Xte)[:, 1], y[te], seqs[te]


def boot_auroc(scores, y, seqs, n=400, seed=0):
    """Sequence-level bootstrap. `scores` may be a dict name -> score array, all
    on the SAME rows, so differences are paired across resamples."""
    single_in = not isinstance(scores, dict)
    sc = {"x": scores} if single_in else scores
    uniq, inv = np.unique(seqs, return_inverse=True)
    rows = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = {k: [] for k in sc}
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows[p] for p in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        for k, v in sc.items():
            out[k].append(roc_auc_score(y[idx], v[idx]))
    res = {k: np.array(v) for k, v in out.items()}
    return res["x"] if single_in else res


def joint(d, mask, feats, label, seqs, kind="gbm", seed=0, n_boot=400):
    """AUROC of a classifier over `feats`, by-sequence split, with a bootstrap SD.

    Rows with a NaN in ANY listed feature are dropped, so that two feature sets
    compared against each other are compared on a common row set only when the
    caller passes a mask that already guarantees it.
    """
    X = np.stack([d[f][mask] for f in feats], 1)
    y = label[mask]
    s = seqs[mask]
    sc, yte, ste = fit_scores(X, y, s, kind=kind, seed=seed)
    au = roc_auc_score(yte, sc)
    b = boot_auroc(sc, yte, ste, n=n_boot, seed=seed)
    return float(au), float(b.std(ddof=1)), int(mask.sum()), int(len(yte))


def joint_pair(d, mask, feats_a, feats_b, label, seqs, kind="gbm", seed=0,
               n_boot=400):
    """Paired comparison of two feature sets on IDENTICAL rows and split.

    Returns (auroc_a, auroc_b, delta, delta_sd, n_test).
    """
    ya = label[mask]
    s = seqs[mask]
    Xa = np.stack([d[f][mask] for f in feats_a], 1)
    Xb = np.stack([d[f][mask] for f in feats_b], 1)
    tr, te = seq_split(s, seed)
    scores = {}
    for nm, X in (("a", Xa), ("b", Xb)):
        if kind == "gbm":
            clf = HistGradientBoostingClassifier(max_iter=200,
                                                 random_state=seed)
            Xtr, Xte = X[tr], X[te]
        else:
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000))
            Xtr, Xte = _clean(X[tr], X[te])
        clf.fit(Xtr, ya[tr])
        scores[nm] = clf.predict_proba(Xte)[:, 1]
    yte, ste = ya[te], s[te]
    au_a = roc_auc_score(yte, scores["a"])
    au_b = roc_auc_score(yte, scores["b"])
    b = boot_auroc(scores, yte, ste, n=n_boot, seed=seed)
    dl = b["b"] - b["a"]
    return (float(au_a), float(au_b), float(au_b - au_a), float(dl.std(ddof=1)),
            int(len(yte)))
