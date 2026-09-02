"""Is u redundant with entropy, or does it add anything?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from induction_run import go

r = go(0)
u, y, c, e, x, p, s = (r[k] for k in "uycexps")
z = lambda a: (a - a.mean()) / (a.std() + 1e-9)

print(f"corr(u, entropy) = {np.corrcoef(u, e)[0,1]:.3f}\n")
print("AUROC -> ERROR, on the most-CONFIDENT (lowest-entropy) steps only.")
print("entropy is definitionally near-blind there; does u see anything?")
for frac in (0.5, 0.33, 0.2):
    lo = e < np.percentile(e, frac * 100)
    if 0 < y[lo].mean() < 1:
        print(f"  bottom {int(frac*100)}% entropy (n={lo.sum():6d}, err rate {y[lo].mean():.3f}): "
              f"u {roc_auc_score(y[lo], u[lo]):.3f}   entropy {roc_auc_score(y[lo], e[lo]):.3f}")

n = len(y); idx = np.random.default_rng(0).permutation(n); tr, te = idx[:n//2], idx[n//2:]
print("\nheld-out AUROC of a logistic model predicting ERROR:")
for nm, cols in (("entropy only", [z(e)]), ("u only", [z(u)]),
                 ("entropy + u", [z(e), z(u)]),
                 ("entropy + maxsoft", [z(e), z(x)]),
                 ("entropy + maxsoft + u", [z(e), z(x), z(u)])):
    X = np.stack(cols, 1)
    lr = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
    print(f"  {nm:24s} {roc_auc_score(y[te], lr.predict_proba(X[te])[:,1]):.4f}")
