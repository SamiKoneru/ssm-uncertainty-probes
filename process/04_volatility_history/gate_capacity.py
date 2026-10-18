"""Is the information ACCESSIBLE to a gate-shaped network, or only to a tree?

separability.py shows a gradient-boosted tree reading the shipped gate's own
three features at pooled AUROC 0.81 while a linear read gets 0.68. That says the
information is present. It does NOT say the gate can get it, for two reasons:

  1. the tree is a far stronger learner than a 3 -> 32 -> 32 -> 1 GELU MLP
  2. the tree is trained DIRECTLY on the class label with ~10k labelled rows,
     while the real gate only ever sees the downstream language-model gradient

This file removes reason 1 and keeps reason 2, by training the EXACT gate
architecture on the class label as a supervised 2-class problem. The gap between
this and the tree is the gate's architectural ceiling; the gap between this and
what the gate actually achieves in training is the optimisation/credit-assignment
cost, which this file does not measure.

Relevant because process/03/README.md:53-61 records an apparent paradox: an
oracle given only the TRUE class labels reaches 5.42% loss and the feature gate
reaches 5.67%, yet an oracle given class labels PLUS the features reaches 8.77%.
If the features predicted the class at AUROC 0.81 and the gate could use that,
class labels should be nearly redundant. They are not, so something between
"present" and "usable" is binding.

    python gate_capacity.py --stream 777
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import auroc as A
import features as FE


def gate_mlp(n_in, width=32):
    """The shipped gate's shape exactly (model.py:78-82), only the input width
    and the final use differ."""
    return nn.Sequential(nn.Linear(n_in, width), nn.GELU(),
                         nn.Linear(width, width), nn.GELU(),
                         nn.Linear(width, 1))


def train_probe(Xtr, ytr, Xte, width=32, steps=3000, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    xe = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    net = gate_mlp(Xtr.shape[1], width)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    n = len(yt)
    for s in range(steps):
        idx = torch.randint(0, n, (512,))
        loss = F.binary_cross_entropy_with_logits(net(xt[idx])[:, 0], yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return net(xe)[:, 0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=int, default=777)
    ap.add_argument("--tag", default="gated_seed0")
    ap.add_argument("--batches", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(2)

    m, ck = FE.load(a.tag)
    d = FE.collect(m, a.stream, batches=a.batches)
    clw, wt, seqs = d["cls_w"], d["wtype"], d["seq"]
    mk = np.isin(clw, [1, 2]) & (wt != FE.UNSEEN)
    lab = (clw == 2).astype(int)
    y, s = lab[mk], seqs[mk]
    tr, te = A.seq_split(s, 0)

    sets = [("shipped gate features [tok, s1, s2]", ["f_tok", "f_s1", "f_s2"]),
            ("all zero-extra-state", FE.INST),
            ("zero-state + exact history", FE.INST + FE.HIST)]

    print(f"model {a.tag}   stream {a.stream}   pooled over all seen writes")
    print(f"n_train {tr.sum()}  n_test {te.sum()}\n")
    print(f"{'feature set':36s}{'width':>7}{'MLP':>9}{'tree':>9}{'linear':>9}")
    for nm, fs in sets:
        X = np.stack([d[f][mk] for f in fs], 1)
        X = np.where(np.isfinite(X), X, np.nan)
        med = np.nanmedian(X[tr], 0)
        Xi = np.where(np.isfinite(X), X, med)
        au_t, _, _, _ = A.joint(d, mk, fs, lab, seqs, kind="gbm")
        au_l, _, _, _ = A.joint(d, mk, fs, lab, seqs, kind="logit")
        for w in (32, 128):
            sc = train_probe(Xi[tr], y[tr], Xi[te], width=w)
            au_m = roc_auc_score(y[te], sc)
            print(f"{nm if w == 32 else '':36s}{w:>7}{au_m:>9.4f}"
                  f"{au_t if w == 32 else float('nan'):>9.4f}"
                  f"{au_l if w == 32 else float('nan'):>9.4f}")
    print("\nMLP is the shipped gate's own 3 -> w -> w -> 1 GELU stack trained\n"
          "directly on the class label. tree and linear columns repeat\n"
          "separability.py PART 5 for the same rows and split.")


if __name__ == "__main__":
    main()
