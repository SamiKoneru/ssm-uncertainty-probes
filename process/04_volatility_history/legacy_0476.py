"""Reproduce the 0.4763 on the model it was actually measured on.

My pre-registered bar was "each recorded value within +/- 0.04 of my measurement
of the quantity I claim it is". On the SHIPPED model the claim failed: replaying
signinit_test's exact procedure gave 0.7133, not 0.4763. Rather than redefine the
bar, this file removes the one factor the shipped model cannot control for, the
MODEL. signinit_test trained

    GatedDeltaLayer(gate="both", a_init=(0,0))

which is a LINEAR gate, beta = sigmoid(W x + a1*log1p(||e||) + a2*s2 + b), with
two scalar coefficients, not the MLP that ships. It reached loss 1.5882 (2.7%)
where the MLP reaches 1.5402 (5.7%), so it writes with a different beta, so the
state it accumulates and the (s1,s2) it produces are different distributions.

Trains only the (0,0) arm and runs only the class-predictability block, which is
the part that produced the number. Everything else about the procedure is copied
from archive/failed_features/signinit_test.py: eval stream 999, 3 batches of 32,
the task's own sn mask, a 3-way HistGradientBoosting on (s1,s2), a token-level
50/50 split in array order, and P(REVISED) as the score.

    python legacy_0476.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))              # process/
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "03_uncertainty_probe"))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

import mixed_task as T
from gated_model import GatedDeltaLayer

sys.path.insert(0, HERE)
import auroc as A


def train(seed=0, steps=800, batch=24):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = GatedDeltaLayer(T.N_SYM, gate="both", a_init=(0.0, 0.0))
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for s in range(steps):
        x, _, _ = T.sample(rng, batch)
        lg, _ = m(x[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s % 200 == 0 or s == steps - 1:
            print(f"  step {s:4d} loss {loss.item():.4f}", flush=True)
    return m


@torch.no_grad()
def extract(m, stream, batches):
    """signinit_test.py:77-99 verbatim in behaviour, plus the bookkeeping this
    directory needs to re-condition the same rows."""
    S1, S2, CW, SN, SEQ, WT = [], [], [], [], [], []
    ev = np.random.default_rng(stream)
    off = 0
    for _ in range(batches):
        x, cl, sn = T.sample(ev, 32)
        B, L = x.shape[0], T.L
        emb = m.embed(x[:, :-1])
        xc = F.silu(m.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))
        sh = F.pad(emb, (0, 0, 1, 0))[:, :L]
        k = F.normalize(m.to_k(sh), dim=-1)
        v = m.to_v(emb)
        beta = torch.sigmoid(m.to_b(xc))[..., 0]
        St = emb.new_zeros(B, 128, 128)
        c1, c2 = [], []
        for t in range(L):
            kt, vt = k[:, t], v[:, t]
            stored = torch.bmm(St, kt.unsqueeze(-1)).squeeze(-1)
            e = stored - vt
            c1.append(torch.log1p(e.norm(dim=-1)))
            c2.append((vt * stored).sum(-1) / (vt.norm(dim=-1) ** 2 + 1e-6))
            St = St - beta[:, t, None, None] * e.unsqueeze(-1) * kt.unsqueeze(1)
        S1.append(torch.stack(c1, 1)); S2.append(torch.stack(c2, 1))
        CW.append(F.pad(cl, (1, 0))[:, :L]); SN.append(sn)
        SEQ.append(torch.from_numpy(
            np.repeat(np.arange(off, off + B)[:, None], L, 1)))
        xs = x.numpy(); wt = np.zeros((B, L), np.int64)
        for b in range(B):
            last = {}
            for t in range(1, L):
                kk, val = int(xs[b, t - 1]), int(xs[b, t])
                wt[b, t] = 0 if kk not in last else (1 if last[kk] == val else 2)
                last[kk] = val
        WT.append(torch.from_numpy(wt))
        off += B
    f = lambda z: torch.cat(z).reshape(-1).numpy()
    return dict(f_s1=f(S1), f_s2=f(S2), cls_w=f(CW).astype(np.int64),
                sn=f(SN).astype(bool), seq=f(SEQ), wtype=f(WT))


if __name__ == "__main__":
    torch.set_num_threads(2)
    print('training the legacy gate="both", a_init=(0,0) model, 800 steps',
          flush=True)
    m = train()
    torch.save({"state_dict": m.state_dict()},
               os.path.join(HERE, "checkpoints", "legacy_both_seed0.pt"))

    d = extract(m, 999, 3)
    sn, y_all = d["sn"], d["cls_w"]
    X = np.stack([d["f_s1"][sn], d["f_s2"][sn]], 1)
    y = y_all[sn]
    n = len(y); tr = np.arange(n) < n // 2
    g = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[tr], y[tr])
    pp = g.predict_proba(X[~tr])
    print("\nsigninit_test's procedure, on signinit_test's model and stream:")
    print(f"  3-way accuracy {accuracy_score(y[~tr], g.predict(X[~tr])):.4f}"
          f"   (chance {max(np.bincount(y)/len(y)):.4f})   "
          f"signinit logged 0.6485 / 0.5053")
    logged = {"STABLE": 0.9317, "NOISY": 0.8345, "REVISED": 0.7396}
    for i, nm in enumerate(("STABLE", "NOISY", "REVISED")):
        au = roc_auc_score((y[~tr] == i).astype(int), pp[:, i])
        print(f"    AUROC {nm:8s} vs rest: {au:.4f}   logged {logged[nm]:.4f}"
              f"   diff {au-logged[nm]:+.4f}")
    rr = np.isin(y[~tr], [1, 2])
    au = roc_auc_score((y[~tr][rr] == 2).astype(int), pp[rr, 2])
    print(f"    AUROC NOISY vs REVISED only: {au:.4f}   logged 0.4763"
          f"   diff {au-0.4763:+.4f}")
    print(f"\n  BAR: |diff| <= 0.04 on the NOISY-vs-REVISED number -> "
          f"{'PASS' if abs(au-0.4763) <= 0.04 else 'FAIL'}")

    print("\nand the SAME legacy model read correctly: s1 alone, 2-class,\n"
          "NOISY vs REVISED, by write type (exact bookkeeping):")
    lab = (y_all == 2).astype(int)
    nr = np.isin(y_all, [1, 2])
    for nm, mk in (("all seen writes", nr & (d["wtype"] != 0)),
                   ("CHANGE writes only", nr & (d["wtype"] == 2)),
                   ("REPEAT writes only", nr & (d["wtype"] == 1))):
        a1, n1 = A.single(d["f_s1"][mk], lab[mk])
        print(f"    {nm:22s} s1 AUROC {a1:.4f}   n {n1}")
    mk = nr & (d["wtype"] != 0)
    for kind in ("logit", "gbm"):
        a2, sd, _, nte = A.joint(d, mk, ["f_s1", "f_s2"], lab, d["seq"],
                                 kind=kind)
        print(f"    pooled, (s1,s2) joint, {kind:5s}  AUROC {a2:.4f} +/- {sd:.4f}")
