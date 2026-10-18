"""AUDIT. Attack the session's strongest claim: that the better the gate, the
less long-term history is worth.

As published in README.md the trend table had THREE models but only TWO points
in the column that decides it, because the legacy model's history delta was
never measured. "Every column is monotone" was therefore not a fair description.
This file measures the missing point.

The legacy model is the LEAST separable instantaneously (0.616 against 0.810 and
0.832), so it has the MOST room for history to add. The claim predicts its
history delta is the LARGEST of the three. If it is the smallest, the trend is
refuted and the session's strongest argument against the volatility memory goes
with it.

Stated before running: the claim predicts delta_legacy > 0.0807 > 0.0634.

Uses features.Bookkeeper directly, so the legacy model gets the IDENTICAL
bookkeeping as the shipped one rather than a copy of it.

    python audit_trend.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "03_uncertainty_probe"))

import auroc as A
import features as FE
import mixed_task as T
from gated_model import GatedDeltaLayer


@torch.no_grad()
def replay_legacy(m, x):
    """The legacy gate='both' recurrence, with features.Bookkeeper attached.

    beta = sigmoid(W x_t + a1*log1p(||e||) + a2*s2 + b), the LINEAR gate, which
    is what signinit_test.py trained.
    """
    tokens = x[:, :-1]
    B, L = tokens.shape
    emb = m.embed(tokens)
    xc = F.silu(m.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))
    sh = F.pad(emb, (0, 0, 1, 0))[:, :L]
    k = F.normalize(m.to_k(sh), dim=-1)
    v = m.to_v(emb)
    base = m.to_b(xc)[..., 0]
    q = F.normalize(m.to_q(emb), dim=-1)

    bk = FE.Bookkeeper(B, m.dv)
    logits = []
    cols = {n: [] for n in FE.INST + FE.HIST + FE.ORACLE + ["beta", "wtype", "ne"]}
    S = emb.new_zeros(B, m.dv, m.dk)
    for t in range(L):
        kt, vt = k[:, t], v[:, t]
        stored = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1)
        e = stored - vt
        ne = e.norm(dim=-1)
        s1 = torch.log1p(ne)
        s2 = (vt * stored).sum(-1) / (vt.norm(dim=-1) ** 2 + 1e-6)
        # gated_model.py:107 exactly: g = g + ga*log1p(||e||) + ga2*prj + gb
        beta = torch.sigmoid(base[:, t] + m.ga[0] * s1 + m.ga2[0] * s2 + m.gb[0])

        cols["f_tok"].append(base[:, t]); cols["f_s1"].append(s1)
        cols["f_s2"].append(s2); cols["f_nstored"].append(stored.norm(dim=-1))
        cols["f_nv"].append(vt.norm(dim=-1))
        cols["beta"].append(beta); cols["ne"].append(ne)
        row = (bk.blank_row() if t == 0
               else bk.step(tokens[:, t - 1], tokens[:, t], e, ne))
        for n in FE.HIST + FE.ORACLE + ["wtype"]:
            cols[n].append(row[n])
        S = S - beta[:, None, None] * e.unsqueeze(-1) * kt.unsqueeze(1)
        logits.append(m.head(m.norm(
            torch.bmm(S, q[:, t].unsqueeze(-1)).squeeze(-1))))

    # the self-check signinit_test.py did not have. Its extraction replayed the
    # model with the INPUT-ONLY beta, so its S was not the trained model's S,
    # and nothing caught it.
    lg = torch.stack(logits, 1)
    ref, _ = m(tokens)
    err = (lg - ref).abs().max().item()
    assert err < 1e-4, f"legacy replay diverged from the real forward: {err:.2e}"
    return {n: torch.stack(c, 1).numpy() for n, c in cols.items()}


def collect_legacy(stream, batches=6, batch=32):
    m = GatedDeltaLayer(T.N_SYM, gate="both", a_init=(0.0, 0.0))
    m.load_state_dict(torch.load(
        os.path.join(HERE, "checkpoints", "legacy_both_seed0.pt"),
        weights_only=False)["state_dict"])
    m.eval()
    ev = np.random.default_rng(stream)
    acc, off = None, 0
    for _ in range(batches):
        x, cl, _ = T.sample(ev, batch)
        d = replay_legacy(m, x)
        d["cls_w"] = F.pad(cl, (1, 0))[:, :T.L].numpy().astype(np.int64)
        d["seq"] = np.repeat(np.arange(off, off + batch)[:, None], T.L, 1)
        off += batch
        if acc is None:
            acc = {kk: [vv] for kk, vv in d.items()}
        else:
            for kk, vv in d.items():
                acc[kk].append(vv)
    return {kk: np.concatenate(vv).reshape(-1) for kk, vv in acc.items()}


if __name__ == "__main__":
    torch.set_num_threads(2)
    print("PREDICTION, stated before the numbers below:")
    print("  the claim 'better gate -> less history value' predicts the legacy")
    print("  model's history delta is the LARGEST of the three, since it is the")
    print("  least separable instantaneously and so has the most room.")
    print("  delta_legacy > +0.0807 (seed 0) > +0.0634 (seed 1)\n")

    rows = []
    for nm, getter in (("legacy linear gate=both", collect_legacy),
                       ("shipped MLP seed 0",
                        lambda s: FE.collect(FE.load("gated_seed0")[0], s)),
                       ("shipped MLP seed 1",
                        lambda s: FE.collect(FE.load("gated_seed1")[0], s))):
        d = getter(777)
        clw, wt, seqs = d["cls_w"], d["wtype"], d["seq"]
        mk = np.isin(clw, [1, 2]) & (wt != FE.UNSEEN)
        lab = (clw == 2).astype(int)
        a_, b_, dl, sd, _ = A.joint_pair(d, mk, FE.INST, FE.INST + FE.HIST,
                                         lab, seqs)
        ao, _, _, _ = A.joint(d, mk, FE.INST + FE.HIST + FE.ORACLE, lab, seqs)
        rows.append((nm, a_, b_, dl, sd, ao))

    print(f"{'model':26s}{'inst':>8}{'+hist':>8}{'delta':>9}{'sd':>8}"
          f"{'+oracle':>10}")
    for nm, a_, b_, dl, sd, ao in rows:
        print(f"{nm:26s}{a_:>8.4f}{b_:>8.4f}{dl:>+9.4f}{sd:>8.4f}{ao:>10.4f}")

    d_leg, d_s0, d_s1 = [r[3] for r in rows]
    print(f"\n  predicted ordering  legacy > seed0 > seed1")
    print(f"  measured            {d_leg:+.4f}  {d_s0:+.4f}  {d_s1:+.4f}")
    if d_leg > d_s0 > d_s1:
        print("  VERDICT: trend SURVIVES (n=3, still exploratory)")
    elif d_leg < d_s0:
        print("  VERDICT: trend REFUTED. The least separable model gains the"
              " LEAST\n           from history, which is the opposite of the"
              " claim.")
    else:
        print("  VERDICT: trend not monotone. Reported as mixed.")
