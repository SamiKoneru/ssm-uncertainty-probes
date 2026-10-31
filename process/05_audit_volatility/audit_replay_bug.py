"""A11.1b. How much does a wrong-beta replay distort separability?

04's AUDIT.md retracts its "the gate manufactures its own signal, 0.61 -> 0.81"
figure: signinit_test.py:88 replays a model trained with gate="both" using
beta = sigmoid(W x) only, dropping the a1*s1 + a2*s2 terms the gate was trained
with, so the S it accumulates is from a recurrence that was never trained.

A11.1 as pre-registered wanted the exact magnitudes (buggy 0.6161 vs corrected
0.7787) reproduced on an independently trained legacy model. That needs one
more 800-step fit and the machine is carrying two other sessions' jobs, so it
is NOT done; see the README. This file tests the same MECHANISM on a model I
already have, which is cheaper and still falsifiable:

replay the shipped GatedDeltaSSM two ways, identical in every respect except
the beta used to accumulate S,

    correct   beta = sigmoid( gate([w.x, s1, s2]) )     what it was trained with
    buggy     beta = sigmoid( w.x )                     the gate MLP dropped

and measure pooled NOISY-vs-REVISED separability of the features read off each
trajectory. If a wrong-beta replay can move separability by a large amount,
the bug class is confirmed as capable of producing 04's artifact.

    python audit_replay_bug.py --stream 505 --tag gated_seed0
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import _paths                                                  # noqa: F401
import auroc as A
import features as FE
import task as T

THREE = ["f_tok", "f_s1", "f_s2"]


@torch.no_grad()
def replay(m, x, buggy):
    """FE.replay's recurrence, with beta either correct or input-only."""
    tokens = x[:, :-1]
    B, L = tokens.shape
    emb = m.embed(tokens)
    ctx = F.silu(m.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))
    shifted = F.pad(emb, (0, 0, 1, 0))[:, :L]
    k = F.normalize(m.to_k(shifted), dim=-1)
    q = F.normalize(m.to_q(emb), dim=-1)
    v = m.to_v(emb)
    bt = m.to_beta(ctx)[..., 0]
    S = emb.new_zeros(B, m.d_v, m.d_k)
    cols = {n: [] for n in FE.INST}
    lg = []
    for t in range(L):
        kt, qt, vt = k[:, t], q[:, t], v[:, t]
        stored = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1)
        e = stored - vt
        s1 = torch.log1p(e.norm(dim=-1))
        s2 = (vt * stored).sum(-1) / (vt.norm(dim=-1) ** 2 + 1e-6)
        cols["f_tok"].append(bt[:, t]); cols["f_s1"].append(s1)
        cols["f_s2"].append(s2)
        cols["f_nstored"].append(stored.norm(dim=-1))
        cols["f_nv"].append(vt.norm(dim=-1))
        if buggy:
            beta = torch.sigmoid(bt[:, t])          # the signinit bug
        else:
            beta = torch.sigmoid(m.gate(
                torch.stack([bt[:, t], s1, s2], -1))[..., 0])
        S = S - beta[:, None, None] * e.unsqueeze(-1) * kt.unsqueeze(1)
        lg.append(m.head(m.norm(torch.bmm(S, qt.unsqueeze(-1)).squeeze(-1))))
    out = {n: torch.stack(c, 1).numpy().astype(np.float64)
           for n, c in cols.items()}
    out["_loss"] = F.cross_entropy(
        torch.stack(lg, 1).reshape(-1, T.N_SYM), x[:, 1:].reshape(-1)).item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=int, default=505)
    ap.add_argument("--tag", default="gated_seed0")
    ap.add_argument("--batches", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(2)
    m, ck = FE.load(a.tag)

    ev = np.random.default_rng(a.stream)
    acc = {}
    for bi in range(a.batches):
        x, cl, _ = T.sample(ev, 32)
        ref = FE.replay(m, x)
        d = {"correct": replay(m, x, False), "buggy": replay(m, x, True)}
        # the correct replay must match the shipped forward pass
        w = max(float(np.abs(d["correct"][n] - ref[n]).max()) for n in FE.INST)
        assert w < 1e-4, f"correct replay diverged: {w:.2e}"
        row = {"cls_w": F.pad(cl, (1, 0))[:, :T.L].numpy(),
               "wtype": ref["wtype"],
               "seq": np.repeat(np.arange(bi*32, bi*32+32)[:, None], T.L, 1)}
        for arm in ("correct", "buggy"):
            for n in FE.INST:
                row[f"{arm}_{n}"] = d[arm][n]
        row["_lc"] = d["correct"]["_loss"]; row["_lb"] = d["buggy"]["_loss"]
        for kk, vv in row.items():
            acc.setdefault(kk, []).append(vv)
    lc = float(np.mean(acc.pop("_lc"))); lb = float(np.mean(acc.pop("_lb")))
    f = {kk: np.concatenate(vv).reshape(-1) for kk, vv in acc.items()}

    mk = (f["wtype"] != FE.UNSEEN) & np.isin(f["cls_w"], [1, 2])
    lab = (f["cls_w"] == 2).astype(int)
    print(f"A11.1b  model {a.tag} (seed {ck['seed']})  stream {a.stream}  "
          f"rows {mk.sum()}")
    print(f"\n  cross-entropy of the trajectory each replay produces:")
    print(f"    correct (what it was trained with)  {lc:.4f}")
    print(f"    buggy   (gate MLP dropped)          {lb:.4f}   "
          f"{lb - lc:+.4f}")
    print(f"\n  pooled NOISY vs REVISED separability read off each trajectory:")
    print(f"  {'feature set':30s}{'correct':>10}{'buggy':>10}{'shift':>10}")
    for nm, fs in (("[s1, s2]", ["f_s1", "f_s2"]),
                   ("[tok, s1, s2]", THREE),
                   ("all five zero-state", FE.INST)):
        r = {}
        for arm in ("correct", "buggy"):
            cols = {f"{arm}_{n}": f[f"{arm}_{n}"] for n in FE.INST}
            cols.update({n: f[n] for n in ("seq",)})
            au, _, _, _ = A.joint(cols, mk, [f"{arm}_{n}" for n in fs],
                                  lab, f["seq"], kind="gbm")
            r[arm] = au
        print(f"  {nm:30s}{r['correct']:>10.4f}{r['buggy']:>10.4f}"
              f"{r['buggy'] - r['correct']:>+10.4f}")
    print(f"\n  04's legacy model, from its AUDIT.md: buggy 0.6161 ->"
          f" correct 0.7787 on [s1,s2], a shift of -0.1626.")
    print("  The sign and the order of magnitude are what matter here: a\n"
          "  wrong-beta replay changes the trajectory, so it changes the\n"
          "  separability of features read off it. Exact agreement is not\n"
          "  expected and is not claimed - this is a different model.")


if __name__ == "__main__":
    main()
