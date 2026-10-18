"""AUDIT. The 0.476 is the product of a BUG in signinit_test.py, and this
session reproduced the bug rather than diagnosing it.

`archive/failed_features/signinit_test.py:88` replays the trained model as

    beta = torch.sigmoid(m.to_b(xc))[..., 0]          # INPUT-ONLY beta

but the model it trained is `gate="both"`, whose actual gate is
`gated_model.py:107`

    beta = sigmoid( W x_t + a1*log1p(||e_t||) + a2*prj_t + b )

with a1 = 0.9790 and a2 = -1.4075 after training (logs/signinit.log:2). So the
state S that the extraction accumulates is NOT the trained model's state, and
the (s1, s2) measured off it come from a recurrence that was never trained. The
file has no check that its replay matches the model's own forward pass, so
nothing caught it.

Consequence for this session: `legacy_0476.py` copied that extraction
deliberately, "verbatim in behaviour", and reproduced 0.4712 against the logged
0.4763. The reproduction was faithful INCLUDING the bug, and I read the match as
evidence that the 0.476 described the legacy model. It does not. It describes a
recurrence that does not exist.

This file isolates it: same model, same stream, same features, same
conditioning, only the replay's beta differs.

    python audit_signinit_bug.py
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
from audit_trend import replay_legacy


def load_legacy():
    m = GatedDeltaLayer(T.N_SYM, gate="both", a_init=(0.0, 0.0))
    m.load_state_dict(torch.load(
        os.path.join(HERE, "checkpoints", "legacy_both_seed0.pt"),
        weights_only=False)["state_dict"])
    m.eval()
    return m


@torch.no_grad()
def replay_inputonly(m, x):
    """signinit_test.py:84-96 verbatim, i.e. with the WRONG beta."""
    tokens = x[:, :-1]
    B, L = tokens.shape
    emb = m.embed(tokens)
    xc = F.silu(m.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))
    sh = F.pad(emb, (0, 0, 1, 0))[:, :L]
    k = F.normalize(m.to_k(sh), dim=-1)
    v = m.to_v(emb)
    beta = torch.sigmoid(m.to_b(xc))[..., 0]          # <-- the bug
    bk = FE.Bookkeeper(B, m.dv)
    cols = {n: [] for n in FE.INST + FE.HIST + FE.ORACLE + ["beta", "wtype", "ne"]}
    S = emb.new_zeros(B, m.dv, m.dk)
    for t in range(L):
        kt, vt = k[:, t], v[:, t]
        stored = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1)
        e = stored - vt
        ne = e.norm(dim=-1)
        cols["f_tok"].append(m.to_b(xc)[..., 0][:, t])
        cols["f_s1"].append(torch.log1p(ne))
        cols["f_s2"].append((vt * stored).sum(-1) / (vt.norm(dim=-1) ** 2 + 1e-6))
        cols["f_nstored"].append(stored.norm(dim=-1))
        cols["f_nv"].append(vt.norm(dim=-1))
        cols["beta"].append(beta[:, t]); cols["ne"].append(ne)
        row = (bk.blank_row() if t == 0
               else bk.step(tokens[:, t - 1], tokens[:, t], e, ne))
        for n in FE.HIST + FE.ORACLE + ["wtype"]:
            cols[n].append(row[n])
        S = S - beta[:, t, None, None] * e.unsqueeze(-1) * kt.unsqueeze(1)
    return {n: torch.stack(c, 1).numpy() for n, c in cols.items()}


def collect(fn, m, stream, batches=6, batch=32):
    ev = np.random.default_rng(stream)
    acc, off = None, 0
    for _ in range(batches):
        x, cl, _ = T.sample(ev, batch)
        d = fn(m, x)
        d["cls_w"] = F.pad(cl, (1, 0))[:, :T.L].numpy().astype(np.int64)
        d["seq"] = np.repeat(np.arange(off, off + batch)[:, None], T.L, 1)
        off += batch
        if acc is None:
            acc = {k: [v] for k, v in d.items()}
        else:
            for k, v in d.items():
                acc[k].append(v)
    return {k: np.concatenate(v).reshape(-1) for k, v in acc.items()}


if __name__ == "__main__":
    torch.set_num_threads(2)
    m = load_legacy()
    print(f"trained gate coefficients: a1 = {m.ga.item():.4f}, "
          f"a2 = {m.ga2.item():.4f}, b = {m.gb.item():.4f}")
    print("signinit's replay drops both a1 and a2, so its beta is the"
          " input-only one.\n")
    print(f"{'replay':46s}{'[s1,s2]':>10}{'[tok,s1,s2]':>13}{'all five':>10}")
    for nm, fn in (("signinit_test.py's: beta = sigmoid(W x)", replay_inputonly),
                   ("correct: beta = sigmoid(W x + a1 s1 + a2 s2 + b)",
                    replay_legacy)):
        d = collect(fn, m, 777)
        clw, wt, seqs = d["cls_w"], d["wtype"], d["seq"]
        mk = np.isin(clw, [1, 2]) & (wt != FE.UNSEEN)
        lab = (clw == 2).astype(int)
        out = []
        for fs in (["f_s1", "f_s2"], ["f_tok", "f_s1", "f_s2"], FE.INST):
            a_, sd, _, _ = A.joint(d, mk, fs, lab, seqs)
            out.append(a_)
        print(f"{nm:46s}{out[0]:>10.4f}{out[1]:>13.4f}{out[2]:>10.4f}")
    print("\nThe correct replay carries an assert against the model's own"
          "\nforward pass. signinit_test.py had none, which is why the bug"
          "\nsurvived, and why legacy_0476.py's 'faithful' reproduction"
          "\nreproduced it instead of catching it.")
