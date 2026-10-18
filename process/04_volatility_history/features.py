"""Replay a trained GatedDeltaSSM and record, for every write, the gate's
instantaneous features plus EXACT per-address history from out-of-model
bookkeeping.

The bookkeeping is deliberately exact rather than a superposition memory. Every
per-key store attempted in this repo died of crosstalk, so an exact history is
an UPPER BOUND: if a signal is absent here it cannot be recovered by any
mechanism, and if it is present, crosstalk is a separate hurdle this file does
not test.

Addressing, which is where this repo has gone wrong before. The write at step t
stores value from token x_t at the address derived from token x_{t-1}:

    k_t  = normalize(W_k emb(x_{t-1}))          address  <- PREVIOUS token
    v_t  = W_v emb(x_t)                         value    <- CURRENT token

so the class of a write is class(x_{t-1}), and "this address was written before"
means x_{t-1} occurred at some position strictly before t-1. Both are computed
here from bookkeeping, not from the task's `seen` array, whose t-th entry is
about x_t and so describes the VALUE token rather than the address.

`replay` asserts its own recurrence reproduces the shipped forward pass.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "gated_delta_ssm"))

import numpy as np
import torch
import torch.nn.functional as F

import task as T
from model import GatedDeltaSSM

HERE = os.path.dirname(os.path.abspath(__file__))

UNSEEN, REPEAT, CHANGE = 0, 1, 2

# every column name this module produces, so downstream code cannot silently
# pick up a feature that was not declared
INST = ["f_tok", "f_s1", "f_s2", "f_nstored", "f_nv"]
HIST = ["h_cos", "h_cosmean", "h_ratio", "h_meane", "h_nvis"]
ORACLE = ["o_memb", "o_ndist"]


def load(tag, gate_width=32):
    ck = torch.load(os.path.join(HERE, "checkpoints", f"{tag}.pt"),
                    weights_only=False)
    m = GatedDeltaSSM(T.N_SYM, memory_gate=ck["memory_gate"],
                      gate_width=ck.get("gate_width", gate_width))
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


class Bookkeeper:
    """Exact per-address history, model-independent.

    Pulled out of `replay` so that a second model class can reuse the IDENTICAL
    bookkeeping instead of a copy, which is how a copy drifts. Verified against
    a naive single-sequence reimplementation in audit_addressing.py.
    """

    def __init__(self, B, dv):
        self.B, self.dv = B, dv
        self.ar = torch.arange(B)
        self.last_e = torch.zeros(B, T.N_SYM, dv)
        self.sum_e = torch.zeros(B, T.N_SYM, dv)
        self.sum_ne = torch.zeros(B, T.N_SYM)
        self.cnt = torch.zeros(B, T.N_SYM)
        self.ndist = torch.zeros(B, T.N_SYM)
        self.last_val = torch.full((B, T.N_SYM), -1, dtype=torch.long)
        self.valset = torch.zeros(B, T.N_SYM, T.N_SYM, dtype=torch.bool)

    def blank_row(self):
        nan = torch.full((self.B,), float("nan"))
        out = {n: nan for n in HIST + ORACLE}
        out["wtype"] = torch.full((self.B,), float(UNSEEN))
        return out

    def step(self, addr, val, e, ne):
        ar = self.ar
        p_last = self.last_e[ar, addr]
        p_sum = self.sum_e[ar, addr]
        p_cnt = self.cnt[ar, addr]
        p_ne = self.sum_ne[ar, addr]
        p_val = self.last_val[ar, addr]
        p_dist = self.ndist[ar, addr]
        memb = self.valset[ar, addr, val].float()
        fresh = p_cnt == 0

        def blank(z):
            return torch.where(fresh, torch.full_like(z, float("nan")), z)

        out = {
            "h_cos": blank(F.cosine_similarity(e, p_last, dim=-1, eps=1e-8)),
            "h_cosmean": blank(
                F.cosine_similarity(e, p_sum, dim=-1, eps=1e-8)),
            "h_ratio": blank(
                torch.log((ne + 1e-6) / (p_last.norm(dim=-1) + 1e-6))),
            "h_meane": blank(p_ne / p_cnt.clamp_min(1)),
            "h_nvis": p_cnt.clone(),
            "o_memb": blank(memb),
            "o_ndist": blank(p_dist),
            "wtype": torch.where(
                fresh, torch.full_like(p_cnt, float(UNSEEN)),
                torch.where(p_val == val, torch.full_like(p_cnt, float(REPEAT)),
                            torch.full_like(p_cnt, float(CHANGE)))),
        }

        self.last_e[ar, addr] = e
        self.sum_e[ar, addr] = p_sum + e
        self.sum_ne[ar, addr] = p_ne + ne
        self.cnt[ar, addr] = p_cnt + 1
        self.ndist[ar, addr] = p_dist + (1.0 - memb)
        self.valset[ar, addr, val] = True
        self.last_val[ar, addr] = val
        return out


@torch.no_grad()
def replay(m, x):
    """x: [B, L+1] token stream. Returns a dict of [B, L] float arrays.

    Reimplements GatedDeltaSSM.forward step by step so the intermediate
    quantities are visible, and checks the reimplementation against the real
    forward pass before returning anything.
    """
    tokens = x[:, :-1]
    B, L = tokens.shape
    dv = m.d_v

    emb = m.embed(tokens)
    ctx = F.silu(m.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))
    shifted = F.pad(emb, (0, 0, 1, 0))[:, :L]
    q = F.normalize(m.to_q(emb), dim=-1)
    k = F.normalize(m.to_k(shifted), dim=-1)
    v = m.to_v(emb)
    beta_token = m.to_beta(ctx)[..., 0]

    bk = Bookkeeper(B, dv)
    cols = {n: [] for n in INST + HIST + ORACLE + ["beta", "wtype", "ne"]}
    S = emb.new_zeros(B, m.d_v, m.d_k)
    logits = []
    for t in range(L):
        kt, qt, vt = k[:, t], q[:, t], v[:, t]
        stored = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1)
        e = stored - vt
        ne = e.norm(dim=-1)

        g = beta_token[:, t]
        s1 = torch.log1p(ne)
        s2 = (vt * stored).sum(-1) / (vt.norm(dim=-1) ** 2 + 1e-6)
        if m.memory_gate:
            g = m.gate(torch.stack([g, s1, s2], -1))[..., 0]
        beta = torch.sigmoid(g)

        cols["f_tok"].append(beta_token[:, t])
        cols["f_s1"].append(s1)
        cols["f_s2"].append(s2)
        cols["f_nstored"].append(stored.norm(dim=-1))
        cols["f_nv"].append(vt.norm(dim=-1))
        cols["beta"].append(beta)
        cols["ne"].append(ne)

        row = (bk.blank_row() if t == 0
               else bk.step(tokens[:, t - 1], tokens[:, t], e, ne))
        for n in HIST + ORACLE + ["wtype"]:
            cols[n].append(row[n])

        S = S - beta[:, None, None] * e.unsqueeze(-1) * kt.unsqueeze(1)
        logits.append(m.head(m.norm(torch.bmm(S, qt.unsqueeze(-1)).squeeze(-1))))

    lg = torch.stack(logits, 1)
    ref = m(tokens)
    err = (lg - ref).abs().max().item()
    assert err < 1e-4, f"replay diverged from the shipped forward pass: {err:.2e}"

    out = {n: torch.stack(c, 1).numpy() for n, c in cols.items()}
    out["logits"] = lg.numpy()
    out["correct"] = (lg.argmax(-1) == x[:, 1:]).numpy()
    return out


def collect(m, seed, batches=6, batch=32):
    """Replay `batches` eval batches and return one flat record per write."""
    ev = np.random.default_rng(seed)
    acc = None
    seq_off = 0
    losses, accs = [], []
    for _ in range(batches):
        x, cl, sn = T.sample(ev, batch)
        d = replay(m, x)
        B, L = batch, T.L
        # class of the write address = class(x_{t-1}); class of the value token
        # = class(x_t). The second is the known labelling trap, kept to size it.
        d["cls_w"] = F.pad(cl, (1, 0))[:, :L].numpy().astype(np.int64)
        d["cls_q"] = cl.numpy().astype(np.int64)
        # the task's own mask, as every earlier script used it (describes x_t)
        d["sn_task"] = sn.numpy()
        # and shifted one step, which is what "the address was written before"
        # actually requires
        d["sn_addr"] = F.pad(sn, (1, 0))[:, :L].numpy()
        d["seq"] = np.repeat(np.arange(seq_off, seq_off + B)[:, None], L, 1)
        d["pos"] = np.repeat(np.arange(L)[None, :], B, 0)
        losses.append(F.cross_entropy(
            torch.from_numpy(d["logits"]).reshape(-1, T.N_SYM),
            x[:, 1:].reshape(-1)).item())
        accs.append(d["correct"].mean())
        d.pop("logits")
        seq_off += B
        if acc is None:
            acc = {kk: [vv] for kk, vv in d.items()}
        else:
            for kk, vv in d.items():
                acc[kk].append(vv)
    flat = {kk: np.concatenate(vv).reshape(-1) for kk, vv in acc.items()}
    flat["_loss"] = float(np.mean(losses))
    flat["_acc"] = float(np.mean(accs))
    return flat


if __name__ == "__main__":
    torch.set_num_threads(2)
    m, ck = load("gated_seed0")
    d = collect(m, 777, batches=1)
    print(f"loss {d['_loss']:.4f}  acc {d['_acc']:.4f}  rows {d['seq'].size}")
    wt = d["wtype"]
    for nm, code in (("unseen", UNSEEN), ("repeat", REPEAT), ("change", CHANGE)):
        print(f"  {nm:7s} {np.mean(wt == code):.4f}")
