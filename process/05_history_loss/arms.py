"""The four arms, and the subclass that puts EXACT per-address history inside
the recurrence so the gate can train on it.

The confound this file exists to avoid is recorded in validation/README.md on
origin/validation-suite: every published comparison in this repo pitted a
3-input MLP gate against NO GATE AT ALL, confounding "the gate can see more"
with "the gate has more parameters". So all three gated arms here get the SAME
gate input width (8), the same parameter count, the same initialisation and the
same FLOPs, and differ only in which columns are zeroed before the MLP sees
them.

    none        memory_gate=False                     input-only baseline
    ours        [tok, s1, s2, 0,0,0,0,0]              masked control
    ours+hist   [tok, s1, s2, h1..h5]                 all five history features
    ours+rate   [tok, s1, s2, 0,0,0,h_meane,0]        the rate memory alone
    ours_pub    [tok, s1, s2]                         literal shipped 3-input gate

The bookkeeping is EXACT: one slot per symbol, no superposition, so it is an
UPPER BOUND on what any real per-address history mechanism could feed the gate.
A real layer would need d_k x d_v of state written state-dependently and would
lose signal to crosstalk. If the exact version buys no loss, nothing cheaper
will.

`test_parity.py` asserts this subclass with n_hist=0 reproduces the shipped
GatedDeltaSSM forward pass exactly, which is the guard against the
reimplemented recurrence drifting from the thing it audits.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "gated_delta_ssm"))

import task as T                                                  # noqa: E402
from model import GatedDeltaSSM                                   # noqa: E402

# order of the five history columns inside the gate input, after [tok, s1, s2]
HIST = ["h_cos", "h_cosmean", "h_ratio", "h_meane", "h_nvis"]

ARMS = {
    #  arm         n_hist  keep (indices into HIST)
    "none":       (0,      ()),
    "ours":       (5,      ()),
    "ours+hist":  (5,      (0, 1, 2, 3, 4)),
    "ours+rate":  (5,      (3,)),            # h_meane only
    "ours_pub":   (0,      ()),
}

RATIO_CLAMP = 10.0


class HistGatedDeltaSSM(GatedDeltaSSM):
    """GatedDeltaSSM whose gate additionally reads exact per-address history.

    n_hist=0 is the shipped model, bit-for-bit (see test_parity.py).
    n_hist=5 widens the gate's first layer to 8 and appends the history block,
    multiplied by a fixed 0/1 keep-mask so every arm is parameter-identical.
    """

    def __init__(self, vocab, n_hist=5, keep=(), gate_width=32,
                 memory_gate=True, n_addr=None, **kw):
        super().__init__(vocab, gate_width=gate_width,
                         memory_gate=memory_gate, **kw)
        self.n_hist = n_hist
        self.n_addr = vocab if n_addr is None else n_addr
        if n_hist:
            # rebuild the gate with the wider input. Done for EVERY arm with
            # n_hist=5, so the torch RNG is consumed identically and the three
            # gated arms share an initialisation distribution.
            self.gate = nn.Sequential(
                nn.Linear(3 + n_hist, gate_width), nn.GELU(),
                nn.Linear(gate_width, gate_width), nn.GELU(),
                nn.Linear(gate_width, 1),
            )
            mask = torch.zeros(n_hist)
            for i in keep:
                mask[i] = 1.0
            self.register_buffer("hist_mask", mask)
        self.n_clamped = 0
        self.n_hist_rows = 0

    def forward(self, tokens, return_beta=False, return_feats=False):
        """tokens: [B, L] -> logits [B, L, vocab].

        Identical to GatedDeltaSSM.forward except for the history block, which
        runs entirely under no_grad and enters the gate detached, exactly as s1
        and s2 already do.
        """
        B, L = tokens.shape
        emb = self.embed(tokens)
        ctx = F.silu(self.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))

        shifted = F.pad(emb, (0, 0, 1, 0))[:, :L]
        q = F.normalize(self.to_q(emb), dim=-1)
        k = F.normalize(self.to_k(shifted), dim=-1)
        v = self.to_v(emb)
        beta_token = self.to_beta(ctx)[..., 0]

        use_hist = self.memory_gate and self.n_hist > 0
        if use_hist:
            dev, dt = emb.device, emb.dtype
            ar = torch.arange(B, device=dev)
            # per-sequence, per-address bookkeeping. Exact, no superposition.
            last_e = torch.zeros(B, self.n_addr, self.d_v, device=dev, dtype=dt)
            sum_e = torch.zeros(B, self.n_addr, self.d_v, device=dev, dtype=dt)
            sum_ne = torch.zeros(B, self.n_addr, device=dev, dtype=dt)
            cnt = torch.zeros(B, self.n_addr, device=dev, dtype=dt)
            zero_h = torch.zeros(B, self.n_hist, device=dev, dtype=dt)

        S = emb.new_zeros(B, self.d_v, self.d_k)
        logits, betas, feats = [], [], []
        for t in range(L):
            kt, qt, vt = k[:, t], q[:, t], v[:, t]

            stored = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1)
            e = stored - vt

            g = beta_token[:, t]
            if self.memory_gate:
                ne = e.norm(dim=-1)
                s1 = torch.log1p(ne).detach()
                s2 = ((vt * stored).sum(-1) /
                      (vt.norm(dim=-1) ** 2 + 1e-6)).detach()
                gin = torch.stack([g, s1, s2], -1)
                if use_hist:
                    with torch.no_grad():
                        h = self._history(t, tokens, e.detach(), ne.detach(),
                                          ar, last_e, sum_e, sum_ne, cnt,
                                          zero_h)
                    gin = torch.cat([gin, h * self.hist_mask], -1)
                if return_feats:
                    feats.append(gin.detach())
                g = self.gate(gin)[..., 0]
            beta = torch.sigmoid(g)

            S = S - beta[:, None, None] * e.unsqueeze(-1) * kt.unsqueeze(1)
            logits.append(self.head(self.norm(
                torch.bmm(S, qt.unsqueeze(-1)).squeeze(-1))))
            if return_beta:
                betas.append(beta)

        lg = torch.stack(logits, 1)
        if return_feats:
            return lg, torch.stack(betas, 1) if return_beta else None, \
                torch.stack(feats, 1)
        return (lg, torch.stack(betas, 1)) if return_beta else lg

    def _history(self, t, tokens, e, ne, ar, last_e, sum_e, sum_ne, cnt,
                 zero_h):
        """The five exact per-address history features at step t, then update.

        The write at step t stores the value from token x_t at the address
        derived from token x_{t-1}, so the bookkeeping key is tokens[:, t-1].
        At t=0 the address is the zero pad, a slot no symbol ever revisits, so
        it is treated as fresh.
        """
        if t == 0:
            return zero_h
        addr = tokens[:, t - 1]
        p_last = last_e[ar, addr]
        p_sum = sum_e[ar, addr]
        p_cnt = cnt[ar, addr]
        p_ne = sum_ne[ar, addr]
        fresh = p_cnt == 0

        def blank(z):
            # 04 emits NaN here. NaN into a gate is a dead model, so fresh gets
            # 0.0; h_nvis = log1p(0) = 0 still identifies the fresh case
            # exactly, so no information is lost.
            return torch.where(fresh, torch.zeros_like(z), z)

        h_cos = blank(F.cosine_similarity(e, p_last, dim=-1, eps=1e-8))
        h_cosmean = blank(F.cosine_similarity(e, p_sum, dim=-1, eps=1e-8))
        raw_ratio = torch.log((ne + 1e-6) / (p_last.norm(dim=-1) + 1e-6))
        clamped = raw_ratio.clamp(-RATIO_CLAMP, RATIO_CLAMP)
        self.n_clamped += int(((raw_ratio != clamped) & ~fresh).sum())
        self.n_hist_rows += int((~fresh).sum())
        h_ratio = blank(clamped)
        # log1p on these two: monotone, so every AUROC in 04 is unchanged, it
        # only conditions the MLP input (04 fed them to a tree, which did not
        # care about scale).
        h_meane = blank(torch.log1p(p_ne / p_cnt.clamp_min(1)))
        h_nvis = torch.log1p(p_cnt)

        last_e[ar, addr] = e
        sum_e[ar, addr] = p_sum + e
        sum_ne[ar, addr] = p_ne + ne
        cnt[ar, addr] = p_cnt + 1
        return torch.stack([h_cos, h_cosmean, h_ratio, h_meane, h_nvis], -1)


def build(arm, vocab=T.N_SYM, gate_width=32):
    """Construct one arm. Must be called with the torch seed already set."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}, expected one of {list(ARMS)}")
    n_hist, keep = ARMS[arm]
    return HistGatedDeltaSSM(vocab, n_hist=n_hist, keep=keep,
                             gate_width=gate_width,
                             memory_gate=(arm != "none"))


if __name__ == "__main__":
    print(f"{'arm':12s}{'params':>10}{'gate in':>9}  keep")
    counts = {}
    for a in ARMS:
        torch.manual_seed(0)
        m = build(a)
        n = sum(p.numel() for p in m.parameters())
        counts[a] = n
        n_hist, keep = ARMS[a]
        print(f"{a:12s}{n:>10,}{3 + n_hist:>9}  "
              f"{[HIST[i] for i in keep] if keep else '-'}")
    gated = ["ours", "ours+hist", "ours+rate"]
    ok = len({counts[a] for a in gated}) == 1
    print(f"\nthe three gated arms: "
          f"{'IDENTICAL param count, control is valid' if ok else 'MISMATCH, control is INVALID'}")
    print(f"none     is {counts['ours'] - counts['none']:+,} vs ours "
          f"(no gate used; mismatch is by design and is reported)")
    print(f"ours_pub is {counts['ours'] - counts['ours_pub']:+,} vs ours "
          f"(the 5 masked input columns' first-layer weights)")
