"""
Gated Delta SSM - a DeltaNet layer whose write strength can see the memory.

Standard DeltaNet decides HOW HARD to write from the incoming token alone:

    beta_t = sigmoid(w_beta . x_t)

which forces one global write policy: it cannot write cautiously to a contested
key and decisively to a stale one in the same sequence. This layer lets beta
also see what the memory already holds at the address being written, using two
scalars the delta rule computes anyway as a byproduct of writing.

    stored_t = S_{t-1} k_t                      already computed (it IS half of e_t)
    e_t      = stored_t - v_t                   the delta rule's residual

    s1 = log(1 + ||e_t||)                       surprise: how wrong was I
    s2 = <v_t, stored_t> / ||v_t||^2            overlap: how much do I already have

    beta_t = sigmoid( MLP([ w_beta . x_t , s1 , s2 ]) )

Learned policy: overwrite decisively when the incoming value is far from and
unlike what is stored; write softly when it is already there.

COST
    extra state       ZERO. s1, s2 are functions of quantities already materialised.
    extra FLOPs       ~1/d_k of the state update. 0.9% at d_k=128, cheaper as
                      heads widen. Two reductions over d_v and a divide.
    extra params      ~1.6k at gate_width=32 (the saturation point).

PARALLELISM
    beta_t now depends on S_{t-1}, so the transition operator is not
    precomputable and the associative scan breaks. Linearity does NOT rescue
    this: with beta = c^T S d the update contains (c^T S d)(S k), quadratic in
    the state. The MLP costs nothing here - it is pointwise, so a linear gate
    has identical parallelism characteristics.

    The fix is granularity: compute s1, s2 against the CHUNK-ENTRY state
    S_{c-1} rather than the running state. The WY form is then untouched and you
    are already sequential at chunk boundaries. This module implements the
    SEQUENTIAL reference recurrence; chunk-frozen is NOT implemented here.

STATUS: the mechanism is prior art. See ../process/README.md. This module is a
clean reference implementation of a replicated ~5.5% loss reduction, not a novel
contribution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedDeltaSSM(nn.Module):
    """One delta-rule layer with an optional memory-conditioned write gate.

    memory_gate=False reproduces the standard input-only baseline exactly.
    """

    def __init__(self, vocab, d_model=128, d_k=128, d_v=128,
                 conv_width=4, gate_width=32, memory_gate=True):
        super().__init__()
        self.d_k, self.d_v, self.memory_gate = d_k, d_v, memory_gate

        self.embed = nn.Embedding(vocab, d_model)
        # depthwise causal conv. Needed ONLY for the write gate's token term:
        # beta must know whether this position is a write, which lives in
        # surrounding context. q, k, v are all routed AROUND it (see below).
        self.conv = nn.Conv1d(d_model, d_model, conv_width,
                              groups=d_model, padding=conv_width - 1)
        self.conv_width = conv_width

        self.to_q = nn.Linear(d_model, d_k)
        self.to_k = nn.Linear(d_model, d_k)
        self.to_v = nn.Linear(d_model, d_v)
        self.to_beta = nn.Linear(d_model, 1)

        # 3 -> gate_width -> gate_width -> 1. A LINEAR gate underperforms:
        # width 16 gives 4.8% loss reduction, 32 gives 5.5%, and it saturates
        # there (64 and 128 add 0.2pp for 3x and 10x the parameters).
        self.gate = nn.Sequential(
            nn.Linear(3, gate_width), nn.GELU(),
            nn.Linear(gate_width, gate_width), nn.GELU(),
            nn.Linear(gate_width, 1),
        )
        self.norm = nn.LayerNorm(d_v)
        self.head = nn.Linear(d_v, vocab)

    def forward(self, tokens, return_beta=False):
        """tokens: [B, L] -> logits [B, L, vocab] (and beta [B, L] if asked)."""
        B, L = tokens.shape
        emb = self.embed(tokens)
        ctx = F.silu(self.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))

        # v and q come from the RAW embedding, k from a HARD SHIFT. Routing any
        # of them through the conv is actively harmful:
        #   v from conv -> the same value token yields vectors at cosine 0.88,
        #                  so a repeat write never gives the delta rule's exact
        #                  zero and there is no zero point to measure against.
        #   k from conv -> leaks the VALUE token into the ADDRESS, so keys that
        #                  change value drift to a different address each time.
        #                  This manufactured most of the apparent signal in the
        #                  first version of this work.
        shifted = F.pad(emb, (0, 0, 1, 0))[:, :L]        # emb of token t-1
        q = F.normalize(self.to_q(emb), dim=-1)
        k = F.normalize(self.to_k(shifted), dim=-1)      # unit keys: required
        v = self.to_v(emb)
        beta_token = self.to_beta(ctx)[..., 0]           # [B, L]

        S = emb.new_zeros(B, self.d_v, self.d_k)
        logits, betas = [], []
        for t in range(L):
            kt, qt, vt = k[:, t], q[:, t], v[:, t]

            stored = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1)   # S_{t-1} k_t
            e = stored - vt                                       # BEFORE the write

            g = beta_token[:, t]
            if self.memory_gate:
                # detached: the gate READS the memory, it does not reshape it
                s1 = torch.log1p(e.norm(dim=-1)).detach()
                s2 = ((vt * stored).sum(-1) /
                      (vt.norm(dim=-1) ** 2 + 1e-6)).detach()
                g = self.gate(torch.stack([g, s1, s2], -1))[..., 0]
            beta = torch.sigmoid(g)                               # [B]

            S = S - beta[:, None, None] * e.unsqueeze(-1) * kt.unsqueeze(1)
            logits.append(self.head(self.norm(
                torch.bmm(S, qt.unsqueeze(-1)).squeeze(-1))))
            betas.append(beta)

        lg = torch.stack(logits, 1)
        return (lg, torch.stack(betas, 1)) if return_beta else lg


# ---------------------------------------------------------------------------
# KNOWN ISSUE, kept visible on purpose.
#
# s2 is ALGEBRAICALLY REDUNDANT given s1 and the two norms. Since e = Sk - v:
#     s2 = ( ||stored||^2 + ||v||^2 - ||e||^2 ) / ( 2 ||v||^2 )
# so conditioned on ||v||, the pair (s1, s2) is a bijection of
# (||e||, ||stored||). It is not "surprise plus overlap"; it is residual norm
# and readout norm in rotated coordinates. Any ablation separating s1 from s2 is
# uninterpretable unless ||stored|| enters as a third feature, and s2 degenerates
# entirely at d_v = 1.
#
# The recommended replacement, UNTESTED, is
#     [ w_beta . x_t , log(1+||e_t||) , ||S_{t-1} k_t|| ]
# which is genuinely three independent quantities. The version above is the one
# that was actually measured, so it is what ships.
# ---------------------------------------------------------------------------
