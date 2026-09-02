"""A minimal but realistic DeltaNet layer, with R as a pure observer."""
import torch, torch.nn as nn, torch.nn.functional as F

class DeltaLayer(nn.Module):
    def __init__(self, vocab, d_model=128, d_k=64, d_v=64, conv_w=4, a=0.99,
                 v_from_embed=True, kq_from_embed=False, fixed_beta=None):
        super().__init__()
        self.dk, self.dv, self.a = d_k, d_v, a
        # v from the RAW embedding: the value token sits at the position being
        # written, so v needs no context.  Taking it from the conv output makes
        # the same value token produce different vectors each time, which leaves
        # a permanent residual floor and destroys the delta rule's exact zero.
        self.v_from_embed = v_from_embed
        # k from a HARD SHIFT of the embedding, q from the current embedding.
        # Taking k from the conv leaks the value token into the address, so a
        # contested key (new value each write) drifts to a different address each
        # time and leaves residual for reasons unrelated to contestation.
        self.kq_from_embed = kq_from_embed
        # fixed_beta=1.0 forces an EXACT overwrite (keep only the latest value).
        # If partial writes are what encode uncertainty, this should preserve
        # accuracy but destroy calibration.
        self.fixed_beta = fixed_beta
        self.embed = nn.Embedding(vocab, d_model)
        # depthwise causal conv: lets a write bind the key token to the value token
        self.conv = nn.Conv1d(d_model, d_model, conv_w, groups=d_model, padding=conv_w - 1)
        self.conv_w = conv_w
        self.to_q = nn.Linear(d_model, d_k)
        self.to_k = nn.Linear(d_model, d_k)
        self.to_v = nn.Linear(d_model, d_v)
        self.to_b = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_v)
        self.head = nn.Linear(d_v, vocab)

    def forward(self, tokens):
        B, L = tokens.shape
        emb = self.embed(tokens)
        x = F.silu(self.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))
        if self.kq_from_embed:
            shifted = F.pad(emb, (0, 0, 1, 0))[:, :L]   # embedding of token t-1
            q = F.normalize(self.to_q(emb), dim=-1)
            k = F.normalize(self.to_k(shifted), dim=-1)
        else:
            q = F.normalize(self.to_q(x), dim=-1)
            k = F.normalize(self.to_k(x), dim=-1)      # unit keys: delta rule needs this
        v = self.to_v(emb if self.v_from_embed else x)  # context-free value
        beta = (torch.sigmoid(self.to_b(x)) if self.fixed_beta is None
                else x.new_full((B, L, 1), self.fixed_beta))

        S = x.new_zeros(B, self.dv, self.dk)
        R = x.new_zeros(B, self.dk)                    # OBSERVER, never touches the loss
        outs, us = [], []
        for t in range(L):
            kt, vt, qt, bt = k[:, t], v[:, t], q[:, t], beta[:, t]
            e = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1) - vt          # [B,dv]
            S = S - bt.unsqueeze(-1) * e.unsqueeze(-1) * kt.unsqueeze(1)  # delta write
            outs.append(torch.bmm(S, qt.unsqueeze(-1)).squeeze(-1))       # read at q
            with torch.no_grad():
                R = self.a * R + e.detach().norm(dim=-1, keepdim=True) * kt.detach()
                us.append((R * qt.detach()).sum(-1).abs())
        o = torch.stack(outs, 1)
        return self.head(self.norm(o)), torch.stack(us, 1)
