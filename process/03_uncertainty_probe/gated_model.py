"""DeltaLayer whose write gate beta can be conditioned on an uncertainty signal.

  gate='none'     beta = sigmoid(W x_t)                 input only (the baseline)
  gate='u'        beta = sigmoid(W x_t + a*log1p(u_t) + b)
                  u_t = |R_{t-1} . k_t|, available BEFORE this step's write
  gate='entropy'  beta = sigmoid(W x_t + a*ent_{t-1} + b)
                  entropy of the PREVIOUS step's prediction (the only entropy
                  that causally exists in time to gate this write)
  gate='ne'       beta = sigmoid(W x_t + a*log1p(||e_t||) + b)
  gate='prj'      beta = sigmoid(W x_t + a*prj_t + b)
  gate='both'     both of the above

e_t = S_{t-1}k_t - v_t is computed BEFORE the write and does not depend on
beta_t, so gating on it is legal.  ||e_t|| and prj_t = <v_t, S_{t-1}k_t>/||v_t||^2
are the raw per-token statistics; unlike |R.k_t| they are not corrupted by the
accumulator's key superposition.  All gate features are detached, so only the
gate weights learn from them.
"""
import torch, torch.nn as nn, torch.nn.functional as F

class GatedDeltaLayer(nn.Module):
    def __init__(self, vocab, d_model=128, d_k=128, d_v=128, conv_w=4, decay=0.99,
                 gate="none", a_init=(0.0, 0.0), width=16):
        super().__init__()
        self.dk, self.dv, self.decay, self.gate = d_k, d_v, decay, gate
        self.embed = nn.Embedding(vocab, d_model)
        self.conv = nn.Conv1d(d_model, d_model, conv_w, groups=d_model, padding=conv_w - 1)
        self.to_q = nn.Linear(d_model, d_k); self.to_k = nn.Linear(d_model, d_k)
        self.to_v = nn.Linear(d_model, d_v); self.to_b = nn.Linear(d_model, 1)
        self.ga = nn.Parameter(torch.full((1,), float(a_init[0])))
        self.gb = nn.Parameter(torch.zeros(1))
        self.ga2 = nn.Parameter(torch.full((1,), float(a_init[1])))
        # non-monotone gate: the correct policy is NOT monotone in ||e||
        # (STABLE low ||e|| -> don't care, REVISED mid -> HIGH beta,
        #  NOISY high -> LOW beta), and sigmoid(linear) cannot express that.
        self.gmlp = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))
        # ORACLE gates: given the TRUE class of the write key. Not a learnable
        # architecture, a CEILING - how much is the class distinction worth at all.
        self.oracle_b = nn.Parameter(torch.zeros(3))
        self.gmlp_o = nn.Sequential(nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 1))
        # RATE MEMORY: a per-key store of how surprising this key usually is.
        # c_t = C_{t-1}.k_t is the 3a-vs-3b signal (isolated change vs churn),
        # which (s1,s2) provably cannot see (AUROC 0.476 NOISY vs REVISED).
        # Written with the DELTA RULE, not accumulation: accumulation is vanilla
        # linear attention and its crosstalk is what killed the earlier R.
        self.gmlp_r = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 1))
        self.p_bc = nn.Parameter(torch.tensor(-1.4))   # sigmoid -> beta_c ~ 0.2
        # The NOISY-vs-REVISED signal in s1 is strong (AUROC 0.834) but ONLY
        # among CHANGE writes; pooled with repeats the relationship inverts
        # (0.353). The gate therefore needs an INTERACTION: its response to s1
        # must flip on whether s2 says 'repeat'. Two ways to supply that:
        self.gmlp_i = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 1))
        self.gmlp_w = nn.Sequential(nn.Linear(3, width), nn.GELU(),
                                    nn.Linear(width, width), nn.GELU(),
                                    nn.Linear(width, 1))
        self.norm = nn.LayerNorm(d_v); self.head = nn.Linear(d_v, vocab)

    def forward(self, tokens, cls_w=None):
        B, L = tokens.shape
        emb = self.embed(tokens)
        xc = F.silu(self.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))
        sh = F.pad(emb, (0, 0, 1, 0))[:, :L]
        q = F.normalize(self.to_q(emb), dim=-1); k = F.normalize(self.to_k(sh), dim=-1)
        v = self.to_v(emb); base = self.to_b(xc)[..., 0]

        S = emb.new_zeros(B, self.dv, self.dk); R = emb.new_zeros(B, self.dk)
        C = emb.new_zeros(B, self.dk)                     # per-key rate memory
        c_all = []
        prev_ent = emb.new_zeros(B)
        lg_all, beta_all = [], []
        for t in range(L):
            kt = k[:, t]
            u_t = (R * kt).sum(-1).abs()                       # from R_{t-1}, at key k_t
            stored = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1)   # S_{t-1} k_t
            e = stored - v[:, t]                                  # BEFORE the write
            g = base[:, t]
            if self.gate in ("mlp", "oracle_class", "oracle_mlp",
                             "rate_delta", "rate_accum", "rate_prj",
                             "mlp_inter", "mlp_wide"):
                ne_ = torch.log1p(e.norm(dim=-1)).detach()
                prj = ((v[:, t] * stored).sum(-1) /
                       (v[:, t].norm(dim=-1) ** 2 + 1e-6)).detach()
                if self.gate == "mlp":
                    g = self.gmlp(torch.stack([g, ne_, prj], -1))[..., 0]
                elif self.gate == "mlp_inter":     # explicit s1*s2 interaction
                    g = self.gmlp_i(torch.stack([g, ne_, prj, ne_ * prj], -1))[..., 0]
                elif self.gate == "mlp_wide":      # capacity control, same features
                    g = self.gmlp_w(torch.stack([g, ne_, prj], -1))[..., 0]
                elif self.gate in ("rate_delta", "rate_accum", "rate_prj"):
                    c_t = (C * kt).sum(-1).detach()
                    g = self.gmlp_r(torch.stack([g, ne_, prj, c_t], -1))[..., 0]
                else:
                    oh = F.one_hot(cls_w[:, t].long(), 3).float()
                    if self.gate == "oracle_class":
                        g = (self.oracle_b[None] * oh).sum(-1)       # beta from TRUE class only
                    else:
                        g = self.gmlp_o(torch.cat(
                            [torch.stack([g, ne_, prj], -1), oh], -1))[..., 0]
            if self.gate == "u":         g = g + self.ga * torch.log1p(u_t) + self.gb
            elif self.gate == "entropy": g = g + self.ga * prev_ent + self.gb
            elif self.gate in ("ne", "prj", "both"):
                ne_ = torch.log1p(e.norm(dim=-1)).detach()
                prj = ((v[:, t] * stored).sum(-1) /
                       (v[:, t].norm(dim=-1) ** 2 + 1e-6)).detach()
                if self.gate == "ne":    g = g + self.ga * ne_ + self.gb
                elif self.gate == "prj": g = g + self.ga * prj + self.gb
                else:                    g = g + self.ga * ne_ + self.ga2 * prj + self.gb

            beta = torch.sigmoid(g)                            # [B]
            S = S - beta[:, None, None] * e.unsqueeze(-1) * kt.unsqueeze(1)
            lg = self.head(self.norm(torch.bmm(S, q[:, t].unsqueeze(-1)).squeeze(-1)))
            lg_all.append(lg); beta_all.append(beta)
            with torch.no_grad():
                nrm = e.norm(dim=-1)
                c_all.append((C * kt).sum(-1))
                bc = torch.sigmoid(self.p_bc)
                if self.gate == "rate_delta":                 # DELTA RULE, stores ||e|| (MAGNITUDE)
                    C = C - bc * ((C * kt).sum(-1, keepdim=True) - nrm.unsqueeze(-1)) * kt
                elif self.gate == "rate_prj":                 # DELTA RULE, stores OVERLAP
                    # ||e|| is confounded with rate: NOISY changes more often AND by
                    # more, so an EMA of ||e|| inherits its wrong class ordering.
                    # s2 is magnitude-free and HAS the right ordering, just too
                    # noisy per token - averaging it per key is what this is for.
                    pr = ((v[:, t] * stored).sum(-1) /
                          (v[:, t].norm(dim=-1) ** 2 + 1e-6))
                    C = C - bc * ((C * kt).sum(-1, keepdim=True) - pr.unsqueeze(-1)) * kt
                elif self.gate == "rate_accum":               # control: plain accumulation
                    C = 0.99 * C + nrm.unsqueeze(-1) * kt
                p = F.softmax(lg, -1)
                prev_ent = -(p * torch.log(p + 1e-9)).sum(-1)
                R = self.decay * R + e.norm(dim=-1, keepdim=True) * kt
        self.last_c = torch.stack(c_all, 1) if c_all else None
        return torch.stack(lg_all, 1), torch.stack(beta_all, 1)
