"""
Head-to-head: our key-side residual gate vs CARVE's query-side readout gate.

We have never compared against the closest prior art. CARVE (arXiv:2606.27229)
conditions its gates on the memory too, but on a DIFFERENT quantity read from a
DIFFERENT address. Verified from its text:

    CARVE Eq. 8:  G_c = [W_U1,b ; W_U1,w] S_{c-1}
                  t_{c,t} = tanh(G_c q_{c,t})              <- QUERY side
                  gate = sigma(x-term + W_U2 . t)
                  the residual e_t never enters the gate

    ours:         s1 = log(1 + ||e_t||)                    <- KEY side, residual
                  s2 = <v_t, S k_t> / ||v_t||^2

Same base layer, same task, same seeds, and the SAME gate head (32-wide MLP) for
every arm, so only the gate's INPUT varies. carve_k reads CARVE's low-rank form
at the KEY, separating "which address" from "residual vs readout".

WHAT THIS IS NOT: a reproduction of CARVE. CARVE is built on GDN-2 (channel-wise
gates) and evaluated at 1.3B on real text. This is a DeltaNet layer with a scalar
beta on a synthetic task that was purpose-built to reward per-key write policies,
and whose induction structure makes k_t and q_t come from DIFFERENT tokens - which
structurally favours key-side reads. Read the result as evidence about the gate
input under induction structure, not as "we beat CARVE".
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import task as T


class ComparisonLayer(nn.Module):
    """DeltaNet layer with a swappable gate INPUT. gate_input='none' is the
    input-only baseline; every arm shares the same gate head."""

    def __init__(self, vocab, gate_input="ours", rank=8, d_model=128,
                 d_k=128, d_v=128, conv_width=4, gate_width=32):
        super().__init__()
        self.d_k, self.d_v, self.gate_input, self.rank = d_k, d_v, gate_input, rank
        self.embed = nn.Embedding(vocab, d_model)
        self.conv = nn.Conv1d(d_model, d_model, conv_width, groups=d_model,
                              padding=conv_width - 1)
        self.to_q = nn.Linear(d_model, d_k); self.to_k = nn.Linear(d_model, d_k)
        self.to_v = nn.Linear(d_model, d_v); self.to_beta = nn.Linear(d_model, 1)

        n_extra = {"none": 0, "ours": 2, "carve_q": rank, "carve_k": rank}[gate_input]
        self.n_in = 1 + n_extra
        if gate_input in ("carve_q", "carve_k"):
            # CARVE's low-rank bottleneck on the state readout
            self.W_lr = nn.Linear(d_v, rank, bias=False)
        if n_extra:
            self.gate = nn.Sequential(
                nn.Linear(self.n_in, gate_width), nn.GELU(),
                nn.Linear(gate_width, gate_width), nn.GELU(),
                nn.Linear(gate_width, 1))
        self.norm = nn.LayerNorm(d_v); self.head = nn.Linear(d_v, vocab)

    def forward(self, tokens):
        B, L = tokens.shape
        emb = self.embed(tokens)
        ctx = F.silu(self.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))
        sh = F.pad(emb, (0, 0, 1, 0))[:, :L]
        q = F.normalize(self.to_q(emb), dim=-1)
        k = F.normalize(self.to_k(sh), dim=-1)
        v = self.to_v(emb)
        base = self.to_beta(ctx)[..., 0]

        S = emb.new_zeros(B, self.d_v, self.d_k)
        outs = []
        for t in range(L):
            kt, qt, vt = k[:, t], q[:, t], v[:, t]
            stored = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1)      # S_{t-1} k_t
            e = stored - vt

            g = base[:, t]
            if self.gate_input == "ours":
                s1 = torch.log1p(e.norm(dim=-1)).detach()
                s2 = ((vt * stored).sum(-1) /
                      (vt.norm(dim=-1) ** 2 + 1e-6)).detach()
                g = self.gate(torch.stack([g, s1, s2], -1))[..., 0]
            elif self.gate_input in ("carve_q", "carve_k"):
                if self.gate_input == "carve_q":
                    # NOTE: an EXTRA matvec. S k_t is already computed for the
                    # delta rule; S q_t is not.
                    rd = torch.bmm(S, qt.unsqueeze(-1)).squeeze(-1)
                else:
                    rd = stored                                      # free
                t_lr = torch.tanh(self.W_lr(rd)).detach()
                g = self.gate(torch.cat([g.unsqueeze(-1), t_lr], -1))[..., 0]
            beta = torch.sigmoid(g)

            S = S - beta[:, None, None] * e.unsqueeze(-1) * kt.unsqueeze(1)
            outs.append(self.head(self.norm(
                torch.bmm(S, qt.unsqueeze(-1)).squeeze(-1))))
        return torch.stack(outs, 1)


def run(gate_input, seed, steps=800, batch=24):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = ComparisonLayer(T.N_SYM, gate_input=gate_input)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for _ in range(steps):
        x, _, _ = T.sample(rng, batch)
        loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, T.N_SYM), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    L_, A = [], []
    ev = np.random.default_rng(12345)
    with torch.no_grad():
        for _ in range(6):
            x, _, _ = T.sample(ev, 32)
            lg = m(x[:, :-1])
            L_.append(F.cross_entropy(lg.reshape(-1, T.N_SYM),
                                      x[:, 1:].reshape(-1)).item())
            A.append((lg.argmax(-1) == x[:, 1:]).float().mean().item())
    ng = sum(p.numel() for n, p in m.named_parameters()
             if n.startswith(("gate", "W_lr")))
    return np.mean(L_), np.mean(A), ng


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    seeds = (0,) if a.quick else (0, 1, 2)
    steps = 300 if a.quick else 800
    arms = [("none", "input-only baseline"),
            ("ours", "ours: key-side residual"),
            ("carve_q", "CARVE form: query-side"),
            ("carve_k", "CARVE form: key-side")]
    print(f"{'arm':28s}{'gate par':>9}{'loss mean':>11}{'sd':>8}{'acc mean':>10}{'sd':>8}{'vs base':>10}")
    base = None
    for gi, lab in arms:
        L_, A_, ng = [], [], 0
        for s in seeds:
            l, ac, ng = run(gi, s, steps=steps); L_.append(l); A_.append(ac)
        L_, A_ = np.array(L_), np.array(A_)
        if base is None: base = L_.mean()
        sdl = L_.std(ddof=1) if len(L_) > 1 else 0.0
        sda = A_.std(ddof=1) if len(A_) > 1 else 0.0
        print(f"{lab:28s}{ng:>9,}{L_.mean():>11.4f}{sdl:>8.4f}{A_.mean():>10.4f}"
              f"{sda:>8.4f}{100*(base-L_.mean())/base:>9.2f}%", flush=True)
