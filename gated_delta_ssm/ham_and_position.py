"""
Two tests that could each kill our result.

TEST 1 - HAM's cosine router as extra arms.

CORRECTED 2026-10-18, after reading the paper instead of these notes. The
claim this docstring used to make was WRONG twice over. Verification and
verbatim quotes are in ../validation/README.md.

HAM (arXiv:2603.22325v2, "Hybrid Associative Memories", Lufkin, Figliolia,
Millidge, Krishnamurthy) does NOT gate the delta rule's write strength at all.
Its routing score decides KV-CACHE ADMISSION for an attention branch:
    Eq 10   e_t = min_h e_t^(h),   m_t = I[e_t >= tau]
    Eq 11   Attn(q_t, {k_i : m_i = 1}, {v_i : m_i = 1})
and the RNN update (Eqs 7-8, Eq 15, Algorithm 1) carries NO routing term:
    S_t <- f(S_{t-1}, k_t^RNN, v_t^RNN)
Its recurrent branch is an UNMODIFIED Gated DeltaNet. HAM's beta is therefore
exactly this file's `none` arm. Its 48.4 vs 49.1 was never measuring our
mechanism, so it was never the opposite of our result.

WRONG CLAIM 2: that cosine loses because it discards magnitude. Our own
results below refute it. ham_cos_mlp scores 5.06% against ours at 5.51%, and
wins on accuracy, so magnitude is worth roughly 8% of the effect, not the
effect.

What survives, and is what these arms actually test, is that HAM's design
space entangles two axes. Its own FLOP accounting labels them as mutually
exclusive options:
    "via RNN state (if not using a learnable router)"
    "via input (if using a learnable router)"
Its state-based score is a hand-designed cosine threshold (Eq 16). Its learned
router is Eq 14, e_t = sigma(MLP(x_t)), fed the TOKEN only. The general form
it simplifies away, D_theta(S_{t-1} k_t, v_t, x_t), is a learned function of
exactly these features, but the paper says it "simplif[ies] this learned
router by only feeding it the input x_t" and never evaluates the general form.

    ham_cos_lin   beta = sigmoid(base + a*(1 - cos(S k, v)) + b)   hand-designed
                  feature, linear gate: HAM's arm, adapted to continuous beta
    ham_cos_mlp   beta = sigmoid(MLP([base, 1 - cos(S k, v)]))     same feature,
                  OUR head: separates "cosine is a weak feature" from
                  "a linear gate is the problem"

If cosine loses to ours with the SAME head, the confound is demonstrated rather
than asserted. If cosine wins, our result was about the MLP and not the residual,
and the whole thing is over.

TEST 2 - position stratification.
SR-TTT (arXiv:2603.06642) is a published corrected negative on amplify-direction
residual gating with a named mechanism: "surprisal gating is systematically
position-biased - the TTT reconstruction loss requires burn-in before a needle
becomes relatively surprising, so early-context needles are stored at near-zero
rates (0-1% containment at depth 0.1)". If our gain is concentrated LATE in the
sequence, it is that artifact.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import task as T


class Layer(nn.Module):
    def __init__(self, vocab, arm="ours", d_model=128, d_k=128, d_v=128,
                 conv_width=4, gate_width=32):
        super().__init__()
        self.d_k, self.d_v, self.arm = d_k, d_v, arm
        self.embed = nn.Embedding(vocab, d_model)
        self.conv = nn.Conv1d(d_model, d_model, conv_width, groups=d_model,
                              padding=conv_width - 1)
        self.to_q = nn.Linear(d_model, d_k); self.to_k = nn.Linear(d_model, d_k)
        self.to_v = nn.Linear(d_model, d_v); self.to_beta = nn.Linear(d_model, 1)
        n_in = {"none": 0, "ours": 3, "ham_cos_lin": 0, "ham_cos_mlp": 2}[arm]
        if arm == "ham_cos_lin":
            self.a = nn.Parameter(torch.zeros(1)); self.b = nn.Parameter(torch.zeros(1))
        if n_in:
            self.gate = nn.Sequential(nn.Linear(n_in, gate_width), nn.GELU(),
                                      nn.Linear(gate_width, gate_width), nn.GELU(),
                                      nn.Linear(gate_width, 1))
        self.norm = nn.LayerNorm(d_v); self.head = nn.Linear(d_v, vocab)

    def forward(self, tokens):
        B, L = tokens.shape
        emb = self.embed(tokens)
        ctx = F.silu(self.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))
        sh = F.pad(emb, (0, 0, 1, 0))[:, :L]
        q = F.normalize(self.to_q(emb), dim=-1); k = F.normalize(self.to_k(sh), dim=-1)
        v = self.to_v(emb); base = self.to_beta(ctx)[..., 0]
        S = emb.new_zeros(B, self.d_v, self.d_k); outs = []
        for t in range(L):
            kt, qt, vt = k[:, t], q[:, t], v[:, t]
            stored = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1)
            e = stored - vt
            g = base[:, t]
            if self.arm == "ours":
                s1 = torch.log1p(e.norm(dim=-1)).detach()
                s2 = ((vt * stored).sum(-1) /
                      (vt.norm(dim=-1) ** 2 + 1e-6)).detach()
                g = self.gate(torch.stack([g, s1, s2], -1))[..., 0]
            elif self.arm in ("ham_cos_lin", "ham_cos_mlp"):
                # HAM's routing feature: 1 - cos(S k, v). MAGNITUDE-FREE.
                cos = ((vt * stored).sum(-1) /
                       (vt.norm(dim=-1) * stored.norm(dim=-1) + 1e-6)).detach()
                score = (1.0 - cos)
                if self.arm == "ham_cos_lin":
                    g = g + self.a * score + self.b
                else:
                    g = self.gate(torch.stack([g, score], -1))[..., 0]
            beta = torch.sigmoid(g)
            S = S - beta[:, None, None] * e.unsqueeze(-1) * kt.unsqueeze(1)
            outs.append(self.head(self.norm(
                torch.bmm(S, qt.unsqueeze(-1)).squeeze(-1))))
        return torch.stack(outs, 1)


def run(arm, seed, steps=800, batch=24):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = Layer(T.N_SYM, arm=arm)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for _ in range(steps):
        x, _, _ = T.sample(rng, batch)
        loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, T.N_SYM), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    L_, A_, POS = [], [], []
    ev = np.random.default_rng(12345)
    with torch.no_grad():
        for _ in range(6):
            x, _, _ = T.sample(ev, 32)
            lg = m(x[:, :-1])
            L_.append(F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1)).item())
            ok = (lg.argmax(-1) == x[:, 1:]).float()          # [B, L]
            A_.append(ok.mean().item()); POS.append(ok.mean(0))  # per-position acc
    return np.mean(L_), np.mean(A_), torch.stack(POS).mean(0).numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    seeds = (0,) if a.quick else (0, 1, 2); steps = 300 if a.quick else 800
    arms = [("none", "input-only baseline"), ("ours", "ours: learned residual"),
            ("ham_cos_lin", "HAM cosine, linear gate"),
            ("ham_cos_mlp", "HAM cosine, OUR head")]
    print("TEST 1: does a hand-designed cosine feature match our learned residual?\n")
    print(f"{'arm':28s}{'loss mean':>11}{'sd':>8}{'acc mean':>10}{'sd':>8}{'vs base':>10}")
    base = None; pos = {}
    for arm, lab in arms:
        L_, A_, P_ = [], [], []
        for s in seeds:
            l, ac, p = run(arm, s, steps=steps); L_.append(l); A_.append(ac); P_.append(p)
        L_, A_ = np.array(L_), np.array(A_); pos[arm] = np.mean(P_, 0)
        if base is None: base = L_.mean()
        print(f"{lab:28s}{L_.mean():>11.4f}{L_.std(ddof=1) if len(L_)>1 else 0:>8.4f}"
              f"{A_.mean():>10.4f}{A_.std(ddof=1) if len(A_)>1 else 0:>8.4f}"
              f"{100*(base-L_.mean())/base:>9.2f}%", flush=True)

    print("\n\nTEST 2: where in the sequence does our gain live?")
    print("(SR-TTT predicts a LATE-concentrated gain if it is the burn-in artifact)\n")
    print(f"{'positions':>14}{'baseline acc':>14}{'ours acc':>11}{'gain':>9}")
    B_, O_ = pos["none"], pos["ours"]
    for lo in range(0, T.L, 32):
        hi = lo + 32
        b, o = B_[lo:hi].mean(), O_[lo:hi].mean()
        print(f"{f'{lo}-{hi-1}':>14}{b:>14.4f}{o:>11.4f}{o-b:>+9.4f}")
    first, second = slice(0, T.L // 2), slice(T.L // 2, T.L)
    gf = (O_[first] - B_[first]).mean(); gs = (O_[second] - B_[second]).mean()
    print(f"\n  first half gain  {gf:+.4f}")
    print(f"  second half gain {gs:+.4f}")
    print(f"  ratio second/first = {gs/gf if abs(gf)>1e-9 else float('nan'):.2f}"
          f"   (>>1 would support the SR-TTT burn-in artifact)")
