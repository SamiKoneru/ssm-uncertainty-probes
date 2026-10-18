"""Is the gate's beta VARIATION doing the work, or just its LEVEL?

Motivated by a seed-0 observation, so everything here is EXPLORATORY: the
`ours+hist` and `ours+oracle` gates collapse to beta ~0.002 at every write
type, which is a near-constant policy. That is not degenerate, because the
readout is head(norm(S q)) and LayerNorm is scale invariant: the stored value
obeys  S_t k = (1-beta) S_{t-1} k + beta v,  an EMA whose rate is beta, so
beta -> 0 turns the memory into a uniform running average over everything
written to that address. For NOISY keys, whose best belief is the average
over their fixed option set, that is the right policy. So a large loss gain
could come from finding a better CONSTANT rather than from reading history.

This separates the two for free, with no retraining:

  as trained      the real gate
  own mean        beta forced to that arm's own mean beta, everywhere
  best constant   beta swept over a grid, the arm's best single constant

If "own mean" matches "as trained", the gate's variation contributes nothing
and the arm is a constant-beta model in disguise. If "best constant" reaches
the trained loss, then nothing about conditioning on anything was needed.

    python const_test.py --stream 5150 --seed 0
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "gated_delta_ssm"))

import task as T                                                  # noqa: E402
import score as SC                                                # noqa: E402

GRID = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15,
        0.2, 0.3, 0.4, 0.6, 0.8, 1.0]


@torch.no_grad()
def run_fixed(m, stream, beta_const=None, batches=6, batch=32):
    """Replays the shipped recurrence; beta_const overrides the gate."""
    ev = np.random.default_rng(stream)
    losses = []
    for _ in range(batches):
        x, _, _ = T.sample(ev, batch)
        tokens = x[:, :-1]
        B, L = tokens.shape
        emb = m.embed(tokens)
        ctx = F.silu(m.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))
        shifted = F.pad(emb, (0, 0, 1, 0))[:, :L]
        q = F.normalize(m.to_q(emb), dim=-1)
        k = F.normalize(m.to_k(shifted), dim=-1)
        v = m.to_v(emb)
        S = emb.new_zeros(B, m.d_v, m.d_k)
        logits = []
        for t in range(L):
            kt, qt, vt = k[:, t], q[:, t], v[:, t]
            e = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1) - vt
            beta = torch.full((B,), float(beta_const))
            S = S - beta[:, None, None] * e.unsqueeze(-1) * kt.unsqueeze(1)
            logits.append(m.head(m.norm(
                torch.bmm(S, qt.unsqueeze(-1)).squeeze(-1))))
        lg = torch.stack(logits, 1)
        losses.append(F.cross_entropy(lg.reshape(-1, T.N_SYM),
                                      x[:, 1:].reshape(-1)).item())
    return float(np.mean(losses))


@torch.no_grad()
def mean_beta(m, stream, batches=2, batch=32):
    ev = np.random.default_rng(stream)
    bs = []
    for _ in range(batches):
        x, _, _ = T.sample(ev, batch)
        bs.append(m(x[:, :-1], return_beta=True)[1].reshape(-1))
    return float(torch.cat(bs).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=int, default=5150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", nargs="+", default=SC.DEFAULT_ARMS)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    print(f"stream {a.stream}   seed {a.seed}   EXPLORATORY, seed-0 motivated\n")
    print(f"{'arm':13s}{'as trained':>12}{'mean beta':>11}{'@ own mean':>12}"
          f"{'variation':>11}{'best const':>12}{'at beta':>9}{'headroom':>10}")
    for arm in a.arms:
        m, ck = SC.load(arm, a.seed)
        if m is None:
            continue
        trained = SC.evaluate(m, a.stream)["loss"]
        mb = mean_beta(m, a.stream)
        at_mean = run_fixed(m, a.stream, mb)
        grid = [(run_fixed(m, a.stream, c), c) for c in GRID]
        best, bc = min(grid)
        print(f"{arm:13s}{trained:>12.4f}{mb:>11.4f}{at_mean:>12.4f}"
              f"{at_mean - trained:>+11.4f}{best:>12.4f}{bc:>9.4f}"
              f"{best - trained:>+10.4f}")
    print("\n  variation = loss(forced to own mean) - loss(as trained).")
    print("              ~0 means the gate is a constant in disguise;")
    print("              large positive means its VARIATION is load bearing.")
    print("  headroom  = loss(best single constant) - loss(as trained).")
    print("              <= 0 would mean no conditioning was needed at all.")


if __name__ == "__main__":
    main()
