"""
ITEM 1. Two checks that can each end the result, run as one sweep.

CHECK A - the missing control.
    Everything published so far compares a 3-input MLP gate against NO gate.
    That confounds "the gate reads the memory" with "the gate has an MLP".
    The `token_only` arm feeds [w_beta.x_t , 0 , 0] through the identical
    32-wide MLP: same parameters (62,058, verified in arms.py), same init,
    same FLOPs. If it recovers most of the 5.5%, the gain was capacity.

CHECK B - is the gap a convergence transient?
    Every number in this repo comes from exactly 800 steps at exactly lr 2e-3.
    If the baseline merely converges more slowly, the gap shrinks with training
    and the headline is an artifact of stopping early. This trains 4x longer and
    evaluates along the way, so the gap is read as a CURVE, not one endpoint.

    Each arm is scored at ITS OWN best learning rate. Tuning one lr on the
    gated arm and reusing it for the baseline would handicap the baseline.

    The task generates fresh data every step, so there is no train set to
    overfit and longer training is purely a convergence question.

    python convergence.py --quick     # 1 arm-triple, 300 steps, wiring check
    python convergence.py             # the real sweep
"""
import argparse
import itertools
import time

import numpy as np
import torch
import torch.nn.functional as F

import arms
import task as T                      # resolved by arms.py's path insert


def evaluate(m, n_batches=6, batch=32):
    """train.py's exact eval protocol: same fixed stream for every config."""
    ev = np.random.default_rng(12345)
    losses, accs = [], []
    with torch.no_grad():
        for _ in range(n_batches):
            x, _, _ = T.sample(ev, batch)
            lg = m(x[:, :-1])
            losses.append(F.cross_entropy(lg.reshape(-1, T.N_SYM),
                                          x[:, 1:].reshape(-1)).item())
            accs.append((lg.argmax(-1) == x[:, 1:]).float().mean().item())
    return float(np.mean(losses)), float(np.mean(accs))


def run(arm, seed, lr, steps, eval_every, batch=24):
    """Train one arm, evaluating at checkpoints. Returns {step: (loss, acc)}."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = arms.build(arm, T.N_SYM)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    curve = {}
    for i in range(1, steps + 1):
        x, _, _ = T.sample(rng, batch)
        loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, T.N_SYM),
                               x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if i % eval_every == 0 or i == steps:
            curve[i] = evaluate(m)
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 2e-3, 4e-3])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.steps, a.eval_every, a.seeds, a.lrs = 300, 150, [0], [2e-3]

    grid = list(itertools.product(arms.ARMS, a.lrs, a.seeds))
    print(f"{len(grid)} runs: {len(arms.ARMS)} arms x {len(a.lrs)} lrs x "
          f"{len(a.seeds)} seeds, {a.steps} steps each\n", flush=True)

    res = {}                                   # (arm, lr) -> list of curves
    t0 = time.time()
    for n, (arm, lr, seed) in enumerate(grid, 1):
        c = run(arm, seed, lr, a.steps, a.eval_every)
        res.setdefault((arm, lr), []).append(c)
        fin = c[max(c)]
        print(f"[{n:>2}/{len(grid)}] {arm:11s} lr {lr:.0e} seed {seed}  "
              f"loss {fin[0]:.4f}  acc {fin[1]:.4f}  "
              f"({(time.time()-t0)/60:.0f} min elapsed)", flush=True)

    ckpts = sorted(next(iter(res.values()))[0].keys())
    mean = {k: {c: (np.mean([cv[c][0] for cv in v]),
                    np.mean([cv[c][1] for cv in v])) for c in ckpts}
            for k, v in res.items()}

    print("\n\nFULL GRID - loss by training step\n")
    print(f"{'arm':12s}{'lr':>7}" + "".join(f"{c:>9}" for c in ckpts), flush=True)
    for arm in arms.ARMS:
        for lr in a.lrs:
            row = "".join(f"{mean[(arm, lr)][c][0]:>9.4f}" for c in ckpts)
            print(f"{arm:12s}{lr:>7.0e}{row}", flush=True)

    # each arm at its own best lr, chosen on FINAL loss
    last = ckpts[-1]
    best = {arm: min(a.lrs, key=lambda l: mean[(arm, l)][last][0])
            for arm in arms.ARMS}
    print("\n\nCHECK B - each arm at ITS OWN best lr. Does the gap close?\n")
    print(f"{'arm':12s}{'best lr':>9}" + "".join(f"{c:>9}" for c in ckpts))
    for arm in arms.ARMS:
        row = "".join(f"{mean[(arm, best[arm])][c][0]:>9.4f}" for c in ckpts)
        print(f"{arm:12s}{best[arm]:>9.0e}{row}", flush=True)
    print(f"\n{'gap vs none (%)':21s}" + "".join(f"{c:>9}" for c in ckpts))
    for arm in ("ours", "token_only"):
        row = ""
        for c in ckpts:
            b = mean[("none", best["none"])][c][0]
            row += f"{100 * (b - mean[(arm, best[arm])][c][0]) / b:>9.2f}"
        print(f"{arm:21s}{row}", flush=True)

    print("\n\nCHECK A - the missing control, at the final checkpoint\n")
    nb = mean[("none", best["none"])][last][0]
    ob = mean[("ours", best["ours"])][last][0]
    tb = mean[("token_only", best["token_only"])][last][0]
    share = 100 * (nb - tb) / (nb - ob) if abs(nb - ob) > 1e-9 else float("nan")
    print(f"  none        {nb:.4f}")
    print(f"  ours        {ob:.4f}   {100*(nb-ob)/nb:+.2f}%")
    print(f"  token_only  {tb:.4f}   {100*(nb-tb)/nb:+.2f}%")
    print(f"\n  token_only recovers {share:.0f}% of the gap.")
    print("  Near 0% means the gain is reading memory. Near 100% means it was"
          "\n  gate capacity all along and the memory feature is decoration.")
    print(f"\n{len(a.seeds)} seeds. Treat differences under ~0.004 loss as noise.")
    print(f"total {(time.time()-t0)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
