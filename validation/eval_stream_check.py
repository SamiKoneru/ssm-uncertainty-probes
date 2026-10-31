"""
AUDIT. Is the headline an artifact of the one eval stream this project reuses?

THE PROBLEM
    14 files in this repo score on np.random.default_rng(12345), including
    train.py, compare_carve.py, ham_and_position.py and convergence.py. That
    is 6 batches of 32, so 192 sequences, and EVERY design decision in the
    project was made while looking at that same sample: which features feed
    the gate, the gate width, the q/k/v routing fix, the CARVE comparison, the
    cosine comparison, and the position stratification.

    Reusing one evaluation sample across dozens of decisions is exactly the
    condition under which a result is exploratory rather than confirmed. The
    project has never once scored on a stream it had not already looked at.
    Note this is NOT train/test leakage: the task is generated fresh every
    training step, so there is no finite training set. The issue is that model
    and feature selection were all validated against a single fixed sample.

PRE-REGISTERED, written before the run
    Question   Does the input-only vs memory-gated gap survive on eval streams
               never used for any decision in this project?
    Metric     Cross-entropy on the synthetic task, PUBLISHED protocol exactly:
               800 steps, lr 2e-3, gate_width 32, 3 seeds. No retuning.
    Streams    12345 (the canonical, reused one) plus 202, 303, 404, 505, 606,
               verified absent from the repo before this file was written.
    PASS BAR   (1) the canonical stream's gap lies INSIDE the [min, max] range
                   of the 5 fresh streams. If it lies outside, the headline is
                   stream-specific and must be requoted as a cross-stream mean.
               (2) the mean gap over the 5 fresh streams stays at or above 4.0%
                   against the 5.47% published figure.
    KILLS IT   mean fresh gap below 4.0%, or the canonical stream turning out
               to be the best of the six.
    ALSO       the cross-stream sd of the gap, a number this project has never
               measured, and whether token_only's null is stream-specific too.

    Reports canonical and fresh side by side. The canonical figure is not
    discarded whatever happens.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

import arms
import task as T

CANONICAL = 12345
FRESH = (202, 303, 404, 505, 606)
STEPS, LR, SEEDS = 800, 2e-3, (0, 1, 2)


def evaluate(m, eval_seed, n_batches=6, batch=32):
    ev = np.random.default_rng(eval_seed)
    ls, ac = [], []
    with torch.no_grad():
        for _ in range(n_batches):
            x, _, _ = T.sample(ev, batch)
            lg = m(x[:, :-1])
            ls.append(F.cross_entropy(lg.reshape(-1, T.N_SYM),
                                      x[:, 1:].reshape(-1)).item())
            ac.append((lg.argmax(-1) == x[:, 1:]).float().mean().item())
    return float(np.mean(ls)), float(np.mean(ac))


def train(arm, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = arms.build(arm, T.N_SYM)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    for _ in range(STEPS):
        x, _, _ = T.sample(rng, 24)
        loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, T.N_SYM),
                               x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


def main():
    streams = (CANONICAL,) + FRESH
    print(f"published protocol: {STEPS} steps, lr {LR}, {len(SEEDS)} seeds")
    print(f"streams: {CANONICAL} (canonical, reused) + {len(FRESH)} fresh "
          f"{FRESH}\n", flush=True)

    t0 = time.time()
    res = {a: {s: [] for s in streams} for a in arms.ARMS}
    for arm in arms.ARMS:
        for seed in SEEDS:
            m = train(arm, seed)
            for s in streams:
                res[arm][s].append(evaluate(m, s))
            print(f"  {arm:11s} seed {seed} done "
                  f"({(time.time()-t0)/60:.0f} min)", flush=True)

    loss = {a: {s: np.mean([v[0] for v in res[a][s]]) for s in streams}
            for a in arms.ARMS}

    print("\n\nLOSS BY EVAL STREAM\n")
    print(f"{'arm':12s}" + "".join(f"{s:>9}" for s in streams))
    for a in arms.ARMS:
        print(f"{a:12s}" + "".join(f"{loss[a][s]:>9.4f}" for s in streams),
              flush=True)

    print("\n\nGAP VS none, BY STREAM (%)\n")
    print(f"{'arm':12s}" + "".join(f"{s:>9}" for s in streams))
    gaps = {}
    for a in ("ours", "token_only"):
        gaps[a] = {s: 100 * (loss["none"][s] - loss[a][s]) / loss["none"][s]
                   for s in streams}
        print(f"{a:12s}" + "".join(f"{gaps[a][s]:>9.2f}" for s in streams),
              flush=True)

    print("\n\nVERDICT against the pre-registered bars\n")
    for a in ("ours", "token_only"):
        c = gaps[a][CANONICAL]
        f = [gaps[a][s] for s in FRESH]
        lo, hi = min(f), max(f)
        print(f"  {a}")
        print(f"    canonical {c:+.2f}%   fresh mean {np.mean(f):+.2f}%  "
              f"sd {np.std(f, ddof=1):.2f}  range [{lo:+.2f}, {hi:+.2f}]")
        inside = lo <= c <= hi
        print(f"    BAR 1 canonical inside the fresh range: "
              f"{'PASS' if inside else 'FAIL, result is stream-specific'}")
        if a == "ours":
            print(f"    BAR 2 fresh mean >= 4.00%: "
                  f"{'PASS' if np.mean(f) >= 4.0 else 'FAIL, headline inflated'}")
            best = max(streams, key=lambda s: gaps[a][s])
            print(f"    KILL  canonical is the best stream: "
                  f"{'yes, FAIL' if best == CANONICAL else f'no (best is {best})'}")
        print(flush=True)
    print(f"total {(time.time()-t0)/60:.0f} min")


if __name__ == "__main__":
    main()
