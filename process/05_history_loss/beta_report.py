"""Mean beta by key class and write type for every arm, and whether giving the
gate history changes the THREE-REGIME non-monotone beta response that
process/04_volatility_history/beta_policy.py recorded on the shipped gate:

    mean beta by s1 decile   0.997 0.864 0.016 0.022 0.033 0.044 0.062
                             0.183 0.258 0.304

    python beta_report.py --stream 5150

Write types follow 04: a write at step t goes to the address from token
x_{t-1} with the value from token x_t, so

    unseen   that address has not been written before in this sequence
    repeat   seen before and x_t equals the value last written there
    change   seen before and x_t differs from it

computed here directly from the token stream, independently of the model's own
bookkeeping, so the two can disagree if either is wrong.
"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "gated_delta_ssm"))

import task as T                                                  # noqa: E402
import arms as A                                                  # noqa: E402
import score as SC                                                # noqa: E402

UNSEEN, REPEAT, CHANGE = 0, 1, 2
NAMES = ("STABLE", "NOISY", "REVISED")


def write_types(tok):
    """[B, L] tokens -> [B, L] write type, computed from the stream alone."""
    B, L = tok.shape
    wt = np.full((B, L), UNSEEN, np.int64)
    for b in range(B):
        last = np.full(T.N_SYM, -1, np.int64)
        for t in range(1, L):
            a, val = int(tok[b, t - 1]), int(tok[b, t])
            if last[a] < 0:
                wt[b, t] = UNSEEN
            else:
                wt[b, t] = REPEAT if last[a] == val else CHANGE
            last[a] = val
    return wt


@torch.no_grad()
def gather(m, stream, batches=6, batch=32):
    ev = np.random.default_rng(stream)
    B_, C_, W_, S1_, NE_ = [], [], [], [], []
    for _ in range(batches):
        x, cl, _ = T.sample(ev, batch)
        tok = x[:, :-1]
        lg, beta, f = m(tok, return_beta=True, return_feats=True)
        B_.append(beta.numpy().ravel())
        C_.append(torch.nn.functional.pad(cl, (1, 0))[:, :T.L].numpy().ravel())
        W_.append(write_types(tok.numpy()).ravel())
        S1_.append(f[..., 1].numpy().ravel())
        NE_.append(np.expm1(f[..., 1].numpy().ravel()))
    return (np.concatenate(B_), np.concatenate(C_), np.concatenate(W_),
            np.concatenate(S1_), np.concatenate(NE_))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=int, default=5150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", nargs="+",
                    default=["ours", "ours+hist", "ours+rate", "ours_pub"])
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    print(f"stream {a.stream}   training seed {a.seed}\n")

    for arm in a.arms:
        m, ck = SC.load(arm, a.seed)
        if m is None:
            print(f"{arm}: no checkpoint, skipped\n")
            continue
        beta, clw, wt, s1, ne = gather(m, a.stream)
        pos = np.ones_like(wt, bool)
        pos.reshape(-1)[::T.L] = False                 # drop t=0, the pad slot

        print(f"=== {arm} (seed {a.seed}) " + "=" * (46 - len(arm)))
        print("mean beta by CLASS of the write address x WRITE TYPE")
        print(f"{'':9s}{'unseen':>20}{'repeat':>20}{'change':>20}")
        print(f"{'':9s}" + "".join(f"{'beta    ||e||      n':>20}"
                                   for _ in range(3)))
        for i, nm in enumerate(NAMES):
            row = ""
            for code in (UNSEEN, REPEAT, CHANGE):
                mk = (clw == i) & (wt == code) & pos
                row += (f"{beta[mk].mean():>8.3f}{ne[mk].mean():>6.2f}"
                        f"{mk.sum():>6d}" if mk.sum() >= 5 else f"{'-':>20}")
            print(f"{nm:9s}{row}")

        print("\nthree-regime check: mean beta by s1 decile, all writes t>0")
        mk = pos
        q = np.quantile(s1[mk], np.linspace(0, 1, 11))
        cells = []
        for j in range(10):
            sel = mk & (s1 >= q[j]) & (s1 <= q[j + 1])
            cells.append(beta[sel].mean() if sel.sum() else np.nan)
        rho = spearmanr(s1[mk], beta[mk]).statistic
        print("  " + " ".join(f"{c:.3f}" for c in cells)
              + f"   spearman {rho:+.3f}")
        print("  decile edges " + " ".join(f"{z:.2f}" for z in q))
        lo, mid, hi = np.mean(cells[:2]), np.mean(cells[2:7]), np.mean(cells[7:])
        print(f"  regimes: low-s1 {lo:.3f}   mid {mid:.3f}   high-s1 {hi:.3f}"
              f"   -> {'NON-MONOTONE (04 shape held)' if lo > mid and hi > mid else 'MONOTONE / shape CHANGED'}")

        print("\nwithin-stratum spearman(beta, s1) and mean beta")
        for i, nm in enumerate(NAMES):
            for code, cn in ((REPEAT, "repeat"), (CHANGE, "change")):
                mk = (clw == i) & (wt == code) & pos
                if mk.sum() < 50:
                    print(f"  {nm:8s} {cn:7s} n={mk.sum():<6d} -")
                    continue
                print(f"  {nm:8s} {cn:7s} n={mk.sum():<6d} "
                      f"spearman {spearmanr(s1[mk], beta[mk]).statistic:+.3f}"
                      f"   mean beta {beta[mk].mean():.3f}")
        print()


if __name__ == "__main__":
    main()
