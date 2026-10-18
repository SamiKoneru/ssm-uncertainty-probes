"""
ITEM 2. Does the 5.5% survive on a task that was not built to require it?

This is the largest open question in the project. Our headline task randomises
each symbol's write policy per sequence so that an input-only gate provably
cannot win. MQAR has no such construction. It is also the setting where the
one much larger published study got the OPPOSITE result to ours, on real
language data, so "works on synthetic, does nothing on ordinary recall" is a
live explanation for that whole disagreement and nothing we have run can
currently rule it out.

Sweeps the capacity axis, because that is the only thing in plain recall that
can make write strength depend on the memory rather than the token.

ORIGINAL PRE-REGISTRATION, WRONG, kept for the record. mqar_task.py predicted
`plain` and `rewrite` would both be near-null because an exact repeat gives
e = S k - v = 0, so beta multiplies zero. Measured instead at dim 32: plain
+39.48%, rewrite +55.81%. The argument holds only for ORTHOGONAL keys, and the
pre-registration contradicted itself by also claiming capacity is the only live
axis, which is exactly the regime where keys cannot be orthogonal.

SECOND PRE-REGISTRATION, written before running dims 24/48/64/128.
    With n = 64 distinct key addresses in R^d, orthogonality is a counting
    question. For n <= d, n mutually orthogonal unit keys EXIST, so the model
    can drive crosstalk to zero, e to zero on repeats, and beta becomes
    irrelevant. For n > d it is impossible by pigeonhole, and the Welch bound
    forces max_{i!=j} |<k_i,k_j>| >= sqrt((n-d)/(d(n-1))):

        d = 128   n <= d   floor 0        predict gain ~ 0
        d =  64   n = d    floor 0        predict gain ~ 0
        d =  48   n > d    floor 0.078    predict gain > 0
        d =  32   n > d    floor 0.126    MEASURED +39% / +56%
        d =  24   n > d    floor 0.165    predict gain > 0
        d =  16   n > d    floor 0.218    MEASURED collapse, 2/3 seeds

    So the prediction is a THRESHOLD at d = n = 64, not a monotone trend.
    Dims 16 and 32 generated this hypothesis; dims 24, 48, 64 and 128 test it.

    KILLS THE EXPLANATION: a substantial gain surviving at d = 128, where the
    keys CAN be made orthogonal and e CAN be driven to zero on repeats. That
    would mean the gate is exploiting something other than crosstalk and this
    whole account is wrong.

    python mqar_eval.py --calibrate   # is MQAR learnable at all here?
    python mqar_eval.py               # the sweep
"""
import argparse
import itertools
import time

import numpy as np
import torch
import torch.nn.functional as F

import arms
import mqar_task as M


def masked(lg, tgt, sel):
    return F.cross_entropy(lg[sel], tgt[sel])


def evaluate(m, n_pairs, rewrite, n_batches=6, batch=32, eval_seed=12345):
    # Identical stream for every arm, but NOT the stream the state dims were
    # selected on. Choosing dims and scoring arms on one sample is selection
    # and scoring on the same data; see README "Audit".
    ev = np.random.default_rng(eval_seed)
    ls, ac = [], []
    with torch.no_grad():
        for _ in range(n_batches):
            x, s = M.sample(ev, batch, n_pairs=n_pairs, rewrite=rewrite)
            lg, tgt = m(x[:, :-1]), x[:, 1:]
            ls.append(masked(lg, tgt, s).item())
            ac.append((lg.argmax(-1)[s] == tgt[s]).float().mean().item())
    return float(np.mean(ls)), float(np.mean(ac))


def run(arm, seed, n_pairs, rewrite, steps, dim=128, lr=2e-3, batch=24,
        trace=False, eval_seed=12345):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # d_model stays 128 so embedding capacity is NOT confounded with memory
    # capacity. Only the state S (d_v x d_k) shrinks.
    m = arms.build(arm, M.VOCAB, d_k=dim, d_v=dim)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for i in range(1, steps + 1):
        x, s = M.sample(rng, batch, n_pairs=n_pairs, rewrite=rewrite)
        loss = masked(m(x[:, :-1]), x[:, 1:], s)    # scored positions only
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if trace and (i % 100 == 0 or i == 1):
            l, a = evaluate(m, n_pairs, rewrite, n_batches=2)
            print(f"    step {i:>4}  loss {l:.4f}  acc {a:.4f}", flush=True)
    return evaluate(m, n_pairs, rewrite, eval_seed=eval_seed)


def calibrate(a):
    """A comparison between arms is meaningless if no arm learns the task."""
    print("CALIBRATION. Does a single delta layer learn MQAR here?")
    print(f"chance accuracy = {1/M.N_VAL:.4f}\n")
    print("A comparison needs headroom: an arm at 1.000 or at chance tells")
    print("us nothing. Looking for state dims where accuracy sits mid-range.\n")
    print(f"{'dim':>5}{'pairs':>7}{'variant':>9}{'loss':>9}{'acc':>8}  verdict")
    for d in a.dims:
        for n in a.pairs:
            for rw in a.variants:
                l, ac = run("ours", 0, n, rw, a.steps, dim=d)
                v = ("SATURATED, no headroom" if ac > 0.97 else
                     "AT CHANCE, unusable" if ac < 4 / M.N_VAL else
                     "USABLE")
                print(f"{d:>5}{n:>7}{'rewrite' if rw else 'plain':>9}"
                      f"{l:>9.4f}{ac:>8.4f}  {v}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--pairs", type=int, nargs="+", default=[64])
    ap.add_argument("--dims", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--variants", type=lambda s: s == "rewrite",
                    nargs="+", default=[False, True])
    ap.add_argument("--eval-seed", type=int, default=12345)
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()

    if a.calibrate:
        a.steps = min(a.steps, 400)
        return calibrate(a)

    grid = list(itertools.product(a.dims, a.pairs, a.variants,
                                 arms.ARMS, a.seeds))
    print(f"{len(grid)} runs: {len(a.dims)} state dims x {len(a.pairs)} "
          f"pair counts x {len(a.variants)} variants x {len(arms.ARMS)} arms "
          f"x {len(a.seeds)} seeds, {a.steps} steps"
          f"\nchance acc {1/M.N_VAL:.4f}\n", flush=True)

    res, t0 = {}, time.time()
    for i, (d, n, rw, arm, seed) in enumerate(grid, 1):
        l, ac = run(arm, seed, n, rw, a.steps, dim=d,
                    eval_seed=a.eval_seed)
        res.setdefault((d, n, rw, arm), []).append((l, ac))
        print(f"[{i:>2}/{len(grid)}] dim {d:>3} pairs {n:>3} "
              f"{'rewrite' if rw else 'plain  '} {arm:11s} seed {seed}  "
              f"loss {l:.4f}  acc {ac:.4f}  ({(time.time()-t0)/60:.0f} min)",
              flush=True)

    print("\n\nMQAR RESULTS\n")
    print(f"{'dim':>5}{'pairs':>7}{'variant':>9}{'arm':>12}{'loss':>9}"
          f"{'sd':>8}{'acc':>8}{'sd':>8}{'vs none':>9}")
    for d in a.dims:
        for n in a.pairs:
            for rw in a.variants:
                base = np.mean([x[0] for x in res[(d, n, rw, "none")]])
                for arm in arms.ARMS:
                    v = np.array(res[(d, n, rw, arm)])
                    sd = v.std(0, ddof=1) if len(v) > 1 else np.zeros(2)
                    print(f"{d:>5}{n:>7}{'rewrite' if rw else 'plain':>9}"
                          f"{arm:>12}{v[:,0].mean():>9.4f}{sd[0]:>8.4f}"
                          f"{v[:,1].mean():>8.4f}{sd[1]:>8.4f}"
                          f"{100*(base-v[:,0].mean())/base:>8.2f}%", flush=True)
                print()

    print("\nREAD THIS AS A SCOPE STATEMENT, not a win or a loss.")
    print("  gain ~0 everywhere      the 5.5% is specific to tasks containing")
    print("                          keys with CONFLICTING optimal write")
    print("                          policies. That is a real limit and it")
    print("                          explains the larger study's disagreement.")
    print("  gain grows as dim falls contested addresses are enough on their")
    print("                          own, no constructed task needed. That")
    print("                          would be the strongest result here.")
    print("  token_only tracks ours  the control failed, see convergence.py.")
    print(f"\n{len(a.seeds)} seeds, eval stream {a.eval_seed}.")
    print(f"total {(time.time()-t0)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
