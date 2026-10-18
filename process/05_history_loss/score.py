"""Score every trained arm on ONE eval stream and apply the pre-registered bars.

    python score.py --stream 5150          # PRIMARY
    python score.py --stream 5151          # HELD OUT, only after the verdict

Streams 5150 and 5151 are fresh: neither appears anywhere on main,
claude/optimistic-easley-09c6ab or validation-suite. The repo's reuse hazard is
that 14 files score on default_rng(12345) and 04 used 777 and 888.

6 batches of 32 sequences, matching gated_delta_ssm/train.py's eval. Every arm
sees the IDENTICAL stream.

TWO loss metrics, both fixed before any run (see PREREGISTRATION.md and its
second appendix):

  overall      mean cross-entropy over all positions. THE PRIMARY. Bar
               +0.0064 nats = 2 x SE_diff at the repo's prior seed sd 0.0039.
  NOISY+REV    mean cross-entropy restricted to positions whose QUERY token is
               NOISY or REVISED. SECONDARY and EXPLORATORY: no prior sd exists
               for it, so its bar is computed from the POOLED WITHIN-ARM seed
               sd by the same formula, which is a nuisance-parameter estimate
               and not the contrast of interest.

The restriction matters because the task has absorbing states: 44% of writes
are STABLE repeats where accuracy is already 1.000, so they dilute any
difference in the overall figure.

The class of the QUERY at position t is class(x_t), because the readout uses
q_t from x_t to predict x_{t+1}. That is a different labelling from the class
of the WRITE at position t, which is class(x_{t-1}); both appear here and are
named cls_q and cls_w to keep them apart.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "gated_delta_ssm"))

import task as T                                                  # noqa: E402
import arms as A                                                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "checkpoints")

BAR = 0.0064          # nats, = 2 * 0.0039 * sqrt(2/3)
PRIOR_SD = 0.0039     # gated_delta_ssm/train.py docstring, n=3
VALIDITY_PCT = 4.00   # validation/README.md's own confirmed bar
CLS = ("STABLE", "NOISY", "REVISED")
DEFAULT_ARMS = ["none", "ours", "ours+hist", "ours+rate", "ours+oracle",
                "ours_pub"]
TEST_ARMS = ["ours+hist", "ours+rate", "ours+oracle"]


def load(arm, seed):
    p = os.path.join(CKPT, f"{arm.replace('+', '_')}_seed{seed}.pt")
    if not os.path.exists(p):
        return None, None
    ck = torch.load(p, weights_only=False)
    m = A.build(arm, gate_width=ck.get("gate_width", 32))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


@torch.no_grad()
def evaluate(m, stream, batches=6, batch=32):
    ev = np.random.default_rng(stream)
    ce, cor, bet, cq, cw, sn = [], [], [], [], [], []
    for _ in range(batches):
        x, cl, seen = T.sample(ev, batch)
        tgt = x[:, 1:]
        lg, b = m(x[:, :-1], return_beta=True)
        ce.append(F.cross_entropy(lg.reshape(-1, T.N_SYM), tgt.reshape(-1),
                                  reduction="none").reshape(batch, T.L))
        cor.append((lg.argmax(-1) == tgt).float())
        bet.append(b)
        cq.append(cl)                                   # class of x_t, the query
        cw.append(F.pad(cl, (1, 0))[:, :T.L])           # class of x_{t-1}, write
        sn.append(seen)
    f = lambda z: torch.cat(z).reshape(-1).numpy()
    ce, cor, cq = f(ce), f(cor), f(cq)
    hard = (cq == 1) | (cq == 2)
    return {
        "loss": float(ce.mean()), "acc": float(cor.mean()),
        "loss_hard": float(ce[hard].mean()), "acc_hard": float(cor[hard].mean()),
        "loss_by_class": [float(ce[cq == i].mean()) for i in range(3)],
        "acc_by_class": [float(cor[cq == i].mean()) for i in range(3)],
        "frac_hard": float(hard.mean()),
        "beta_by_class": [float(f(bet)[(f(cw) == i) & f(sn).astype(bool)].mean())
                          for i in range(3)],
    }


def table(res, key, label, bar, arms, note=""):
    print(f"\n{label}   bar {bar:+.4f} nats {note}")
    print(f"  {'arm':13s}{'s0':>9}{'s1':>9}{'s2':>9}{'mean':>9}{'sd':>8}"
          f"{'gain vs ours':>14}{'per-seed gain':>28}{'verdict':>9}")
    base = res.get("ours")
    for arm, r in res.items():
        v = np.array([run[key] for run in r["runs"]])
        sd = v.std(ddof=1) if len(v) > 1 else float("nan")
        cells = "".join(f"{z:>9.4f}" for z in v) + " " * (9 * (3 - len(v)))
        g, ps, verd = "", "", ""
        if base is not None and arm in arms:
            bv = np.array([run[key] for run in base["runs"]])
            gain = bv.mean() - v.mean()
            g = f"{gain:>+14.4f}"
            n = min(len(bv), len(v))
            ps = "".join(f"{bv[i] - v[i]:>+9.4f}" for i in range(n)) \
                + " " * (9 * (3 - n))
            verd = "PASS" if gain >= bar else "KILL"
            r[f"{key}_gain"] = float(gain)
            r[f"{key}_verdict"] = verd
        else:
            g, ps = " " * 14, " " * 27
        print(f"  {arm:13s}{cells}{v.mean():>9.4f}{sd:>8.4f}{g}{ps}{verd:>9}")
        r[f"{key}_mean"], r[f"{key}_sd"] = float(v.mean()), float(sd)


def pooled_sd(res, key):
    """Within-arm seed sd pooled across arms: a nuisance-parameter estimate,
    not the contrast of interest."""
    sq, df = 0.0, 0
    for r in res.values():
        v = np.array([run[key] for run in r["runs"]])
        if len(v) > 1:
            sq += ((v - v.mean()) ** 2).sum()
            df += len(v) - 1
    return float(np.sqrt(sq / df)) if df else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    print(f"EVAL STREAM {a.stream}   seeds {a.seeds}   "
          f"6 batches x 32 sequences")
    res = {}
    for arm in a.arms:
        runs = []
        for s in a.seeds:
            m, ck = load(arm, s)
            if m is None:
                continue
            r = evaluate(m, a.stream)
            r.update(seed=s, params=ck["params"],
                     secs=round(ck["train_secs"], 1),
                     clamp=ck["h_ratio_clamp_rate"])
            runs.append(r)
        if runs:
            res[arm] = {"runs": runs}
    if not res:
        print("no checkpoints found")
        return
    print(f"fraction of positions with a NOISY or REVISED query: "
          f"{res[list(res)[0]]['runs'][0]['frac_hard']:.4f}")
    print(f"\n{'arm':13s}{'params':>9}{'train min':>11}")
    for arm, r in res.items():
        print(f"{arm:13s}{r['runs'][0]['params']:>9,}"
              f"{np.mean([x['secs'] for x in r['runs']]) / 60:>11.1f}")

    table(res, "loss", "PRIMARY: overall eval loss (nats)", BAR, TEST_ARMS,
          "(pre-registered, from the repo's prior seed sd 0.0039)")

    psd = pooled_sd(res, "loss_hard")
    bar_hard = 2 * psd * np.sqrt(2 / 3)
    table(res, "loss_hard",
          "SECONDARY, EXPLORATORY: loss on NOISY+REVISED queries only",
          bar_hard, TEST_ARMS,
          f"(from pooled within-arm sd {psd:.4f}, computed not asserted)")

    psd_o = pooled_sd(res, "loss")
    if np.isnan(psd_o):
        print("\nCROSS-CHECK on the primary: not available, needs >1 seed")
    else:
        imp = 2 * psd_o * np.sqrt(2 / 3)
        print(f"\nCROSS-CHECK on the primary: pooled within-arm sd {psd_o:.4f} "
              f"-> implied bar {imp:+.4f} vs the pre-registered {BAR:+.4f}")
        print("  the pre-registered bar is "
              + ("CONSERVATIVE: this run was QUIETER than the prior sd, so the "
                 "bar is harder to pass than the data requires"
                 if BAR >= imp else
                 "OPTIMISTIC: this run was NOISIER than the prior sd, so the "
                 "test is UNDERPOWERED. Reported, bar NOT moved."))

    print("\nVALIDITY GATE: the masked `ours` must reproduce the published effect")
    if "none" in res and "ours" in res:
        n, o = res["none"]["loss_mean"], res["ours"]["loss_mean"]
        pct = 100 * (n - o) / n
        print(f"  none {n:.4f}  ours {o:.4f}  -> {pct:+.2f}%   "
              f"bar >= {VALIDITY_PCT:.2f}%   "
              f"{'PASS' if pct >= VALIDITY_PCT else 'VOID: the harness is broken'}")
        res["_validity"] = {"pct": float(pct), "pass": bool(pct >= VALIDITY_PCT)}

    print(f"\n{'arm':13s}{'acc':>8}{'acc hard':>10}   "
          f"loss by class {CLS}")
    for arm, r in res.items():
        if arm.startswith("_"):
            continue
        lb = np.mean([x["loss_by_class"] for x in r["runs"]], 0)
        print(f"{arm:13s}{np.mean([x['acc'] for x in r['runs']]):>8.4f}"
              f"{np.mean([x['acc_hard'] for x in r['runs']]):>10.4f}   "
              + "".join(f"{z:>9.4f}" for z in lb))

    print(f"\n{'arm':13s}mean beta by key class on seen writes "
          f"{CLS}")
    for arm, r in res.items():
        if arm.startswith("_"):
            continue
        b = np.mean([x["beta_by_class"] for x in r["runs"]], 0)
        print(f"{arm:13s}" + "".join(f"{z:>10.3f}" for z in b))

    p = os.path.join(HERE, f"results_{a.stream}.json")
    with open(p, "w") as fh:
        json.dump({"stream": a.stream, "seeds": a.seeds, "bar": BAR,
                   "prior_sd": PRIOR_SD, "arms": res}, fh, indent=1,
                  default=float)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
