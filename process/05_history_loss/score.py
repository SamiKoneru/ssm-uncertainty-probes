"""Score every trained arm on ONE eval stream and apply the pre-registered bars.

    python score.py --stream 5150          # PRIMARY
    python score.py --stream 5151          # HELD OUT, only after the verdict

Streams 5150 and 5151 are fresh: neither appears anywhere on main,
claude/optimistic-easley-09c6ab or validation-suite. The repo's reuse hazard is
that 14 files score on default_rng(12345) and 04 used 777 and 888.

6 batches of 32 sequences, matching gated_delta_ssm/train.py's eval. Every arm
sees the IDENTICAL stream.

Bars, from PREREGISTRATION.md, fixed before any run:

    BAR       mean_loss(ours) - mean_loss(arm) >= +0.0064 nats
              = 2 x SE_diff at the repo's measured seed sd of 0.0039, n=3
    VALIDITY  mean_loss(none) - mean_loss(ours) >= 4.00% of mean_loss(none)
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
    """Eval loss/acc plus beta tagged by key class and the task's seen mask."""
    ev = np.random.default_rng(stream)
    losses, accs, betas, cls_w, seen = [], [], [], [], []
    for _ in range(batches):
        x, cl, sn = T.sample(ev, batch)
        lg, b = m(x[:, :-1], return_beta=True)
        losses.append(F.cross_entropy(lg.reshape(-1, T.N_SYM),
                                      x[:, 1:].reshape(-1)).item())
        accs.append((lg.argmax(-1) == x[:, 1:]).float().mean().item())
        betas.append(b)
        # beta_t writes to the key from token t-1, so label it with that
        cls_w.append(F.pad(cl, (1, 0))[:, :T.L])
        seen.append(sn)
    f = lambda z: torch.cat(z).reshape(-1).numpy()
    return (float(np.mean(losses)), float(np.mean(accs)),
            f(betas), f(cls_w), f(seen).astype(bool))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+",
                    default=["none", "ours", "ours+hist", "ours+rate",
                             "ours_pub"])
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    print(f"EVAL STREAM {a.stream}   seeds {a.seeds}   "
          f"6 batches x 32 sequences\n")
    res = {}
    for arm in a.arms:
        L, Ac, meta = [], [], []
        for s in a.seeds:
            m, ck = load(arm, s)
            if m is None:
                continue
            loss, acc, beta, clw, sn = evaluate(m, a.stream)
            L.append(loss); Ac.append(acc)
            meta.append({"seed": s, "loss": loss, "acc": acc,
                         "params": ck["params"],
                         "secs": round(ck["train_secs"], 1),
                         "clamp": ck["h_ratio_clamp_rate"],
                         "beta_by_class": [
                             float(beta[(clw == i) & sn].mean())
                             for i in range(3)]})
        if L:
            res[arm] = {"loss": L, "acc": Ac, "runs": meta}

    print(f"{'arm':12s}{'params':>8}{'loss s0':>10}{'s1':>9}{'s2':>9}"
          f"{'mean':>9}{'sd':>8}{'acc':>9}{'min/step':>10}")
    for arm, r in res.items():
        L = np.array(r["loss"]); Ac = np.array(r["acc"])
        sd = L.std(ddof=1) if len(L) > 1 else float("nan")
        cells = "".join(f"{v:>9.4f}" for v in L) + " " * (9 * (3 - len(L)))
        secs = np.mean([m["secs"] for m in r["runs"]])
        print(f"{arm:12s}{r['runs'][0]['params']:>8,}{cells}"
              f"{L.mean():>9.4f}{sd:>8.4f}{Ac.mean():>9.4f}"
              f"{secs / 60:>10.1f}")
        r["mean"], r["sd"], r["acc_mean"] = float(L.mean()), float(sd), \
            float(Ac.mean())

    out = {"stream": a.stream, "seeds": a.seeds, "bar": BAR,
           "prior_sd": PRIOR_SD, "arms": res}

    # ------------------------------------------------------- validity gate
    print("\nVALIDITY GATE (pre-registered): the masked 8-input `ours` must "
          "reproduce the published effect")
    if "none" in res and "ours" in res:
        n, o = res["none"]["mean"], res["ours"]["mean"]
        pct = 100 * (n - o) / n
        ok = pct >= VALIDITY_PCT
        print(f"  none {n:.4f}  ours {o:.4f}  -> {pct:+.2f}%   "
              f"bar >= {VALIDITY_PCT:.2f}%   "
              f"{'PASS' if ok else 'VOID: the harness is broken'}")
        out["validity"] = {"pct": pct, "pass": bool(ok)}

    # -------------------------------------------------------- primary bars
    print(f"\nPRIMARY BARS (pre-registered): gain over `ours` must be "
          f">= {BAR:+.4f} nats")
    print(f"  {'arm':12s}{'mean':>9}{'gain vs ours':>15}{'per-seed gain':>26}"
          f"{'verdict':>12}")
    if "ours" in res:
        base = res["ours"]
        for arm in ("ours+hist", "ours+rate"):
            if arm not in res:
                continue
            g = base["mean"] - res[arm]["mean"]
            paired = [base["loss"][i] - res[arm]["loss"][i]
                      for i in range(min(len(base["loss"]),
                                         len(res[arm]["loss"])))]
            v = "PASS" if g >= BAR else "KILL"
            print(f"  {arm:12s}{res[arm]['mean']:>9.4f}{g:>+15.4f}"
                  + "".join(f"{p:>+9.4f}" for p in paired)
                  + " " * (9 * (3 - len(paired))) + f"{v:>12}")
            res[arm]["gain_vs_ours"] = float(g)
            res[arm]["paired"] = [float(p) for p in paired]
            res[arm]["verdict"] = v

    # ------------------------------------------- realized sd vs the prior
    print("\nREALIZED seed sd vs the 0.0039 the bar was derived from")
    for arm, r in res.items():
        if np.isnan(r["sd"]):
            continue
        se_d = r["sd"] * np.sqrt(2 / 3)
        print(f"  {arm:12s} sd {r['sd']:.4f}   implied 2xSE_diff "
              f"{2 * se_d:+.4f}   "
              f"{'bar was FAIR or generous' if r['sd'] <= PRIOR_SD else 'bar was OPTIMISTIC: underpowered'}")

    print("\nmean beta by key class on seen writes (STABLE / NOISY / REVISED)")
    for arm, r in res.items():
        b = np.array([m["beta_by_class"] for m in r["runs"]]).mean(0)
        print(f"  {arm:12s}{b[0]:>9.3f}{b[1]:>9.3f}{b[2]:>9.3f}")

    p = os.path.join(HERE, f"results_{a.stream}.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
