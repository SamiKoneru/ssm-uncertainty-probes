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
DEFAULT_ARMS = ["none", "ours", "ours+hist", "ours+rate", "ours+cheap",
                "ours+oracle", "ours+bayes", "ours+nvis", "ours+fresh",
                "ours_pub"]
# ours+nvis and ours+fresh were added after seeing seed 0, as CONTROLS that
# can only explain the headline gain away. Flagged in every table.
# ours+cheap is ALSO post-hoc but is NOT of that kind: it could support a
# positive feasibility claim, so on this stream it is exploratory and its
# verdict is not final until stream 7771 reproduces it. See
# PREREGISTRATION_cheap.md.
TEST_ARMS = ["ours+hist", "ours+rate", "ours+cheap", "ours+oracle",
             "ours+bayes", "ours+nvis", "ours+fresh"]
POST_HOC = {"ours+nvis", "ours+fresh", "ours+cheap"}
# Bar 2 for ours+cheap only: 50% of ours+hist's measured +0.0551. Fixed in
# PREREGISTRATION_cheap.md before the arm existed.
SUFFICIENCY_BAR = 0.0276
SUFFICIENCY_REF = "ours+hist"
# Identifiability is steeply visit-count dependent: the exact Bayes AUROC for
# NOISY vs REVISED is 0.687 at 1 prior write to the address, 0.786 at 2, 0.890
# at 3-4, 0.977 at 5-8 and 0.9997 at 9 or more. So a gain, if one exists, need
# not be uniform, and a flat null is a different result from a null overall
# with a gain in an identifiable slice. Reported, no bar: EXPLORATORY.
VISIT_BINS = [(0, 0, "0 (fresh)"), (1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"),
              (5, 8, "5-8"), (9, 10 ** 9, "9+")]


def query_visits(tok):
    """[B, L] tokens -> [B, L] prior writes to the QUERY address.

    The readout at position t uses q_t from x_t to predict x_{t+1}, so what
    matters for that prediction is how many times x_t has already been written
    to AS AN ADDRESS, i.e. how often x_t appeared at some position before t.
    """
    B, L = tok.shape
    out = np.zeros((B, L), np.int64)
    for b in range(B):
        c = {}
        row = tok[b]
        for t in range(L):
            out[b, t] = c.get(int(row[t]), 0)
            if t >= 1:
                a = int(row[t - 1])
                c[a] = c.get(a, 0) + 1
    return out


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
    ce, cor, bet, cq, cw, sn, qv = [], [], [], [], [], [], []
    for _ in range(batches):
        x, cl, seen = T.sample(ev, batch)
        tgt = x[:, 1:]
        qv.append(torch.from_numpy(query_visits(x[:, :-1].numpy())))
        lg, b = m(x[:, :-1], return_beta=True)
        ce.append(F.cross_entropy(lg.reshape(-1, T.N_SYM), tgt.reshape(-1),
                                  reduction="none").reshape(batch, T.L))
        cor.append((lg.argmax(-1) == tgt).float())
        bet.append(b)
        cq.append(cl)                                   # class of x_t, the query
        cw.append(F.pad(cl, (1, 0))[:, :T.L])           # class of x_{t-1}, write
        sn.append(seen)
    f = lambda z: torch.cat(z).reshape(-1).numpy()
    ce, cor, cq, qv = f(ce), f(cor), f(cq), f(qv)
    hard = (cq == 1) | (cq == 2)
    by_visit = {}
    for lo, hi, nm in VISIT_BINS:
        m = hard & (qv >= lo) & (qv <= hi)
        by_visit[nm] = [float(ce[m].mean()) if m.sum() else float("nan"),
                        int(m.sum())]
    return {
        "loss_by_visit": by_visit,
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
    print("  (* = control arm added after seeing seed 0; can only explain "
          "the gain away)")
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
        flag = " *" if arm in POST_HOC else ""
        print(f"  {arm:13s}{cells}{v.mean():>9.4f}{sd:>8.4f}{g}{ps}"
              f"{verd:>9}{flag}")
        r[f"{key}_mean"], r[f"{key}_sd"] = float(v.mean()), float(sd)


def cheap_verdict(res, stream):
    """Apply the TWO pre-registered bars for `ours+cheap` and name the outcome.

    Bar 1 is the repo's existing BAR: does the cheap block help at all.
    Bar 2 is SUFFICIENCY_BAR: does it keep at least half of what the full
    five-feature block gets, which is the question that decides whether the
    d_k x d_v residual-direction state can be deleted.

    Both were fixed in PREREGISTRATION_cheap.md before the arm existed. This
    function only compares, it never chooses a bar.
    """
    need = ("ours", "ours+cheap", SUFFICIENCY_REF)
    if any(k not in res for k in need):
        print(f"\nCHEAP-ARM VERDICT: skipped, needs {need}")
        return
    base = res["ours"]["loss_mean"]
    gain = base - res["ours+cheap"]["loss_mean"]
    gain_ref = base - res[SUFFICIENCY_REF]["loss_mean"]
    kept = gain / gain_ref if gain_ref else float("nan")
    print("\nCHEAP-ARM VERDICT   (PREREGISTRATION_cheap.md, bars fixed before "
          "the arm existed)")
    print(f"  ours        {base:.4f}")
    print(f"  ours+cheap  {res['ours+cheap']['loss_mean']:.4f}   "
          f"gain {gain:+.4f}")
    print(f"  {SUFFICIENCY_REF:11s} {res[SUFFICIENCY_REF]['loss_mean']:.4f}   "
          f"gain {gain_ref:+.4f}   <- the full five-feature block")
    b1 = gain >= BAR
    b2 = gain >= SUFFICIENCY_BAR
    print(f"  bar 1 existence     gain >= {BAR:+.4f}   "
          f"{'PASS' if b1 else 'KILL'}")
    print(f"  bar 2 sufficiency   gain >= {SUFFICIENCY_BAR:+.4f}   "
          f"{'PASS' if b2 else 'FAIL'}   "
          f"(50% of {SUFFICIENCY_REF}'s +0.0551)")
    print(f"  fraction of the full block's gain retained: {100 * kept:.1f}%")
    if b2:
        out = ("VIABLE. The d_k-only features carry the bulk of it, so the "
               "d_k x d_v direction state can be dropped. Next: the "
               "chunk-frozen version, measured against this as its ceiling.")
    elif b1:
        out = ("REAL BUT MINORITY. The direction features carry most of the "
               "gain; the d_k x d_v cost stays load-bearing and feasibility "
               "stays open.")
    else:
        out = ("KILL. Worthless alone AND worthless together: "
               "superadditivity does not rescue the cheap block, the "
               "direction features are necessary, and the d_k x d_v cost is "
               "unavoidable.")
    print(f"  OUTCOME: {out}")
    status = ("CONFIRMATION" if stream == 7771 else
              "EXPLORATORY (this arm was specified after reading 5150's "
              "ablation; not final until 7771 reproduces it)")
    print(f"  STATUS: {status}")
    res["_cheap"] = {"gain": float(gain), "gain_ref": float(gain_ref),
                     "kept": float(kept), "bar1": bool(b1), "bar2": bool(b2),
                     "stream": int(stream), "confirmation": stream == 7771}


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

    cheap_verdict(res, a.stream)

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

    # Every gain above is measured against `ours`, the param-matched masked
    # control. If masking 6 columns handicaps it relative to the literal
    # shipped 3-input gate, those gains are inflated by exactly that much, so
    # the same contrast is shown against ours_pub.
    if "ours" in res and "ours_pub" in res:
        o, pub = res["ours"]["loss_mean"], res["ours_pub"]["loss_mean"]
        print(f"\nBASELINE CHECK: ours {o:.4f} vs ours_pub {pub:.4f} "
              f"-> masking the 6 columns costs {o - pub:+.4f} nats")
        print(f"  {'arm':13s}{'vs ours':>10}{'vs ours_pub':>14}"
              f"{'inflation':>11}")
        for arm in TEST_ARMS:
            if arm not in res:
                continue
            v = res[arm]["loss_mean"]
            print(f"  {arm:13s}{o - v:>+10.4f}{pub - v:>+14.4f}"
                  f"{(o - v) - (pub - v):>+11.4f}")

    print("\nEXPLORATORY, no bar: loss on NOISY+REVISED queries, sliced by "
          "prior writes to the QUERY address")
    print("  (Bayes AUROC for the class at those counts: 0.687 / 0.786 / "
          "0.890 / 0.977 / 0.9997 at 1 / 2 / 3-4 / 5-8 / 9+)")
    names = [nm for _, _, nm in VISIT_BINS]
    print(f"  {'arm':13s}" + "".join(f"{nm:>11}" for nm in names))
    print(f"  {'n per seed':13s}"
          + "".join(f"{res[list(res)[0]]['runs'][0]['loss_by_visit'][nm][1]:>11,}"
                    for nm in names))
    base_v = None
    for arm, r in res.items():
        if arm.startswith("_"):
            continue
        v = [float(np.mean([x["loss_by_visit"][nm][0] for x in r["runs"]]))
             for nm in names]
        if arm == "ours":
            base_v = v
        print(f"  {arm:13s}" + "".join(f"{z:>11.4f}" for z in v))
    if base_v is not None:
        print(f"\n  gain vs ours (positive = history helped)")
        for arm, r in res.items():
            if arm.startswith("_") or arm not in TEST_ARMS:
                continue
            v = [float(np.mean([x["loss_by_visit"][nm][0] for x in r["runs"]]))
                 for nm in names]
            print(f"  {arm:13s}"
                  + "".join(f"{base_v[i] - v[i]:>+11.4f}"
                            for i in range(len(names))))

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
