"""The SHAPE of beta, not its mean. See PREREGISTRATION_beta_shape.md.

    python beta_shape.py --streams 5150 3310

`ours+hist` wins +0.0578 nats while its MEAN beta collapses to 0.001/0.002/0.002
with no class ordering at all, and `ours+bayes`, the one arm whose ordering is
correct, gains less. A mean of 0.002 is produced equally well by two policies
that are nothing alike:

    FLAT    beta sits near 0.002 everywhere. Then the gain is not coming from
            write decisions at all and the 04/05 framing is wrong.
    SPIKY   beta is ~0 almost everywhere and near 1 on a small fraction of
            writes. Then the algorithm is write-once plus rare overwrite, which
            is a different algorithm from per-class rate tuning.

Four metrics, all fixed before any number was seen:

  shape          quantiles of beta, frac(beta>0.5), frac(beta>0.9)
  concentration  top1_share, the share of total write mass in the largest 1% of
                 beta. Scale-free: a constant gate gives exactly 0.01, all the
                 mass in 1% of positions gives 1.00. This is what compares arms
                 whose mean beta differs by 100x.
  targeting      enrichment of (REVISED & change) among those top-1% positions
  identifiability  the same enrichment within each visit-count bin, taken over
                 the top 1% WITHIN that bin so the bins stay comparable

`change` is the one position where the two volatile classes want OPPOSITE
actions: for REVISED the stored value just became wrong (overwrite), for NOISY
it is a fresh draw from the same fixed option set and old observations are still
valid (keep averaging). So a targeted policy spikes on REVISED & change
specifically, not on change in general.

All conditioning is on the WRITE at step t, which sends the value from token x_t
to the address from token x_{t-1}. That is one position off from the query
labelling score.py uses for loss and the two must not be mixed. Position t=0 is
dropped: its key comes from the zero pad, not from a token.

VALIDITY: beta collected here must reproduce score.py's published
`beta_by_class` from results_<stream>.json to 1e-6, using score.py's own masking
convention. Asserted, not eyeballed. No reference JSON exists for a fresh
stream, so there the assert is skipped and said to be skipped.

EXPLORATORY and POST-HOC: designed after seeing the mean-beta table, and it
tests no loss claim.
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
import score as SC                                                # noqa: E402
import beta_report as BR                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ("STABLE", "NOISY", "REVISED")
WTN = ("unseen", "repeat", "change")
ARMS = ["none", "ours", "ours+hist", "ours+bayes", "ours+cheap",
        "ours+oracle", "ours_pub"]
SEEDS = [0, 1, 2]
BATCHES = 6          # matches score.py; the validity assert needs this exact value
TOP = 0.01                     # the top 1%

# bars, from PREREGISTRATION_beta_shape.md, fixed before the first number
SPIKY_SHARE, SPIKY_FRAC = 0.25, 0.001
FLAT_SHARE, FLAT_MAX = 0.10, 0.50
TARGETED_BAR, UNTARGETED_BAR = 2.0, 1.2
SUBJECT = "ours+hist"
# the identifiability bins, copied from score.py so the Bayes AUROC per bin
# (0.687 / 0.786 / 0.890 / 0.977 / 0.9997) lines up with them. Bin 0 is dropped:
# a fresh address has no history, so REVISED & change cannot occur in it.
VISIT_BINS = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 8, "5-8"),
              (9, 10 ** 9, "9+")]
BIN_AUROC = {"1": 0.687, "2": 0.786, "3-4": 0.890, "5-8": 0.977, "9+": 0.9997}


def write_visits(tok):
    """[B, L] tokens -> [B, L] prior writes to the address written at step t.

    The address written at t is x_{t-1}, so this counts how many of the writes
    at steps 1..t-1 went to that same symbol. Written independently of
    beta_report.write_types and asserted against it: a write is `unseen`
    exactly where this is 0.
    """
    B, L = tok.shape
    out = np.zeros((B, L), np.int64)
    for b in range(B):
        c = np.zeros(T.N_SYM, np.int64)
        row = tok[b]
        for t in range(1, L):
            a = int(row[t - 1])
            out[b, t] = c[a]
            c[a] += 1
    return out


@torch.no_grad()
def gather(m, stream, batches=6, batch=32):
    """One arm, one stream -> flat per-position arrays.

    Consumes the rng exactly as score.evaluate does (same batches, same batch
    size, same order), so the positions line up with the published numbers.
    """
    ev = np.random.default_rng(stream)
    cols = {k: [] for k in ("beta", "cls_w", "wt", "vis", "seen")}
    for _ in range(batches):
        x, cl, seen = T.sample(ev, batch)
        tok = x[:, :-1]                        # [B, L], score.py's own input
        _, beta = m(tok, return_beta=True)
        tn = tok.numpy()
        cols["beta"].append(beta.numpy())
        cols["cls_w"].append(F.pad(cl, (1, 0))[:, :T.L].numpy())
        cols["wt"].append(BR.write_types(tn))
        cols["vis"].append(write_visits(tn))
        cols["seen"].append(seen.numpy())
    d = {k: np.concatenate(v).reshape(-1) for k, v in cols.items()}
    unseen = d["wt"] == BR.UNSEEN
    assert np.array_equal(unseen, d["vis"] == 0), \
        "write_types and write_visits disagree on which writes are fresh"
    return d


def shape_stats(beta):
    q = np.quantile(beta, [0.5, 0.9, 0.99, 0.999])
    order = np.argsort(beta)[::-1]
    n_top = max(1, int(round(TOP * beta.size)))
    top = order[:n_top]
    mass = beta.sum()
    return {
        "mean": float(beta.mean()), "p50": float(q[0]), "p90": float(q[1]),
        "p99": float(q[2]), "p999": float(q[3]), "max": float(beta.max()),
        "frac_gt_50": float((beta > 0.5).mean()),
        "frac_gt_90": float((beta > 0.9).mean()),
        "top1_share": float(beta[top].sum() / mass) if mass > 0 else float("nan"),
        "n_top": int(n_top), "top_idx": top,
    }


def enrichment(beta, target):
    """P(target | top 1% of beta) / P(target).  target is a bool mask."""
    base = target.mean()
    n_top = max(1, int(round(TOP * beta.size)))
    top = np.argsort(beta)[::-1][:n_top]
    hit = target[top].mean()
    return (float(hit / base) if base > 0 else float("nan"),
            float(hit), float(base), int(target[top].sum()), int(n_top))


def run_stream(stream, threads):
    print("=" * 78)
    print(f"STREAM {stream}")
    out = {"stream": stream, "arms": {}}
    ref = os.path.join(HERE, f"results_{stream}.json")
    pub = json.load(open(ref))["arms"] if os.path.exists(ref) else None
    if pub and BATCHES != 6:
        # a reduced batch count reads a PREFIX of the stream, so the means
        # cannot match the published ones. Disable the check rather than let
        # it fail for the wrong reason, and say so.
        print(f"validity reference: results_{stream}.json EXISTS but "
              f"--batches {BATCHES} != 6 reads only a prefix of the stream, "
              f"so the assert is DISABLED. This run is a smoke test, not a "
              f"result.")
        pub = None
    else:
        print(f"validity reference: "
              f"{'results_%d.json' % stream if pub else 'NONE (fresh stream, assert skipped)'}")

    data = {}
    worst = 0.0
    for arm in ARMS:
        for sd in SEEDS:
            m, _ = SC.load(arm, sd)
            if m is None:
                print(f"  {arm} seed {sd}: no checkpoint, skipped")
                continue
            d = gather(m, stream, BATCHES)
            data[(arm, sd)] = d
            if pub and arm in pub:
                # score.py's own convention: class of x_{t-1} AND the task's
                # `seen` flag for x_t. Reproduced exactly, not approximated.
                mine = [float(d["beta"][(d["cls_w"] == i)
                                        & d["seen"].astype(bool)].mean())
                        for i in range(3)]
                run = [r for r in pub[arm]["runs"] if r["seed"] == sd]
                if run:
                    diff = max(abs(a - b) for a, b in
                               zip(mine, run[0]["beta_by_class"]))
                    worst = max(worst, diff)
    if pub:
        print(f"VALIDITY: beta_by_class reproduced, max |diff| {worst:.2e}", end="  ")
        assert worst < 1e-6, (f"beta collection does not reproduce the "
                              f"published beta_by_class (max |diff| {worst:.2e}); "
                              f"this report is VOID")
        print("PASS")
    print()

    # positions: every write with t >= 1. t=0's key is the zero pad.
    def keep(d):
        pos = np.ones(d["beta"].size, bool)
        pos[::T.L] = False
        return pos

    st = {}
    for (arm, sd), d in data.items():
        p = keep(d)
        st[(arm, sd)] = shape_stats(d["beta"][p])
    n_pos = int(keep(next(iter(data.values()))).sum())
    print(f"writes scored per seed: {n_pos:,}   "
          f"top 1% = {max(1, int(round(TOP * n_pos)))} positions")

    print(f"\nSHAPE of beta over those writes")
    print(f"  {'arm':13s}{'seed':>5}{'mean':>9}{'p50':>9}{'p90':>9}{'p99':>9}"
          f"{'p99.9':>9}{'max':>9}{'f>0.5':>9}{'f>0.9':>9}")
    for arm in ARMS:
        for sd in SEEDS:
            s = st.get((arm, sd))
            if s is None:
                continue
            print(f"  {arm:13s}{sd:>5}{s['mean']:>9.4f}{s['p50']:>9.4f}"
                  f"{s['p90']:>9.4f}{s['p99']:>9.4f}{s['p999']:>9.4f}"
                  f"{s['max']:>9.4f}{s['frac_gt_50']:>9.4f}"
                  f"{s['frac_gt_90']:>9.4f}")

    print(f"\nCONCENTRATION: top1_share, the share of total write mass in the "
          f"largest 1% of beta")
    print(f"  a constant gate gives exactly {TOP:.4f}; all mass in 1% of "
          f"positions gives 1.0000")
    print(f"  {'arm':13s}{'s0':>9}{'s1':>9}{'s2':>9}{'mean':>9}")
    for arm in ARMS:
        v = [st[(arm, sd)]["top1_share"] for sd in SEEDS if (arm, sd) in st]
        if not v:
            continue
        cells = "".join(f"{z:>9.4f}" for z in v) + " " * (9 * (3 - len(v)))
        print(f"  {arm:13s}{cells}{np.mean(v):>9.4f}")

    print(f"\nTARGETING: enrichment of (REVISED & change) among the top 1% of "
          f"beta")
    print(f"  enrich = P(REVISED & change | top 1%) / P(REVISED & change)."
          f"  1.00 = blind selection")
    print(f"  {'arm':13s}{'s0':>9}{'s1':>9}{'s2':>9}{'mean':>9}"
          f"{'base rate':>11}{'n in top1%':>12}")
    enr = {}
    for arm in ARMS:
        row, base, cnt = [], float("nan"), []
        for sd in SEEDS:
            d = data.get((arm, sd))
            if d is None:
                continue
            p = keep(d)
            tgt = ((d["cls_w"] == 2) & (d["wt"] == BR.CHANGE))[p]
            e, _, base, k, _ = enrichment(d["beta"][p], tgt)
            row.append(e)
            cnt.append(k)
        if not row:
            continue
        enr[arm] = row
        cells = "".join(f"{z:>9.3f}" for z in row) + " " * (9 * (3 - len(row)))
        print(f"  {arm:13s}{cells}{np.mean(row):>9.3f}{base:>11.4f}"
              f"{'/'.join(str(c) for c in cnt):>12}")

    print(f"\nIDENTIFIABILITY: the same enrichment WITHIN each visit-count bin")
    print(f"  top 1% taken within the bin, so bins stay comparable. Bayes "
          f"AUROC for NOISY vs REVISED at that count in brackets.")
    hdr = "".join(f"{nm + ' (' + str(BIN_AUROC[nm]) + ')':>16}"
                  for _, _, nm in VISIT_BINS)
    print(f"  {'arm':13s}{hdr}")
    for arm in ARMS:
        cells = ""
        for lo, hi, nm in VISIT_BINS:
            vals = []
            for sd in SEEDS:
                d = data.get((arm, sd))
                if d is None:
                    continue
                p = keep(d) & (d["vis"] >= lo) & (d["vis"] <= hi)
                if p.sum() < 500:
                    continue
                tgt = ((d["cls_w"] == 2) & (d["wt"] == BR.CHANGE))[p]
                e, _, _, _, _ = enrichment(d["beta"][p], tgt)
                vals.append(e)
            cells += f"{np.mean(vals):>16.3f}" if vals else f"{'-':>16}"
        if arm in enr:
            print(f"  {arm:13s}{cells}")

    print(f"\nMEAN vs p99 by class of the write address x write type")
    print(f"  the mean/shape contrast in one place: a flat policy has p99 "
          f"close to the mean, a spiky one does not")
    print(f"  {'arm':13s}{'':9}" + "".join(f"{nm:>22}" for nm in WTN))
    print(f"  {'':13s}{'':9}" + "".join(f"{'mean     p99       n':>22}"
                                        for _ in WTN))
    for arm in ARMS:
        for i, nm in enumerate(NAMES):
            row = ""
            for code in range(3):
                vals, p99s, ns = [], [], []
                for sd in SEEDS:
                    d = data.get((arm, sd))
                    if d is None:
                        continue
                    p = keep(d) & (d["cls_w"] == i) & (d["wt"] == code)
                    if p.sum() < 50:
                        continue
                    b = d["beta"][p]
                    vals.append(b.mean())
                    p99s.append(np.quantile(b, 0.99))
                    ns.append(p.sum())
                row += (f"{np.mean(vals):>8.3f}{np.mean(p99s):>8.3f}"
                        f"{int(np.mean(ns)):>6d}" if vals else f"{'-':>22}")
            if (arm, SEEDS[0]) in data:
                print(f"  {arm:13s}{nm:9s}{row}")

    # ---- the pre-registered verdict, subject arm only -------------------
    print(f"\nVERDICT for {SUBJECT}   (bars from PREREGISTRATION_beta_shape.md)")
    share = [st[(SUBJECT, sd)]["top1_share"] for sd in SEEDS
             if (SUBJECT, sd) in st]
    fgt = [st[(SUBJECT, sd)]["frac_gt_50"] for sd in SEEDS
           if (SUBJECT, sd) in st]
    mx = [st[(SUBJECT, sd)]["max"] for sd in SEEDS if (SUBJECT, sd) in st]
    e = enr.get(SUBJECT, [])
    spiky = (all(z >= SPIKY_SHARE for z in share)
             and all(z >= SPIKY_FRAC for z in fgt))
    flat = all(z <= FLAT_SHARE for z in share) and all(z <= FLAT_MAX for z in mx)
    shape_v = "SPIKY" if spiky else "FLAT" if flat else "INTERMEDIATE"
    targ = ("TARGETED" if all(z >= TARGETED_BAR for z in e)
            else "UNTARGETED" if all(z <= UNTARGETED_BAR for z in e)
            else "INTERMEDIATE")
    print(f"  top1_share   {' '.join(f'{z:.4f}' for z in share)}"
          f"   SPIKY needs all >= {SPIKY_SHARE}, FLAT needs all <= {FLAT_SHARE}")
    print(f"  frac>0.5     {' '.join(f'{z:.4f}' for z in fgt)}"
          f"   SPIKY needs all >= {SPIKY_FRAC}")
    print(f"  max beta     {' '.join(f'{z:.4f}' for z in mx)}"
          f"   FLAT needs all <= {FLAT_MAX}")
    print(f"  enrichment   {' '.join(f'{z:.3f}' for z in e)}"
          f"   TARGETED needs all >= {TARGETED_BAR}, UNTARGETED all "
          f"<= {UNTARGETED_BAR}")
    print(f"  SHAPE {shape_v}   TARGETING {targ}")
    read = {("SPIKY", "TARGETED"):
            "write-once plus targeted overwrite. Reframes the project and "
            "motivates a cheap trigger.",
            ("SPIKY", "UNTARGETED"):
            "the spikes are NOT about volatility class. An anomaly to chase, "
            "NOT a licence to build a trigger.",
            }.get((shape_v, targ))
    if shape_v == "FLAT":
        read = ("the gain is NOT coming from write decisions. The 04/05 "
                "framing is wrong, and this is the most consequential outcome.")
    print(f"  READING: {read or 'no pre-registered reading for this pair; '
                                'reported, claims nothing.'}")
    print(f"  REFRAME: {'SURVIVES' if (shape_v, targ) == ('SPIKY', 'TARGETED') else 'KILLED' if shape_v == 'FLAT' or targ == 'UNTARGETED' else 'UNDECIDED'}")

    # calibration note the pre-registration demands
    ns = np.mean([st[("none", sd)]["top1_share"] for sd in SEEDS
                  if ("none", sd) in st])
    print(f"\n  calibration: `none` has NO memory conditioning and its "
          f"top1_share is {ns:.4f}.")
    if ns >= SPIKY_SHARE:
        print(f"  WARNING: that is at or above the SPIKY bar {SPIKY_SHARE}, so "
              f"the concentration bar does not separate memory-conditioned "
              f"gating from token-only gating. Said out loud as required.")

    for arm in ARMS:
        out["arms"][arm] = {
            "top1_share": [st[(arm, sd)]["top1_share"] for sd in SEEDS
                           if (arm, sd) in st],
            "shape": [{k: v for k, v in st[(arm, sd)].items()
                       if k != "top_idx"} for sd in SEEDS if (arm, sd) in st],
            "enrichment": enr.get(arm, []),
        }
    out["verdict"] = {"subject": SUBJECT, "shape": shape_v, "targeting": targ,
                      "none_top1_share": float(ns)}
    p = os.path.join(HERE, f"beta_shape_{stream}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nwrote {p}")
    return out["verdict"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, nargs="+", default=[5150, 3310])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--arms", nargs="+", default=None, help="smoke test only")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="smoke test only")
    ap.add_argument("--batches", type=int, default=6,
                    help="smoke test only; anything but 6 disables the "
                         "validity assert and the run is not a result")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    global ARMS, SEEDS, BATCHES
    if a.arms:
        ARMS = a.arms
    if a.seeds:
        SEEDS = a.seeds
    BATCHES = a.batches
    v = {s: run_stream(s, a.threads) for s in a.streams}
    print("\n" + "=" * 78)
    print("AGREEMENT ACROSS STREAMS (a shape verdict counts as settled only "
          "if both agree)")
    for s, d in v.items():
        print(f"  {s}: SHAPE {d['shape']:14s} TARGETING {d['targeting']}")
    pairs = {(d["shape"], d["targeting"]) for d in v.values()}
    print(f"  -> {'AGREE' if len(pairs) == 1 else 'DISAGREE, nothing settled'}")


if __name__ == "__main__":
    main()
