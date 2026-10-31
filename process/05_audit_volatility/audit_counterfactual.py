"""A1b. The counterfactual LOSS, which 04 said it could not afford.

04: "What would settle it directly, and was not run: a loss measurement: wire
the exact bookkeeping history into the gate and train. It was not run because a
trustworthy answer needs 3 seeds x 2 arms at ~28 min per 800-step run, about 6
hours."

It does not need training. Everything in the model except beta is frozen; beta
is the only thing a better gate would change. So: replay the frozen model and
SUBSTITUTE beta from a fitted policy, online, recomputing every feature
(including the per-address bookkeeping) under the substituted beta at each step,
and measure actual cross-entropy.

DECLARED LIMITATION (pre-registration A1b, written before running): the policy
is fitted on features generated under the SHIPPED beta and then applied where it
changes those features, and the readout is tuned to the beta distribution it
trained with. This is a LOWER bound on what a better policy is worth, not an
estimate of what retraining would give. A negative here is weaker evidence than
a positive.

Control that must pass or the whole file is void: replaying with the gate's own
beta must reproduce the shipped forward pass's loss to < 1e-4, and must
reproduce every feature column FE.replay records.

    python audit_counterfactual.py --stream 505 --tag gated_seed0
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


import _paths                                                  # noqa: F401
import beta_optimal as BO                                      # noqa: E402
import features as FE                                          # noqa: E402
import task as T                                               # noqa: E402

THREE = ["f_tok", "f_s1", "f_s2"]
INST, HIST = FE.INST, FE.HIST
ALLF = INST + HIST
GATE = "the gate's own beta (shipped)"


def hdr(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}", flush=True)


@torch.no_grad()
def replay_policy(m, x, mode="gate", policy=None, feats=None, cls_w=None,
                  v_full=None, tgt=None, want_cols=False):
    """Replay the frozen model with beta from `mode`, online.

    mode:
      "gate"     the shipped gate (control)
      "const"    a fixed scalar `policy`
      "policy"   a fitted BO.Policy over `feats`
      "class"    dict class -> beta, using cls_w
      "bstar"    the per-write optimum, recomputed against the CURRENT
                 trajectory at every step. Non-causal (reads future tokens),
                 so it is a ceiling, not a mechanism.

    Mirrors features.py's recurrence and its per-address bookkeeping exactly;
    `want_cols` returns the recorded columns so the control can compare them.
    """
    tokens = x[:, :-1]
    B, L = tokens.shape
    dv = m.d_v
    emb = m.embed(tokens)
    ctx = F.silu(m.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))
    shifted = F.pad(emb, (0, 0, 1, 0))[:, :L]
    q = F.normalize(m.to_q(emb), dim=-1)
    k = F.normalize(m.to_k(shifted), dim=-1)
    v = m.to_v(emb)
    beta_token = m.to_beta(ctx)[..., 0]

    ar = torch.arange(B)
    last_e = torch.zeros(B, T.N_SYM, dv)
    sum_e = torch.zeros(B, T.N_SYM, dv)
    sum_ne = torch.zeros(B, T.N_SYM)
    cnt = torch.zeros(B, T.N_SYM)

    S = emb.new_zeros(B, dv, m.d_k)
    logits, cols, betas = [], {n: [] for n in ALLF}, []
    nanv = torch.full((B,), float("nan"))
    for t in range(L):
        kt, qt, vt = k[:, t], q[:, t], v[:, t]
        stored = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1)
        e = stored - vt
        ne = e.norm(dim=-1)
        s1 = torch.log1p(ne)
        s2 = (vt * stored).sum(-1) / (vt.norm(dim=-1) ** 2 + 1e-6)

        row = {"f_tok": beta_token[:, t], "f_s1": s1, "f_s2": s2,
               "f_nstored": stored.norm(dim=-1), "f_nv": vt.norm(dim=-1)}
        if t == 0:
            for n in HIST:
                row[n] = nanv.clone()
            row["h_nvis"] = torch.zeros(B)
        else:
            addr = tokens[:, t - 1]
            p_last, p_sum = last_e[ar, addr], sum_e[ar, addr]
            p_cnt, p_ne = cnt[ar, addr], sum_ne[ar, addr]
            fresh = p_cnt == 0
            bl = lambda z: torch.where(fresh, torch.full_like(z, float("nan")),
                                       z)
            row["h_cos"] = bl(F.cosine_similarity(e, p_last, dim=-1, eps=1e-8))
            row["h_cosmean"] = bl(
                F.cosine_similarity(e, p_sum, dim=-1, eps=1e-8))
            row["h_ratio"] = bl(torch.log(
                (ne + 1e-6) / (p_last.norm(dim=-1) + 1e-6)))
            row["h_meane"] = bl(p_ne / p_cnt.clamp_min(1))
            row["h_nvis"] = p_cnt.clone()

        if mode == "gate":
            g = beta_token[:, t]
            if m.memory_gate:
                g = m.gate(torch.stack([g, s1, s2], -1))[..., 0]
            beta = torch.sigmoid(g)
        elif mode == "const":
            beta = torch.full((B,), float(policy))
        elif mode == "class":
            beta = torch.tensor(
                [policy.get(int(c), 0.0) for c in cls_w[:, t]],
                dtype=torch.float32)
        elif mode == "bstar":
            ti = tgt[:, t]
            ok = torch.from_numpy(ti >= 0)
            vn = v_full[ar, torch.from_numpy(np.where(ti >= 0, ti, 0))]
            Aq = (ne.double() ** 2)
            Bq = (-(e.double() * (vn - vt.double())).sum(-1)) + Aq
            bs = torch.clamp(Bq / Aq.clamp_min(1e-30), 0.0, 1.0)
            beta = torch.where(ok & (Aq > 0), bs, torch.zeros_like(bs)).float()
        else:
            X = np.stack([row[n].numpy().astype(np.float64) for n in feats], 1)
            beta = torch.from_numpy(policy.predict(X)).float()

        if want_cols:
            for n in ALLF:
                cols[n].append(row[n])
        betas.append(beta)

        if t > 0:
            addr = tokens[:, t - 1]
            last_e[ar, addr] = e
            sum_e[ar, addr] = sum_e[ar, addr] + e
            sum_ne[ar, addr] = sum_ne[ar, addr] + ne
            cnt[ar, addr] = cnt[ar, addr] + 1

        S = S - beta[:, None, None] * e.unsqueeze(-1) * kt.unsqueeze(1)
        logits.append(m.head(m.norm(torch.bmm(S, qt.unsqueeze(-1)).squeeze(-1))))

    lg = torch.stack(logits, 1)
    ce = F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1)).item()
    acc = (lg.argmax(-1) == x[:, 1:]).float().mean().item()
    out = {"ce": ce, "acc": acc,
           "beta": torch.stack(betas, 1).numpy()}
    if want_cols:
        out["cols"] = {n: torch.stack(c, 1).numpy() for n, c in cols.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=int, default=505)
    ap.add_argument("--tag", default="gated_seed0")
    ap.add_argument("--fit-batches", type=int, default=6)
    ap.add_argument("--eval-batches", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(2)

    m, ck = FE.load(a.tag)
    print(f"A1b  model {a.tag} (seed {ck['seed']})   stream {a.stream}")
    print(f"  fit on {a.fit_batches} batches of 32, evaluate counterfactually "
          f"on {a.eval_batches} DISJOINT batches drawn later from the same "
          f"stream")

    # ---------------------------------------------------------------- control
    hdr("A1b.0  control: the reimplementation must reproduce the shipped model")
    ev = np.random.default_rng(a.stream)
    xs = [T.sample(ev, 32) for _ in range(a.fit_batches + a.eval_batches)]
    x0, cl0, _ = xs[0]
    ref = FE.replay(m, x0)
    got = replay_policy(m, x0, mode="gate", want_cols=True)
    ce_ref = F.cross_entropy(
        torch.from_numpy(ref["logits"]).reshape(-1, T.N_SYM),
        x0[:, 1:].reshape(-1)).item()
    print(f"  CE  shipped {ce_ref:.6f}   mine {got['ce']:.6f}   "
          f"|diff| {abs(ce_ref - got['ce']):.3e}   (bar < 1e-4)")
    worst = max(
        float(np.nanmax(np.abs(got["cols"][n] - ref[n]))) for n in ALLF)
    print(f"  max |feature column difference| over all 10 columns {worst:.3e}")
    bmax = float(np.abs(got["beta"] - ref["beta"]).max())
    print(f"  max |beta difference| {bmax:.3e}")
    assert abs(ce_ref - got["ce"]) < 1e-4 and worst < 1e-4 and bmax < 1e-5, \
        "A1b control FAILED - the reimplementation is not the shipped model"
    print("  PASS")

    # ------------------------------------------------------------------- fit
    d = BO.collect(m, a.stream, batches=a.fit_batches)
    Aa, Bb = d["A"], d["B"]
    fitmask = d["valid"]
    policies = {}
    for nm, feats, kind in (("3 feats, linear", THREE, "linear"),
                            ("3 feats, GBT", THREE, "gbm"),
                            ("5 zero-state, GBT", INST, "gbm"),
                            ("5 zero-state + history, GBT", ALLF, "gbm")):
        X = np.stack([d[f][fitmask] for f in feats], 1)
        policies[nm] = (BO.Policy(kind, feats, seed=0).fit(
            X, Aa[fitmask], Bb[fitmask]), feats)
    b_const = BO.const_beta(Aa[fitmask], Bb[fitmask])
    pc = BO.per_class_beta(Aa[fitmask], Bb[fitmask], d["cls_w"][fitmask])
    print(f"\n  fitted on {fitmask.sum()} writes with a defined target")
    print(f"  best constant beta {b_const:.4f}")
    print(f"  per-class beta  STABLE {pc.get(0, float('nan')):.4f}  "
          f"NOISY {pc.get(1, float('nan')):.4f}  "
          f"REVISED {pc.get(2, float('nan')):.4f}")

    # -------------------------------------------------------------- evaluate
    hdr("A1b.1  actual cross-entropy on held-out sequences, frozen model, "
        "beta substituted")
    runs = []
    for bi in range(a.fit_batches, a.fit_batches + a.eval_batches):
        x, cl, _ = xs[bi]
        cw = F.pad(cl, (1, 0))[:, :T.L].numpy()
        vf = m.to_v(m.embed(x)).double()
        tg = BO.target_index(x.numpy())
        r = {}
        r[GATE] = replay_policy(m, x, "gate")
        r["best constant beta"] = replay_policy(m, x, "const", policy=b_const)
        for nm, (p, feats) in policies.items():
            r[nm] = replay_policy(m, x, "policy", policy=p, feats=feats)
        r["TRUE CLASS LABEL (oracle)"] = replay_policy(
            m, x, "class", policy=pc, cls_w=cw)
        r["per-write beta* (non-causal ceiling)"] = replay_policy(
            m, x, "bstar", v_full=vf, tgt=tg)
        runs.append(r)
        print(f"  batch {bi} done", flush=True)

    keys = list(runs[0].keys())
    ce = {k: np.mean([r[k]["ce"] for r in runs]) for k in keys}
    acc = {k: np.mean([r[k]["acc"] for r in runs]) for k in keys}
    mb = {k: np.mean([r[k]["beta"].mean() for r in runs]) for k in keys}
    base = ce["best constant beta"]
    print(f"\n  {'beta policy':40s}{'CE':>10}{'acc':>8}{'mean b':>9}"
          f"{'CE gain vs const':>18}")
    for k in keys:
        print(f"  {k:40s}{ce[k]:>10.4f}{acc[k]:>8.4f}{mb[k]:>9.4f}"
              f"{base - ce[k]:>+18.4f}")
    spread = (max(r[GATE]["ce"] for r in runs)
              - min(r[GATE]["ce"] for r in runs))
    print(f"\n  per-batch CE spread (max-min) for the shipped gate: "
          f"{spread:.4f}")

    hdr("A1b.2  the pre-registered loss bars")
    g_hist = ce["5 zero-state, GBT"] - ce["5 zero-state + history, GBT"]
    g_non = ce["3 feats, linear"] - ce["3 feats, GBT"]
    print(f"  CE(5 feats) - CE(5 feats + history)   {g_hist:>+9.4f}   "
          f"<- history's loss gain")
    print(f"  CE(3 linear) - CE(3 GBT)              {g_non:>+9.4f}   "
          f"<- nonlinearity's loss gain")
    print(f"  repo seed sd on loss: 0.0026 to 0.0039")
    print(f"\n  bar (b) COLLAPSES on loss: g_hist >= 0.0039 AND "
          f"g_hist >= g_non")
    print(f"       -> {'FIRES' if (g_hist >= 0.0039 and g_hist >= g_non) else 'does not fire'}")
    print(f"  bar (d)'s prediction HOLDS: g_hist <= 0.0013")
    print(f"       -> {'FIRES' if g_hist <= 0.0013 else 'does not fire'}")
    print(f"\n  context: total CE on the table from beta alone "
          f"(const -> non-causal per-write optimum) "
          f"{base - ce['per-write beta* (non-causal ceiling)']:>+.4f}")
    print(f"           the shipped gate captures {base - ce[GATE]:>+.4f}"
          f" of that")


if __name__ == "__main__":
    main()
