"""A6. Claim (d): did the built rate memory really carry ~93% of the available
separation?

04 compares `rate_delta`'s logged `c_t` AUROC of 0.3690 (03/logs/rate.log)
against an exact `h_meane` AUROC of 0.3189 measured in 04, and concludes "the
delta-rule rate memory recovered 0.631 of an available 0.681, about 93% of the
separation on that axis", therefore "crosstalk was not the problem".

Two separate problems.

A6.1  THE ARITHMETIC. 93% is (1-0.3690)/(1-0.3189), a ratio of raw AUROCs.
      AUROC's null is 0.5, not 0. Pure arithmetic, no data needed.

A6.2  THE COMPARISON. The two numbers come from different model classes
      (GatedDeltaLayer vs GatedDeltaSSM), different gates, different eval
      streams (12345 vs 777/888) and different masks (`sn` vs `sn_addr`).
      `mixed_task.py` and `gated_delta_ssm/task.py` are byte-identical, so the
      TASK is not a confound; everything else is.

      Rather than retrain (which would reintroduce the model confound this is
      criticising), the rate memory is run as a READ-ONLY SIDE CHANNEL on the
      shipped model: the same `e_t` sequence, the same stream, the same mask,
      the same labelling. The only difference between the arms is then the
      thing (d) is actually about.

      Three arms, which decompose the gap:

        h_meane   exact per-address MEAN of ||e||        04's ceiling
        h_ema     exact per-address EMA of ||e||, same   isolates the EMA form
                  write strength bc, NO crosstalk
        c_t       <C_{t-1}, k_t> with C written by the   the BUILT mechanism
                  delta rule at the superposed address   (EMA + crosstalk)

      So (h_meane -> h_ema) is the cost of the EMA form and (h_ema -> c_t) is
      the cost of crosstalk, which is exactly the quantity (d) rules out.

Bar: recovered fraction of signal above chance >= 0.80 -> (d) survives;
<= 0.70 -> (d) fails as stated.

    python audit_claim_d.py --stream 505 --tag gated_seed0
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import auroc as A                                              # noqa: E402
import features as FE                                          # noqa: E402
import task as T                                               # noqa: E402

BC_LOGGED = 0.198          # sigmoid(p_bc) as logged in 03/logs/rate.log


def hdr(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}", flush=True)


@torch.no_grad()
def rate_channels(m, x, bc=BC_LOGGED):
    """Run the shipped model and carry three rate-memory arms alongside it.

    Nothing here feeds back into beta or S, so the model's trajectory is
    identical to the shipped forward pass and all three arms see the same e_t.
    """
    tokens = x[:, :-1]
    B, L = tokens.shape
    emb = m.embed(tokens)
    ctx = F.silu(m.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))
    shifted = F.pad(emb, (0, 0, 1, 0))[:, :L]
    k = F.normalize(m.to_k(shifted), dim=-1)
    v = m.to_v(emb)
    beta_token = m.to_beta(ctx)[..., 0]

    ar = torch.arange(B)
    S = emb.new_zeros(B, m.d_v, m.d_k)
    C = emb.new_zeros(B, m.d_k)                  # superposed rate memory
    ema = torch.zeros(B, T.N_SYM)                # exact per-address EMA
    seenad = torch.zeros(B, T.N_SYM, dtype=torch.bool)
    c_all, ema_all = [], []
    for t in range(L):
        kt, vt = k[:, t], v[:, t]
        stored = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1)
        e = stored - vt
        ne = e.norm(dim=-1)

        # read BEFORE this step's write, as gated_model.py does
        c_all.append((C * kt).sum(-1).clone())
        if t == 0:
            ema_all.append(torch.full((B,), float("nan")))
        else:
            addr = tokens[:, t - 1]
            ema_all.append(torch.where(seenad[ar, addr], ema[ar, addr],
                                       torch.full((B,), float("nan"))))

        g = beta_token[:, t]
        if m.memory_gate:
            s1 = torch.log1p(ne)
            s2 = (vt * stored).sum(-1) / (vt.norm(dim=-1) ** 2 + 1e-6)
            g = m.gate(torch.stack([g, s1, s2], -1))[..., 0]
        beta = torch.sigmoid(g)

        # the exact same delta-rule scalar store, at the exact address
        C = C - bc * ((C * kt).sum(-1, keepdim=True) - ne.unsqueeze(-1)) * kt
        if t > 0:
            addr = tokens[:, t - 1]
            prev = torch.where(seenad[ar, addr], ema[ar, addr], ne)
            ema[ar, addr] = prev - bc * (prev - ne)
            seenad[ar, addr] = True

        S = S - beta[:, None, None] * e.unsqueeze(-1) * kt.unsqueeze(1)
    return (torch.stack(c_all, 1).numpy().astype(np.float64),
            torch.stack(ema_all, 1).numpy().astype(np.float64))


def recovered(au_mech, au_ceil):
    """Fraction of the ceiling's signal ABOVE CHANCE that the mechanism has."""
    return abs(au_mech - 0.5) / max(abs(au_ceil - 0.5), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=int, default=505)
    ap.add_argument("--tag", default="gated_seed0")
    ap.add_argument("--batches", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(2)

    hdr("A6.1  the arithmetic, no data needed")
    rc, rh = 0.3690, 0.3189
    print(f"  logged  rate_delta c_t AUROC            {rc:.4f}")
    print(f"  04's    exact h_meane AUROC            {rh:.4f}")
    print(f"  04's figure: (1-{rc:.4f})/(1-{rh:.4f}) = "
          f"{(1 - rc) / (1 - rh):.4f}   'about 93%'")
    print(f"  that is a ratio of RAW AUROCs. AUROC's null is 0.5, not 0.")
    print(f"\n  signal above chance:  c_t {abs(rc - .5):.4f}   "
          f"exact {abs(rh - .5):.4f}")
    frac = recovered(rc, rh)
    print(f"  fraction of SIGNAL recovered = {abs(rc-.5):.4f}/"
          f"{abs(rh-.5):.4f} = {frac:.4f}")
    print(f"\n  bar: <= 0.80 -> (d) as stated is WRONG")
    print(f"  -> {frac:.3f}, so {'(d) AS STATED IS WRONG' if frac <= 0.80 else '(d) survives A6.1'}")

    # --------------------------------------------------------------- A6.2
    m, ck = FE.load(a.tag)
    print(f"\nA6.2  model {a.tag} (seed {ck['seed']})   stream {a.stream}")
    ev = np.random.default_rng(a.stream)
    cols = {n: [] for n in ("c_t", "h_ema", "h_meane", "cls_w", "wtype",
                            "seq", "sn_task", "sn_addr")}
    for bi in range(a.batches):
        x, cl, sn = T.sample(ev, 32)
        d = FE.replay(m, x)
        ct, hema = rate_channels(m, x)
        L = T.L
        cols["c_t"].append(ct)
        cols["h_ema"].append(hema)
        cols["h_meane"].append(d["h_meane"])
        cols["wtype"].append(d["wtype"])
        cols["cls_w"].append(F.pad(cl, (1, 0))[:, :L].numpy())
        cols["sn_task"].append(sn.numpy())
        cols["sn_addr"].append(F.pad(sn, (1, 0))[:, :L].numpy())
        cols["seq"].append(np.repeat(
            np.arange(bi * 32, bi * 32 + 32)[:, None], L, 1))
    d2 = {k: np.concatenate(v).reshape(-1) for k, v in cols.items()}

    seen = d2["wtype"] != FE.UNSEEN
    nr = np.isin(d2["cls_w"].astype(int), [1, 2])
    lab = (d2["cls_w"].astype(int) == 2).astype(int)

    hdr("A6.2  all three arms on ONE model, ONE stream, ONE mask, ONE labelling")
    print("  NOISY vs REVISED on writes to an address written before.\n"
          "  AUROC < 0.5 means the statistic reads LOWER on REVISED; the\n"
          "  'flipped' column is 1-AUROC, as 04 reports it.")
    masks = (("sn_addr (correct)", seen & nr),
             ("sn_task (the mask rate_test.py used)",
              d2["sn_task"].astype(bool) & nr))
    for mname, mk in masks:
        print(f"\n  mask {mname}, n = {mk.sum()}")
        print(f"    {'arm':34s}{'AUROC':>9}{'flipped':>9}{'|above chance|':>16}")
        res = {}
        for nm in ("h_meane", "h_ema", "c_t"):
            au, n = A.single(d2[nm][mk], lab[mk])
            res[nm] = au
            print(f"    {nm:34s}{au:>9.4f}{1 - au:>9.4f}"
                  f"{abs(au - .5):>16.4f}")
        print(f"\n    decomposition of the gap from the exact ceiling:")
        print(f"      exact MEAN -> exact EMA (the EMA form) "
              f"{recovered(res['h_ema'], res['h_meane']):>8.4f} recovered")
        print(f"      exact EMA  -> superposed c_t (CROSSTALK)"
              f"{recovered(res['c_t'], res['h_ema']):>8.4f} recovered")
        print(f"      exact MEAN -> superposed c_t (BOTH)     "
              f"{recovered(res['c_t'], res['h_meane']):>8.4f} recovered")
        tot = recovered(res["c_t"], res["h_meane"])
        print(f"\n    bar: >= 0.80 (d) survives;  <= 0.70 (d) fails as stated")
        print(f"    -> {tot:.3f}: "
              f"{'(d) SURVIVES' if tot >= 0.80 else '(d) FAILS as stated' if tot <= 0.70 else 'AMBIGUOUS'}")

    hdr("A6.2 note")
    print("  The side channel holds the model fixed, so it isolates\n"
          "  superposition from every other difference. It does NOT reproduce\n"
          "  rate_delta's own training, where the gate consumes c_t and so\n"
          "  changes the e_t distribution the memory sees. Retraining was not\n"
          "  run (one more 800-step fit on a loaded shared machine) and would\n"
          "  reintroduce the model confound this section exists to remove.")


if __name__ == "__main__":
    main()
