"""
Does crosstalk explain the MQAR capacity gain? A direct measurement.

THE CLAIM UNDER TEST, post-hoc and so far unverified
    The delta rule writes S_t = S_{t-1} - beta_t e_t k_t^T with e_t = S_{t-1}k_t
    - v_t. With n stored pairs the readout at address k_j is

        S k_j = v_j + sum_{i != j} v_i <k_i, k_j>
                ^^^      ^^^^^^^^^^^^^^^^^^^^^^^^
                signal   crosstalk from every other address

    so e_t on an EXACT REPEAT write is zero only if the keys are orthogonal.
    Orthogonality is a counting question: n mutually orthogonal unit vectors
    exist in R^d iff n <= d. With n = 64 distinct keys:

        d >= 64   achievable   crosstalk -> 0, e -> 0 on repeats, beta inert
        d <  64   impossible   Welch bound forces coherence
                               max_{i!=j} |<k_i,k_j>| >= sqrt((n-d)/(d(n-1)))

    CORRECTED: the Welch expression is a LOWER BOUND ON THE MAXIMUM pairwise
    coherence, and comparing it to the MEAN is invalid, since
    mean <= rms <= max. The same expression is however the EXACT, achievable
    floor for the RMS coherence, from the Gram matrix of K = [k_1 ... k_n]:

        ||G||_F^2 = n + sum_{i!=j} c_ij^2,   and   ||G||_F^2 = sum_i lambda_i^2
        trace(K K^T) = n spread over d eigenvalues  =>  sum lambda_i^2 >= n^2/d
        =>  sum_{i!=j} c_ij^2 >= n(n-d)/d
        =>  rms coherence >= sqrt((n-d)/(d(n-1))),  equality for a tight frame

    That same inequality bounds the quantity actually of interest. Averaging
    ||e_j||^2 ~ ||v||^2 sum_{i!=j} c_ij^2 over slots:

        ||e|| / ||v||  >=  sqrt( (n - d) / d )

        d=128  0.000   d=64  0.000   d=48  0.577
        d= 32  1.000   <- contamination is NECESSARILY at least the signal
        d= 24  1.291   d=16  1.732

    The floor crosses 1 at exactly d = n/2 = 32, which is where the largest
    gain was measured, and d = 16 sits at 1.73, which is where it collapsed.

PRE-REGISTERED PREDICTIONS, written before running
    P1  RMS key coherence stays near the Welch floor (not the mean: see above).
        This is a DIAGNOSTIC, not the gain prediction. It separates "geometry
        forces e > 0" from "SGD failed to find orthogonal keys it could have
        had", which are different failures with different consequences.
    P2  the relative residual ||e||/||v|| on EXACT REPEAT writes is near 0 at
        d >= 64 and at or above the sqrt((n-d)/d) floor below it.
    P3  the gain measured by mqar_eval.py tracks P2, and is ~0 at d >= 64.

WHAT BETA HAS LEFT TO DO EVEN AT THE FLOOR
    Sitting at the floor minimises the AVERAGE contamination. It does not make
    contamination uniform. e_j = sum_{i!=j} v_i c_ij depends on which other
    addresses have actually been written, with what values, in what order, so
    two slots with identical geometry can carry very different e. Geometry sets
    the average; history sets the instantaneous value, and that is what beta
    responds to. Consistent with the position stratification already measured,
    where the gain is zero over the first 32 positions and flat afterwards,
    turning on exactly as the memory fills.
    CONTRAST  ||e||/||v|| on FIRST writes should be ~1 at every d, since
        nothing useful is stored at that address yet. If it is not ~1, the
        measurement is wrong rather than the theory.
    KILLS IT  repeat-write ||e||/||v|| staying high at d = 128, or the gain
        surviving there. Either means the gate exploits something that is not
        crosstalk.

Measured on the `none` baseline, so the environment the gate operates in is
characterised without the gate contaminating it.

METHOD NOTE
    ||e_t|| is internal to the shipped forward pass and not exposed. This file
    replays the recurrence to expose it, then ASSERTS the replay reproduces the
    shipped model's own logits. If the assert passes, the replay is faithful
    and the residuals are the real ones. If it fails, the measurement is void.
"""
import numpy as np
import torch
import torch.nn.functional as F

import arms
import mqar_task as M

DIMS = (16, 24, 32, 48, 64, 128)
SEEDS = (0, 1)
STEPS, N_PAIRS = 800, 64


def welch(n, d):
    """Exact achievable floor on RMS pairwise coherence (also the Welch bound
    on the maximum). Zero when n <= d, where orthogonality is achievable."""
    return 0.0 if n <= d else float(np.sqrt((n - d) / (d * (n - 1))))


def e_floor(n, d):
    """Implied floor on ||e||/||v|| for an exact repeat write, from the same
    Gram-matrix inequality under the uncorrelated-values approximation."""
    return 0.0 if n <= d else float(np.sqrt((n - d) / d))


def replay(m, x):
    """Rerun the shipped recurrence, returning (logits, ||e||/||v|| per step)."""
    assert not m.memory_gate, "replay only implements the ungated baseline"
    B, L = x.shape
    emb = m.embed(x)
    ctx = F.silu(m.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))
    shifted = F.pad(emb, (0, 0, 1, 0))[:, :L]
    q = F.normalize(m.to_q(emb), dim=-1)
    k = F.normalize(m.to_k(shifted), dim=-1)
    v = m.to_v(emb)
    base = m.to_beta(ctx)[..., 0]

    S = emb.new_zeros(B, m.d_v, m.d_k)
    logits, rel = [], []
    for t in range(L):
        kt, qt, vt = k[:, t], q[:, t], v[:, t]
        stored = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1)
        e = stored - vt
        beta = torch.sigmoid(base[:, t])
        S = S - beta[:, None, None] * e.unsqueeze(-1) * kt.unsqueeze(1)
        logits.append(m.head(m.norm(torch.bmm(S, qt.unsqueeze(-1)).squeeze(-1))))
        rel.append(e.norm(dim=-1) / (vt.norm(dim=-1) + 1e-9))
    return torch.stack(logits, 1), torch.stack(rel, 1)


def write_kinds(xin):
    """The write at step t uses key emb(x[t-1]) and value emb(x[t]).

    So it writes the ORDERED PAIR (x[t-1], x[t]). An exact repeat is that same
    pair having been written earlier in the sequence. t=0 is skipped: its key
    comes from the zero pad, not from a token.
    """
    B, L = xin.shape
    first = torch.zeros(B, L, dtype=torch.bool)
    rep = torch.zeros(B, L, dtype=torch.bool)
    for b in range(B):
        seen = set()
        for t in range(1, L):
            pair = (int(xin[b, t - 1]), int(xin[b, t]))
            if pair in seen:
                rep[b, t] = True
            else:
                first[b, t] = True
                seen.add(pair)
    return first, rep


def coherence(m):
    """Off-diagonal |cos| among the learned keys: (mean, RMS, max).

    RMS is the one with an exact achievable bound, so it is what P1 tests.
    """
    with torch.no_grad():
        kv = F.normalize(m.to_k(m.embed(torch.arange(M.N_KEY))), dim=-1)
        G = (kv @ kv.T).abs()
    n = G.shape[0]
    off = ~torch.eye(n, dtype=torch.bool)
    g = G[off]
    return g.mean().item(), g.pow(2).mean().sqrt().item(), g.max().item()


def train_baseline(dim, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = arms.build("none", M.VOCAB, d_k=dim, d_v=dim)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for _ in range(STEPS):
        x, s = M.sample(rng, 24, n_pairs=N_PAIRS, rewrite=False)
        loss = F.cross_entropy(m(x[:, :-1])[s], x[:, 1:][s])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


def main():
    print(f"n = {N_PAIRS} distinct key addresses, measured on the `none` "
          f"baseline, {len(SEEDS)} seeds\n")
    print(f"{'d':>5}{'n<=d':>6}{'floor':>7}{'coh rms':>9}{'coh mean':>9}"
          f"{'coh max':>8}{'e floor':>9}{'first':>8}{'REPEAT':>8}")
    rows = []
    for d in DIMS:
        co, cr, cx, rf, rr = [], [], [], [], []
        for seed in SEEDS:
            m = train_baseline(d, seed)
            cm, crm, cmx = coherence(m)
            co.append(cm); cr.append(crm); cx.append(cmx)
            ev = np.random.default_rng(101)
            f_, r_ = [], []
            with torch.no_grad():
                for _ in range(3):
                    x, _ = M.sample(ev, 16, n_pairs=N_PAIRS, rewrite=False)
                    xin = x[:, :-1]
                    lg_replay, rel = replay(m, xin)
                    ref = m(xin)
                    assert torch.allclose(lg_replay, ref, atol=1e-4), \
                        "replay diverged from the shipped forward, measurement void"
                    fm, rm = write_kinds(xin)
                    f_.append(rel[fm]); r_.append(rel[rm])
            rf.append(torch.cat(f_).mean().item())
            rr.append(torch.cat(r_).mean().item())
        row = (d, N_PAIRS <= d, welch(N_PAIRS, d), np.mean(cr), np.mean(co),
               np.mean(cx), e_floor(N_PAIRS, d), np.mean(rf), np.mean(rr))
        rows.append(row)
        print(f"{d:>5}{str(row[1]):>6}{row[2]:>7.3f}{row[3]:>9.4f}"
              f"{row[4]:>9.4f}{row[5]:>8.4f}{row[6]:>9.3f}{row[7]:>8.3f}"
              f"{row[8]:>8.3f}", flush=True)

    print("\n\nAGAINST THE PRE-REGISTERED PREDICTIONS\n")
    big = [r for r in rows if r[1]]
    small = [r for r in rows if not r[1]]
    print("  P1 RMS coherence against its exact achievable floor")
    for r in rows:
        slack = r[3] - r[2]
        print(f"    d={r[0]:>4}  floor {r[2]:.4f}  measured {r[3]:.4f}  "
              f"slack {slack:+.4f}"
              f"{'   AT FLOOR' if abs(slack) < 0.02 else ''}")
    print(f"    n<=d mean {np.mean([r[3] for r in big]):.4f}  vs  "
          f"n>d mean {np.mean([r[3] for r in small]):.4f}  "
          f"{'PASS' if np.mean([r[3] for r in big]) < np.mean([r[3] for r in small]) else 'FAIL'}")

    print("\n  P2 repeat-write ||e||/||v|| against its implied floor")
    for r in rows:
        print(f"    d={r[0]:>4}  floor {r[6]:.3f}  measured {r[8]:.3f}")
    print(f"    n<=d mean {np.mean([r[8] for r in big]):.4f}  vs  "
          f"n>d mean {np.mean([r[8] for r in small]):.4f}  "
          f"{'PASS' if np.mean([r[8] for r in big]) < np.mean([r[8] for r in small]) else 'FAIL'}")

    d128 = [r for r in rows if r[0] == 128]
    if d128:
        print(f"\n  KILL CHECK repeat ||e||/||v|| at d=128: {d128[0][8]:.4f}")
        print("    near 0 supports crosstalk. Still large means either SGD did"
              "\n    not find the orthogonal keys available to it (check P1 at"
              "\n    d=128), or the gate's gain is not about crosstalk at all.")
    print(f"\n  CONTRAST first-write ||e||/||v|| should be ~1 at every d: "
          f"{[round(r[7], 3) for r in rows]}")
    print("    far from 1 means the measurement is broken, not the theory.")


if __name__ == "__main__":
    main()
