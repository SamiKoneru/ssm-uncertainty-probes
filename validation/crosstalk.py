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

PRE-REGISTERED PREDICTIONS, written before running
    P1  mean key coherence stays near its Welch floor: small at d >= 64, rising
        as d drops below 64.
    P2  the relative residual ||e||/||v|| on EXACT REPEAT writes tracks
        coherence: near 0 at d >= 64, growing as d falls.
    P3  therefore the gain measured by mqar_eval.py tracks P2, and is ~0 at
        d >= 64.
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
    return 0.0 if n <= d else float(np.sqrt((n - d) / (d * (n - 1))))


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
    """Pairwise |cos| among the learned keys of all N_KEY key tokens."""
    with torch.no_grad():
        kv = F.normalize(m.to_k(m.embed(torch.arange(M.N_KEY))), dim=-1)
        G = (kv @ kv.T).abs()
    n = G.shape[0]
    off = ~torch.eye(n, dtype=torch.bool)
    return G[off].mean().item(), G[off].max().item()


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
    print(f"{'d':>5}{'n<=d':>7}{'welch':>8}{'coh mean':>10}{'coh max':>9}"
          f"{'|e|/|v| first':>15}{'|e|/|v| REPEAT':>16}")
    rows = []
    for d in DIMS:
        co, cx, rf, rr = [], [], [], []
        for seed in SEEDS:
            m = train_baseline(d, seed)
            cm, cmx = coherence(m)
            co.append(cm); cx.append(cmx)
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
        row = (d, N_PAIRS <= d, welch(N_PAIRS, d), np.mean(co), np.mean(cx),
               np.mean(rf), np.mean(rr))
        rows.append(row)
        print(f"{d:>5}{str(row[1]):>7}{row[2]:>8.3f}{row[3]:>10.4f}"
              f"{row[4]:>9.4f}{row[5]:>15.4f}{row[6]:>16.4f}", flush=True)

    print("\n\nAGAINST THE PRE-REGISTERED PREDICTIONS\n")
    big = [r for r in rows if r[1]]
    small = [r for r in rows if not r[1]]
    print(f"  P1 coherence: mean {np.mean([r[3] for r in big]):.4f} at n<=d "
          f"vs {np.mean([r[3] for r in small]):.4f} at n>d  "
          f"{'PASS' if np.mean([r[3] for r in big]) < np.mean([r[3] for r in small]) else 'FAIL'}")
    print(f"  P2 repeat ||e||/||v||: {np.mean([r[6] for r in big]):.4f} at n<=d "
          f"vs {np.mean([r[6] for r in small]):.4f} at n>d  "
          f"{'PASS' if np.mean([r[6] for r in big]) < np.mean([r[6] for r in small]) else 'FAIL'}")
    d128 = [r for r in rows if r[0] == 128]
    if d128:
        print(f"\n  KILL CHECK repeat ||e||/||v|| at d=128: {d128[0][6]:.4f}")
        print("    near 0 supports crosstalk. Still large means the gate's gain"
              "\n    at low d is NOT about crosstalk and this account is wrong.")
    print(f"\n  CONTRAST first-write ||e||/||v|| should be ~1 at every d: "
          f"{[round(r[5], 3) for r in rows]}")
    print("    far from 1 means the measurement is broken, not the theory.")


if __name__ == "__main__":
    main()
