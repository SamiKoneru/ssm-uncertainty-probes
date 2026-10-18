r"""
Multi-query associative recall. A STANDARD recall task, not built to favour us.

WHY THIS IS THE REAL TEST
    gated_delta_ssm/task.py randomises each symbol's class per sequence
    specifically so an input-only gate PROVABLY cannot differentiate. That is
    sound ablation hygiene and it is also why the 5.5% cannot be quoted as a
    general result: it is measured on a task constructed to require the
    mechanism. MQAR is the ordinary version of the question.

WHAT A SINGLE GLOBAL BETA WANTS HERE, which is the whole subtlety
    plain      every write is a first write or an EXACT repeat. A repeat gives
               e = S k - v = 0, so beta multiplies zero and is irrelevant.
               A global beta is already optimal. Expect NO gain. This is the
               honest null.
    rewrite    a key's value is replaced once. That write needs high beta, and
               nothing anywhere wants low beta, so "beta = 1 always" is optimal
               and an input-only gate learns that constant trivially. Also
               expect little gain, which is itself worth having on record.
    CAPACITY   the axis that can actually separate the arms. Keys are only
               approximately orthogonal, so every write bleeds into other
               addresses. Once n_pairs is large relative to what the state
               holds cleanly, how hard to write depends on how contested the
               address already is, which is a property of the MEMORY and is
               invisible to the token. This pressure is not constructed in our
               favour, it is just what makes recall hard.

    So the experiment sweeps n_pairs, and the prediction under test is that any
    gain appears with capacity pressure and not otherwise.

FORMAT  keys are tokens [0, N_KEY), values are tokens [N_KEY, N_KEY + N_VAL),
        so a key can never be mistaken for a value.

            k1 v1  k2 v2 ... kN vN   [ki vi' ...]   kq vq  kq' vq' ...
            \------ write phase -----/\-rewrites-/  \--- query phase ---/

        Scored at query KEY positions only: the target of position p is x[p+1],
        the key's LATEST value. This matches the layer's native routing, where
        a write at t stores key emb(x_{t-1}) -> value emb(x_t) and a read at t
        queries with emb(x_t), so recall is next-token prediction.
"""
import numpy as np
import torch

N_KEY, N_VAL = 64, 64
VOCAB = N_KEY + N_VAL
L = 256


def sample(rng, B, n_pairs=32, rewrite=False, frac_rewrite=0.5, L=L):
    """-> x [B, L+1] int64, score [B, L] bool (positions whose target is scored)."""
    n_rw = int(round(n_pairs * frac_rewrite)) if rewrite else 0
    n_write = 2 * (n_pairs + n_rw)
    if n_write >= L:
        raise ValueError(f"write phase {n_write} leaves no room in L={L}")

    x = np.zeros((B, L + 1), np.int64)
    score = np.zeros((B, L), bool)
    for b in range(B):
        keys = rng.choice(N_KEY, size=n_pairs, replace=False)
        vals = rng.integers(N_VAL, size=n_pairs) + N_KEY

        toks = []
        for i in range(n_pairs):
            toks += [int(keys[i]), int(vals[i])]

        if n_rw:
            for i in rng.choice(n_pairs, size=n_rw, replace=False):
                nv = rng.integers(N_VAL) + N_KEY
                while nv == vals[i]:                 # must actually change
                    nv = rng.integers(N_VAL) + N_KEY
                vals[i] = nv                         # queries want the LATEST
                toks += [int(keys[i]), int(nv)]

        qpos = []
        while len(toks) + 2 <= L + 1:
            j = rng.integers(n_pairs)
            qpos.append(len(toks))
            toks += [int(keys[j]), int(vals[j])]
        while len(toks) < L + 1:                     # odd-length tail filler
            toks.append(int(keys[rng.integers(n_pairs)]))

        x[b] = toks[:L + 1]
        for p in qpos:
            if p < L:
                score[b, p] = True
    return torch.from_numpy(x), torch.from_numpy(score)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"vocab {VOCAB}  (keys 0-{N_KEY-1}, values {N_KEY}-{VOCAB-1})  L {L}")
    print(f"chance accuracy on a scored position: {1/N_VAL:.4f}\n")
    print(f"{'n_pairs':>8}{'rewrite':>9}{'write toks':>12}{'queries':>9}"
          f"{'scored/seq':>12}")
    for n in (16, 32, 64):
        for rw in (False, True):
            x, s = sample(rng, 8, n_pairs=n, rewrite=rw)
            nrw = int(round(n * 0.5)) if rw else 0
            print(f"{n:>8}{str(rw):>9}{2*(n+nrw):>12}"
                  f"{int(s[0].sum()):>9}{s.float().sum(1).mean():>12.1f}")

    # the stated contract: a scored position's target must be its key's latest value
    x, s = sample(np.random.default_rng(1), 1, n_pairs=16, rewrite=True)
    seq, latest, bad = x[0].tolist(), {}, 0
    for t in range(L):
        if seq[t] < N_KEY and t + 1 <= L and seq[t + 1] >= N_KEY:
            if s[0, t] and latest.get(seq[t]) != seq[t + 1]:
                bad += 1
            latest[seq[t]] = seq[t + 1]
    print(f"\nscored targets disagreeing with the key's latest value: {bad} "
          f"-> {'OK' if bad == 0 else 'TASK IS BROKEN'}")
