"""
In-context associative recall (induction), the thing these layers actually do.

Each sequence gets a FRESH successor map, so the rules cannot be memorised in
the weights and must be learned in-context from the sequence itself.

  STABLE symbol s    -> always followed by one fixed symbol
  CONTESTED symbol s -> followed by a uniform draw from N_OPT options

A stable symbol is perfectly predictable after its first occurrence.  A
contested symbol is fundamentally unpredictable (ceiling 1/N_OPT), so errors
are REAL and correlate with contestation by construction.  That finally makes
"does u predict error" a testable question.
"""
import numpy as np, torch

N_SYM, N_OPT, L = 40, 4, 256
N_STABLE = N_SYM // 2
# Which symbols are contested is resampled PER SEQUENCE.  If it were fixed
# (0..19 always stable), the model could learn "symbol >= 20 -> be uncertain"
# from token identity in its weights and never consult the memory at all,
# which would make output entropy look like an uncertainty tracker for free.
RANDOMIZE_CONTESTED = True

def sample(rng, B):
    seq = np.zeros((B, L + 1), np.int64)
    con_sym = np.zeros((B, N_SYM), bool)
    for b in range(B):
        if RANDOMIZE_CONTESTED:
            con_sym[b, rng.permutation(N_SYM)[:N_SYM - N_STABLE]] = True
        else:
            con_sym[b, N_STABLE:] = True
        succ = rng.integers(N_SYM, size=N_SYM)                      # stable successor per symbol
        opts = rng.integers(N_SYM, size=(N_SYM, N_OPT))             # contested option set per symbol
        s = rng.integers(N_SYM)
        for t in range(L + 1):
            seq[b, t] = s
            s = opts[s, rng.integers(N_OPT)] if con_sym[b, s] else succ[s]
    x = torch.from_numpy(seq)
    cur = x[:, :-1]
    contested = torch.from_numpy(con_sym)[torch.arange(B)[:, None], cur]  # per-position truth
    # has this symbol appeared earlier in the sequence? (first occurrence is
    # unpredictable regardless of stability, so it must be controlled for)
    seen = torch.zeros_like(cur, dtype=torch.bool)
    for b in range(cur.shape[0]):
        s_ = set()
        for t in range(L):
            v = int(cur[b, t]); seen[b, t] = v in s_; s_.add(v)
    return x, contested, seen

if __name__ == "__main__":
    x, c, s = sample(np.random.default_rng(0), 8)
    print("seq", tuple(x.shape), "| contested frac", c.float().mean().item())
    print("repeat-symbol frac", s.float().mean().item())
    print(f"ceiling: stable->1.0, contested->{1/N_OPT:.2f}, so max acc approx "
          f"{(1-c.float().mean())*1 + c.float().mean()/N_OPT:.3f}")
