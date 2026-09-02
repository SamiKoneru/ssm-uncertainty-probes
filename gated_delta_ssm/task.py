"""
Keys that want OPPOSITE write policies, in the SAME sequence.

  STABLE   successor never changes            -> beta irrelevant
  NOISY    successor drawn fresh each time from a FIXED per-symbol option set.
           Old observations stay VALID, so the best belief is the average over
           the set  ->  wants LOW beta (accumulate / average)
  REVISED  successor PERMANENTLY changes with prob P_REV and stays until the
           next revision.  Old observations become WRONG, so the best belief is
           the latest value only  ->  wants HIGH beta (overwrite)

A single input-dependent beta must choose one global policy. Only a beta that
can condition on the key's own history can serve both at once.
"""
import numpy as np, torch

N_SYM, L, N_OPT, P_REV = 40, 256, 4, 0.35
STABLE, NOISY, REVISED = 0, 1, 2

def sample(rng, B):
    seq = np.zeros((B, L + 1), np.int64)
    cls = np.zeros((B, N_SYM), np.int64)
    seen = np.zeros((B, L), bool)
    for b in range(B):
        cls[b] = rng.integers(3, size=N_SYM)
        succ = rng.integers(N_SYM, size=N_SYM)
        opts = rng.integers(N_SYM, size=(N_SYM, N_OPT))
        shown = np.zeros(N_SYM, bool)
        s = rng.integers(N_SYM)
        for t in range(L + 1):
            seq[b, t] = s
            if t < L: seen[b, t] = shown[s]
            c = cls[b, s]
            if c == NOISY:
                nxt = opts[s, rng.integers(N_OPT)]          # fresh draw, set is fixed
            else:
                nxt = succ[s]
                if c == REVISED and rng.random() < P_REV:
                    succ[s] = rng.integers(N_SYM)           # permanent revision
            shown[s] = True
            s = nxt
    x = torch.from_numpy(seq); cur = x[:, :-1]
    cl = torch.from_numpy(cls)[torch.arange(B)[:, None], cur]
    return x, cl, torch.from_numpy(seen)

if __name__ == "__main__":
    x, cl, sn = sample(np.random.default_rng(0), 16)
    for i, nm in enumerate(("STABLE", "NOISY", "REVISED")):
        m = (cl == i) & sn
        print(f"{nm:8s} frac {(cl==i).float().mean():.3f}   wants beta "
              f"{'irrelevant' if i==0 else 'LOW (average)' if i==1 else 'HIGH (overwrite)'}")
    print(f"\nceiling: STABLE 1.00, NOISY {1/N_OPT:.2f}, REVISED 1.00 when fresh")
