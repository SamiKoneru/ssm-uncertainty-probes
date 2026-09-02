"""
NON-STATIONARY recall.  Unlike induction_task, a key's successor can be
PERMANENTLY REVISED, so old observations become WRONG rather than being
alternatives worth remembering.  This is the regime that creates stale
confident beliefs, which no earlier task in this repo produced.

Per sequence, each symbol is assigned a revision rate:
    STABLE   never revised           -> model should be confident and right
    SLOW     revised w.p. 0.15       -> occasionally CONFIDENTLY WRONG
    VOLATILE revised every time      -> unpredictable, model should hedge

Flags recorded per position:
    contested  : symbol is SLOW or VOLATILE
    stale      : succ[s] was revised since the last time s appeared, so the
                 model's memory is out of date and it SHOULD be wrong
"""
import numpy as np, torch

N_SYM, L = 40, 256
P_REV = np.array([0.0, 0.15, 1.0])            # stable, slow, volatile

def sample(rng, B, p_rev=None):
    seq = np.zeros((B, L + 1), np.int64)
    cls = np.zeros((B, N_SYM), np.int64)
    stale = np.zeros((B, L), bool)
    seen = np.zeros((B, L), bool)
    for b in range(B):
        pr = P_REV if p_rev is None else p_rev
        cls[b] = rng.integers(3, size=N_SYM)
        succ = rng.integers(N_SYM, size=N_SYM)
        revised = np.zeros(N_SYM, bool)       # revised since last shown?
        shown = np.zeros(N_SYM, bool)
        s = rng.integers(N_SYM)
        for t in range(L + 1):
            seq[b, t] = s
            if t < L:
                stale[b, t] = revised[s] and shown[s]
                seen[b, t] = shown[s]
            nxt = succ[s]
            revised[s] = False; shown[s] = True
            if rng.random() < pr[cls[b, s]]:               # permanent revision
                succ[s] = rng.integers(N_SYM); revised[s] = True
            s = nxt
    x = torch.from_numpy(seq)
    cur = x[:, :-1]
    cl = torch.from_numpy(cls)[torch.arange(B)[:, None], cur]      # class of current symbol
    return x, cl, torch.from_numpy(stale), torch.from_numpy(seen)

if __name__ == "__main__":
    x, cl, st, sn = sample(np.random.default_rng(0), 16)
    for i, nm in enumerate(("stable", "slow", "volatile")):
        m = cl == i
        print(f"{nm:9s} frac {m.float().mean():.3f}   stale-when-seen "
              f"{st[m & sn].float().mean():.3f}")
    print(f"\nrepeat frac {sn.float().mean():.3f}")
    print("SLOW symbols are the interesting ones: usually the memory is correct,")
    print("occasionally it is silently out of date -> confidently wrong.")
