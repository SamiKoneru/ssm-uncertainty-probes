"""
Contested-key task.  Stream of ops, each 3 tokens: [OP, key, value]
  WRITE key v   -> key's current value becomes v
  QUERY key v   -> v is the CORRECT current value of key (the prediction target)

Half the keys are STABLE (always written the same value); half are CONTESTED
(each write assigns a fresh random value).  Both are deterministic given perfect
tracking, but contested keys demand recency tracking under interference, so
they are genuinely harder.  Ground truth per query: was the key contested.
"""
import numpy as np, torch

N_KEYS, N_VALS, N_OPS = 16, 32, 80          # overridden by configure()
WRITE, QUERY = 0, 1
TOK_OP, TOK_KEY, TOK_VAL = 0, 2, 2 + N_KEYS          # token id offsets
VOCAB = 2 + N_KEYS + N_VALS
L = 3 * N_OPS

def configure(n_keys, n_vals, n_ops):
    global N_KEYS, N_VALS, N_OPS, TOK_VAL, VOCAB, L
    N_KEYS, N_VALS, N_OPS = n_keys, n_vals, n_ops
    TOK_VAL = 2 + N_KEYS
    VOCAB = 2 + N_KEYS + N_VALS
    L = 3 * N_OPS
    return VOCAB, L


def sample(rng, B):
    seq = np.zeros((B, L), np.int64)
    tgt_pos = np.zeros((B, L), bool)                  # positions scored
    contested_at = np.zeros((B, L), bool)             # ground truth at those positions
    stable_val = rng.integers(N_VALS, size=(B, N_KEYS))
    is_con = np.zeros((B, N_KEYS), bool); is_con[:, N_KEYS // 2:] = True
    for b in range(B):
        cur = stable_val[b].copy()
        written = np.zeros(N_KEYS, bool)
        for o in range(N_OPS):
            k = rng.integers(N_KEYS)
            # query only keys already written, else write
            op = QUERY if (written[k] and rng.random() < 0.45) else WRITE
            if op == WRITE:
                cur[k] = rng.integers(N_VALS) if is_con[b, k] else stable_val[b, k]
                written[k] = True
            i = 3 * o
            seq[b, i], seq[b, i+1], seq[b, i+2] = TOK_OP + op, TOK_KEY + k, TOK_VAL + cur[k]
            if op == QUERY:
                tgt_pos[b, i+1] = True                # predicting token i+2 from i+1
                contested_at[b, i+1] = is_con[b, k]
    return (torch.from_numpy(seq), torch.from_numpy(tgt_pos),
            torch.from_numpy(contested_at))

if __name__ == "__main__":
    s, m, c = sample(np.random.default_rng(0), 4)
    print("vocab", VOCAB, "seq len", L, "scored positions per seq:", m[0].sum().item())
    print("fraction of scored queries on contested keys:", c[m].float().mean().item())
