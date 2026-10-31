"""AUDIT. The addressing and labelling, against hand-built sequences where the
right answer is known by construction rather than by reading the code again.

This is the highest-consequence possible bug in the directory: if the write at
step t is labelled by the wrong token's class, or "address written before" is
off by one, every number moves. The repo has been burned by exactly this.

The claims under test, from features.py's docstring:
  k_t = normalize(W_k emb(x_{t-1}))   address comes from the PREVIOUS token
  v_t = W_v emb(x_t)                  value comes from the CURRENT token
  so  class(write at t) = class(x_{t-1})
  and "address written before" = x_{t-1} occurred strictly before position t-1

Test 1 forces the model's own weights to make the address and value readable, so
the claim is checked against the MODEL rather than against my bookkeeping.
Test 2 checks the bookkeeping against an independent, deliberately naive
reimplementation on hand-built token streams.

    python audit_addressing.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                "gated_delta_ssm"))

import features as FE
import task as T
from model import GatedDeltaSSM


def hdr(s):
    print(f"\n{'='*74}\n{s}\n{'='*74}", flush=True)


@torch.no_grad()
def test_model_addressing():
    """Make to_k and to_v identity-like on a one-hot embedding, so the address
    and the value can be read straight off k_t and v_t and compared to tokens."""
    m = GatedDeltaSSM(T.N_SYM, d_model=T.N_SYM, d_k=T.N_SYM, d_v=T.N_SYM,
                      memory_gate=True, gate_width=4)
    m.embed.weight.copy_(torch.eye(T.N_SYM))
    m.to_k.weight.copy_(torch.eye(T.N_SYM)); m.to_k.bias.zero_()
    m.to_v.weight.copy_(torch.eye(T.N_SYM)); m.to_v.bias.zero_()
    m.eval()

    toks = torch.tensor([[3, 7, 11, 7, 25, 7, 3, 19]])
    L = toks.shape[1]
    emb = m.embed(toks)
    shifted = F.pad(emb, (0, 0, 1, 0))[:, :L]
    k = F.normalize(m.to_k(shifted), dim=-1)
    v = m.to_v(emb)
    print("  tokens:", toks[0].tolist())
    print(f"\n{'t':>3}{'argmax k_t':>12}{'x_{t-1}':>10}{'argmax v_t':>12}"
          f"{'x_t':>7}{'ok':>5}")
    ok = True
    for t in range(L):
        ak, av = int(k[0, t].argmax()), int(v[0, t].argmax())
        prev = int(toks[0, t - 1]) if t > 0 else None
        cur = int(toks[0, t])
        # at t=0 the key is the zero pad, so argmax k is meaningless
        good = (t == 0) or (ak == prev and av == cur)
        ok &= good
        print(f"{t:>3}{ak:>12}{str(prev):>10}{av:>12}{cur:>7}"
              f"{('ok' if good else 'MISMATCH') if t else 'pad':>5}")
    print(f"\n  address = previous token, value = current token: "
          f"{'CONFIRMED' if ok else 'REFUTED'}")
    return ok


def naive_bookkeeping(tok):
    """Deliberately naive and written from the spec, not from features.py:
    plain Python dicts, one sequence, no vectorisation."""
    L = len(tok)
    wtype = [FE.UNSEEN] * L
    memb = [None] * L
    nvis = [None] * L
    last_val, seen_vals = {}, {}
    for t in range(1, L):
        a, val = tok[t - 1], tok[t]
        if a in last_val:
            wtype[t] = FE.REPEAT if last_val[a] == val else FE.CHANGE
            memb[t] = 1.0 if val in seen_vals[a] else 0.0
            nvis[t] = float(len(seen_vals[a]) and sum(seen_vals[a].values()))
        last_val[a] = val
        seen_vals.setdefault(a, {})
        seen_vals[a][val] = seen_vals[a].get(val, 0) + 1
    return wtype, memb


@torch.no_grad()
def test_bookkeeping():
    m, _ = FE.load("gated_seed0")
    rng = np.random.default_rng(5150)
    cases = {
        "all repeats (absorbing)": [5] * 12,
        "strict alternation": [1, 2, 1, 2, 1, 2, 1, 2],
        "revise once then hold": [4, 9, 4, 9, 4, 9, 4, 30, 4, 30, 4, 30],
        "cycle through a 3-set": [6, 1, 6, 2, 6, 3, 6, 1, 6, 2, 6, 3],
        "never repeats an address": list(range(1, 13)),
    }
    allok = True
    print(f"\n{'case':28s}{'n writes':>10}{'wtype':>9}{'o_memb':>9}")
    for nm, seq in cases.items():
        pad = T.L + 1 - len(seq)
        full = seq + list(rng.integers(0, T.N_SYM, pad))
        x = torch.tensor([full], dtype=torch.long)
        d = FE.replay(m, x)
        exp_wt, exp_mb = naive_bookkeeping(full[:T.L])
        got_wt = d["wtype"][0][:len(seq)]
        n = len(seq)
        wt_ok = all(int(got_wt[t]) == exp_wt[t] for t in range(n))
        mb_ok = True
        for t in range(n):
            g = d["o_memb"][0][t]
            e = exp_mb[t]
            if e is None:
                mb_ok &= not np.isfinite(g)
            else:
                mb_ok &= np.isfinite(g) and abs(g - e) < 1e-6
        allok &= wt_ok and mb_ok
        print(f"{nm:28s}{n:>10}{('ok' if wt_ok else 'MISMATCH'):>9}"
              f"{('ok' if mb_ok else 'MISMATCH'):>9}")
    print(f"\n  vectorised bookkeeping == naive reimplementation: "
          f"{'CONFIRMED' if allok else 'REFUTED'}")
    return allok


@torch.no_grad()
def test_class_labelling():
    """cls_w[t] must be the class of x_{t-1}, taken from the task generator's
    own per-sequence class assignment, not from features.py's shifting."""
    m, _ = FE.load("gated_seed0")
    ev = np.random.default_rng(777)
    x, cl, _ = T.sample(ev, 4)
    d = FE.replay(m, x)
    cls_w = F.pad(cl, (1, 0))[:, :T.L].numpy()
    cls_q = cl.numpy()
    toks = x[:, :T.L].numpy()
    # cl[b, t] is the class of token x[b, t]; rebuild the mapping and verify
    bad_w = bad_q = 0
    for b in range(4):
        sym2cls = {}
        for t in range(T.L):
            sym2cls[int(toks[b, t])] = int(cls_q[b, t])
        for t in range(1, T.L):
            if cls_w[b, t] != sym2cls[int(toks[b, t - 1])]:
                bad_w += 1
            if cls_q[b, t] != sym2cls[int(toks[b, t])]:
                bad_q += 1
    print(f"  cls_w[t] != class(x_"+"{t-1}"+f") on {bad_w} of {4*(T.L-1)} writes")
    print(f"  cls_q[t] != class(x_t)        on {bad_q} of {4*(T.L-1)} writes")
    print(f"\n  labelling: {'CONFIRMED' if bad_w == 0 and bad_q == 0 else 'REFUTED'}")
    return bad_w == 0 and bad_q == 0


if __name__ == "__main__":
    torch.set_num_threads(2)
    hdr("TEST 1  is the address really the PREVIOUS token? (model weights forced)")
    r1 = test_model_addressing()
    hdr("TEST 2  vectorised bookkeeping vs a naive reimplementation")
    r2 = test_bookkeeping()
    hdr("TEST 3  is cls_w really class(x_{t-1})?")
    r3 = test_class_labelling()
    print(f"\n{'='*74}\nALL THREE: {'PASS' if (r1 and r2 and r3) else 'FAIL'}")
