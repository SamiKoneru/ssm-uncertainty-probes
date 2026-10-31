"""A2. Does `features.py`'s addressing do what its docstring claims, and does
its `assert` constrain what it is cited for?

Pre-registered in PREREGISTRATION.md, section A2. Five checks:

  A2.1  bookkeeping `cnt == 0` must equal `~sn_addr` on every row
  A2.2  mutation test: corrupt the addressing and see if the assert still passes
  A2.3  independent reimplementation of all seven history/oracle columns,
        written from the task definition rather than from features.py
  A2.4  `cls_w[t]` must equal the generator's class of symbol `x_{t-1}`
  A2.5  off-by-one sensitivity is in audit_offbyone.py (needs a trained model)

None of this needs a trained model: the bookkeeping is a function of the token
stream and of `e_t`, and `e_t` itself is covered by the replay assert (it feeds
`S`, which feeds the logits that the assert compares). So a randomly
initialised model is used, which also keeps the check independent of training.

    python audit_addressing.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
FOUR = os.path.join(os.path.dirname(HERE), "04_volatility_history")
sys.path.insert(0, FOUR)
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(HERE)), "gated_delta_ssm"))

import features as FE                                          # noqa: E402
import task as T                                               # noqa: E402
from model import GatedDeltaSSM                                 # noqa: E402

HIST7 = FE.HIST + FE.ORACLE


def hdr(s):
    print(f"\n{'=' * 74}\n{s}\n{'=' * 74}", flush=True)


def independent_history(seq, e):
    """Recompute every history/oracle column with an explicit per-sequence loop.

    Written from `task.py` + `model.py` directly: the write at step t puts the
    value token x_t at the address of token x_{t-1}, so the prior writes to that
    address are the earlier steps t' with x_{t'-1} == x_{t-1}.

    seq: [B, L+1] int token stream.  e: [B, L, d_v] residuals from the replay.
    Returns dict name -> [B, L] float64, NaN where features.py blanks.
    """
    B, L, _ = e.shape
    out = {n: np.full((B, L), np.nan) for n in HIST7}
    for b in range(B):
        res = {}          # address -> list of residual vectors, in write order
        vals = {}         # address -> list of value tokens, in write order
        for t in range(1, L):
            addr = int(seq[b, t - 1])
            val = int(seq[b, t])
            prior_e = res.get(addr, [])
            prior_v = vals.get(addr, [])
            et = e[b, t]
            net = float(np.linalg.norm(et))

            # h_nvis is the only column features.py does NOT blank on a fresh
            # address, so it is 0.0 rather than NaN there.
            out["h_nvis"][b, t] = float(len(prior_e))

            if prior_e:
                last = prior_e[-1]
                mean = np.sum(prior_e, axis=0)
                # clamp EACH norm at eps, as torch.cosine_similarity does.
                # Clamping the product instead crushes the cosine on repeat
                # writes, where ||e_t|| falls to ~1e-5 and the product of the
                # two norms goes below 1e-8 while neither norm does.
                cos = lambda a, c: float(
                    np.dot(a.astype(np.float64), c.astype(np.float64))
                    / (max(np.linalg.norm(a), 1e-8)
                       * max(np.linalg.norm(c), 1e-8)))
                out["h_cos"][b, t] = cos(et, last)
                out["h_cosmean"][b, t] = cos(et, mean)
                out["h_ratio"][b, t] = float(np.log(
                    (net + 1e-6) / (np.linalg.norm(last) + 1e-6)))
                out["h_meane"][b, t] = float(
                    np.mean([np.linalg.norm(p) for p in prior_e]))
                out["o_memb"][b, t] = float(val in set(prior_v))
                out["o_ndist"][b, t] = float(len(set(prior_v)))

            res.setdefault(addr, []).append(et)
            vals.setdefault(addr, []).append(val)
    return out


def replay_with_e(m, x):
    """features.py's replay, plus the raw residual vectors it does not return.

    Only the residual tensor is added; every number compared below still comes
    out of the real `FE.replay`.
    """
    tokens = x[:, :-1]
    B, L = tokens.shape
    with torch.no_grad():
        emb = m.embed(tokens)
        shifted = F.pad(emb, (0, 0, 1, 0))[:, :L]
        k = F.normalize(m.to_k(shifted), dim=-1)
        v = m.to_v(emb)
        ctx = F.silu(m.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))
        beta_token = m.to_beta(ctx)[..., 0]
        S = emb.new_zeros(B, m.d_v, m.d_k)
        es = []
        for t in range(L):
            kt, vt = k[:, t], v[:, t]
            stored = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1)
            e = stored - vt
            es.append(e)
            g = beta_token[:, t]
            if m.memory_gate:
                s1 = torch.log1p(e.norm(dim=-1))
                s2 = (vt * stored).sum(-1) / (vt.norm(dim=-1) ** 2 + 1e-6)
                g = m.gate(torch.stack([g, s1, s2], -1))[..., 0]
            beta = torch.sigmoid(g)
            S = S - beta[:, None, None] * e.unsqueeze(-1) * kt.unsqueeze(1)
    return torch.stack(es, 1).numpy()


MUTANT = '''
import os, sys, numpy as np, torch
sys.path.insert(0, {four!r})
sys.path.insert(0, {gds!r})
import features as FE, task as T
from model import GatedDeltaSSM
src = open(os.path.join({four!r}, "features.py")).read()
# the single corruption: write to the CURRENT token's address, not the previous
old = "            addr = tokens[:, t - 1]"
assert old in src, "patch target not found"
src = src.replace(old, "            addr = tokens[:, t]")
ns = {{"__name__": "mutant_features", "__file__":
       os.path.join({four!r}, "features.py")}}
exec(compile(src, "features_MUTANT.py", "exec"), ns)
torch.manual_seed(0)
m = GatedDeltaSSM(T.N_SYM, memory_gate=True, gate_width=32).eval()
x, _, _ = T.sample(np.random.default_rng(0), 4)
try:
    ns["replay"](m, x)
    print("MUTANT_ASSERT_PASSED")
except AssertionError as ex:
    print("MUTANT_ASSERT_FIRED", ex)
'''


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    m = GatedDeltaSSM(T.N_SYM, memory_gate=True, gate_width=32).eval()

    B, nb = 16, 2
    rng = np.random.default_rng(505)
    print("A2 addressing audit. Randomly initialised model (the bookkeeping is\n"
          "weight-independent), 2 batches of 16 sequences, stream 505.")

    ok = {k: True for k in ("A2.1", "A2.3", "A2.4")}
    worst = {n: 0.0 for n in HIST7}
    nanmismatch = {n: 0 for n in HIST7}
    n_rows = 0

    for bi in range(nb):
        x, cl, sn = T.sample(rng, B)
        d = FE.replay(m, x)                       # the real thing, assert runs
        e = replay_with_e(m, x)
        L = T.L
        sn_addr = F.pad(sn, (1, 0))[:, :L].numpy()
        cls_w = F.pad(cl, (1, 0))[:, :L].numpy()
        seq = x.numpy()
        n_rows += B * L

        # ---------------------------------------------------------- A2.1
        # "written before" from bookkeeping vs from the task's own seen array
        fresh_book = d["wtype"] == FE.UNSEEN
        fresh_task = ~sn_addr.astype(bool)
        # t=0 is the zero-pad address: features.py calls it UNSEEN by fiat and
        # sn_addr is False there too, so the two agree by construction.
        mism = (fresh_book != fresh_task)
        if mism.any():
            ok["A2.1"] = False
            print(f"  A2.1 batch {bi}: {mism.sum()} mismatches")

        # ---------------------------------------------------------- A2.4
        # rebuild symbol -> class from (seq, cl) pairs, independently
        for b in range(B):
            m_sym = {}
            for t in range(L):
                s_, c_ = int(seq[b, t]), int(cl[b, t])
                if m_sym.setdefault(s_, c_) != c_:
                    ok["A2.4"] = False
                    print(f"  A2.4 inconsistent class for symbol {s_}")
            for t in range(1, L):
                if m_sym[int(seq[b, t - 1])] != int(cls_w[b, t]):
                    ok["A2.4"] = False

        # ---------------------------------------------------------- A2.3
        ind = independent_history(seq, e)
        for n in HIST7:
            a, bb = d[n], ind[n]
            na, nb_ = ~np.isfinite(a), ~np.isfinite(bb)
            bad = int((na != nb_).sum())
            nanmismatch[n] += bad
            if bad:
                ok["A2.3"] = False
            both = np.isfinite(a) & np.isfinite(bb)
            if both.any():
                worst[n] = max(worst[n],
                               float(np.abs(a[both] - bb[both]).max()))

    hdr("A2.1  bookkeeping 'address written before' == task's shifted `seen`")
    print(f"  rows checked {n_rows}")
    print(f"  {'PASS' if ok['A2.1'] else 'FAIL'}: the two definitions agree "
          f"{'on every row' if ok['A2.1'] else 'NOT everywhere'}")

    hdr("A2.4  cls_w[t] == class of symbol x_{t-1}, from an independent map")
    print(f"  {'PASS' if ok['A2.4'] else 'FAIL'}")

    hdr("A2.3  independent reimplementation of every history/oracle column")
    print("  features.py uses vectorised scatter/gather over [B, N_SYM, d_v]\n"
          "  buffers; the reference below is a plain per-sequence Python loop\n"
          "  over dicts, written from task.py and model.py.")
    print(f"\n  {'column':12s}{'max abs diff':>16}{'NaN-pattern diffs':>20}")
    for n in HIST7:
        print(f"  {n:12s}{worst[n]:>16.3e}{nanmismatch[n]:>20d}")
    allok = all(worst[n] < 1e-5 for n in HIST7) and not any(
        nanmismatch.values())
    print(f"\n  bar: max abs diff < 1e-5 and identical NaN patterns")
    print(f"  {'PASS' if allok else 'FAIL'}")

    hdr("A2.2  mutation test: does the replay assert constrain the addressing?")
    print("  features.py docstring: '`replay` asserts its own recurrence\n"
          "  reproduces the shipped forward pass.' The question is whether that\n"
          "  assert gives any coverage of the ADDRESSING claim stated two\n"
          "  paragraphs above it in the same docstring.\n")
    print("  corruption applied:  addr = tokens[:, t - 1]  ->  tokens[:, t]")
    gds = os.path.join(os.path.dirname(os.path.dirname(HERE)), "gated_delta_ssm")
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "run_mutant.py")
        with open(p, "w") as f:
            f.write(MUTANT.format(four=FOUR, gds=gds))
        r = subprocess.run([sys.executable, "-I", p], capture_output=True,
                           text=True, timeout=600)
        out = (r.stdout + r.stderr).strip()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n  result: {out.splitlines()[-1] if out else '(no output)'}")
    passed = "MUTANT_ASSERT_PASSED" in out
    print(f"\n  {'DEFECT CONFIRMED' if passed else 'assert does cover it'}: "
          f"the assert {'does NOT' if passed else 'DOES'} detect a corrupted "
          f"address.")
    if passed:
        print("  The bookkeeping arrays (last_e, sum_e, cnt, ndist, valset,\n"
              "  last_val) are pure side outputs: nothing in them feeds S or\n"
              "  the logits, so the assert cannot see them. It verifies the\n"
              "  recurrence and gives ZERO coverage of the addressing.")

    hdr("A2 VERDICT")
    print(f"  A2.1 bookkeeping vs task `seen`      "
          f"{'PASS' if ok['A2.1'] else 'FAIL'}")
    print(f"  A2.3 independent reimplementation    "
          f"{'PASS' if allok else 'FAIL'}")
    print(f"  A2.4 class labelling                 "
          f"{'PASS' if ok['A2.4'] else 'FAIL'}")
    print(f"  A2.2 assert coverage of addressing   "
          f"{'DEFECT (no coverage)' if passed else 'covered'}")
    print("\n  So: the addressing as SHIPPED is correct, and the assert cited\n"
          "  for it proves nothing about it." if (ok["A2.1"] and allok
                                                  and ok["A2.4"] and passed)
          else "\n  See above.")


if __name__ == "__main__":
    main()
