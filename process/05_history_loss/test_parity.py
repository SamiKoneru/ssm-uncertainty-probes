"""Guards against the reimplemented recurrence in arms.py drifting from the
shipped GatedDeltaSSM, and against the history block leaking gradient or
changing the forward pass when it is masked off.

    python test_parity.py

CHECK 1  arms.py with n_hist=0 reproduces the SHIPPED forward pass exactly.
CHECK 2  the masked `ours` arm's history columns are exactly zero, so it is
         information-restricted and not capacity-restricted.
CHECK 3  the history block opens NO gradient path of its own. With the gate's
         history-column weights zeroed, ours+hist and ours compute the same
         function, so EVERY parameter must get an identical gradient EXCEPT
         gate.0.weight[:, 3:], which is the one path by which the gate learns
         to use history and must therefore differ.
         (An earlier version of this check demanded that ALL gradients match,
         including that column. That assertion was false: if it held, the
         history feature could never be learned at all. Corrected, not
         weakened: the check is now strictly stronger, because it localises
         the difference instead of only bounding its size.)
CHECK 4  ours+hist and ours differ in their logits once the history weights
         are NOT zero, i.e. the feature actually reaches the output.
CHECK 5  the bookkeeping agrees with an independent naive recomputation on
         BOTH visit count and o_memb. 04's own bookkeeping lives in
         features.Bookkeeper on an unmerged branch, so it cannot be imported
         here; this checks the same way its audit did, against a
         reimplementation written from the spec rather than copied.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "gated_delta_ssm"))

import task as T                                                  # noqa: E402
from model import GatedDeltaSSM                                   # noqa: E402
import arms as A                                                  # noqa: E402
import bayes as BY                                                # noqa: E402

torch.set_num_threads(2)
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAIL.append(name)


x, _, _ = T.sample(np.random.default_rng(99), 6)
tok = x[:, :-1]

# ---------------------------------------------------------------- CHECK 1
print("CHECK 1  n_hist=0 reproduces the shipped forward pass")
for mg in (True, False):
    torch.manual_seed(7)
    ship = GatedDeltaSSM(T.N_SYM, memory_gate=mg, gate_width=32)
    mine = A.HistGatedDeltaSSM(T.N_SYM, n_hist=0, memory_gate=mg,
                               gate_width=32)
    mine.load_state_dict(ship.state_dict())
    ship.eval(); mine.eval()
    with torch.no_grad():
        err = (mine(tok) - ship(tok)).abs().max().item()
    check(f"memory_gate={mg}", err == 0.0, f"max|diff| {err:.3e}")

# ---------------------------------------------------------------- CHECK 2
print("\nCHECK 2  the masked control really is masked")
torch.manual_seed(7)
m_ours = A.build("ours")
m_ours.eval()
with torch.no_grad():
    _, _, f = m_ours(tok, return_beta=True, return_feats=True)
hist_block = f[..., 3:]
check("all five history columns are exactly 0 in `ours`",
      bool((hist_block == 0).all()), f"max|h| {hist_block.abs().max():.3e}")
check("the three instantaneous columns are NOT all zero",
      bool(f[..., :3].abs().max() > 0))

torch.manual_seed(7)
m_rate = A.build("ours+rate")
m_rate.eval()
with torch.no_grad():
    _, _, fr = m_rate(tok, return_beta=True, return_feats=True)
live = [i for i in range(5) if fr[..., 3 + i].abs().max() > 0]
check("`ours+rate` has exactly one live history column", live == [3],
      f"live columns {[A.HIST[i] for i in live]}")

torch.manual_seed(7)
m_orc = A.build("ours+oracle")
m_orc.eval()
with torch.no_grad():
    _, _, fo = m_orc(tok, return_beta=True, return_feats=True)
live = [i for i in range(A.N_EXTRA) if fo[..., 3 + i].abs().max() > 0]
check("`ours+oracle` has every column live, o_memb included",
      live == list(range(A.N_EXTRA)),
      f"live columns {[A.HIST[i] for i in live]}")
check("o_memb is a 0/1 indicator",
      bool(((fo[..., 8] == 0) | (fo[..., 8] == 1)).all()))

torch.manual_seed(7)
m_by = A.build("ours+bayes")
m_by.eval()
with torch.no_grad():
    _, _, fb = m_by(tok, return_beta=True, return_feats=True)
bp = fb[..., 9]
check("`ours+bayes` adds a 10th column in [0, 1]",
      bool((bp >= 0).all() and (bp <= 1).all()),
      f"range [{bp.min():.4f}, {bp.max():.4f}]")
# a fresh address and t=0 carry no evidence, so the flat posterior 0.5
nvis0 = torch.expm1(fb[..., 7]).round().long() == 0
check("b_post is exactly 0.5 wherever there is no history",
      bool((bp[nvis0] == 0.5).all()),
      f"{int(nvis0.sum())} such positions")
check("b_post is NOT 0.5 everywhere else",
      bool((bp[~nvis0] != 0.5).float().mean() > 0.9),
      f"{100 * (bp[~nvis0] != 0.5).float().mean():.1f}% informative")
check("b_post matches bayes.posterior_batch recomputed outside the model",
      bool(torch.allclose(bp, BY.posterior_batch(tok)[0], atol=1e-6)))
ok_sc, pa, pb, pref = BY.selfcheck()
check("bayes.py likelihoods normalise to 1/N_SYM on one observation", ok_sc,
      f"{pa:.10f} / {pb:.10f} vs {pref:.10f}")

# ---------------------------------------------------------------- CHECK 3
print("\nCHECK 3  history carries no gradient of its own")
torch.manual_seed(7)
m_h = A.build("ours+hist")
torch.manual_seed(7)
m_o = A.build("ours")
# zero the gate's first-layer weights on the history columns in BOTH, so the
# two models compute the identical function. Any gradient difference then
# means the history block leaked a path into the graph.
with torch.no_grad():
    m_h.gate[0].weight[:, 3:] = 0.0
    m_o.gate[0].weight[:, 3:] = 0.0
for m in (m_h, m_o):
    m.zero_grad()
    m(tok).square().mean().backward()
worst, where = 0.0, "-"
for (n, a), (_, b) in zip(m_h.named_parameters(), m_o.named_parameters()):
    if n == "gate.0.weight":
        continue                       # handled separately below
    d = (a.grad - b.grad).abs().max().item()
    if d > worst:
        worst, where = d, n
check("no gradient reaches any parameter through the bookkeeping",
      worst == 0.0, f"worst {where} max|dgrad| {worst:.3e}")
gw = (m_h.gate[0].weight.grad - m_o.gate[0].weight.grad).abs()
check("gate.0.weight instantaneous columns [0:3] also identical",
      gw[:, :3].max().item() == 0.0, f"max|dgrad| {gw[:, :3].max():.3e}")
check("gate.0.weight history columns [3:8] DO differ, so history is learnable",
      gw[:, 3:].max().item() > 0.0, f"max|dgrad| {gw[:, 3:].max():.3e}")

# ---------------------------------------------------------------- CHECK 4
print("\nCHECK 4  history actually reaches the output when it is not masked")
torch.manual_seed(7)
m_h = A.build("ours+hist")
torch.manual_seed(7)
m_o = A.build("ours")
m_h.eval(); m_o.eval()
with torch.no_grad():
    diff = (m_h(tok) - m_o(tok)).abs().max().item()
check("ours+hist differs from ours at identical init", diff > 1e-6,
      f"max|diff| {diff:.4f}")
check("h_ratio clamp rate is reported",
      m_h.n_hist_rows > 0,
      f"{m_h.n_clamped}/{m_h.n_hist_rows} = "
      f"{100 * m_h.n_clamped / max(m_h.n_hist_rows, 1):.2f}% of seen writes")

# ---------------------------------------------------------------- CHECK 5
print("\nCHECK 5  bookkeeping vs an independent naive reimplementation")
# Written from the spec, one sequence at a time, no vectorisation and no
# reference to arms.py: the write at step t goes to the address from token
# x_{t-1} carrying the value from token x_t.
with torch.no_grad():
    _, _, fh = m_orc(tok, return_beta=True, return_feats=True)
mine_nvis = torch.expm1(fh[..., 7]).round().long().numpy()
mine_memb = fh[..., 8].numpy()
B, L = tok.shape
ref_nvis = np.zeros((B, L), np.int64)
ref_memb = np.zeros((B, L), np.float64)
for b in range(B):
    cnt = {}
    seen_vals = {}
    for t in range(1, L):
        a, val = int(tok[b, t - 1]), int(tok[b, t])
        ref_nvis[b, t] = cnt.get(a, 0)
        # vacuously false on a fresh address, matching the blanking convention
        ref_memb[b, t] = 1.0 if val in seen_vals.get(a, set()) else 0.0
        cnt[a] = cnt.get(a, 0) + 1
        seen_vals.setdefault(a, set()).add(val)
check("visit counts match the naive reimplementation",
      bool((mine_nvis == ref_nvis).all()),
      f"mismatches {int((mine_nvis != ref_nvis).sum())}/{B * L}")
check("o_memb matches the naive reimplementation",
      bool((mine_memb == ref_memb).all()),
      f"mismatches {int((mine_memb != ref_memb).sum())}/{B * L}, "
      f"base rate {ref_memb.mean():.3f}")

print(f"\n{'ALL CHECKS PASS' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
