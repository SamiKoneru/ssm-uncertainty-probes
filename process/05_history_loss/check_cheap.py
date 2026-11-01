"""Pre-flight checks for `ours+cheap`. Run this BEFORE training the arm.

    python check_cheap.py

Four assertions, every one of which can fail:

  1. PARAM IDENTITY. `ours+cheap` must have exactly as many parameters as
     `ours`, `ours+hist`, `ours+rate` and `ours+oracle`. If it does not, the
     arm is not a control and no gain measured against `ours` means anything.

  2. BLIND to the expensive features. Perturbing `h_cos`, `h_cosmean` or
     `o_memb` inside the recurrence must leave the logits BIT-IDENTICAL. This
     is what licenses the claim "this arm needs only d_k of state".

  3. SIGHTED on the cheap features. Perturbing `h_ratio`, `h_meane` or
     `h_nvis` must CHANGE the logits. Check 2 on its own passes trivially for
     a model that reads nothing at all, which is the failure mode that would
     make the whole arm meaningless, so this check is not optional.

  4. SMOKE. A few optimiser steps must run and produce a finite loss, so a
     crash is found in seconds rather than after 3 x 800 steps.

The perturbation goes through the REAL forward pass, not through the gate MLP
in isolation. That matters: a changed gate input changes beta, which changes
the state S, which changes every later residual, so a sighted column
compounds and is easy to detect, while a masked column must leave the whole
trajectory untouched.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "gated_delta_ssm"))

import task as T                                                  # noqa: E402
import arms as A                                                  # noqa: E402

ARM = "ours+cheap"
CHEAP = (2, 3, 4)            # h_ratio, h_meane, h_nvis      cost d_k
EXPENSIVE = (0, 1, 5)        # h_cos, h_cosmean, o_memb      cost d_k x d_v


def perturbed(model, col, scale=3.0, seed=7):
    """Return a forward pass with history column `col` replaced by noise.

    Wraps `_history` rather than editing it, so the audited code path is the
    one that actually trains.
    """
    real = model._history
    g = torch.Generator().manual_seed(seed)

    def hooked(*args, **kw):
        h = real(*args, **kw).clone()       # clone: zero_h is reused across t
        h[:, col] = torch.randn(h.shape[0], generator=g) * scale
        return h

    model._history = hooked
    try:
        with torch.no_grad():
            return model(X)
    finally:
        model._history = real


def fail(msg):
    print(f"\nFAILED: {msg}")
    sys.exit(1)


torch.manual_seed(0)
X = T.sample(np.random.default_rng(0), 4)[0][:, :-1]

# ---- 1. parameter identity -------------------------------------------------
print("1. PARAM IDENTITY")
counts = {}
for arm in ("ours", "ours+hist", "ours+rate", ARM, "ours+oracle"):
    torch.manual_seed(0)
    counts[arm] = sum(p.numel() for p in A.build(arm).parameters())
    print(f"   {arm:13s}{counts[arm]:>9,}")
if len(set(counts.values())) != 1:
    fail(f"parameter counts differ: {counts}. ours+cheap is NOT a control.")
print(f"   all identical at {next(iter(counts.values())):,}  OK")

torch.manual_seed(0)
m = A.build(ARM)
m.eval()

# ---- the mask itself -------------------------------------------------------
want = torch.zeros(A.N_EXTRA)
for i in CHEAP:
    want[i] = 1.0
if not torch.equal(m.hist_mask, want):
    fail(f"hist_mask is {m.hist_mask.tolist()}, expected {want.tolist()}")
print(f"   hist_mask {m.hist_mask.tolist()}  "
      f"keeps {[A.HIST[i] for i in CHEAP]}  OK")

with torch.no_grad():
    ref = m(X)
print(f"   reference logits {tuple(ref.shape)}  "
      f"mean {ref.mean().item():+.6f}")

# ---- 2. blind to the expensive features ------------------------------------
print("\n2. BLIND to the d_k x d_v features (must be bit-identical)")
for col in EXPENSIVE:
    d = (perturbed(m, col) - ref).abs().max().item()
    print(f"   perturb {A.HIST[col]:11s} max |dlogit| {d:.3e}"
          f"   {'OK' if d == 0.0 else 'LEAK'}")
    if d != 0.0:
        fail(f"{A.HIST[col]} leaks into the gate: max |dlogit| {d:.3e}. "
             f"The d_k-only cost claim is false.")
print("   provably blind: no residual-direction information reaches the gate")

# ---- 3. sighted on the cheap features --------------------------------------
print("\n3. SIGHTED on the d_k features (must change the logits)")
for col in CHEAP:
    d = (perturbed(m, col) - ref).abs().max().item()
    print(f"   perturb {A.HIST[col]:11s} max |dlogit| {d:.3e}"
          f"   {'OK' if d > 0.0 else 'DEAD'}")
    if d == 0.0:
        fail(f"{A.HIST[col]} does NOT reach the gate. The arm reads nothing, "
             f"so check 2 passed vacuously and the whole arm is meaningless.")
print("   provably sighted: every cheap column moves the output")

# ---- 4. smoke --------------------------------------------------------------
print("\n4. SMOKE (5 steps at the published lr)")
torch.manual_seed(0)
m = A.build(ARM)
opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
rng = np.random.default_rng(0)
for s in range(5):
    x, _, _ = T.sample(rng, 4)
    loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, T.N_SYM),
                           x[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    print(f"   step {s}  loss {loss.item():.4f}")
    if not np.isfinite(loss.item()):
        fail(f"loss is {loss.item()} at step {s}")

print("\nALL CHECKS PASS. ours+cheap is param-matched, provably blind to the "
      "expensive features, provably sighted on the cheap ones, and trains.")
