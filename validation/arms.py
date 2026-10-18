"""
Shared arm construction for the validation sweeps.

Imports the SHIPPED model from ../gated_delta_ssm/model.py. Nothing here is a
copy of it, so these scripts cannot drift from the thing they are auditing.

THREE ARMS
    none         memory_gate=False. The standard input-only baseline.
    ours         memory_gate=True. [w_beta.x_t , s1 , s2] through the 32-wide MLP.
    token_only   memory_gate=True, but the two memory-derived columns are ZEROED
                 before the MLP sees them, i.e. [w_beta.x_t , 0 , 0].

token_only is the control that was missing. It holds EVERYTHING fixed except
the information: same MLP, same parameter count, same initialisation, same
FLOPs (s1 and s2 are still computed, just masked). The only difference between
it and `ours` is whether the gate can see the memory. If token_only recovers
most of the gap, the gain came from gate capacity and not from reading state.

Implemented by wrapping the gate module rather than editing model.py, so the
shipped architecture is untouched.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "gated_delta_ssm"))
from model import GatedDeltaSSM                                   # noqa: E402

ARMS = ("none", "ours", "token_only")


class MaskMemoryFeatures(nn.Module):
    """Zeroes the memory-derived columns of the gate input, keeping the MLP.

    The real gate is held as a submodule, so parameters, their count and their
    initialisation are all identical to `ours`.
    """

    def __init__(self, inner, n_in=3, keep=(0,)):
        super().__init__()
        self.inner = inner
        mask = torch.zeros(n_in)
        for i in keep:
            mask[i] = 1.0
        self.register_buffer("mask", mask)

    def forward(self, x):
        return self.inner(x * self.mask)


def build(arm, vocab, gate_width=32, **kw):
    """Construct one arm. Wrapping happens BEFORE the optimiser is built."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
    m = GatedDeltaSSM(vocab, memory_gate=(arm != "none"),
                      gate_width=gate_width, **kw)
    if arm == "token_only":
        m.gate = MaskMemoryFeatures(m.gate)
    return m


if __name__ == "__main__":
    for a in ARMS:
        m = build(a, 40)
        n = sum(p.numel() for p in m.parameters())
        print(f"{a:12s} params {n:,}")
    # the control must be information-restricted, NOT capacity-restricted
    a, b = build("ours", 40), build("token_only", 40)
    na = sum(p.numel() for p in a.parameters())
    nb = sum(p.numel() for p in b.parameters())
    print(f"\nours vs token_only param count: {na:,} vs {nb:,} -> "
          f"{'IDENTICAL' if na == nb else 'MISMATCH, control is invalid'}")
