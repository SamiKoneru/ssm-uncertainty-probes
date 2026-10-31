"""Put 04's harness and the model package on sys.path.

Every script here reuses 04's own `auroc.py`, `features.py` and the
`gated_delta_ssm` package unchanged, so that the audit measures 04's code
rather than a reimplementation of it. Import this first.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FOUR = os.path.join(os.path.dirname(HERE), "04_volatility_history")
GDS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "gated_delta_ssm")
for p in (HERE, FOUR, GDS):
    if p not in sys.path:
        sys.path.insert(0, p)
