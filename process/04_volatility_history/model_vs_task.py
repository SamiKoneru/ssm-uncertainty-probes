"""Is the separability of NOISY from REVISED in (s1,s2) a property of the TASK
or of the TRAINED GATE?

legacy_0476.py reproduced signinit_test's 0.4763 to within 0.005 on the model it
was measured on, a LINEAR gate. The shipped MLP gate gives 0.713 under the same
procedure and 0.81 under correct conditioning. So the 0.237 gap is the MODEL,
not the bookkeeping. If that holds with stream and batch count held fixed, then
beta shapes S, S shapes (s1,s2), and a better gate MANUFACTURES a more separable
feature distribution for itself.

Runs both trained models over the SAME eval streams with the SAME conditioning,
so the only thing that differs is the gate.

    python model_vs_task.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "03_uncertainty_probe"))

import auroc as A
import features as FE
from legacy_0476 import extract
from gated_model import GatedDeltaLayer
import mixed_task as T


def legacy_rows(stream, batches):
    m = GatedDeltaLayer(T.N_SYM, gate="both", a_init=(0.0, 0.0))
    m.load_state_dict(torch.load(
        os.path.join(HERE, "checkpoints", "legacy_both_seed0.pt"),
        weights_only=False)["state_dict"])
    m.eval()
    return extract(m, stream, batches)


def shipped_rows(stream, batches):
    m, _ = FE.load("gated_seed0")
    return FE.collect(m, stream, batches=batches)


if __name__ == "__main__":
    torch.set_num_threads(2)
    print("NOISY vs REVISED, exact bookkeeping, class(x_{t-1}) labelling.")
    print("Both models over the same streams and the same 6 batches of 32.\n")
    print(f"{'model':28s}{'stream':>8}{'s1 pooled':>11}{'s1 change':>11}"
          f"{'(s1,s2) gbm pooled':>21}")
    for nm, fn in (("legacy linear gate=both", legacy_rows),
                   ("shipped MLP 3->32->32->1", shipped_rows)):
        for stream in (777, 888, 999):
            d = fn(stream, 6)
            clw, wt = d["cls_w"], d["wtype"]
            lab = (clw == 2).astype(int)
            nr = np.isin(clw, [1, 2])
            seen = nr & (wt != 0)
            chg = nr & (wt == 2)
            a_p, _ = A.single(d["f_s1"][seen], lab[seen])
            a_c, _ = A.single(d["f_s1"][chg], lab[chg])
            a_j, sd, _, _ = A.joint(d, seen, ["f_s1", "f_s2"], lab, d["seq"])
            print(f"{nm:28s}{stream:>8}{a_p:>11.4f}{a_c:>11.4f}"
                  f"{a_j:>15.4f} +/-{sd:.4f}")
    print("\nand the loss each model reaches, for context:")
    print("  legacy linear gate=both   1.5882   (+2.74% vs the 1.6329 baseline)")
    print("  shipped MLP w=32          1.5428   (+5.47%)")
