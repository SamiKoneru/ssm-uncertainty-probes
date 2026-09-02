"""
Does the magnitude/coherence of a norm-preserving (unitary) SSM state predict
per-step prediction error BETTER than the cheap baselines (output entropy,
max-softmax)?  This is the single load-bearing question.

Architecture: the SIMPLEST unitary SSM -- a complex diagonal state whose
transition is pure phase rotation (magnitude-preserving by construction).
If the readout doesn't beat entropy here, it won't on DeltaProduct/AUSSM either.

Task (Option A): a stochastic finite-state process with a DIFFICULTY GRADIENT.
Some steps are deterministic (genuinely predictable) and some are random
(genuinely uncertain), and WE KNOW WHICH -- that's the ground truth we test against.

Runs on CPU in minutes. No GPU needed.

    pip install torch numpy scikit-learn
    python uncertainty_test.py
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# ============================================================================
# TASK: stochastic finite-state process with known per-step uncertainty
# ============================================================================
#
# Vocab of V symbols. The process alternates between two kinds of steps:
#   - DETERMINISTIC runs: next symbol is a fixed function of current (predictable)
#   - RANDOM steps:       next symbol is uniformly random  (genuinely uncertain)
# We record, per step, whether it was a random step. That's ground-truth "should
# be uncertain". We also record whether the model actually predicted correctly.
#
# The key test: on steps the model got WRONG but where output-entropy was LOW
# (confidently wrong), does the STATE MAGNITUDE still flag them?

class StochasticProcess:
    def __init__(self, vocab=8, det_run_len=(3, 8), seed=0):
        self.V = vocab
        self.det_run_len = det_run_len
        self.rng = np.random.default_rng(seed)
        # A fixed deterministic transition map (used during deterministic runs).
        self.det_next = self.rng.permutation(self.V)

    def sample(self, length):
        seq = np.zeros(length, dtype=np.int64)
        is_random = np.zeros(length, dtype=np.int64)  # ground truth: was this step's TARGET random?
        seq[0] = self.rng.integers(self.V)
        run_left = self.rng.integers(*self.det_run_len)
        in_det = True
        for t in range(1, length):
            if in_det and run_left > 0:
                seq[t] = self.det_next[seq[t - 1]]   # predictable
                is_random[t] = 0
                run_left -= 1
                if run_left == 0:
                    in_det = False
            else:
                seq[t] = self.rng.integers(self.V)   # genuinely random
                is_random[t] = 1
                # after a random step, maybe start a new deterministic run
                if self.rng.random() < 0.6:
                    in_det = True
                    run_left = self.rng.integers(*self.det_run_len)
        return seq, is_random


def make_batch(proc, batch, length):
    seqs, rands = [], []
    for _ in range(batch):
        s, r = proc.sample(length)
        seqs.append(s); rands.append(r)
    return (torch.tensor(np.stack(seqs)),
            torch.tensor(np.stack(rands)))


# ============================================================================
# MODEL: simplest unitary SSM -- complex diagonal state, pure-rotation transition
# ============================================================================

class UnitarySSM(nn.Module):
    """
    Complex diagonal state h (N complex numbers). Transition is multiplication
    by e^{i*theta} per channel -- a pure rotation, so |h| is preserved by the
    dynamics and only changes via the input write. That makes |h| a clean
    readout of accumulated input interference.
    """
    def __init__(self, vocab=8, d_model=64, n_state=64):
        super().__init__()
        self.V = vocab
        self.N = n_state
        self.embed = nn.Embedding(vocab, d_model)
        # per-channel rotation angles (fixed spectrum, learnable)
        self.theta = nn.Parameter(torch.randn(n_state) * 0.5)
        # input write: real token embedding -> complex write vector (real, imag)
        self.write_re = nn.Linear(d_model, n_state)
        self.write_im = nn.Linear(d_model, n_state)
        # readout: complex state (concat re, im) -> logits
        self.readout = nn.Linear(2 * n_state, vocab)

    def forward(self, x):
        """
        x: [B, L] token ids.
        Returns logits [B, L, V] and the per-step state magnitude [B, L].
        """
        B, L = x.shape
        emb = self.embed(x)                      # [B, L, d]
        hr = torch.zeros(B, self.N)              # real part of state
        hi = torch.zeros(B, self.N)              # imag part of state
        cos, sin = torch.cos(self.theta), torch.sin(self.theta)

        logits_all, mag_all = [], []
        for t in range(L):
            # --- rotate state (magnitude-preserving) ---
            nr = hr * cos - hi * sin
            ni = hr * sin + hi * cos
            hr, hi = nr, ni
            # --- write input (this is what changes magnitude) ---
            wr = self.write_re(emb[:, t, :])
            wi = self.write_im(emb[:, t, :])
            hr = hr + wr
            hi = hi + wi
            # slight per-step magnitude control so it doesn't grow unbounded:
            # normalize by a running scale (keeps |h| meaningful, still lets
            # interference show up as relative magnitude)
            # --- readout ---
            state = torch.cat([hr, hi], dim=-1)  # [B, 2N]
            logits_all.append(self.readout(state))
            mag_all.append(torch.sqrt(hr**2 + hi**2 + 1e-8).mean(dim=-1))  # [B]

        return torch.stack(logits_all, 1), torch.stack(mag_all, 1)


# ============================================================================
# TRAIN
# ============================================================================

def train(model, proc, steps=2000, batch=64, length=120, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(steps):
        x, _ = make_batch(proc, batch, length + 1)
        inp, tgt = x[:, :-1], x[:, 1:]
        logits, _ = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, model.V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"  step {step+1:4d}  loss {loss.item():.4f}")


# ============================================================================
# EVALUATE: compare signals at predicting per-step error
# ============================================================================

@torch.no_grad()
def evaluate(model, proc, n_seqs=200, length=200):
    model.eval()
    x, is_random = make_batch(proc, n_seqs, length + 1)
    inp, tgt = x[:, :-1], x[:, 1:]
    logits, mag = model(inp)                      # [B,L,V], [B,L]

    probs = F.softmax(logits, dim=-1)
    pred = probs.argmax(-1)
    wrong = (pred != tgt).float()                 # ground-truth error, per step

    # --- candidate signal (ours) ---
    state_mag = mag                               # [B,L] magnitude of state

    # --- baselines ---
    entropy = -(probs * torch.log(probs + 1e-9)).sum(-1)   # output entropy
    max_prob = probs.max(-1).values
    softmax_uncert = 1.0 - max_prob                          # 1 - top prob

    # flatten (drop first few steps where state is warming up)
    warm = 5
    def flat(z): return z[:, warm:].reshape(-1).numpy()
    y = flat(wrong)                               # 1 = model was wrong

    signals = {
        "state_magnitude (OURS)": flat(state_mag),
        "output_entropy":         flat(entropy),
        "softmax_uncertainty":    flat(softmax_uncert),
    }

    print("\n=== AUROC: does the signal predict per-step ERROR? (0.5=useless, 1=perfect) ===")
    aurocs = {}
    for name, sig in signals.items():
        # a signal predicts error if higher signal -> more likely wrong
        try:
            auc = roc_auc_score(y, sig)
        except ValueError:
            auc = float("nan")
        aurocs[name] = auc
        print(f"  {name:26s}  AUROC = {auc:.3f}")

    # --- THE KEY TEST: confidently-wrong steps ---
    # steps where the model was WRONG but output entropy was LOW (confident).
    # Does our state magnitude flag these where entropy fails?
    ent = flat(entropy)
    mags = flat(state_mag)
    low_ent = ent < np.percentile(ent, 33)        # most-confident third
    wrong_mask = y > 0.5
    conf_wrong = low_ent & wrong_mask             # confidently wrong
    conf_right = low_ent & (~wrong_mask)          # confidently right

    print("\n=== KEY TEST: on CONFIDENT steps, does state magnitude separate wrong from right? ===")
    n_cw, n_cr = conf_wrong.sum(), conf_right.sum()
    print(f"  confident-and-wrong steps: {n_cw}    confident-and-right steps: {n_cr}")
    if n_cw > 10 and n_cr > 10:
        # among confident steps only, can state magnitude tell wrong from right?
        sub_y = np.concatenate([np.ones(n_cw), np.zeros(n_cr)])
        sub_sig = np.concatenate([mags[conf_wrong], mags[conf_right]])
        try:
            sub_auc = roc_auc_score(sub_y, sub_sig)
            print(f"  state_magnitude AUROC on confident-only steps = {sub_auc:.3f}")
            print("  (>0.6 here = magnitude catches errors entropy is blind to. THIS is the claim.)")
        except ValueError:
            print("  (not enough variation to compute)")
    else:
        print("  (not enough confident-wrong steps; model may be too accurate or too poor)")

    return aurocs


# ============================================================================
# MAIN
# ============================================================================

def main():
    torch.manual_seed(0); np.random.seed(0)
    print("Building stochastic process (deterministic runs + random steps)...")
    proc = StochasticProcess(vocab=8, seed=0)

    model = UnitarySSM(vocab=8, d_model=64, n_state=64)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: unitary complex-diagonal SSM, {n_params:,} params\n")

    print("Training (a few minutes on CPU)...")
    train(model, proc, steps=2000, batch=64, length=120)

    print("\nEvaluating signals...")
    evaluate(model, proc)

    print("\n" + "=" * 70)
    print("HOW TO READ THIS:")
    print("- If state_magnitude AUROC >> output_entropy AUROC: the readout is real.")
    print("- If they're about equal: the state signal is redundant with entropy.")
    print("  (the likely outcome -- and an honest negative result worth knowing.)")
    print("- The KEY TEST (confident-only AUROC) is the decisive one: does magnitude")
    print("  catch confidently-wrong steps that entropy, by definition, cannot flag?")
    print("=" * 70)


if __name__ == "__main__":
    main()
