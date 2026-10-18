# validation

Open checks on the claims in [`../gated_delta_ssm/`](../gated_delta_ssm). These
can falsify that directory's headline, which is why they live outside it.

Nothing here copies the model. `arms.py` imports the shipped
`GatedDeltaSSM` directly, so these scripts cannot drift from the thing they
audit, and the shipped architecture needed no edits to support them.

## Why these two, and not the other eight

The headline is a 5.5% loss reduction from letting the write gate read the
memory. Two specific things could make that number mean nothing, and neither
had been tested:

| | risk | status |
|---|---|---|
| 1 | the gain is **gate capacity**, not memory access | `convergence.py` CHECK A |
| 2 | the gain is a **convergence transient** at 800 steps | `convergence.py` CHECK B |
| 3 | the gain needs a task **built to require it** | `mqar_eval.py` |

Everything else on the backlog (the `s2` replacement, the volatility memory,
chunk-freezing, oracle seeds, probe bootstraps) refines a result that these
three checks have to leave standing first.

## The three arms

Identical parameter count, 62,058. Verified by `python arms.py`.

```
none         memory_gate=False                 standard input-only baseline
ours         MLP([ w_beta.x_t , s1 , s2 ])     the published configuration
token_only   MLP([ w_beta.x_t ,  0 ,  0 ])     the control that was missing
```

`token_only` is the whole point of CHECK A. Every published comparison so far
pitted a 3-input MLP gate against **no gate at all**, which confounds "the gate
can see the memory" with "the gate has an MLP in it". This arm holds the MLP,
its parameter count, its initialisation and its FLOPs fixed, and removes only
the information, by zeroing the two memory-derived columns before the MLP sees
them. It is implemented as a wrapper around the gate module
(`MaskMemoryFeatures`) rather than a branch inside `model.py`.

## Files

```
arms.py            the three arms + the param-parity assertion. Run it directly.
convergence.py     CHECK A and CHECK B as one sweep: 3 arms x 3 lrs x 2 seeds,
                   3000 steps, evaluated every 500 so the gap is read as a
                   curve. Each arm is scored at ITS OWN best lr, because tuning
                   one lr on the gated arm and reusing it for the baseline
                   would handicap the baseline.
mqar_task.py       multi-query associative recall. Run it directly for the
                   format and a check that every scored target really is its
                   key's latest value.
mqar_eval.py       the capacity sweep on MQAR.
```

## The subtlety in MQAR, written down before the runs

MQAR is the honest version of "does this hold on an ordinary task", but two of
its three settings are **expected to show no gain, for reasons that are not a
failure of the mechanism**:

- **plain**: every write is a first write or an exact repeat. A repeat gives
  `e = S k - v = 0`, so beta multiplies zero and cannot matter. A global beta
  is already optimal.
- **rewrite**: a replaced value wants high beta and nothing anywhere wants low
  beta, so "beta = 1 always" is optimal and an input-only gate learns that
  constant immediately.
- **capacity**: the one axis that can separate the arms. Keys are only
  approximately orthogonal, so writes bleed across addresses. Once `n_pairs`
  is large relative to what the state holds cleanly, the right write strength
  depends on how contested the address already is, which is a property of the
  memory and invisible to the token. That pressure is not constructed in our
  favour; it is just what makes recall hard.

So the sweep is over `n_pairs`, and the prediction under test is that a gain
appears with capacity pressure or not at all.

A near-zero result here is **not** a failed experiment. It converts the
headline from a general claim into a scoped one, and it is also the simplest
available explanation for why the one much larger published study got the
opposite ordering to ours on real language data.

## Results

Not yet in. Logs land here as `*.log`, tracked, same convention as the rest of
the repo.

## Audit

Checked against the experiment rules in `~/.claude/CLAUDE.md`. Shortfalls are
recorded here rather than quietly fixed.

### Design changes made after seeing numbers

Both are logged with their reason, and the original result is kept.

1. **MQAR capacity axis: `n_pairs` -> state dim.** The first design swept
   `n_pairs` at full width. Calibration showed `n_pairs=32` reaches **accuracy
   1.0000 by step 100**, and `n_pairs=64` at dim 64 also saturates, so no arm
   could be distinguished from any other. Switched to sweeping `d_k = d_v`
   while holding `d_model=128`, which isolates memory capacity from embedding
   capacity. The change was forced by absence of headroom, not by an outcome.
   Usable window turned out narrow: dim 16 both variants, dim 32 rewrite only.

2. **MQAR eval stream: 12345 -> 101.** The state dims were SELECTED using
   stream 12345, and the sweep was going to SCORE on stream 12345. That is
   selection and scoring on one sample. Relaunched on fresh stream 101 after
   3 minutes and before any arm comparison existed.

### Known methodological flaws, not yet fixed

- **One reused eval stream.** 14 files score on `default_rng(12345)`, 192
  sequences. Every decision in the project was made while looking at it. See
  `eval_stream_check.py`, which is the first confirmation on streams never
  looked at. Until it reports, every number in this repo is **exploratory**.
- **lr selected and scored on the same stream.** `convergence.py` picks each
  arm's best lr by final loss on 12345 and then reports the gap on 12345. All
  three arms chose 1e-3 so the effect is probably minor, but it is the same
  flaw in miniature. `eval_stream_check.py` avoids it by fixing lr at the
  published 2e-3 with no selection.
- **No numeric pass bars were pre-registered** for `convergence.py` or
  `mqar_eval.py`. Their docstrings state the question and the kill condition
  but only a directional bar ("near 0% / near 100%"). The stated noise floor
  of 0.004 loss was asserted rather than derived, and the measured within-cell
  seed sd is 0.0001 to 0.0016, so it was 3 to 10x too conservative.
  `eval_stream_check.py` is the first file here with real numeric bars.
- **2 seeds per cell** in `convergence.py`. Adequate for a 5.9% effect,
  not for the 0.16% sub-effect noted below.
- **CHECK B does not reach convergence.** `ours` was still descending at step
  3000 (1.5342 at 2500 to 1.5309 at 3000). The check establishes that the gap
  is not closing by 3000 steps. It cannot rule out closure at 10k or beyond.
- **The synthetic task wins by construction.** `../gated_delta_ssm/task.py`
  says so in its own docstring. That is sound as an existence proof and is
  exactly why MQAR exists, but the 5.9% is not a general number.

### Anomalies observed in passing

- `token_only` beats `none` by +0.00255 loss, 6/6 cells same sign, sd 0.00032,
  which is above the measured seed noise. Gate capacity contributes something
  real but worth 2.6% of the total effect. **Exploratory**: the pre-registered
  check was a yes/no on the control, not a test for a small capacity term, and
  calling it significant uses a noise floor computed after the fact.
- Best lr is 1e-3 for all three arms, not the published 2e-3. Every number in
  the repo was taken at a slightly suboptimal lr. Not in our favour: `none`
  also prefers 1e-3, so the fair per-arm-best gap is 5.92% against 5.47%.
- `ours` at lr 2e-3 has seed sd 0.0016, 3 to 5x every other cell (0.0001 to
  0.0006). One anomalously spread cell, unexplained.
- `none` is non-monotone at the 0.001 to 0.003 scale late in training
  (lr 2e-3: 1.6310 at 1500, 1.6312 at 2000; lr 1e-3: 1.6262 at 2500, 1.6272
  at 3000). Consistent with a converged model plus eval noise.
- MQAR dim 32 rewrite sits at 0.9196 accuracy, close to the 0.97 saturation
  cutoff, so it has limited headroom and may discriminate poorly.
- At dim 16, `rewrite` is EASIER than `plain` (0.2798 vs 0.1953), which is
  backwards. Likely recency: rewritten keys are written later so they suffer
  less crosstalk erosion before the queries. The rewrite variant also has 32
  query positions against plain's 64. `plain` and `rewrite` therefore differ
  in three ways at once and **must not be compared to each other**; only
  across-arm comparisons within one variant are valid.

### Recommended: a separate adversarial session

Per the rule on headline results, this should be run by a fresh session whose
only job is to break these numbers, with these targets:

1. Re-derive the 5.9% from scratch without reading this repo's conclusions.
2. Attack `token_only`. Is masking the gate input genuinely equivalent to
   withholding the information, or does the zeroed MLP behave differently from
   a 1-input MLP in a way that matters?
3. Attack the eval protocol. 6 batches of 32 at one seed, reused everywhere.
4. Attack the task. Does the class randomisation really make input-only gating
   impossible, or merely hard?
5. Check whether the larger contrary published study's router was actually
   linear. That assumption is still load-bearing and still unverified.

## Verified: HAM does not contradict this work, and never could have

Checked 2026-10-18 against the paper itself (arXiv:2603.22325v2, "Hybrid
Associative Memories", Lufkin, Figliolia, Millidge, Krishnamurthy; v2 revised
27 March 2026), not against earlier notes. Two claims this project had been
carrying for a month were wrong.

### WRONG 1: that HAM ran our ablation and got the opposite answer

HAM's routing score controls **KV-cache admission to an attention branch**. It
does not touch the delta rule's write strength.

```
Eq 10   e_t = min_h e_t^(h),   m_t = I[e_t >= tau]
Eq 11   Attn(q_t, {k_i : m_i = 1}, {v_i : m_i = 1})
```

The RNN update carries no routing term. Eqs 7-8, Eq 15 and Algorithm 1 all
give `S_t <- f(S_{t-1}, k_t^RNN, v_t^RNN)`. The recurrent branch is an
**unmodified Gated DeltaNet**, described as already having "learnable decay
gates and write strengths", with no change to its gate equations. For the
learned routers the score enters the graph by "scaling the KV values", which
again is the attention path and not the state.

Asked directly whether the paper anywhere proposes using the prediction error,
the readout `S_{t-1}k_t`, or the residual to modulate the RNN's own write
strength, step size, beta or update gate: **no such passage exists**, in the
body or the appendices.

So HAM's beta is precisely this suite's `none` arm, and its 48.4 vs 49.1 was
never measuring our mechanism. Treating that table as a contradiction was an
error on our side, not a finding.

### WRONG 2: that cosine loses because it discards magnitude

Refuted by our own `ham_and_position.py` before this check: `ham_cos_mlp`
scores 5.06% against `ours` at 5.51% and wins on accuracy. Magnitude is worth
about 8% of the effect.

### What is true, and is what the cosine arms actually test

HAM's design space entangles two axes, and its own FLOP accounting labels them
as mutually exclusive options:

```
"via RNN state (if not using a learnable router)"
"via input   (if using a learnable router)"
```

- state-based score: hand-designed cosine distance, Eq 16, no learned weights,
  used only through `m_t = I[e_t >= tau]`. tau may be fixed or learned.
- learned router: Eq 14, `e_t = sigma(MLP(x_t))`, fed the **token only**.
  Appendix A.1 gives either `sigma(W_r x_t)` or a 3-layer GELU MLP, width 256.
- the general form it simplifies away:
  `e_t^(h) = D_theta(S_{t-1}^(h) k_t^(h), v_t^(h), x_t)`, "parametrized as a
  MLP with a sigmoid at the end". The paper states it "simplif[ies] this
  learned router by only feeding it the input x_t" and never evaluates the
  general form.
- the `Learned router + EDA` row's learned per-layer coefficient gamma_l,
  `e_t^l <- gamma_l e_t^l + (1 - gamma_l) e_t^{l-1}`, averages the LEARNED
  router's scores across layers. It is not a learned function of the residual.

So HAM writes down a learned function of exactly our features, never runs it,
and its two evaluated arms differ in both information source and learnedness
at once. Our `ham_cos_lin` 1.93% vs `ham_cos_mlp` 5.06% independently
reproduces the half of that which is about learnedness.

### Scale, stated plainly

HAM: ~800M params, 50B tokens, 16,384 context, real language, 50% KV cache.
This repo: 62,058 params, one layer, synthetic task, 256 context. Four orders
of magnitude apart. Nothing here transfers by default, which is the whole
reason MQAR and eventually real language data matter.

HAM's Table 1 reports **no seeds or variance**, so its own 0.7-point average
gap is unqualified. Its RULER gaps are much larger (MK1 39/28/15 for Fixed tau
against 95/90/69 for the learned router).

### Residual uncertainty

Verified by targeted queries against the arXiv HTML, cross-checked three ways:
the main text equations, the appendix FLOP tables (a different part of the
document, written for a different purpose, saying the same thing), and an
explicit negative-existence query. The appendix labels are the strongest
corroboration. Not verified by reading the PDF end to end personally.
