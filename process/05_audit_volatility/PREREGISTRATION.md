# PRE-REGISTRATION - adversarial audit of 04_volatility_history

**Written 2026-10-18, before any measurement in this directory.** Nothing here
was edited after seeing a result. Changes, if any, are appended at the bottom
with a reason, original text kept.

My job is to find how `process/04_volatility_history/`'s negative conclusion
could be WRONG. I am not trying to confirm it. "The negative holds" is an
acceptable outcome, but it only counts if the attacks below could have broken it
and did not.

---

## The four claims under attack

As stated in `04/README.md`:

- **(a)** The shipped gate's own three features `[tok, s1, s2]` already separate
  NOISY from REVISED at pooled AUROC **0.80**.
- **(b)** A nonlinear read of those same three features is worth **+0.134**
  AUROC, where all exact per-address history is worth only **+0.081**. So the
  free nonlinear headroom is 1.7x everything history can buy.
- **(c)** The gate's own `3 -> 32 -> 32 -> 1` shape trained directly on class
  labels reaches **0.8127**, so the remaining oracle gap is credit assignment,
  not information and not capacity.
- **(d)** The rate memory in `03/archive/failed_features/rate_test.py` carried
  about **93%** of the available separation on its statistic and still bought no
  loss.

---

## Shared setup

Model and task unchanged: `gated_delta_ssm/model.py`, `gated_delta_ssm/task.py`,
published training protocol (800 steps, batch 24, AdamW lr 2e-3, clip 1.0,
`gate_width=32`, `memory_gate=True`). 2 threads, shared machine.

`04/checkpoints/` is gitignored and absent, so **every model used here is
retrained by me**. Seeds 0 and 1 are retrains of 04's two models; seeds 2 and 3
are new. Bit-exact reproduction of 04's checkpoints is not expected (CPU
reduction order); any drift is reported.

**Eval streams.** The repo's reuse hazard is `default_rng(12345)`, and 04 has now
also spent 777 and 888. So:

    505    MY PRIMARY audit stream. Fresh, never used in this repo.
    606    MY HELD-OUT stream. Not looked at until my verdict is committed.
    777    04's primary. Used here ONLY to reproduce 04's numbers.
    888    04's confirmation. Used here ONLY to reproduce 04's numbers.
    12345  reproduction of 03-era numbers only.

**Row set.** For exact comparability with 04, the primary row set is identical to
04's: writes whose address was written before in the same sequence
(`wtype != UNSEEN`), class of the write address in {NOISY, REVISED}, labelled
`class(x_{t-1})`. Splits are BY SEQUENCE, 50/50. A declared secondary row set is
all writes with a defined target (below), including STABLE, because that is what
the gate actually faces.

---

## A1 - the proxy attack. Does history predict the OPTIMAL BETA better than it
## predicts the CLASS?

04's own pre-registration lists this as a known, unmeasured limit: "AUROC on the
class label is a PROXY for 'did the gate choose the right beta'". Claims (a) and
(b) are entirely denominated in that proxy. If history is much more useful for
the optimal beta than for the class, (a) and (b) collapse.

### The optimal beta is computable in closed form

In `model.py` the write at step `t` is
`S_t = S_{t-1} - beta_t e_t k_t^T` with `e_t = stored_t - v_t` and unit `k_t`, so

    content at this address after the write  =  (1 - beta_t) stored_t + beta_t v_t

beta is a pure mixing coefficient. Let

    stored_t  = S_{t-1} k_t       what the address already holds, d_v vector
    v_t       = W_v emb(x_t)      the value being written
    u_t       = v_t - stored_t    = -e_t, the direction beta moves the content
    v_next    = W_v emb(x_{j+1}), where j = min{ i >= t : x_i = x_{t-1} }
                i.e. the value the NEXT write to this same address will store
    w_t       = v_next - stored_t

`v_next` is the right target because the delta rule's own objective at the next
read of this address is to hold the value that the next write would store there.
The squared error of the stored content against that target is

    err_t(beta) = || beta u_t - w_t ||^2 = A_t beta^2 - 2 B_t beta + C_t
    where  A_t = ||u_t||^2 = ||e_t||^2 ,  B_t = <u_t, w_t> ,  C_t = ||w_t||^2

**where:**
- `A_t` (scalar, >= 0) is the leverage of beta at this write. `A_t ~ 0` means
  beta multiplies almost nothing and cannot matter. This is the weighting the
  class-label AUROC throws away, and it is the single biggest reason the proxy
  could be wrong.
- `B_t` (scalar) is how much of the needed correction `w_t` lies along the
  direction beta actually moves.
- `C_t` (scalar, >= 0) is the error you are left with at beta = 0.

Because `A beta^2 - 2B beta + C = A (beta - B/A)^2 + (C - B^2/A)`, minimising
mean `err` over a feature set is EXACTLY a weighted least-squares fit of beta
against target `B_t/A_t` with sample weight `A_t`. No custom loss is needed.
Fitting target clipped to `[-1, 2]` (predictions are clipped to `[0,1]` anyway);
evaluation always uses exact `A_t, B_t, C_t`, so the clip cannot flatter a result.

### A1 primary metric

Relative error reduction against the best single global beta, on held-out
sequences:

    RER(F) = 1 - E[err(beta_F)] / E[err(beta_const)]

`beta_const = clip(sum B / sum A, 0, 1)` fitted on train sequences; this is the
"one global write policy" the task claims is insufficient. Then

    gain_nonlin = RER(3 feats, GBT) - RER(3 feats, linear)
    gain_hist   = RER(5 zero-state + 5 history, GBT) - RER(5 zero-state, GBT)
    R_beta      = gain_hist / gain_nonlin

04's class-AUROC answer is `R_class = 0.081 / 0.134 = 0.60`.

**Bars, on stream 505, seed-0 model:**

- **(a)+(b) COLLAPSE** if `R_beta >= 1.00`. The ordering 04 relies on flips on
  the quantity beta actually controls.
- **(a)+(b) SURVIVE this attack** if `R_beta <= 0.75`, i.e. the same ordering as
  the class proxy with room to spare.
- **AMBIGUOUS** if `0.75 < R_beta < 1.00`. Reported as ambiguous.

**Secondary bar, history's share of what a perfect class detector is worth:**

    share_hist = gain_hist / ( RER(true class label) - RER(5 zero-state, GBT) )

- `share_hist >= 0.75` -> history delivers most of the class label's beta value,
  so "history adds little" is false in beta terms.
- `share_hist <= 0.40` -> it does not.
- **Degenerate branch, declared now:** if `RER(class) <= RER(5 feats, GBT)` the
  denominator is non-positive. That means the true class label is worth nothing
  beyond the instantaneous features on the beta target, which INDEPENDENTLY
  vindicates (a); I report it that way and do not compute the ratio.

### A1b - the counterfactual loss, which 04 said it could not afford

04 says a loss measurement "would settle it directly" and was skipped at ~6
hours of training. It does not need training: the rest of the model is frozen
and only beta changes. I replay the frozen seed-0 model substituting beta from a
fitted policy ONLINE (features, including the per-address bookkeeping, are
recomputed under the substituted beta at every step) and measure actual
cross-entropy on held-out sequences.

    loss_gain(F) = CE(beta_const) - CE(beta_F)

**Declared limitation:** the policy is fitted on features generated under the
SHIPPED beta and then applied where it changes those features (off-policy
shift), and the readout is tuned to the beta distribution it trained on. This
makes the measurement a LOWER bound on the value of a better policy, not an
estimate of what retraining would give. Stated before running.

**Bars (loss units, nats). The repo's own seed sd on loss is 0.0026 to 0.0039,
which is the noise floor I must beat:**

- **(b) COLLAPSES on loss** if `CE(5 feats) - CE(5 feats + history) >= 0.0039`
  (one full seed sd) AND that gain is `>=` the nonlinearity gain
  `CE(3 linear) - CE(3 GBT)`.
- **(d)'s prediction HOLDS** if `CE(5 feats) - CE(5 feats + history) <= 0.0013`
  (one third of a seed sd).
- **AMBIGUOUS** between.
- Control that must pass or A1b is void: replaying with the gate's own beta must
  reproduce the shipped loss to `< 1e-4`.

---

## A2 - the addressing and the replay assert

`features.py` claims the write at step `t` stores `value(x_t)` at
`address(x_{t-1})`, so its class is `class(x_{t-1})` and "written before" means
`x_{t-1}` occurred strictly before `t-1`. It also says `replay` "asserts its own
recurrence reproduces the shipped forward pass". These are pass/fail code facts,
not statistics.

- **A2.1** Bookkeeping `cnt == 0` must equal `~sn_addr` (the task's own `seen`,
  shifted) on EVERY row. **Fail on any mismatch.**
- **A2.2** Mutation test on the assert. Deliberately corrupt the addressing
  (`addr = tokens[:, t]`) and re-run. **If the assert still passes, it gives
  zero coverage of the addressing claim** and the docstring's implied guarantee
  is overstated. Reported as a verification defect regardless of whether the
  shipped addressing is correct.
- **A2.3** Independent reimplementation. Recompute `h_cos, h_cosmean, h_ratio,
  h_meane, h_nvis, o_memb, o_ndist` for 2 batches with a separate, explicit
  per-sequence Python loop written from the task definition, not from
  `features.py`. **Bar: max abs difference < 1e-5** on non-NaN rows, and
  identical NaN patterns.
- **A2.4** `cls_w[t]` must equal the generator's `cls[seq[t-1]]` on every row.
- **A2.5** Off-by-one sensitivity. Report what the headline pooled AUROC becomes
  under each single-step shift of the label and of the mask, so the size of a
  hypothetical off-by-one is on record.

---

## A3 - is the +0.081 history gain capacity rather than information?

04's only capacity control was raw position (+0.003), which is one feature and
low entropy. I add FIVE junk features, matching the five history features in
count, and two of them match their GROUPED structure exactly:

    j_gauss      iid N(0,1) per row
    j_unif       iid U(0,1) per row
    j_heavy      iid Student-t(2) per row, to match h_ratio's tail
    j_seqconst   one N(0,1) draw per SEQUENCE, repeated on its rows
    j_addrconst  one N(0,1) draw per (sequence, address), repeated on its rows

`j_addrconst` is the sharpest control available: it has exactly the per-address
grouped structure of a volatility memory and carries zero class information.
`j_seqconst` doubles as a check that the by-sequence split works.

**Bars, stream 505, paired on identical rows and split:**

- **(b) WEAKENED** if `gain_junk >= +0.02` (a quarter of the +0.081 is
  reachable with noise, so part of the history gain is capacity).
- **capacity REJECTED as the explanation** if `gain_junk <= +0.01`.
- **AMBIGUOUS** between.

---

## A4 - are the error bars honest, and does the split leak?

- **A4.1 Split-seed variability.** `auroc.py`'s bootstrap resamples only TEST
  sequences with the classifier and the split held FIXED, so it cannot see
  train/test-split or fitting variability. I recompute the `inst -> inst+history`
  delta across 20 independent split seeds.
  **Bar:** `ratio_sd = SD(delta across splits) / mean(reported bootstrap SD)`.
  Error bars **UNDERSTATED** if `ratio_sd >= 2.0`; **honest** if
  `ratio_sd <= 1.5`; ambiguous between.
- **A4.2 Leakage probe.** Replace the real label with a random class drawn
  independently per (sequence, address) and rerun the full pipeline. No feature
  can predict an independent label, so **bar: |AUROC - 0.5| <= 0.02** for both
  the instantaneous and the history feature sets. Anything larger is leakage.

---

## A5 - is any of this stable across trained models?

Separability is a property of a trained model's feature distributions. 04 rests
on seed 0 with a partial seed-1 check.

Retrain seeds 0, 1 (reproductions) and 2, 3 (fresh). On stream 505 report, for
each: loss, pooled 3-feature AUROC, `delta_hist`, `gain_nonlin`, and `R_beta`.

**Bars:**

- **(b) COLLAPSES robustly** if `R_beta >= 1.00` on at least 3 of 4 seeds.
- **(b)'s ordering is SEED-DEPENDENT** if `delta_hist > gain_nonlin` on the
  class scale for at least one of the four seeds.
- 04's claim that "the 0.80 is not a fact about the task" is already conceded by
  04, so the 4-seed range of the pooled 3-feature AUROC is REPORTED, not barred.

---

## A6 - claim (d), measured on one model and one stream

04 compares `rate_delta`'s logged `c_t` AUROC 0.3690 against an exact `h_meane`
AUROC of 0.3189 from a different model on a different stream, and calls it "93%
recovered".

Two separate problems, tested separately:

- **A6.1 The arithmetic.** "93%" is `(1-0.3690)/(1-0.3189) = 0.926`, a ratio of
  raw AUROCs. AUROC's null is 0.5, not 0, so the fraction of SIGNAL recovered is
  `|0.3690-0.5| / |0.3189-0.5| = 0.131/0.181 = 0.723`. **Bar: if the correct
  normalisation gives <= 0.80, claim (d) as stated ("about 93%") is wrong** and
  must be restated, independent of any new measurement.
- **A6.2 The comparison.** Retrain `rate_delta` and the shipped `mlp` gate under
  the same protocol and measure `c_t` and exact `h_meane` on the SAME eval
  stream (505), same mask, same labelling, each on its own model, and also
  measure exact `h_meane` ON THE `rate_delta` MODEL so the channel and its
  ceiling come from one model.
  **Bar: recovered fraction (excess over chance) `>= 0.80` -> (d) survives;
  `<= 0.70` -> (d) fails as stated.**
  Note already established by reading: `mixed_task.py` and
  `gated_delta_ssm/task.py` are byte-identical, so the TASK is not a confound;
  the model class (`GatedDeltaLayer` vs `GatedDeltaSSM`), the gate, the stream
  (12345 vs 777/888) and the mask (`sn` vs `sn_addr`) all are.

---

## A7 - is "the premise bar is indecisive" honest?

04 reports 0.8095 on 777 and 0.7983 on 888 against a pre-registered 0.80 bar and
calls it indecisive because the two straddle it by 0.0017.

777 and 888 are the same size and both fresh, and git history confirms 777 was
designated primary before any number was produced. So neither is the "sounder"
one on design grounds, and the straddle is an eval-sampling question, which more
eval data answers.

**Test:** pool 777 and 888 (and 505, 606) into one estimate on the seed-0 model
and bootstrap a 95% CI over sequences.

- **"indecisive" was HONEST** if the pooled 95% CI contains 0.80.
- **"indecisive" was WRONG** if the pooled CI excludes 0.80, in whichever
  direction; the bar was then resolvable with data 04 already had.

---

## A8 - a reasoning defect in (b), no measurement needed

(b) compares `+0.134` (linear -> nonlinear read of the gate's 3 features)
against `+0.081` (nonlinear read, with vs without history). These are increments
from DIFFERENT baselines, and the first is not an investment the shipped
architecture can make: the gate already IS a `3 -> 32 -> 32 -> 1` GELU MLP, and
04's own `gate_capacity.py` shows that exact shape reaches 0.8127 against the
tree's 0.8095. The nonlinearity is therefore already inside the shipped
architecture's reach, not an alternative to building history.

On the headroom that actually remains above the gate's features, history
supplies `0.081` of the `0.92 - 0.81 = 0.11` available, i.e. about three
quarters of it.

**This is an argument, not a measurement**, and it is recorded as a defect in
(b)'s framing whatever the numbers do. It does not by itself touch (c), which is
the load-bearing claim and makes a different argument.

---

## What would make MY audit wrong

Stated now, so it is not invented later:

- The optimal-beta target `v_next` ignores the readout path (`q` vs `k`
  alignment, LayerNorm, the head) and crosstalk from other addresses. It is a
  better proxy than the class label, not the truth. A1b is there because it is
  not a proxy at all.
- The counterfactual replay is off-policy (above). A negative there is weaker
  evidence than a positive.
- Four seeds is still few. Anything I notice after seeing numbers is
  EXPLORATORY and gets labelled so, and confirmed on stream 606 or not at all.
- I retrain every model, so any disagreement with 04's figures could be my
  retrain rather than 04's measurement. A2/A6 reproductions are therefore
  reported with both numbers side by side.
