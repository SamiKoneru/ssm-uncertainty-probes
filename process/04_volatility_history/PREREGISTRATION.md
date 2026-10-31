# PRE-REGISTRATION - written before any number was produced

Session question: **does long-term per-address volatility history add anything to
delta-rule write strength (beta) beyond what the INSTANTANEOUS residual already
provides?**

Written 2026-10-18, before running `separability.py`. Nothing in this file was
edited after seeing a result. Changes, if any, are appended at the bottom with a
reason and the original text kept.

---

## Shared setup

Model: `gated_delta_ssm/model.py`, `GatedDeltaSSM(vocab=40, memory_gate=True,
gate_width=32)`, the shipped configuration. Trained with the published protocol
(`train.py`): 800 steps, batch 24, AdamW lr 2e-3, grad clip 1.0.

Task: `gated_delta_ssm/task.py` unmodified.

**Eval streams.** The repo's reuse hazard is that 14 files score on
`default_rng(12345)` and every past design decision was made against that one
192-sequence sample. So:

    seed 777   PRIMARY measurement stream. Fresh, never used in this repo.
    seed 12345 reproduction only, to explain the three recorded numbers.
    seed 888   HELD OUT. Not looked at until the final confirmation of whatever
               conclusion this session reaches. If I look at it early I will say so.

**Labelling.** `beta_t` writes to the address derived from token `x_{t-1}` with
value from `x_t`. The class of the write is therefore `class(x_{t-1})`, i.e.
`F.pad(cl, (1,0))[:, :L]`. Labelling by `class(x_t)` is the known trap and is
measured here only to size it.

**Write types.** A write at step `t` to address `a = x_{t-1}`:

    unseen   address `a` has not been written before in this sequence
    repeat   `a` seen before and `x_t` equals the value last written at `a`
    change   `a` seen before and `x_t` differs from the value last written at `a`

All AUROCs are NOISY (negative class) vs REVISED (positive class), so AUROC > 0.5
means the statistic reads higher on REVISED.

**Classifier protocol** where a classifier is used: logistic regression on
standardised features and a gradient-boosted tree, split **BY SEQUENCE** (not by
token), 50/50. A per-token split leaks, because writes inside one sequence share
the address's own option set and revision history.

---

## TASK 1 - which of the three recorded numbers is right

The repo records three values for how well instantaneous features separate the
two hard classes:

    0.834   process/README.md:109                      "s1 ... on CHANGE writes"
    0.353   process/README.md:110                      "s1 ... POOLED"
    0.476   archive/failed_features/rate_test.py:11     "(s1,s2) gets 0.476 = chance"

**Question.** Are these three measurements of the same quantity (so at most one
can be right), or of different quantities (so the disagreement is bookkeeping)?

**Metric.** AUROC, NOISY vs REVISED, under the full cross of
{feature set} x {conditioning} x {labelling}.

**Prediction, stated now.** They are different quantities and all three
reproduce: 0.834 is s1 alone on change writes, 0.353 is s1 alone pooled over all
seen writes, 0.476 is a 3-way `(s1,s2)` gradient-boosted classifier's
`P(REVISED)` pooled over all seen writes, on a different trained model and a
different eval seed.

**Pass bar for "reproduced".** Each recorded value within +/- 0.04 of my
measurement of the quantity I claim it is, on stream 12345 (the stream those
numbers were taken on). Outside that, the explanation is wrong and I say so.

### The premise test that actually matters

The gate must choose beta at every write without being told whether that write
is a repeat or a change. So the operationally relevant number is the POOLED one,
with a classifier free to recover the repeat/change distinction from the features
it has.

    INSTANTANEOUS feature set, everything the gate can see with ZERO extra state:
        w_beta . x_t        the token term the baseline already has
        log(1+||e_t||)      s1
        <v_t,stored>/||v||^2  s2
        ||S_{t-1} k_t||     the recommended s2 replacement (model.py:145)
        ||v_t||

**PASS (premise dies).** Pooled AUROC >= 0.80. Instantaneous information is then
close to sufficient, the remaining oracle gap is an optimisation or capacity
problem, not an information problem, and a volatility memory is not worth
building. **This is the outcome that kills the session's own idea, and it is live:
the set includes two features the shipped gate does not use.**

**FAIL (premise supported).** Pooled AUROC <= 0.70. There is an information gap
for history to fill.

**AMBIGUOUS.** 0.70 to 0.80. Reported as ambiguous, no mechanism recommended.

### Secondary, same run, no separate bar

- fraction of seen writes that are repeats, per class. The repo asserts
  "NOISY = ~all changes"; with `N_OPT=4` a redraw repeats about 1 in 4, so I
  expect ~0.3, not ~0.0. Reported as an anomaly if it differs from the claim.
- mean `||e_t||` on repeat vs change writes. If `||e||` on repeats is not small,
  then "beta multiplies zero on repeats" is false at `d_k=128` and the pooled
  conditioning is not merely diluted but actively load-bearing.
- AUROC under the WRONG labelling, to size the trap.

---

## TASK 2 - is the history signal there at all

**Hypothesis.** Successive residual DIRECTIONS at the SAME address separate the
classes: a NOISY address redraws, so its residuals scatter; a REVISED address has
one persistent error being slowly corrected under the low beta the gate actually
learns (0.13 to 0.16 overall), so its residuals stay aligned.

**Measured with exact out-of-model bookkeeping**, a dict of per-sequence
per-address history. This is deliberately an UPPER BOUND: it has no superposition
crosstalk, which is what killed every per-key store in this repo. If the signal
is absent even here, no mechanism can recover it and the question is closed. If
it is present, crosstalk is the next hurdle and is not tested here.

    HISTORY features, each needing NEW STATE to compute inside a layer:
        h_cos      cos(e_t, e_prev)          e_prev = residual at the previous
                                             write to this address
        h_cosmean  cos(e_t, mean of all previous residuals at this address)
        h_ratio    ||e_t|| / ||e_prev||
        h_meane    running mean of ||e|| at this address  (the rate memory's
                                             ground truth, crosstalk-free)
        h_nvis     number of previous writes to this address

    ORACLE-ISH history, reported separately and NEVER counted as a mechanism:
        o_memb     is v_t a member of the set of values this address has shown
                   before?  NOISY redraws from a fixed set so yes; REVISED moves
                   to a fresh symbol so no. This is close to the class label by
                   construction and is here only as a ceiling.
        o_ndist    number of distinct values seen at this address

**Predicted direction.** `h_cos` higher for REVISED than NOISY. If it comes out
the other way that is still informative and gets reported as a direction error,
not as a success.

**Primary metric.** Incremental AUROC on stream 777, pooled over all seen writes,
by-sequence split:

    delta = AUROC(instantaneous + history) - AUROC(instantaneous)

**PASS.** `delta >= +0.05` AND absolute AUROC(inst+hist) `>= 0.70`.
Justification for +0.05 rather than something smaller: the mechanism costs
`d_k x d_v` of new state, as large as the memory itself, and its update is
state-dependent, so it is strictly worse for parallelism than the current gate,
which is already a sequential Python loop. The repo dismissed a +0.007 AUROC gain
as "almost nothing" in `A_uncertainty/probe_test.py`; +0.05 is 7x that.

**KILL.** `delta < +0.02` or absolute AUROC `< 0.60`. Report the negative, do not
build a volatility memory, stop.

**AMBIGUOUS.** Anything between. Recommend against building, given the
parallelism cost.

**Also killed** if `h_cos` only separates because it is a proxy for repeat-vs-
change: tested by reporting AUROC of each history feature WITHIN change writes
and WITHIN repeat writes. A feature that is at chance inside both strata has no
information beyond the write type, which the instantaneous features already
carry.

---

## What would make this whole session's premise collapse

The target is the oracle's 1.4pp accuracy edge and the `oracle_mlp` 8.77% vs
5.67% loss gap. That target assumes the class label is the thing worth knowing.
It is worth checking rather than assuming: if `o_memb` and `o_ndist`, which are
nearly the label itself, cannot separate the classes from per-address history at
`L=256` (about 6 visits per address), then "the class label IS each key's
long-term behaviour" is false at this sequence length, and no history mechanism
of any kind can reach the oracle. Reported either way.

## Known limits of this design, stated now

- One trained model, one seed, for the primary measurement. Separability is a
  property of the trained model's feature distributions, so a second seed is a
  real risk and is run only if the first clears or nearly clears a bar.
- AUROC on the class label is a PROXY for "did the gate choose the right beta".
  A gate can be right without knowing the class: on writes where `||e||` is
  small, beta multiplies almost nothing. No loss number is produced by this
  session unless a bar is cleared.
- The machine is shared with another session's jobs, so everything here runs at
  2 threads and the numbers are not timings.
