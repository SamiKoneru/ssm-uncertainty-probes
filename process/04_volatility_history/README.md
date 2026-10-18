# 04_volatility_history - does long-term volatility add anything to beta?

**ANSWER: NO. Not worth building.** The information gap is real and
reproducible, but it is smaller than the headroom already sitting unused inside
the three features the gate has, the one mechanism that was built for it
already delivered the signal and bought no loss, and it would cost state as
large as the memory itself.

Pre-registered in [`PREREGISTRATION.md`](PREREGISTRATION.md) before any
measurement. Conclusion and confirmation bars fixed in
[`CONFIRMATION_PLAN.md`](CONFIRMATION_PLAN.md) before the held-out stream was
touched. All five confirmation bars passed on stream 888.

---

## The question

`gated_delta_ssm/task.py` has two classes that want opposite write policies and
produce a large instantaneous residual for both:

    NOISY    successor redrawn from a FIXED set   old obs still VALID   wants LOW beta
    REVISED  successor PERMANENTLY changes        old obs now WRONG     wants HIGH beta

The session's premise was that what separates them is the TIME STRUCTURE of the
changes, so no single-step feature can recover it, so the oracle's remaining
1.4pp accuracy edge needs a new per-address volatility memory.

**The premise is about three-quarters wrong, for a reason that was in front of
us the whole time: `S_{t-1}` IS a per-address history summary.** A "single-step"
gate feature is a function of the entire write history, compressed. The shipped
gate's own three features separate the two classes at pooled AUROC **0.80**.

## Headline numbers

One trained model (`gated_seed0`, published protocol), NOISY vs REVISED, pooled
over all writes to an address written before, by-sequence split, sequence-level
bootstrap. Two eval streams, both fresh: **777** primary, **888** held out until
the conclusion was committed.

```
                                                      777      888
read of the shipped gate's own [tok, s1, s2]
  linear                                            0.6758   0.6613
  gradient-boosted tree                             0.8095   0.7983
  spline basis over the SAME three features         0.8112        -
+ the two other zero-extra-state features             0.8122   0.8015
+ EXACT crosstalk-free per-address history          0.8929   0.8830
+ near-label oracle (per-address value set)         0.9206   0.9154
```

so, as increments on the pooled scale:

```
  nonlinearity on features the gate ALREADY HAS     +0.134   +0.137   zero new state
  two more zero-extra-state features                +0.003   +0.003   zero new state
  ALL exact per-address history                     +0.081   +0.082   d_k x d_v new state
  a near-label oracle on top of that                +0.028   +0.032   not a mechanism
```

**The free nonlinear headroom is 1.7x everything history can buy.** That is the
same answer `process/03` reached from the loss side ("the actual deficit was gate
CAPACITY"), now reached independently from the information side.

## Every pre-registered bar, and the one that does not resolve

```
TASK 1 premise test   pooled AUROC of the shipped gate's own features
  bar  >= 0.80 premise DIES / <= 0.70 premise SUPPORTED / between AMBIGUOUS
  777  0.8095 +/- 0.0059   fires: premise dies
  888  0.7983 +/- 0.0064   does NOT fire
  INDECISIVE. The quantity sits ON the bar, and the two streams straddle it by
  0.0017, a quarter of one bootstrap SD. I set a bar at a value this data cannot
  resolve. Not retroactively moved; recorded as mis-specified.

TASK 1 reproduction   each recorded value within +/- 0.04 of my measurement
  0.834 = s1 alone, CHANGE writes         0.8073 / 0.8121    PASS
  0.353 = s1 alone, all seen writes       0.3569 / 0.3589    PASS
  0.476 = 3-way GBM P(REVISED), pooled    0.7133 / 0.7118    FAIL by 0.237

TASK 2 primary metric   delta from adding exact per-address history
  bar  PASS >= +0.05 and absolute >= 0.70 / KILL < +0.02 or absolute < 0.60
  777  +0.0807 +/- 0.0029, absolute 0.8929    PASS
  888  +0.0815 +/- 0.0040, absolute 0.8830    PASS

TASK 2 proxy kill   a history feature at chance inside BOTH strata is a
                    write-type proxy and adds nothing
  h_cos  0.742 pooled / 0.742 inside change writes / 0.468 inside repeats
  survives: it carries real information inside the change stratum
```

**Both tasks' bars fired, in opposite directions, and both are true.** The
pre-registration conflated two statements at the 0.80 bar: "instantaneous is
close to sufficient" and "history adds nothing" are not the same claim. The
first is roughly right and the second is false. The recommendation does not rest
on the premise bar; it rests on the increments and the cost.

### The 0.476 is the one recorded number that is wrong

`0.834` and `0.353` are the same measurement under two conditionings and both
READMEs quote them correctly: `logs/recurrence.log:12,17` has them in one table
from one run, s1 alone, NOISY vs REVISED, all seen writes 0.3530 and change
writes only 0.8337. Both reproduce here on two fresh streams.

`0.476` is `logs/signinit.log:17`, and it is a **3-way** gradient-boosted
classifier's `P(REVISED)`, scored pooled on a token-level split. Replaying that
exact procedure gives 0.713 and 0.712 here, so the 0.237 gap is not the
procedure. `legacy_0476.py` tests the remaining factor, the model: signinit
trained a LINEAR gate, `sigmoid(W x + a1*s1 + a2*s2 + b)`, which reached loss
1.5882 where the shipped MLP reaches 1.5402, so it writes with a different beta
and accumulates a different state.

Separately, **`logs/signinit.log:15` is a trap**: `AUROC NOISY vs rest: 0.8345`
is a different contrast from the 0.834 in the READMEs, which is NOISY vs
REVISED on change writes. The agreement to three decimals is a coincidence.

Also measured: the known labelling trap does not invert the signal, it
**destroys** it. Labelling a write by `class(x_t)` instead of `class(x_{t-1})`
takes s1 on change writes from 0.8073 to 0.5201 (777) and 0.8121 to ~0.49 (888).
And the token term `w_beta . x_t` sits at 0.4989 / 0.5038, so the task's
per-sequence class randomisation really does leave zero token information, as
designed. A third, smaller slip: every earlier script masks with the task's
`seen` array, whose `t`-th entry describes `x_t` and so the VALUE token, while
labelling with `class(x_{t-1})`. Worth 0.010 of AUROC (0.3473 vs 0.3569).

## The history feature that works is not the one we hypothesised

Pre-registered hypothesis: successive residual DIRECTIONS at one address
separate the classes, because a NOISY address scatters while a REVISED address
carries one persistent error being slowly corrected. **Direction confirmed,
magnitude disappointing.** `cos(e_t, e_prev)` is 0.036 on NOISY and 0.579 on
REVISED, a large mean separation, 0.74-0.75 AUROC on its own, and it survives
the "it is only a proxy for write type" kill (0.75 inside change writes). But as
an increment it is nearly worthless:

```
increment over all zero-extra-state features        777      888
  h_cos      cos(e_t, e_prev) at the address      +0.017   +0.018
  h_meane    mean ||e|| at the address            +0.036   +0.035
  all five history features together             +0.081   +0.082
  position t (uninformative control)             +0.003   +0.004
```

`h_meane` is **the rate memory**, and it is worth double what the residual
direction is worth. The direction hypothesis is the weaker half of the idea.

## The decisive fact, and it was already in the repo

`archive/failed_features/rate_test.py` BUILT the `h_meane` mechanism as a
delta-rule-written rate memory. Comparing its logged diagnostic against the
exact crosstalk-free ceiling measured here:

```
  rate_delta  c_t  pooled AUROC 0.3690  ->  0.631 with the sign flipped   (rate.log)
  rate_accum  c_t  pooled AUROC 0.4644  ->  0.536 flipped
  h_meane  EXACT   pooled AUROC 0.3189  ->  0.681 flipped                 (this dir)
```

**The delta-rule rate memory recovered 0.631 of an available 0.681, about 93% of
the separation on that axis.** And its loss was 5.12% against the plain feature
gate's 4.93%, a +0.19pp difference inside this repo's own seed sd of 0.0026 to
0.0039 on loss (0.16 to 0.24% of gain).

So the mechanism was built, it worked as an information channel, and it bought
nothing. `process/03/README.md:97-103` files this under "superposition crosstalk
kills signals of this size". For this signal **crosstalk was not the problem**;
the information simply is not worth much to the loss. That is a stronger and
more useful negative than the one on file.

Caveat: different trained models (rate_delta vs the shipped gate), one seed
each, and the two AUROCs come from different eval streams. Indicative, not an
exact ceiling comparison.

## What this says about the oracle gap we were chasing

The target was `oracle_class`'s 1.4pp accuracy edge and `oracle_mlp`'s 8.77% vs
the feature gate's 5.67%. Two findings bear on it:

1. **At L=256 the class label is only partly recoverable from per-address
   history at all.** NOISY and REVISED show the SAME mean number of distinct
   values per address, **2.72 vs 2.72**, because a NOISY address makes about 3.3
   distinct draws from 4 options in ~7 visits while a REVISED address makes
   1 + Binomial(7, 0.35) ~ 3.4. The near-label oracle tops out at 0.92, not 1.0.
2. **The gap is neither informational nor architectural. It is credit
   assignment.** `gate_capacity.py` trains the shipped gate's EXACT shape,
   `3 -> w -> w -> 1` GELU, directly on the class label as a supervised
   problem, which removes the "a tree is a stronger learner than the gate"
   objection while keeping everything else:

   ```
   feature set                           width      MLP     tree   linear
   shipped gate features [tok, s1, s2]      32   0.8127   0.8095   0.6760
                                           128   0.8114
   all zero-extra-state                     32   0.8146   0.8122   0.7640
                                           128   0.8112
   ```

   The gate's own architecture at its own shipped width reaches the full
   information content, **slightly beating the tree**, and 4x the width adds
   nothing. So the remaining oracle gap is not that the information is absent
   (0.81 of a 0.92 near-label ceiling), and not that a `3 -> 32 -> 32 -> 1` MLP
   cannot represent the required function. What the real gate lacks is the
   **learning signal**: it only ever sees the downstream language-model
   gradient, never a class label, and 800 steps of that does not find the
   function direct supervision finds. That is consistent with
   `validation/convergence.log`, where `ours` was still descending at step 3000.

   **The actionable lever is therefore the learning signal, not new state and
   not more width.** An auxiliary loss on the gate, a longer schedule, or a
   better initialisation are all cheaper than a `d_k x d_v` volatility memory
   and attack the part that is actually binding. Untested here.

## Reportable anomalies

- **The task has absorbing states, and they dominate it.** `task.py` draws
  `succ = rng.integers(N_SYM, size=N_SYM)`, so `succ[s] == s` with probability
  1/40 per symbol, and a STABLE symbol pointing at itself never releases the
  walk. Measured: the median sequence sends **28 of its 256 writes to a single
  address**, the mean is 68.8, and **21% of sequences send more than 120**.
  44% of all writes are STABLE repeats. Every loss and accuracy number in this
  repo is dominated by that sub-population, where prediction is trivial
  (STABLE accuracy is logged at 1.000).
- **"NOISY = ~all changes" is wrong.** Among writes to a seen address, NOISY is
  **28%** repeats, not ~0%. REVISED's logged 65% is exactly right (measured
  0.656). The repo's own `logs/s3.log` already contained the refutation, in its
  run lengths: NOISY 1.350, REVISED 2.292.
- **A Simpson's paradox of my own, caught and corrected.** Mean beta is 0.310
  on repeat writes and 0.195 on change writes, which looks like the exact
  opposite of the policy every README here states. It is not: 21,492 STABLE
  repeat writes sit at beta 0.436, and inside every stratum where beta is
  identified the gate AMPLIFIES, `spearman(beta, s1)` = **+0.974** on NOISY
  changes and **+0.840** on REVISED changes. The stated policy holds and C4
  "amplify" is correctly classified. See `beta_policy.py`.
- **The gate learned a three-regime non-monotone response**, which is the
  mechanism behind the linear-vs-spline gap above. Mean beta by s1 decile:
  `0.997 0.864 0.016 0.022 0.033 0.044 0.062 0.183 0.258 0.304`. It parks beta
  near 1.0 in the bottom two deciles, where `||e|| ~ 0` and beta cannot matter,
  holds at ~0.02 through the middle, and rises to 0.31 at large `||e||`.
- **`s3 = ||stored||/||v||` is genuinely null, now for a good reason.**
  `archive/failed_features/s3_test.py` filed it on a pooled difference of means
  (-0.0248) measured on an UNGATED model. Measured properly here it is 0.424 to
  0.436 AUROC on its own and adds +0.002 to +0.003 over the other zero-state
  features, inside one bootstrap SD. The verdict was right; the evidence for it
  was not.
- **`||stored||` reads LOWER on REVISED** (AUROC 0.43), the opposite of
  `s3_test.py`'s stated prediction that a committed REVISED address holds a
  large-norm single value while a NOISY one holds a cancelling blend.

## Parallelism cost of what was NOT built

Worse than the current gate, which is already a sequential Python loop:

- `h_cos` needs a per-address residual store, `d_k x d_v`, **as large as `S`
  itself**, written state-dependently.
- `h_meane` needs only `d_k`, which is cheap, and is the one already built.
- CARVE's chunk-entry fix does not rescue the direction form: `cos(e_t, e_prev)`
  needs the previous write to the SAME address, which can fall inside the
  current chunk, so freezing at chunk entry loses exactly the comparison.

## What would settle it directly, and was not run

A loss measurement: wire the exact bookkeeping history into the gate and train.
The pre-registration licenses this because the Task 2 bar cleared. It was not
run because a trustworthy answer needs 3 seeds x 2 arms at ~28 min per 800-step
run, about 6 hours of 2-thread CPU, and the machine was shared with another
session's jobs throughout. The prediction, recorded here: **no loss gain beyond
seed noise**, because the rate memory already ran that experiment for the
strongest single history feature and got +0.19pp.

## Files

```
PREREGISTRATION.md    question, metric, bars, kill conditions. Written first.
CONFIRMATION_PLAN.md  the conclusion and the 5 confirmation bars, written
                      before stream 888 was touched.
fit_model.py          trains and saves the shipped model once
features.py           replays it and records instantaneous + EXACT per-address
                      history. Asserts the replay matches the shipped forward.
auroc.py              by-sequence split, paired sequence-level bootstrap
separability.py       TASK 1: the factorial, the labelling trap, the premise test
history_probe.py      TASK 2: history features and the pre-registered delta
beta_policy.py        the beta-vs-residual direction anomaly, and its resolution
gate_capacity.py      is the information reachable by a gate-shaped MLP?
legacy_0476.py        the one reproduction that failed its bar
logs/                 one log per run, streams 777 and 888
```

## Reuse hazard, handled

14 files in this repo score on `default_rng(12345)`. Nothing here is scored on
it except the explicit reproduction attempts. Primary stream **777**,
confirmation stream **888** held out and untouched until the conclusion was
committed (`a7df0ff`).
