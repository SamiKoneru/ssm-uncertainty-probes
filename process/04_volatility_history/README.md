# 04_volatility_history - does long-term volatility add anything to beta?

> **SETTLED BY `../05_history_loss/`, AND THIS DIRECTORY'S RECOMMENDATION IS
> OVERTURNED.** Exact per-address history in the gate cuts loss by **+0.0551
> nats**, 8.6x that session's pre-registered bar, 3 seeds, confirmed on a
> held-out stream (+0.0563) with eight arms across two streams and zero
> disagreements. My recorded prediction of "no loss gain beyond seed noise" was
> wrong. The mechanism, however, is not what this directory theorised, and
> **two separate effects turned out to be tangled together**: see "THE FINAL
> ANSWER" at the end of this file. Read it before anything else here.

> **READ [`AUDIT.md`](AUDIT.md) FIRST.** A post-hoc audit retracted one headline
> figure of this directory as a bug artifact, retracted one argument outright,
> and found a third misapplied. The separability measurements all survived,
> including a random-feature control. The RECOMMENDATION is weaker than this
> README originally stated and the sections below are corrected accordingly.

**ANSWER: a real but bounded information gap exists, and the separability
evidence here does NOT establish that closing it is worth it.** Exact
crosstalk-free per-address history adds +0.063 to +0.085 AUROC over everything
the gate can compute with zero extra state, confirmed on a held-out eval stream.
A ceiling bounds the gap, though NOT at the 0.92 this README first claimed: the
exact Bayes ceiling given an address's value history is **~0.945**
(`audit_bayes_ceiling.py`), so 0.92 was a property of my FEATURE SET and the
headroom for history is +0.133 rather than +0.109. See `AUDIT.md`. **Recommendation: do not build it** - but that is a judgement resting
on the cost of `d_k x d_v` of new state plus one prior null loss result, not a
measurement of loss on this question. The loss experiment was not run here.

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
+ oracle features (per-address value set)           0.9206   0.9154
EXACT BAYES ceiling given the value history        0.9449   0.9469
```

so, as increments on the pooled scale:

```
  two more zero-extra-state features                +0.003   +0.003   zero new state
  ALL exact per-address history                     +0.081   +0.082   d_k x d_v new state
  the oracle features on top of that                +0.028   +0.032   not a mechanism
  everything left over, up to exact Bayes           +0.024   +0.032   not a mechanism
```

**RETRACTED** (see `AUDIT.md`): this table originally carried a row reading
"nonlinearity on features the gate ALREADY HAS +0.134", with the claim that the
free nonlinear headroom was 1.7x what history can buy. The linear-vs-nonlinear
gap is real (0.676 to 0.810) but it is **already banked**: `gate_capacity.py`
shows the shipped gate's own `3 -> 32 -> 32 -> 1` stack reaches 0.8127, so that
+0.134 is part of the published 5.5% and is not available to spend. Comparing an
already-realised gain against a prospective one favoured my own conclusion.

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
                                          on the SHIPPED model, and the failure
                                          was the model, not the procedure:
          on the LEGACY model it was measured on, 0.4712 vs 0.4763,  PASS
          with all three one-vs-rest AUROCs inside 0.0020.
  ALL THREE RECORDED NUMBERS ARE CORRECT. None of them is a bookkeeping error.

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

### Why the three numbers disagree: two conditionings and two models

`0.834` and `0.353` are the same measurement under two conditionings, and both
READMEs quote them correctly: `logs/recurrence.log:12,17` has them in one table
from one run, s1 alone, NOISY vs REVISED, all seen writes 0.3530 and change
writes only 0.8337. Both reproduce here on two fresh streams.

`0.476` is `logs/signinit.log:17`, a **3-way** classifier's `P(REVISED)` scored
pooled on a token-level split. Replaying that exact procedure on the SHIPPED
model gives 0.713, so conditioning alone does not explain it. `legacy_0476.py`
removes the last factor by training the model signinit actually used, a LINEAR
gate `sigmoid(W x + a1*s1 + a2*s2 + b)`, and reproduces the whole block:

```
                           measured   logged    diff
  3-way accuracy             0.6545   0.6485
  AUROC STABLE  vs rest      0.9337   0.9317   +0.0020
  AUROC NOISY   vs rest      0.8365   0.8345   +0.0020
  AUROC REVISED vs rest      0.7385   0.7396   -0.0011
  AUROC NOISY vs REVISED     0.4712   0.4763   -0.0051
```

**So all three recorded numbers are right.** The disagreement is two
conditionings and two models, and nothing in the repo stated which model each
number came from.

### CORRECTED: the gate shapes its own features, but only a little

This section originally claimed separability goes 0.61 (legacy linear gate) to
0.81 (shipped MLP), called that +0.19 "the strongest single argument against the
volatility memory", and attributed it to beta shaping `S` shaping `(s1,s2)`.

**The 0.61 was a bug artifact.** `signinit_test.py:88` replays the trained model
with the INPUT-ONLY beta, dropping the a1 and a2 terms its own gate was trained
with, and has no check against the model's forward pass. `logs/model_vs_task.log`
inherited that replay. Corrected (`logs/audit_signinit_bug.log`): the legacy
gate's `[s1,s2]` is **0.7787**, not 0.6161.

The corrected comparison, all five zero-extra-state features, stream 777:

```
gate                       loss      pooled AUROC
legacy linear gate="both" 1.5882        0.7910
shipped MLP seed 0        1.5786        0.8122
shipped MLP seed 1        1.5732        0.8370
```

Still monotone in loss, so the effect is real and in the claimed direction, but
**+0.046 across the full range of gate quality available here, not +0.19.** It
is a small exploratory trend, not an argument that carries a recommendation.

It also means `signinit_test.py`'s "the gap is INFORMATION-limited" conclusion
was not even right about its own model: read correctly, that model's features
carry 0.79.

### And the better the gate, the less history is worth. EXPLORATORY, n=3 models

A second training seed was run as a stability check. It came out BETTER on loss,
MORE separable instantaneously, and gained LESS from history. Lining up all
three models measured here, every column is monotone:

```
model                        loss     (s1,s2) pooled   history delta
legacy linear gate="both"  1.5882          0.616              -
shipped MLP seed 0         1.5786          0.810         +0.0807
shipped MLP seed 1         1.5732          0.832         +0.0634
```

(losses on stream 777; the legacy figure is its logged 12345 loss, so it is not
strictly comparable, but it is the worst model by a wide margin on any stream.)

**EXPLORATORY**: three models, no seeds behind any single point, and the
ordering was noticed after the fact rather than predicted. But the direction is
the one the mechanism above predicts, and it is the direction that matters for
the decision: the marginal value of a volatility memory SHRINKS as the gate
improves. Confirming it needs a proper seed sweep with the separability and the
loss read off the same runs, which was not done.

### Seed stability of the two headline numbers

```
                              seed 0 / 777   seed 0 / 888   seed 1 / 777
pooled, shipped features            0.8095         0.7983         0.8320
pooled, all zero-state              0.8122         0.8015         0.8370
  the same, linear read             0.6760         0.6613         0.6933
s1 alone, CHANGE writes             0.8073         0.8121         0.7956
history delta                      +0.0807        +0.0815        +0.0634
  of which h_cos                   +0.0168        +0.0180        +0.0116
  of which h_meane                 +0.0361        +0.0351        +0.0344
oracle-feature ceiling              0.9206         0.9154         0.9231
  (exact Bayes ceiling is 0.9449 / 0.9469; see AUDIT.md RETRACTED 3)
position control                   +0.0030        +0.0042        -0.0000
```

The history delta clears its +0.05 bar on all three. `h_meane` beats `h_cos` on
all three. The pre-registered 0.80 premise bar is cleared by two of three, which
is the same indecision reported above, now across training seeds as well as eval
streams. **The 0.80 is not a fact about the task.** It is a fact about one
trained gate, and the spread across three gates here is 0.61 to 0.83.

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

**RETRACTED, see `AUDIT.md`. The "93%" is an arithmetic error of mine and the
measured figure is about half that.** 0.631/0.681 is a ratio of raw AUROCs, but
AUROC's null is 0.5, not 0, so the excess-over-chance ratio is
|0.3690-0.5|/|0.3189-0.5| = **0.723**, and measured properly on one model the
rate memory recovers **0.513** (0.481 on a second). The conclusion it supported
still holds and for a better-identified reason: the auditing session decomposes
the shortfall into roughly 40% from using an EMA rather than an exact per-address
mean, and only 12 to 21% from superposition crosstalk. So crosstalk was indeed
not the main culprit, but the mechanism is the discount form, not the store.

Its loss was 5.12% against the plain feature
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
   1 + Binomial(7, 0.35) ~ 3.4. BUT this is weaker than it looks: the exact
   Bayes ceiling is 0.945, and by prior visits to the address it runs 0.687 at
   1 visit, 0.786 at 2, 0.890 at 3-4, 0.977 at 5-8 and 0.9997 at 9 or more. The
   class IS recoverable from history, just not early. See AUDIT.md.
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

   ```
   zero-state + exact history               32   0.9017   0.8929   0.8398
   ```

   The gate's own architecture at its own shipped width reaches the full
   information content, **slightly beating the tree**, and 4x the width adds
   nothing. So the remaining oracle gap is not that the information is absent
   (0.81 of a 0.945 exact Bayes ceiling), and not that a `3 -> 32 -> 32 -> 1` MLP
   cannot represent the required function. What the real gate lacks is the
   **learning signal**: it only ever sees the downstream language-model
   gradient, never a class label. That is consistent with
   `validation/convergence.log`, where `ours` was still descending at step 3000.

   **MISAPPLIED, see `AUDIT.md`.** I originally used this as an argument against
   history. It does not discriminate: the same probe shows the gate-shaped MLP
   reaches the history-augmented **0.9017** too, so the architecture represents
   both mappings and "the gate cannot learn this from the LM gradient" applies
   equally to both. It explains the oracle gap; it says nothing about whether
   history is worth adding. The lever it does point at - an auxiliary loss on
   the gate, a longer schedule, a better initialisation - is cheaper than a
   `d_k x d_v` memory, and is untested here.

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

### THAT PREDICTION IS FALSIFIED. 3 SEEDS, COMMITTED

`process/05_history_loss/` (branch `claude/sharp-dubinsky-67ebc6`, commit
`47c8e68`) ran it. 3 seeds, stream 5150, all gated arms param-matched at 62,250,
`h_ratio` clamp rate 0.000%, pre-registered bar 0.0064 nats:

```
arm          loss mean      sd     acc      sd   vs none   vs ours   x bar
none            1.6420  0.0014  0.6340  0.0016     0.00%
ours            1.5531  0.0055  0.6455  0.0045     5.42%        --     0.0
ours_pub        1.5509  0.0035  0.6451  0.0059     5.55%   +0.0022     0.3
ours+rate       1.5470  0.0054  0.6432  0.0042     5.79%   +0.0060     0.9
ours+hist       1.4979  0.0017  0.6413  0.0023     8.78%   +0.0551     8.6
ours+oracle     1.4832  0.0005  0.6484  0.0015     9.67%   +0.0698    10.9
```

`ours_pub` at 5.55% reproduces the published 5.47%, so the harness is sound. The
loss effect is not marginal: `ours+hist` per-seed is 1.4966 / 1.4998 / 1.4974,
sd 0.0017, the tightest of any arm, against a gap of +0.0551. **`ours+rate`
lands at 0.9x the bar, BELOW it**, so the rate memory is confirmed null across
three seeds.

On NOISY/REVISED writes only: none 2.7539, ours 2.6154, `ours+hist` 2.5429,
`ours+oracle` 2.5176. So history closes about half the remaining distance from
the gate to the oracle on the writes that matter.

**ACCURACY, and it is the opposite sign.** `ours+hist` is 0.6413 against `ours`
at 0.6455, lower on 3 of 3 seeds (-0.0001, -0.0062, -0.0063). **But the
magnitude, -0.42pp, is within one seed-sd of `ours` itself (0.45pp), and a sign
test on n=3 is p=0.125.** So: directionally consistent, NOT established. Stated
that way because the first version of this note called it a regression, which
n=3 does not support. What IS clear is the contrast with `ours+oracle`, which
gains accuracy (+0.29pp at sd 0.0015) as well as loss. **History buys loss;
only the oracle buys both.** Since the 1.4pp accuracy edge is what motivated
this directory, that distinction is the one that matters, and history is not a
substitute for class knowledge.

### AND THE GAIN IS NOT AN INFORMATION EFFECT

Their provisional verdict (`e5f0f15`) adds the arm that settles what kind of
effect this is. `ours+bayes` carries every feature `ours+oracle` has **plus**
`b_post`, the exact posterior `P(REVISED | the address's value history)` under a
flat prior, i.e. a genuine sufficient statistic for the history (cross-checked
against this directory's `audit_bayes_ceiling.py` to four decimals). Gains over
`ours`:

```
ours+hist     +0.0551    8.6x bar
ours+oracle   +0.0698   10.9x bar
ours+bayes    +0.0393    6.1x bar     <- STRICT SUPERSET of ours+oracle
```

**A strict superset of the features gains LESS.** Gains are not monotone in
information, so whatever produces the +0.0551 is not the gate learning more
about the address's volatility. And the gain sits in STABLE and NOISY; on
**REVISED it is worse than `ours`**, and REVISED-versus-NOISY is the single
discrimination this entire hypothesis was built on.

So the practical conclusion of this directory may hold after all, on a third set
of grounds and none of mine: not "there is no information" (there is, +0.08
AUROC), not "it does not help loss" (it does, 8.6x the bar), but **the help is
not an information effect and does not act on the classes the hypothesis was
about.** There is no volatility mechanism here to build. Pending the two
controls and the held-out stream 5151, which that session is correctly keeping
untouched until the controls land.

My prediction was wrong, as was my ranking:
`h_meane`, which this directory measured as the strongest history feature on
separability, is the arm that lands AT the bar and does nothing, while the
residual-direction family that I called "the weak half" carries the whole gain.

**So the class-AUROC metric this entire directory is built on mis-ranked the
features it was used to rank.** That is a defect in the method, not in the
arithmetic, and it is the exact failure the auditing session suspected under its
attack surface A1 and could not demonstrate. The loss experiment demonstrated it.

### But the gain is not where volatility history would put it

Gain of `ours+hist` over `ours` by prior visits to the write address, against
the exact Bayes identifiability from `audit_bayes_ceiling.py`:

```
prior visits      n     loss gain    Bayes identifiability
   0 (fresh)   3837       +0.0707    no history exists
           1   3133       +0.2723    0.6868
           2   2724       +0.1196    0.7858
         3-4   4347       +0.0406    0.8896
         5-8   5461       +0.0159    0.9773
          9+   5994       -0.0126    0.9997
```

The gain is largest where the class is LEAST recoverable from history and
NEGATIVE where it is almost perfectly recoverable, and it is +0.0707 on fresh
writes where by construction there is no history. Two things explain that better
than volatility does:

- **`ours` is worse than `none` on early writes** (2.9707 vs 2.7407 at one prior
  visit) and only overtakes it from three visits on. So the shipped memory gate
  mis-writes early and much of this gain is repair, not new capability.
- their `arms.py:239` is `h_nvis = log1p(p_cnt)`, an unblanked **per-address
  visit counter**, and their blanking convention makes "all other features
  exactly 0.0" a first-write indicator. A visit counter is `d_k` of state, one
  scalar per address, not the `d_k x d_v` residual store this directory priced
  the mechanism at, and it is not volatility.

Two controls are running to separate these: `ours+nvis` (the counter alone,
param-matched) and `ours+fresh` (a single seen/fresh bit). **Both were designed
after seeing the visit-count profile, by me, so they are exploratory by
construction** and that session has logged the amendment as the first one not
written blind. Whatever they show needs confirmation on data not used to design
them.

And the headline that survives all of it: the 1.4pp accuracy edge that motivated
this directory is moved by the oracle and not by history, so the thing it set out
to close is still open. See the accuracy paragraph above for why that is stated
as a contrast rather than as a regression.

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
legacy_0476.py        the reproduction that reproduced a BUG; see AUDIT.md
model_vs_task.py      both models over the same streams. Its legacy column is
                      the buggy replay; audit_signinit_bug.py supersedes it.
AUDIT.md              the post-hoc audit: what survived and what did not
audit_sanity.py       published-loss reproduction, random-feature control,
                      NaN-branch and split checks
audit_addressing.py   addressing and labelling against hand-built cases
audit_trend.py        the legacy model's history delta, with a stated prediction
audit_s2_swap.py      the repo's untested s2 -> ||Sk|| recommendation
audit_ftok.py         is the token term an interaction or a leak? (neither)
audit_signinit_bug.py isolates the input-only-beta replay bug
logs/                 one log per run, streams 777 and 888
```

## Reuse hazard, handled

14 files in this repo score on `default_rng(12345)`. Nothing here is scored on
it except the explicit reproduction attempts. Primary stream **777**,
confirmation stream **888** held out and untouched until the conclusion was
committed (`a7df0ff`).


---

# THE FINAL ANSWER (`../05_history_loss`, branch `claude/sharp-dubinsky-67ebc6`)

Confirmed on held-out stream 5151: **eight arms, two streams, zero
disagreements.**

```
arm              5150 gain   5151 gain   verdict
ours+hist          +0.0551     +0.0563      PASS     8.6x the bar
ours+oracle        +0.0698     +0.0731      PASS
ours+bayes         +0.0393     +0.0426      PASS
ours+rate          +0.0060     +0.0052      KILL     misses on both
ours+nvis          -0.0009     -0.0027      KILL
ours+fresh         +0.0032     +0.0028      KILL
VALIDITY GATE        +5.42%      +5.60%      PASS
```

## My visit-counter explanation is REFUTED

After seeing the gain concentrate at low visit counts I proposed that the whole
effect was a per-address write counter, `d_k` of state rather than `d_k x d_v`,
and nothing to do with volatility. That got its own arm instead of an argument,
and it is **wrong**: `h_nvis` alone is worth **-0.0009** and a single
"ever written" bit is worth **+0.0032**, against the five together at +0.0551.
Neither comes close. Seventh retraction of mine in this session.

What the features do instead is **superadditive**. `h_meane` alone is worth
+0.0060 as an arm, but removing it from the trained five costs **+0.0934**, a
15x gap. The gate co-adapts to the whole block, so no single feature's
standalone value predicts its contribution.

## There are TWO effects, and only one of them is volatility

This is the result worth keeping, and it is the single most informative
comparison either session produced. Gain over `ours` by prior writes, against
the exact Bayes identifiability at that visit count:

```
gain vs ours      0 (fresh)        1        2      3-4      5-8       9+
ours+hist           +0.0826  +0.3131  +0.1402  +0.0587  +0.0163  -0.0294
ours+bayes          -0.0145  +0.1222  +0.0506  +0.0279  +0.0683  +0.0928
Bayes AUROC               -    0.687    0.786    0.890    0.977   0.9997
```

- **`ours+bayes` is the volatility mechanism, and it works exactly as this
  directory theorised.** Its gain GROWS with identifiability, from -0.0145 on
  fresh writes to +0.0928 at nine or more prior writes. Worth **+0.0393**. But
  it carries `b_post`, the exact posterior `P(REVISED | value history)`, which
  requires reading `N_OPT` and `P_REV` off the task generator. It is a ceiling,
  not a buildable mechanism.
- **`ours+hist` is something else, and it is the bigger effect.** Its gain runs
  the OTHER way, five sixths of it at 0 to 2 prior writes, negative at 9+. It is
  not identifying the key's class.

So the volatility hypothesis is **vindicated in principle and worth +0.039**,
while the +0.055 that a buildable feature set actually delivers comes from a
different mechanism. Those two were tangled together in every number this
directory reported, because class-AUROC cannot tell them apart.

## What the big effect actually is

Beta collapses to **~0.002 everywhere** in `ours+hist` and `ours+oracle`. Since
`S_t k = (1-beta) S_{t-1} k + beta v` is an EMA with rate beta, driving beta
toward zero makes the memory a **uniform running average over everything ever
written to that address**: exactly right for NOISY, exactly wrong for REVISED.
The per-class losses follow: NOISY improves 2.8751 to 2.7094, **REVISED gets
worse**, 2.3705 to 2.3859.

It is not a constant policy. Forcing `ours+hist` to its own mean beta costs
+0.2564, and the best constant over a 16-point grid reaches only 1.7517,
because in EMA terms moving between 0.0005 and 0.01 is a 20x swing in update
rate. That check validates itself on `none`, where forcing the mean costs
+0.0001.

## And the original target is still not closed

```
arm            acc
ours        0.6455
ours+hist   0.6413     <- buys 0.055 nats, LOSES 0.4pp of accuracy
ours+oracle 0.6484     not buildable
ours+bayes  0.6491     not buildable
```

The 1.4pp accuracy edge that motivated this entire directory is moved only by
the oracle arms. **Nothing buildable moves it.** The question this directory was
convened to answer is, in the end, still open.

## What this directory got right, and what it got wrong

Right, and all of it survived an 18-attack adversarial audit: the measurement
machinery, the +0.08 AUROC of information, the 0.945 Bayes ceiling, the
visit-count identifiability curve, the task's absorbing states, NOISY being 28%
repeats, the labelling trap, the signinit replay bug and its blast radius.

Wrong: every inference drawn from class-AUROC to a decision. The metric
mis-ranked its own features (`h_meane` first, the direction family last; loss
reverses it) and could not separate the two effects above. **The lesson is not
about volatility. It is that a proxy metric validated only against itself will
rank confidently and wrongly, and that the only thing that caught it was running
the expensive experiment the proxy was supposed to make unnecessary.**
