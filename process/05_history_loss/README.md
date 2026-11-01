# 05_history_loss - does per-address history in the gate reduce LOSS?

**ANSWER: YES, and it overturns `process/04_volatility_history`'s
recommendation. But NOT for the reason that directory proposed, and the
mechanism it found is not the one it was looking for.**

`ours+hist` beats the param-matched control by **+0.0551 nats**, 8.6x the
pre-registered bar, with all three seeds positive and a seed sd of 0.0017.
04's recorded prediction was "no loss gain beyond seed noise". That
prediction is wrong.

Then the diagnostics take it apart. The gain is concentrated where the
address's class is LEAST identifiable from its history, is negative where it
is almost perfectly identifiable, makes accuracy slightly WORSE, and makes the
REVISED class - the one the whole volatility hypothesis was about - worse too.
And the arm holding a **sufficient statistic** for the value history gains
LESS than two arms holding strictly less information. Gains are not monotone
in information, so this is not an information result.

Pre-registered in [`PREREGISTRATION.md`](PREREGISTRATION.md). Three appendices
were written before any number existed; the fourth is marked as the first one
written after seeing data, and adds only controls that can shrink the
headline.

> **STATUS: CONFIRMED.** Held-out stream 5151 was scored only after the 5150
> verdict was committed (`821dfb4`). **Every verdict reproduced**, including
> both KILLs. See [Confirmation](#confirmation-on-held-out-stream-5151).

---

## Headline

3 seeds, published protocol (800 steps, batch 24, AdamW 2e-3, clip 1.0,
gate_width 32), eval stream **5150**, fresh and never used for any decision in
this repo. Bar **+0.0064 nats** = `2 x 0.0039 x sqrt(2/3)`, derived from the
repo's own prior seed sd before anything ran.

```
arm              params      s0       s1       s2     mean      sd   gain    verdict
none             62,058  1.6433   1.6423   1.6405   1.6420  0.0014      -
ours             62,250  1.5577   1.5544   1.5470   1.5531  0.0055      -
ours+hist        62,250  1.4966   1.4998   1.4974   1.4979  0.0017  +0.0551   PASS
ours+rate        62,250  1.5511   1.5489   1.5410   1.5470  0.0054  +0.0060   KILL
ours+oracle      62,250  1.4829   1.4830   1.4838   1.4832  0.0005  +0.0698   PASS
ours+bayes       62,282  1.5129   1.5180   1.5103   1.5137  0.0039  +0.0393   PASS
ours+nvis    *   62,250  1.5305   1.5982   1.5331   1.5539  0.0384  -0.0009   KILL
ours+fresh   *   62,314  1.5496   1.5502   1.5498   1.5499  0.0003  +0.0032   KILL
ours_pub         62,058  1.5544   1.5509   1.5474   1.5509  0.0035      -
```

`*` = control arm added after seeing seed 0, able only to shrink the headline.

All four gated arms in the primary set are **parameter-identical at 62,250**
and differ only in a 0/1 keep-mask over the history columns, so this does not
repeat the confound `validation/README.md` records, where every published
comparison pitted a 3-input MLP gate against no gate at all.

**Validity gate PASSES**: `ours` beats `none` by **+5.42%**, against the 4.00%
bar and the published 5.47%. The harness reproduces the shipped result.

**The bar was not rescued by noise.** Realized pooled within-arm sd is
**0.0036**, against the 0.0039 the bar was derived from, so the pre-registered
bar is slightly harder to pass than this data requires.

### `ours+rate` misses by 0.0004 nats

+0.0060 against a +0.0064 bar, with all three seeds positive (+0.0066,
+0.0055, +0.0061). **Recorded as KILL. The bar is not moved.** It is the one
result whose verdict the bar's exact placement decides, and the baseline check
below pushes it further down rather than up: against `ours_pub` the gain is
+0.0039. 04's strongest single separability feature is worth almost nothing on
loss by itself.

### The visit-counter explanation is refuted

The gain profile below concentrates at low visit counts, which fits "the gate
learned a per-address visit counter" better than it fits volatility. A counter
costs `d_k`, not `d_k x d_v`, and has nothing to do with volatility, so this
needed its own arm rather than an argument.

```
ours+nvis    h_nvis alone, param-matched      -0.0009    KILL
ours+fresh   a single "ever written" bit      +0.0032    KILL
ours+hist    all five together                +0.0551    PASS
```

**The counter alone is worth nothing and the fresh bit is worth half the
bar.** Neither comes near +0.0551, so the gain is not a visit counter and is
not the implicit first-write indicator either.

What it is instead is **superadditive**. The two strongest single features,
measured alone as arms, buy +0.0060 (`h_meane`) and -0.0009 (`h_nvis`); the
five together buy +0.0551. Eval-time ablation on the trained `ours+hist`, 3
seeds, shows the same from the other side:

```
drop h_meane      +0.0934        drop h_cos      +0.0299
drop h_nvis       +0.0778        drop h_ratio    +0.0006
drop h_cosmean    +0.0544        drop ALL        +0.2480
```

`h_meane` is worth +0.0060 as a standalone arm and +0.0934 to remove from the
trained set, a 15x gap. The features are not independent contributions; the
gate co-adapts to the whole block.

**Do not read that table as attribution.** It is tempting to note that the
two `d_k`-cost columns (`h_meane` +0.0934, `h_nvis` +0.0778) outweigh the two
`d_k x d_v` direction columns (`h_cosmean` +0.0544, `h_cos` +0.0299) and
conclude a cheap version exists. `ours+cheap` was built to test exactly that
reading and refuted it: all three cheap features together score **-0.0005**
against `ours` on a fresh stream, KILL on both pre-registered bars, and they
degrade the beta discrimination `ours` already had. See `RESULTS_cheap.md`.
The standalone arms were the honest signal; the knockout ranks co-adaptation,
which is what this file's own `ablate.py` docstring says it measures.

### Every gain is inflated by +0.0022

`ours` (1.5531) is slightly worse than `ours_pub` (1.5509), so masking six
columns handicaps the baseline every gain is measured against:

```
arm            vs ours   vs ours_pub   inflation
ours+hist      +0.0551       +0.0530      +0.0022
ours+rate      +0.0060       +0.0039      +0.0022
ours+oracle    +0.0698       +0.0677      +0.0022
ours+bayes     +0.0393       +0.0372      +0.0022
```

The headline survives it comfortably; `ours+rate` does not.

---

## Why this is not the volatility result it looks like

### 1. More information buys LESS

`ours+bayes` carries every feature `ours+oracle` has **plus** `b_post`, the
exact posterior `P(REVISED | the address's value history)` under a flat prior,
which is a **sufficient statistic** for that history (AUROC 0.9449, verified
against 04's `audit_bayes_ceiling.py` to four decimals). It gains **+0.0393**
against `ours+oracle`'s +0.0698.

A strictly larger feature set does strictly worse. Whatever is producing these
gains, it is not the amount of class information in the gate's input.

### 2. The gain lands where the class is least identifiable

`ours+hist` minus `ours`, by prior writes to the query address, against the
exact Bayes identifiability of the class at that visit count:

```
prior writes      0 (fresh)        1        2      3-4      5-8       9+
n per seed            3,837    3,133    2,724    4,347    5,461    5,994
gain                +0.0826  +0.3131  +0.1402  +0.0587  +0.0163  -0.0294
Bayes AUROC               -    0.687    0.786    0.890    0.977   0.9997
```

Backwards. Five sixths of the gain sits at 0 to 2 prior visits, and the sign
flips negative at 9+ where the class is essentially certain. A mechanism that
worked by identifying a key's long-run behaviour would do the opposite.

**And `ours+bayes` does exactly the opposite, which is the most informative
single comparison here:**

```
gain vs ours      0 (fresh)        1        2      3-4      5-8       9+
ours+hist           +0.0826  +0.3131  +0.1402  +0.0587  +0.0163  -0.0294
ours+bayes          -0.0145  +0.1222  +0.0506  +0.0279  +0.0683  +0.0928
Bayes AUROC               -    0.687    0.786    0.890    0.977   0.9997
```

The arm holding the sufficient statistic gains **where identifiability is
highest** and loses on fresh writes. That is the volatility mechanism behaving
exactly as 04 theorised. So the honest reading is that **there are two
different effects here**: a genuine identifiability effect, visible in
`ours+bayes`, worth +0.0393 and requiring a feature that reads `N_OPT` and
`P_REV` off the task generator; and a larger effect in `ours+hist` that runs
the other way and is not about identifying the key's class at all.

Caveat on this table, which matters: the bins slice by prior writes to the
QUERY address `x_t`, while the gate's features describe the WRITE address
`x_{t-1}`. They correlate but are not the same conditioning, so the fresh-bin
row is "predictions about never-queried symbols improved", not "the gate
gained where it had no history".

### 3. Accuracy gets WORSE, and REVISED gets worse

```
arm              acc   acc hard    STABLE    NOISY   REVISED
none          0.6340     0.3633    0.4437   3.1287    2.4006
ours          0.6455     0.3857    0.4081   2.8751    2.3705
ours+hist     0.6413     0.3774    0.3717   2.7094    2.3859
ours+rate     0.6432     0.3810    0.4013   2.8639    2.3708
ours+oracle   0.6484     0.3907    0.3684   2.6528    2.3902
ours+bayes    0.6491     0.3928    0.3905   2.7809    2.3439
```

`ours+hist` buys 0.055 nats while **losing** 0.4pp of accuracy. Its gain is
entirely in STABLE and NOISY; on **REVISED it is worse than `ours`**, and
REVISED versus NOISY is the discrimination the whole hypothesis was built on.

This also closes 04's original target. The 1.4pp accuracy gap that motivated
the volatility memory is moved only by the ORACLE arms (`ours+oracle` 0.6484,
`ours+bayes` 0.6491), which are not buildable. Nothing buildable moves it.

### 4. What the arms actually learned: beta collapses to an average

Mean beta on seen writes, and what forcing the gate to a constant costs:

```
arm            beta S/N/R         as trained   @ own mean   best constant
none        0.387 0.387 0.387         1.6420      +0.0001         1.6434
ours        0.023 0.138 0.114         1.5531      +0.1504         1.6631
ours+hist   0.001 0.002 0.002         1.4979      +0.2564         1.7517
ours+oracle 0.001 0.001 0.002         1.4832      +0.3224         1.8092
ours_pub    0.169 0.132 0.109         1.5509      +0.1123         1.6667
```

`ours+hist` and `ours+oracle` drive beta to ~0.002 everywhere. That is not
degenerate: the readout is `head(norm(S q))` and LayerNorm is scale
invariant, while `S_t k = (1-beta) S_{t-1} k + beta v` is an EMA whose rate is
beta. So **beta -> 0 turns the memory into a uniform running average over
everything ever written to that address**, which is exactly optimal for NOISY
keys and exactly wrong for REVISED keys. That is precisely the per-class
pattern above.

It is also not a constant policy: forcing each arm to its own mean beta costs
`ours+hist` +0.2564 and the best constant over a 16-point grid reaches only
1.7517. Their beta VARIATION is worth more than anyone else's, because in EMA
terms moving between 0.0005 and 0.01 is a 20x swing in update rate.

> **Where that variation actually is, measured later** (`RESULTS_beta_shape.md`,
> both streams, three seeds). NOT in a tail: `frac(beta > 0.5)` is exactly
> 0.0000 and **not one write out of 48,960 exceeds beta 0.094**, so the policy
> is near-flat and no spike explains the gain. And NOT across classes, despite
> the table above: `ours+hist`'s NOISY and REVISED means are identical, and the
> contrast is across WRITE TYPES inside a class, a REVISED `change` getting 3.5x
> a REVISED `repeat` and 2.3x a NOISY `change`. Averaging over write types
> inside a class destroys the contrast, which is why this table looks flat.
> Two further things that section 4 does not predict: the KILLED `ours+cheap`
> targets `REVISED & change` as accurately as `ours+hist` does (enrichment 12.33
> against 12.20, at a ceiling of 12.92), so targeting does not separate the
> arms; and `ours+cheap` seed 1 reaches mean beta 0.0039, inside this regime,
> and is its worst seed. The gain itself localizes to addresses seen **once or
> twice**: +0.3301 nats at one prior write, -0.0257 at nine or more.

**The constant test validates itself on `none`**: forcing the input-only gate
to its own mean costs **+0.0001**, so it really is a constant in disguise,
which is what the task's per-sequence class randomisation guarantees. That
check could have failed and did not.

---

## The three-regime beta response: destroyed, but not by history

04 recorded the shipped gate parking beta near 1.0 at low residual:
`0.997 0.864 0.016 0.022 0.033 0.044 0.062 0.183 0.258 0.304` by s1 decile.

```
arm           mean beta by s1 decile                                     spearman
ours_pub      0.997 ... 0.043 ... 0.247   NON-MONOTONE, 04's shape HELD
ours          0.024 0.007 0.010 0.015 0.027 0.039 0.061 0.227 0.329 0.359   +0.890
ours+rate     0.004 0.006 0.020 0.037 0.046 0.052 0.069 0.180 0.235 0.268   +0.974
ours+bayes    0.006 0.006 0.033 0.044 0.043 0.048 0.088 0.150 0.195 0.205   +0.678
ours+hist     0.001 0.002 0.002 0.002 0.003 0.002 0.002 0.002 0.002 0.002   -0.078
ours+oracle   0.004 0.002 0.002 0.002 0.002 0.001 0.001 0.001 0.001 0.001   -0.438
```

**Only `ours_pub`, the literal shipped 3-input gate, reproduces it**, and it
reproduces the 0.997 exactly. Its bottom decile edges are `0.00 0.00 0.04`:
over 20% of its writes have `||e|| ~ 0`.

So the three-regime shape is a **consequence of a high-beta policy**, not a
learned preference. Write hard, and exact repeats produce a zero residual
where beta cannot matter, and the gate parks it anywhere. Arms that find the
low-beta averaging solution never generate those zero-residual writes and have
no bottom regime to park in.

`ours+hist` does destroy the shape, but so does `ours`, so **history is not
what destroys it**. The masked control and the shipped gate reach nearly
identical loss (1.5531 vs 1.5509) by qualitatively different policies -
`ours` writes 0.016 on STABLE repeats, `ours_pub` writes 0.471 - which is
worth knowing before treating either as "the" gate's policy.

`ours+hist` and `ours+oracle` also invert the sign: beta DECREASES with
residual norm. 04 lists "policy direction: AMPLIFY on large residual" as a
taken position; these two arms damp instead.

---

## Cost

`cost.py` times configurations alternating inside one process, because a
between-process timing on this shared machine is worthless: the same arm
measured ten minutes apart differed by 2.5x while other sessions took the load
average from 2 to 185 on 10 cores.

```
                                  median s/step    vs no bookkeeping
none                                      0.963               -5.1%
gate, no bookkeeping (n_hist=0)           1.015               +0.0%
gate + bookkeeping (n_hist=5)             1.082               +6.6%
```

Two runs of this gave +2.1% and +6.6%, so the honest claim is **single-digit
percent wall-clock on CPU**, and nothing tighter.

**That is not the real cost and should not be quoted as if it were.** The
bookkeeping here is exact and crosstalk-free, one slot per symbol. A real
layer needs `d_k x d_v` of state for the residual-direction features, as large
as `S` itself, written state-dependently, which breaks the associative scan
the same way the existing gate already does. `h_meane` and `h_nvis` need only
`d_k`. Wall-clock on a sequential Python loop at `d_k = 128` says nothing
about any of that.

---

## Confirmation on held-out stream 5151

Stream 5151 was fixed in the pre-registration, never touched during any
decision, and scored once, after the 5150 verdict was committed at `821dfb4`.
The pre-registration requires a PASS to clear the bar on **both** streams.

```
arm              5150 gain   5151 gain   verdict   agree
none                     -           -         -
ours                     -           -         -
ours+hist          +0.0551     +0.0563      PASS     yes
ours+rate          +0.0060     +0.0052      KILL     yes
ours+oracle        +0.0698     +0.0731      PASS     yes
ours+bayes         +0.0393     +0.0426      PASS     yes
ours+nvis    *     -0.0009     -0.0027      KILL     yes
ours+fresh   *     +0.0032     +0.0028      KILL     yes

VALIDITY GATE        +5.42%      +5.60%      PASS     yes
```

**Eight arms, two streams, zero disagreements.** `ours+rate` misses on both
(+0.0060 and +0.0052 against +0.0064), so its KILL does not depend on which
stream it was read off.

### One honest wrinkle in the sd cross-check

The cross-check printed "UNDERPOWERED" on 5151, because the pooled within-arm
sd came out at 0.0128 against the 0.0039 the bar assumes. That is entirely
`ours+nvis`:

```
per-arm seed sd, 5151:  none 0.0027   ours 0.0051   +hist 0.0014   +rate 0.0052
                        +oracle 0.0011   +bayes 0.0022   +fresh 0.0028
                        +nvis 0.0370   <-- 7 to 100x everyone else
```

Pooling over all arms:          sd 0.0128  ->  implied bar +0.0208
Pooling excluding `ours+nvis`:  sd 0.0035  ->  implied bar +0.0056

So the pre-registered +0.0064 is conservative against every stable arm, and
the "underpowered" flag is an artifact of including a single unstable control
in a nuisance-parameter estimate. **The bar was not moved either way**, and it
does not matter: `ours+hist` (+0.0563), `ours+oracle` (+0.0731) and
`ours+bayes` (+0.0426) clear even the inflated +0.0208 bar on both streams.

---

## Every pre-registered bar

```
                                            5150      5151   verdict
VALIDITY  ours beats none by >= 4.00%     +5.42%    +5.60%     PASS
PRIMARY   ours+hist   >= +0.0064         +0.0551   +0.0563     PASS   8.8x
          ours+rate   >= +0.0064         +0.0060   +0.0052     KILL   0.8x
          ours+oracle >= +0.0064         +0.0698   +0.0731     PASS  11.4x
          ours+bayes  >= +0.0064         +0.0393   +0.0426     PASS   6.7x
CONTROLS  ours+nvis   >= +0.0064         -0.0009   -0.0027     KILL
          ours+fresh  >= +0.0064         +0.0032   +0.0028     KILL
SD CHECK  pooled sd excluding ours+nvis    0.0034    0.0035   conservative
```

### What this means for 04's recommendation

04 concluded "NO. Not worth building", on a recorded prediction of **no loss
gain beyond seed noise**. Its own audit then withdrew most of the support for
that, leaving one cross-model single-seed null and the state cost.

**The prediction is wrong.** `ours+hist` buys +0.0551 and +0.0563 nats on two
fresh streams, 8.8x the bar, every seed positive, using only features
computable from the model's own residuals plus per-address bookkeeping - no
oracle. For scale, that is **62% of the entire published memory-gate effect**
(`ours` over `none` is 0.0889 nats), bought on top of it.

**But 04's mechanism is not what delivers it**, and the directory should not
be read as vindicating the volatility hypothesis:

- the identifiability mechanism 04 theorised is real and isolated here, in
  `ours+bayes`, whose gain tracks visit-count identifiability. It is worth
  **+0.0393**, needs a feature that reads `N_OPT` and `P_REV` off the task
  generator, and is beaten by two arms with less information.
- the larger `ours+hist` effect runs the opposite way across visit counts,
  makes accuracy worse, makes REVISED worse, and is best explained by beta
  collapsing to a uniform running average.
- `h_meane`, 04's strongest single separability feature, **fails the bar on
  both streams** as a standalone arm.

So: history reduces loss, the recommendation to not build is overturned, and
the reason it works is not the reason that was proposed.

---

## Reportable anomalies

- **`ours` and `ours_pub` reach the same loss by different policies.** 1.5531
  vs 1.5509, but STABLE-repeat beta 0.016 vs 0.471 and a qualitatively
  different beta-vs-residual response. Widening the gate's input from 3 to 9
  and masking six columns does not just cost 0.0022 nats, it moves the
  solution to a different basin. Any "the gate's learned policy is X" claim in
  this repo is a claim about one basin.
- **More information can hurt.** `ours+bayes` strictly contains
  `ours+oracle`'s features and loses 0.0305 nats to it.
- **`h_ratio` is dead weight.** Zeroing it at eval time costs +0.0005 on
  `ours+hist` and +0.0001 on `ours+oracle`, the only column that is free to
  remove.
- The `h_ratio` clamp never bound in training: 0.000% on every run.
- **`ours+nvis` is unstable.** Seed sd 0.0384 and 0.0370 across the two
  streams, 7 to 100x every other arm (seeds 1.5305 / 1.5982 / 1.5331). Giving
  the gate a bare visit counter and nothing else makes training erratic. It is
  also the arm that inflates the pooled-sd cross-check.
- **`ours+bayes` has the best accuracy and the best REVISED loss** (0.6491 and
  2.3439) while having only the third-best overall loss. The arms trade off
  differently, so "which arm is best" depends on the metric.
- The task's absorbing states dominate as 04 reported: 48% of positions have a
  STABLE query, and `none` already scores 0.4437 nats there against 3.1287 on
  NOISY.

## Files

```
PREREGISTRATION.md   question, metric, bars, kill conditions, and four
                     appendices. The first three were written before any
                     number existed; the fourth says in its first line that
                     it was not, and adds only controls that can shrink the
                     headline.
arms.py              the nine arms. All four primary gated arms are
                     62,250 parameters and differ only in a 0/1 keep-mask.
bayes.py             the exact Bayes posterior from task.py's generative
                     process. Verified against 04's audit_bayes_ceiling.py:
                     AUROC 0.9449 vs 0.9449, and the whole AUROC-by-visit
                     table to three decimals.
test_parity.py       the drift guard. n_hist=0 reproduces the shipped
                     GatedDeltaSSM bit-for-bit; the masked control's columns
                     are exactly zero; no gradient reaches any parameter
                     through the bookkeeping; o_memb and the visit counts
                     match a naive reimplementation written from the spec.
run_arm.py           train one arm at one seed, published protocol
run_all.sh           the 18 primary runs, two at a time
run_bayes.sh         the ceiling arm, queued after run_all
run_controls.sh      the two post-hoc controls, queued after run_bayes
score.py             both loss metrics, the bars, the validity gate, the
                     baseline-inflation check and the visit-count slices
const_test.py        is it the beta LEVEL or the beta VARIATION?
ablate.py            zero one history column at a time in a trained model
gate_use.py          d beta / d feature, by autograd on real inputs
beta_report.py       beta by class x write type, and the three-regime shape
cost.py              load-robust timing, alternating inside one process
logs/                one log per run, plus every analysis above
results_5150.json    primary stream
results_5151.json    held-out stream
```

## Reuse hazard, handled

14 files in this repo score on `default_rng(12345)`, 04 used 777 and 888, and
`validation-suite` used 202/303/404/505/606. Nothing here touches any of them.
**5150** is the primary stream and **5151** was held out until the verdict was
committed at `821dfb4`. 5152 was used once to smoke-test the scoring path on
throwaway 2-step models and no decision rests on it.

## What was NOT checked

- **Crosstalk.** The bookkeeping is exact, one slot per symbol. A real layer
  must hold this in superposition, and every per-key store attempted in this
  repo died of crosstalk. These numbers are an UPPER BOUND on what a real
  mechanism could deliver, and the gap is untested.
- **Why the gain inverts across visit count** between `ours+hist` and
  `ours+bayes`. The two profiles are the most interesting thing here and
  neither is explained.
- **Whether the beta collapse is reachable without history.** `const_test.py`
  rules out a constant, but not a different parameterisation or schedule that
  reaches the same low-beta averaging regime more cheaply. That is the obvious
  next experiment and it is not run here.
- **800 steps only.** `validation/convergence.log` records `ours` still
  descending at 3000. All of this is the published operating point, not an
  asymptote.
- Anything at a different `d_k`, `d_v`, sequence length or `N_OPT`.
