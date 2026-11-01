# Pre-registration: the SHAPE of beta, not its mean

Written BEFORE `beta_shape.py` exists. Nothing in here may be changed after the
first number is seen. If something must change, it gets logged with a reason and
the original is reported alongside, per the project rule.

## The anomaly that motivates this

From `logs/score_7771_cheap.log`, mean beta by the class of the write address:

```
arm            STABLE    NOISY  REVISED     loss gain vs ours
ours            0.022    0.137    0.116            -
ours+hist       0.001    0.002    0.002       +0.0578   <- best arm
ours+bayes      0.028    0.065    0.108       +0.0401
ours+cheap      0.011    0.053    0.053       -0.0005
none            0.387    0.386    0.387            -
```

The project's thesis is that a gate should read memory state in order to pick a
different write rate per volatility class: low for NOISY (average the redraws),
high for REVISED (overwrite the stale value).

**The best arm does not do that.** `ours+hist` collapses beta to 0.001-0.002 and
shows no class ordering at all. `ours+bayes` is the only arm whose ordering is
correct (NOISY 0.065 < REVISED 0.108) and it gains less. So the winning arm wins
by something other than per-class rate selection, and the project does not know
what.

Those are MEANS. A mean of 0.002 is produced equally well by two policies that
are nothing alike, and nobody has looked at which one it is.

## Question

Does `ours+hist` keep beta uniformly near 0.002, or does it keep beta at
essentially zero almost everywhere and spike it on a small fraction of writes?

## Why it matters before anything else gets built

- **Flat and low** means the gain is not coming from write decisions at all. The
  framing of 04 and 05 would be wrong, and the planned low-rank direction-memory
  experiment would be built on a false premise.
- **Spiky** means the algorithm is write-once plus rare overwrite, which is a
  different algorithm from per-class rate tuning. A trigger that fires on 1% of
  writes may be implementable far more cheaply than a feature that has to
  regress a continuous rate at every step.

Free to answer: the 30 checkpoints already exist, this is scoring only.

## Definitions

All conditioning is on the WRITE at step t, which sends the value from token
`x_t` to the address from token `x_{t-1}`. That is one position offset from the
query labelling `score.py` uses for loss, and the two must not be mixed.

- `cls_w[t] = class(x_{t-1})`, the volatility class of the address written.
- `write_type[t]`, computed from the token stream alone, independently of the
  model's bookkeeping, reusing `beta_report.py`'s definition:
  - `unseen`: this address has not been written before in this sequence.
  - `repeat`: written before, and `x_t` equals the value last written there.
  - `change`: written before, and `x_t` differs from it.
- `visits[t]`: prior writes to this address in this sequence.

`change` is the position where the two classes want OPPOSITE actions, which is
what makes it the target of interest:

| class of address | a `change` means | correct beta |
|---|---|---|
| REVISED | a permanent revision; the stored value is now WRONG | high, overwrite |
| NOISY | a fresh draw from the same fixed 4-option set; old observations still VALID | low, keep averaging |
| STABLE | cannot happen | n/a |

So the ideal targeted policy spikes on `REVISED & change` specifically, not on
`change` in general.

## Metrics

Over all write positions `t >= 1`, per arm, per training seed, reported
separately for all three seeds (not pooled, so a single-seed artifact stays
visible: `ours+cheap`'s seed 1 was 10x the sd of every other arm).

1. **Shape.** Quantiles of beta: p50, p90, p99, p99.9, max. Plus
   `frac(beta > 0.5)` and `frac(beta > 0.9)`.
2. **Concentration.** `top1_share` = (sum of the largest 1% of beta values) /
   (sum of all beta values). A constant gate gives exactly 0.01. All the mass in
   1% of positions gives 1.00. This is scale-free, so it compares arms whose
   mean beta differs by 100x.
3. **Targeting.** Among the top 1% of beta positions,
   `enrich = P(REVISED & change | top 1%) / P(REVISED & change | all writes)`.
4. **Targeting vs identifiability.** Metric 3 within each visit-count bin. The
   exact Bayes AUROC for NOISY vs REVISED at those counts is already known
   (0.687 / 0.786 / 0.890 / 0.977 / 0.9997 at 1 / 2 / 3-4 / 5-8 / 9+), so if the
   spikes are a real inference rather than noise, their enrichment should RISE
   with visit count.

## Bars, fixed now

Verdicts require the condition to hold on ALL THREE seeds. Two out of three is
reported as a split and claims nothing.

**Shape of `ours+hist`:**
- `SPIKY` if `top1_share >= 0.25` and `frac(beta > 0.5) >= 0.001`.
- `FLAT` if `top1_share <= 0.10` and `max(beta) <= 0.5`.
- anything else: `INTERMEDIATE`, reported, no claim.

**Targeting of `ours+hist`:**
- `TARGETED` if `enrich >= 2.0`.
- `UNTARGETED` if `enrich <= 1.2`.
- anything else: `INTERMEDIATE`.

The two bars are independent, so there are three outcomes that matter:

| shape | targeting | reading |
|---|---|---|
| SPIKY | TARGETED | write-once plus targeted overwrite. Reframes the project and motivates a cheap trigger. |
| SPIKY | UNTARGETED | the spikes are not about volatility class. An anomaly to chase, NOT a licence to build a trigger. |
| FLAT | either | the gain is not from write decisions. The 04/05 framing is wrong, and this is the most consequential of the three. |

## What would kill the reframe

`FLAT`, or `SPIKY + UNTARGETED`. In either case "rare targeted overwrite" is the
wrong description and I do not get to propose a cheap trigger built on it.

## Reference arms, and why each is here

| arm | why |
|---|---|
| `none` | no memory features, mean beta flat at 0.387. Calibrates what `top1_share` looks like for a gate that CANNOT condition on history. If `none` already scores above 0.25 the concentration bar is meaningless and I say so. |
| `ours` | the shipped gate. Has the class ordering `ours+hist` lacks. |
| `ours+bayes` | the only arm with the CORRECT ordering. If spiking is the winning mechanism, this arm should spike too; if it does not, spiking is not what separates winners from losers. |
| `ours+cheap` | the KILLED arm. Its NOISY and REVISED means are identical (0.053 / 0.053). Shape may show whether it failed by being flat or by spiking in the wrong places. |
| `ours+oracle` | collapses beta like `ours+hist` (0.001/0.002/0.002) and gains the most. Should behave the same way; if it does not, `ours+hist`'s shape is not the mechanism. |
| `ours_pub` | the unmasked published gate, mean beta 0.174/0.132/0.111. |

`none` and `ours_pub` make the concentration bar able to fail, which is the
point of including them.

## Validity check that can fail

`beta_shape.py` collects beta through its own code path. It must reproduce
`score.py`'s published `beta_by_class` numbers, using score.py's own masking
convention (`class(x_{t-1})` AND the task's `seen` flag for `x_t`), to within
1e-6 for every arm and seed. If it does not, the collection is wrong and the
whole report is void. This is asserted in the script, not eyeballed.

## Streams

- `5150` PRIMARY but already looked at, so EXPLORATORY.
- `3310` CONFIRMATION. Verified absent from every `.py`, `.md` and `.sh` in this
  worktree and in all three sibling worktrees before being chosen. Used streams
  elsewhere in the repo: 0, 1, 11, 31, 99, 101, 321, 505, 777, 888, 999, 4242,
  5150, 5151, 7771, 12345, 31337.

A shape verdict counts as settled only if both streams agree.

## Status of this whole exercise

EXPLORATORY and POST-HOC. It was designed after seeing the mean-beta table, and
it tests no loss claim. The bars make the shape question falsifiable; they do not
make the investigation confirmatory. Nothing here promotes or demotes any
existing verdict in `README.md` or `RESULTS_cheap.md`.
