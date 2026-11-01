# Result: the shape of beta. Not spiky, maximally targeted, and the targeting is worthless

Pre-registered in `PREREGISTRATION_beta_shape.md`. Run: `beta_shape.py --streams
5150 3310`, log `logs/beta_shape.log`, JSON `beta_shape_{5150,3310}.json`.
No bar was changed.

## Verdict as the bars read it

```
                 5150                    3310 (CONFIRMATION)
SHAPE            INTERMEDIATE            INTERMEDIATE
TARGETING        TARGETED                TARGETED
streams agree    yes
validity         beta_by_class reproduced from results_5150.json, max |diff| 0.00e+00
```

`ours+hist`, three seeds, both streams:

| metric | 5150 | 3310 | bar |
|---|---|---|---|
| `top1_share` | 0.0920 / 0.1009 / 0.1153 | 0.0870 / 0.0974 / 0.1061 | SPIKY >= 0.25, FLAT <= 0.10 |
| `frac(beta > 0.5)` | 0.0000 / 0.0000 / 0.0000 | same | SPIKY >= 0.001 |
| `max beta` | 0.0729 / 0.0501 / 0.0939 | 0.0627 / 0.0525 / 0.0825 | FLAT <= 0.50 |
| `enrich(REVISED & change)` | 12.447 / 11.788 / 12.368 | 11.901 / 11.221 / 11.901 | TARGETED >= 2.0 |

**A gap in my own pre-registration.** It enumerated readings for SPIKY and for
FLAT and never for INTERMEDIATE, so by the letter of the document this pair
decides nothing, and that is what the script printed (`REFRAME: UNDECIDED`). The
script is left as it stands rather than re-judged after the fact.

On the substance the reframe is dead anyway, because it required SPIKY and SPIKY
failed on all three of its conditions and not narrowly:

- concentration 0.10 against a bar of 0.25,
- `frac(beta > 0.5)` exactly 0.0000 on every seed and both streams,
- **not one write out of 48,960 has beta above 0.094.**

"Write-once plus rare targeted overwrite" is not what `ours+hist` does. FLAT
misses only on `top1_share`, by 0.001 to 0.015 on two of three seeds, with
`max beta` passing by a factor of 5. It is a near-flat policy, and I do not get
to relabel it FLAT.

## The finding that actually matters: the targeting is maximal and worthless

The base rate of `REVISED & change` is 0.0774, so the **largest possible
enrichment is 1 / 0.0774 = 12.92**. Every arm with history features sits at the
ceiling:

| arm | enrich (5150) | of the top 490 positions, n that are REVISED & change | loss gain vs `ours` (7771) |
|---|---|---|---|
| `none` | 1.04 | 49 / 39 / 30 | - |
| `ours_pub` | 1.39 | 0 / 140 / 18 | +0.0027 is the MASKING COST, not a gain |
| `ours` | 2.05 | 0 / 103 / 130 | reference |
| `ours+cheap` | **12.33** | 475 / 452 / 476 | **-0.0005  KILLED** |
| `ours+oracle` | 12.11 | 455 / 451 / 472 | +0.0730 |
| `ours+hist` | 12.20 | 472 / 447 / 469 | +0.0578 |
| `ours+bayes` | 12.90 | **490 / 489 / 489** | +0.0401 |

`ours+bayes` puts **all 490** of its highest-beta writes on `REVISED & change`.

**`ours+cheap`, the arm that KILLED on loss, targets as well as `ours+hist`.**
12.33 against 12.20, reproduced on the fresh stream (12.04 against 11.68). So
hitting the right positions with the top of the beta distribution is NOT what
separates the arms that win from the arm that does not, and metric 3 cannot
discriminate among history arms at all: it is saturated. That is an instrument
limitation of this experiment, reported, not worked around.

## Nor does the discrimination ratio

The mean-beta ratio between the two write types that want opposite actions:

| arm | mean beta (s0) | REVISED&change | NOISY&change | ratio | loss gain |
|---|---|---|---|---|---|
| `none` | 0.3878 | 0.387 | 0.387 | 1.00 | - |
| `ours` | 0.1097 | 0.287 | 0.182 | 1.58 | reference |
| `ours+cheap` | 0.0661 | 0.144 | 0.070 | 2.06 | -0.0005 |
| `ours+hist` | 0.0021 | 0.007 | 0.003 | 2.33 | +0.0578 |
| `ours+oracle` | 0.0017 | 0.006 | 0.002 | 3.00 | +0.0730 |
| `ours+bayes` | 0.0819 | 0.279 | 0.086 | **3.24** | +0.0401 |

`ours+bayes` has the best ratio and a middling gain. `ours+cheap` beats `ours`
on ratio and loses to it on loss. So relative discrimination does not predict
the gain either.

What this experiment has therefore established is mostly negative, and that is
its value: **neither the tail of the beta distribution nor the per-class
contrast explains why `ours+hist` wins.** Every top-1% instrument built here is
the wrong instrument, which rules out the whole family of spike and
tail explanations and pushes the mechanism into the bulk of the distribution.

## Where the mechanism must live instead

Two tables already in the repo locate it, and they agree with each other.

The identifiability table from this run, enrichment within each visit-count bin:

```
arm             1 visit   2    3-4    5-8     9+        (Bayes AUROC .687 .786 .890 .977 .9997)
none              1.09  1.35  1.10   1.18   1.04
ours              2.31  3.84  4.81   6.57  16.29
ours+hist         2.55  3.56  4.10   6.81  18.68
```

`ours` reaches enrichment 16.3 at 9+ prior writes **with no history features at
all**, and is level with `ours+hist` from the second visit onward. The reason is
mechanical: at a well-averaged address a REVISED revision produces a large
`||e_t||` while a NOISY redraw produces a small one, so `s1 = log1p(||e_t||)`,
which every memory-gated arm already has, separates the classes on its own once
the address has been visited enough.

And the loss-by-visit table from `logs/score_7771_cheap.log`, `ours+hist` gain
over `ours`:

```
prior writes   0 (fresh)      1       2     3-4     5-8      9+
gain            +0.0899  +0.3301  +0.1512  +0.0583  +0.0242  -0.0257
```

The gain is concentrated at **1 to 2 prior writes** and turns **negative** at 9
or more. The two tables say the same thing: the history features buy nothing
where `||e_t||` already works, and everything in the narrow band where the
address has been seen once or twice and `||e_t||` is still ambiguous.

That is the first real localization of the +0.0578 in this directory.

## Correction: the accumulator story is already in the README, not new here

An earlier draft of this file presented the following as a hypothesis generated
by this run. It is not. `README.md` section 4 already states it, and already has
stronger evidence for it than anything measured here:

> `beta -> 0` turns the memory into a uniform running average over everything
> ever written to that address

with the scale-invariance of `head(norm(S q))` spelled out, and with
`const_test.py`'s result that forcing `ours+hist` to its own mean beta costs
**+0.2564** nats while the best constant over a 16-point grid only reaches
1.7517 against 1.4979 as trained. That is direct evidence that the beta
VARIATION does the work, and it predates this run.

What this run adds to that section is a refinement and a problem.

**The refinement.** The README reads the per-class means (`ours+hist` 0.001 /
0.002 / 0.002) as "precisely the per-class pattern" the accumulator story
predicts, but NOISY and REVISED are IDENTICAL in those means. The variation is
not across classes at all, it is across WRITE TYPES within a class:

```
ours+hist      unseen   repeat   change
  STABLE        0.003    0.001       -
  NOISY         0.003    0.001    0.003
  REVISED       0.003    0.002    0.007
```

A REVISED `change` gets 3.5x the weight of a REVISED `repeat` and 2.3x that of a
NOISY `change`. Averaging over write types inside a class destroys exactly the
contrast that matters, which is why the published per-class table looks flat.

**The problem.** `ours+cheap` at seed 1 has mean beta 0.0039, inside the
accumulator regime, and it is its WORST seed by loss (-0.0243 on 5150, -0.0277
on 7771). Being in that regime is therefore not sufficient, and the README's
section 4 does not account for an arm that reaches the regime and loses.

The sharp test, which costs no training and is not yet run: rescale
`ours+bayes`'s beta by a constant at eval time, preserving its ratios exactly.
Its ratio (3.24) is better than `ours+hist`'s (2.33) and its gain is worse, so if
the regime is what matters, scaling it down should recover the difference. If it
does nothing or hurts, the regime account is incomplete in a second way.

## Anomalies noticed in passing

1. **`ours_pub` seed 0 is a different policy from its own seeds 1 and 2.** Mean
   beta 0.2947, p90 0.9971, `frac(beta > 0.9) = 0.2026`, max 0.9981. Seeds 1 and
   2: mean 0.098, max 0.36, `frac(beta > 0.9) = 0.0000`. One in five writes at
   beta above 0.9 on seed 0 and none at all on the other two, from the same
   published configuration. Reproduced identically on 3310 (0.1913).
2. **`ours` seed 0** is milder but the same direction: max 0.5389 and
   `frac(beta > 0.5) = 0.0010`, against max 0.30 to 0.32 and 0.0000 for seeds 1
   and 2.
3. **Seed 0 enrichment of exactly 0.000** for `ours` and `ours_pub`: their entire
   top 1% lands on `unseen` writes, which can never be `REVISED & change`. Their
   highest mean beta is on fresh addresses (0.313 and 0.297) above
   `REVISED & change` (0.287 and 0.284). The shipped gate spends its largest
   writes on addresses it knows nothing about.
4. The `ours+cheap` seed-1 collapse now has a shape as well as a loss: mean beta
   0.0039 against 0.066 and 0.049 for seeds 0 and 2, and `top1_share` 0.1501
   against 0.0576 and 0.0670. Still unexplained. Fourth seed-1 oddity in this
   project.

## Not settled

- Why `ours+hist` wins. This run ruled out the tail explanations and localized
  the gain to 1 to 2 prior writes. It did not identify the mechanism.
- Whether the regime hypothesis survives the eval-time rescaling test.
- `ours_pub` seed 0, and `ours+cheap` seed 1.
- Metric 3 is saturated, so a discriminating version (ranking quality over all
  writes rather than the top 1%, and mean beta per visit-count bin, which this
  run does not report) is still needed.

## Status

EXPLORATORY and POST-HOC throughout, as pre-registered. It tests no loss claim
and changes no verdict in `README.md` or `RESULTS_cheap.md`. What it does change
is the direction of the open question, which now points at the bulk of the
distribution at low visit counts rather than at a spike.
