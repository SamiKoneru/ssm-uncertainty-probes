# `ours+cheap` result: KILL, confirmed on a fresh stream

Verdict against the bars in `PREREGISTRATION_cheap.md`, which were committed
in c0dfffb before `arms.py` gained the arm. Nothing below moves a bar.

## Headline

```
                      stream 5150        stream 7771 (CONFIRMATION)
ours                     1.5531             1.5761
ours+cheap               1.5548             1.5766
  gain vs ours          -0.0018            -0.0005
ours+hist                1.4979             1.5183
  gain vs ours          +0.0551            +0.0578

bar 1  existence   >= +0.0064              KILL on both
bar 2  sufficiency >= +0.0276              FAIL on both
fraction of the full block's gain retained  -3.2%  /  -0.8%
```

Stream 7771 appears nowhere in this repo or any of the three sibling
worktrees; verified by grep before it was chosen. 5151 was already spent.

**The pre-registered outcome is the third one: KILL.** The three cheap,
chunk-freezable features together do no better than either of them alone
(`ours+rate` +0.0059, `ours+nvis` +0.0005 on 7771). Superadditivity does not
happen here.

## The prediction was wrong, and it was mine

Before running this, in writing: "Given the ablation puts two thirds of the
weight on exactly those cheap features, and given that this block is strongly
superadditive (worthless alone, valuable together), I think that arm is more
likely to succeed than not."

It failed. Recorded here because the protocol says report every attempt, and
because the prediction erred toward the answer I wanted.

**The reasoning error, specifically.** I treated `ablate.py`'s eval-time
column knockout as an attribution of what each feature *contributes*. It is
not. Zeroing a column in a model that trained with all five measures how much
that model has come to lean on it, which is co-adaptation. `ablate.py`'s own
docstring says so ("a trained gate has co-adapted to all its columns, so
knocking one out at eval time overstates that column's importance relative to
training without it. Read the two together") and README.md says it again. The
standalone arms were the honest signal and they said KILL. This arm is what
it takes to tell those two readings apart, and it sided with the arms.

So `VOLATILITY.md`'s claim that the residual-**direction** family carries the
gain stands, and the correction I made against it on the strength of the
ablation table was wrong.

## What the recovered diagnostics add: this is not a null, it is a degradation

Mean beta by key class on seen writes, stream 7771. The task wants
NOISY < REVISED, because NOISY values stay valid and should be averaged while
REVISED values go stale and should be overwritten.

```
arm          STABLE   NOISY  REVISED
none          0.387   0.386   0.387    no gate: one global policy, as designed
ours          0.022   0.137   0.116    discriminates STABLE from the rest
ours+rate     0.032   0.135   0.122    preserves the `ours` profile
ours+fresh    0.024   0.131   0.111    preserves it
ours+cheap    0.011   0.053   0.053    NOISY and REVISED now IDENTICAL
ours+nvis     0.005   0.024   0.030    also collapsed
ours+hist     0.001   0.002   0.002    total collapse (already documented)
ours+bayes    0.028   0.065   0.108    the CORRECT ordering, NOISY < REVISED
```

Two things here, both new:

1. **The cheap block destroys discrimination that `ours` already had.** It
   pushes beta down about 2.5x (NOISY 0.137 -> 0.053, REVISED 0.116 -> 0.053)
   and makes the two classes indistinguishable. Adding the features is worse
   than not having them, which is why the gain is negative rather than zero.
   `acc_hard` agrees: 0.3587 against `ours`'s 0.3767.

2. **The correct policy is reachable.** `ours+bayes`, holding the exact
   sufficient statistic for the value history, is the only arm that gets
   NOISY below REVISED. So the task's optimal beta ordering is attainable in
   this architecture, and the cheap features lead away from it rather than
   toward it.

## Where `ours+cheap` actually goes wrong: visit count

Loss on NOISY+REVISED queries, gain vs `ours`, sliced by prior writes to the
query address. Stream 7771, exploratory, no bar.

```
                 0(fresh)       1        2      3-4      5-8       9+
ours+hist         +0.0899  +0.3301  +0.1512  +0.0583  +0.0242  -0.0257
ours+cheap        -0.0049  +0.2013  +0.0627  -0.0563  -0.0768  -0.0116
```

The cheap block **helps at 1 to 2 prior visits** (+0.2013, +0.0627) and
**hurts from 3 visits on** (-0.0563, -0.0768). It nets to zero by
cancellation, not by carrying no information.

That profile is a mechanism for the headline. At one or two visits there is
almost no history, so "how new is this address" is most of the available
signal and a counter captures it. Once enough history exists for the
direction features to say something, scalars alone start actively misleading
the gate. The full block keeps the low-count gain and does not pay the
high-count penalty.

## Power: the KILL on bar 1 is "indistinguishable from zero", not "proven zero"

```
pooled within-arm sd 0.0129  ->  implied bar +0.0210
pre-registered bar                           +0.0064
```

This run was 3.3x noisier than the prior sd the bar was built from, driven by
`ours+cheap` (0.0204) and `ours+nvis` (0.0344). The bar was NOT moved. The
honest consequence:

- **Bar 2 (+0.0276) fails robustly.** Even the upper end of what this run
  could hide, about +0.021, is below it. The sufficiency conclusion holds.
- **Bar 1 (+0.0064) sits under the noise floor**, so a true effect anywhere
  in roughly [-0.02, +0.02] is consistent with what was measured. The
  measured mean is negative on both streams, so this is not a missed positive
  effect, but "exactly zero" is not established either.

Either way the deployment question is settled: nothing in that interval buys
enough to justify a second `d_k x d_v` state.

## Anomaly, flagged and not explained

`ours+cheap` sd is 0.0204 (7771) and 0.0239 (5150), about ten times every
other arm. Per-seed gains are +0.0108 / **-0.0243** / +0.0120.

```
arm           sd(7771)   s0 / s1 / s2 gain
ours+hist      0.0011    +0.0613  +0.0561  +0.0561   most stable arm in the table
ours+rate      0.0041    +0.0055  +0.0035  +0.0087
ours+fresh     0.0026    +0.0077  +0.0041  -0.0016
ours+cheap     0.0204    +0.0108  -0.0243  +0.0120   seed 1 breaks
ours+nvis      0.0344    +0.0244  -0.0397  +0.0167   seed 1 breaks
```

The two unstable arms are exactly the two carrying `h_nvis` **without** the
direction features. The same `h_nvis` inside the full block gives the most
stable arm here. Same seed, both streams, so it is an interaction and not a
bad seed. Seed 1's final train loss (1.5501) is mid-range, so this is not a
training divergence.

Two arms is not a pattern worth betting on, but this is the third seed-1
collapse in the project (`validation/dim_sweep.log` at d=24 is another, sd
1.1013), so it deserves its own look rather than a footnote.

**It does not rescue the verdict.** The two non-collapsing seeds average
about +0.011, which clears bar 1 but is nowhere near bar 2, and selecting
seeds after seeing which ones worked is exactly the move the protocol
forbids. KILL stands as pre-registered.

## Harness checks, all passing

```
check_cheap.py   param identity 62,250 across all five N_EXTRA arms
                 BLIND: perturbing h_cos / h_cosmean / o_memb moves the
                        logits by exactly 0.000e+00
                 SIGHTED: perturbing h_ratio / h_meane / h_nvis moves them
                        by 1.5e-01 / 1.2e-01 / 1.2e-01
                 5-step smoke run finite
test_parity.py   unchanged, ALL CHECKS PASS
validity gate    none 1.6671 -> ours 1.5761 = +5.45% on 7771, bar 4.00%, PASS
```

That last line is a second result worth keeping: the published ~5.5% headline
and `ours+hist`'s history gain (+0.0551 -> **+0.0578**) both reproduce on a
stream that has never been looked at.

## Consequence for the long-term volatility idea

```
does per-address history help?    YES, +0.0578, now confirmed on fresh data
which part carries it?            the residual DIRECTION features
what do they cost?                d_k x d_v, doubling the layer's state
are they chunk-freezable?         NO. cos(e_t, e_prev) needs the previous
                                  write to the SAME address, which routinely
                                  falls inside the current chunk (the median
                                  sequence sends 28 of 256 writes to one
                                  address), so freezing at chunk entry loses
                                  exactly the comparison.
```

The science works and the deployable version does not exist in this form.

## Not settled by this

- **One task, one size, 800 steps.** The task is built so NOISY addresses
  reward averaging. "Direction features are necessary" is established for
  this task, not in general.
- **Why `ours+hist` wins while collapsing beta to 0.002** everywhere. It does
  not win by discrimination, and `ours+bayes` discriminates correctly while
  gaining less. Still unexplained, and still the most interesting thing in
  this directory.
- **The seed-1 / `h_nvis` interaction** above.
- **A learned (crosstalk-bearing) version of the cheap features** was never
  built. This arm used exact per-symbol bookkeeping, which is an upper bound,
  so the KILL is a KILL for the learned version too. That direction is closed,
  not untested.
