# PRE-REGISTRATION - written before any number was produced

Session question: **does exact per-address volatility history in the write gate
actually reduce LOSS, or only separability?**

`process/04_volatility_history` measured separability only. It found that exact
crosstalk-free per-address history adds **+0.081 AUROC** for NOISY vs REVISED
over the five zero-extra-state features (0.812 -> 0.893), confirmed on a
held-out stream, and it cleared its own pre-registered +0.05 bar. Its
pre-registration licensed a loss measurement once that bar cleared. The loss
measurement was then skipped for machine time. **This directory is that
measurement.**

04's own recorded prediction, which this file does not get to edit:

> The prediction, recorded here: **no loss gain beyond seed noise**, because the
> rate memory already ran that experiment for the strongest single history
> feature and got +0.19pp.

Written 2026-10-18, before `arms.py` was run on anything. Nothing in this file
was edited after seeing a result. Changes, if any, are appended at the bottom
with a reason and the original text kept.

---

## Shared setup

**Protocol, the published one, unchanged** (`gated_delta_ssm/train.py`): 800
steps, batch 24, AdamW lr 2e-3, grad clip 1.0, `gate_width=32`. Task
`gated_delta_ssm/task.py` unmodified. 3 seeds (0, 1, 2) per arm.

For a given seed every arm sees the **identical training stream**, because the
data comes from `np.random.default_rng(seed)` and `T.sample` consumes the same
draws whatever the gate is. Model initialisation is only partially paired: the
arms have different gate input widths, so the torch RNG is consumed
differently. The primary analysis is therefore **unpaired**; per-seed paired
differences are reported as a secondary read.

**Eval streams.** 14 files in this repo score on `default_rng(12345)`, and 04
used 777 and 888. Fresh seeds, confirmed absent from every branch
(`main`, `claude/optimistic-easley-09c6ab`, `validation-suite`):

    seed 5150   PRIMARY. Every headline number is read off this stream.
    seed 5151   HELD OUT. Not touched until the verdict is written and
                committed. If I look at it early I will say so.

6 batches of 32 sequences, matching `train.py`'s eval.

## The four arms

The confound this directory must not repeat is recorded in
`validation/README.md` on `origin/validation-suite`: *"Every published
comparison so far pitted a 3-input MLP gate against no gate at all, which
confounds 'the gate can see the memory' with 'the gate has an MLP in it'."*

So all three **gated** arms get the **same gate input width (8)**, the same
parameter count, the same initialisation distribution and the same FLOPs. They
differ only in which columns are zeroed before the MLP sees them, in the style
of that directory's `token_only` arm.

```
none        memory_gate=False                       the input-only baseline.
                                                    Shipped construction, so
                                                    param count differs from
                                                    the gated arms BY DESIGN.
ours        MLP([tok, s1, s2,  0,0,0,0,0 ])         masked control = the
                                                    shipped gate's information
ours+hist   MLP([tok, s1, s2,  h1..h5 ])            all five exact history
                                                    features live
ours+rate   MLP([tok, s1, s2,  0,0,0,h_meane,0 ])   only the strongest single
                                                    history feature (04: +0.036
                                                    of the +0.081, double what
                                                    the residual direction buys)
```

Secondary arm, **not** part of the primary comparison, run only if machine time
allows:

```
ours_pub    MLP([tok, s1, s2])                      literal 3-input shipped gate,
                                                    to check that widening the
                                                    input to 8 and masking does
                                                    not itself move the result
```

**Param-count statement, made now rather than after the fact.** `ours`,
`ours+hist` and `ours+rate` are exactly equal. `none` is smaller by
construction and `ours_pub` by 160 unused weights; both mismatches are stated
in the results table rather than hidden.

## The five history features

Taken from `process/04_volatility_history/features.py`, reusing its exact
per-address bookkeeping (previous residual, residual sum, running norm sum,
visit count), moved inside the training loop and run under `no_grad`. The gate
reads them detached, exactly as it already reads `s1` and `s2`.

```
h_cos      cos(e_t, e_prev)              previous residual at THIS address
h_cosmean  cos(e_t, sum of prev e)       mean direction at this address
h_ratio    log(||e_t|| / ||e_prev||)
h_meane    log1p(mean ||e|| at address)  THE RATE MEMORY
h_nvis     log1p(visits to address)
```

Addressing follows 04: the write at step `t` stores the value from token `x_t`
at the address from token `x_{t-1}`, so the bookkeeping key is `tokens[:, t-1]`.

**This is an UPPER BOUND, not a mechanism.** The bookkeeping is exact and
crosstalk-free: one slot per symbol, no superposition. A real layer would have
to hold this in a `d_k x d_v` store written state-dependently and would lose
signal to crosstalk. So if `ours+hist` does not beat `ours` here, no real
per-address history mechanism will.

**Three deviations from 04's feature code, all stated now.**

1. `h_meane` and `h_nvis` get a `log1p`, and in 04 they were raw. Monotone, so
   every AUROC in 04 is unchanged; it only conditions the MLP input, which in
   04 was a tree and did not care.
2. 04 emits `NaN` for a fresh address and at `t=0`. NaN into a gate is a dead
   model, so fresh/`t=0` gets **0.0** on the four history-dependent columns.
   No information is lost: `h_nvis = log1p(0) = 0` identifies the fresh case
   exactly, and the MLP can read it.
3. `h_ratio` is clamped to `[-10, 10]`. An exact repeat write makes
   `||e_prev|| ~ 0` and the raw log then reaches +15, which is bad MLP input.
   The fraction of writes where the clamp binds is reported.

A parity test (`test_parity.py`) constructs the shipped `GatedDeltaSSM`, copies
its weights into this directory's subclass with `n_hist=0`, and asserts
identical logits. That is the guard against the reimplemented recurrence
drifting from the thing it audits.

---

## PRIMARY METRIC AND BARS

**Metric.** Mean eval cross-entropy in nats on stream 5150, averaged over 3
seeds. Primary contrast: `ours` minus `ours+hist`, i.e. **positive means
history helped**.

### Deriving the bar from measured seed sd, not asserting it

The repo's own measured seed sd on loss at exactly this protocol, n=3:

```
gated_delta_ssm/train.py docstring   input-only baseline  sd 0.0018
                                     memory-gated w=32    sd 0.0039
```

The task brief quotes the repo range as 0.0026 to 0.0039. I take the **largest**,
`sd = 0.0039`, as the conservative choice, and derive:

```
SE of one arm's mean of 3 seeds       0.0039 / sqrt(3)      = 0.00225
SE of the DIFFERENCE of two arm means 0.0039 * sqrt(2/3)    = 0.00318
BAR = 2 x SE_diff                                           = 0.0064 nats
```

which is **0.41%** of a ~1.54 loss. Everything below is in those units.

```
PASS       mean_loss(ours) - mean_loss(arm) >= +0.0064 nats on stream 5150
           AND the same arm clears +0.0064 on held-out stream 5151.
           This OVERTURNS 04's recommendation.

KILL       mean_loss(ours) - mean_loss(arm) < +0.0064 nats on stream 5150.
           History does not pay for itself. 04's prediction is confirmed, the
           question is closed, and no volatility memory should be built.

AMBIGUOUS  clears +0.0064 on 5150 but not on 5151, or vice versa. Reported as
           ambiguous, no mechanism recommended, given the parallelism cost.
```

The same bar applies to `ours+rate`.

**The bar can fail in both directions and I do not control which.** 04 measured
a large, reproducible, held-out-confirmed information gain for exactly these
features. If information were the binding constraint, +0.081 AUROC should buy
far more than 0.0064 nats.

### Validity gate, which can void the whole directory

Before any history number is interpreted:

```
mean_loss(none) - mean_loss(ours) >= 4.00% of mean_loss(none)
```

4.00% is `validation/README.md`'s own confirmed bar on fresh eval streams
(there: fresh mean 5.49%, range 5.26 to 5.79 over 5 streams). If the masked
8-input `ours` does not reproduce the published effect, my harness is broken
and nothing else here means anything. Reported as VOID, not worked around.

### Reported, with no separate bar

- realized per-arm seed sd, and whether 0.0039 was a fair prior. If the
  realized sd is much larger the test was underpowered, and I say so **without
  moving the bar**.
- per-seed paired differences and their sign pattern. Three seeds all agreeing
  in sign happens 25% of the time under the null, so sign alone decides
  nothing; it is context for the magnitude.
- accuracy alongside loss.
- mean beta by key class x write type, per arm.
- whether `ours+hist` changes the **three-regime non-monotone beta response**
  that 04's `beta_policy.py` recorded on the shipped gate:
  `0.997 0.864 0.016 0.022 0.033 0.044 0.062 0.183 0.258 0.304` by s1 decile.
- wall-clock cost per step of the bookkeeping, since parallelism is half the
  argument against the mechanism.
- clamp rate on `h_ratio`.

## My prediction, recorded now

**No gain beyond the bar**, same as 04, and for a reason 04 states better than
it knew: `gate_capacity.py` showed the gate's own architecture already reaches
the full information content of the features it has (0.8127 MLP vs 0.8095
tree), and that what is missing is the **learning signal**, not the
information. Adding more information to an input the gate is already failing to
exploit should do nothing.

The live risk to that prediction, stated so it is not a hindsight story: 04's
one prior loss datapoint (`rate_test.py`, +0.19pp worse) used a **delta-rule
rate memory with crosstalk**, and this arm is **exact**. If crosstalk really was
what killed it, `ours+hist` can clear the bar.

## Known limits of this design, stated now

- 3 seeds per arm is thin. The bar is derived to match that, not to flatter it.
- The history is exact bookkeeping, so a PASS would be an upper bound and would
  still need a crosstalk-tolerant mechanism before anything could be built.
- 800 steps is the published protocol, and `validation/convergence.log` records
  `ours` still descending at step 3000. A feature that only pays off later
  would be missed here. This measures the published operating point.
- The machine is shared; everything runs at 2 threads and no number here is a
  timing claim except the explicitly labelled per-step cost.

---

## APPENDED 2026-10-18, before any result was produced: a factual correction

**What is corrected.** Under "My prediction, recorded now" the original text
reads:

> The live risk to that prediction, stated so it is not a hindsight story: 04's
> one prior loss datapoint (`rate_test.py`, +0.19pp worse) used a **delta-rule
> rate memory with crosstalk**, and this arm is **exact**. If crosstalk really
> was what killed it, `ours+hist` can clear the bar.

**`+0.19pp worse` is wrong, and the sign is backwards.** I read
`process/03_uncertainty_probe/logs/rate.log` after writing the above:

```
        gate     loss     acc     gain
         mlp   1.5524  0.6416    4.93%
  rate_delta   1.5492  0.6379    5.12%
```

4.93% and 5.12% are **gains over the baseline, not losses**. `rate_delta`'s
loss is 1.5492 against the plain gate's 1.5524, so the delta-rule rate memory
was **0.0032 nats BETTER**, not worse. 04's README states this correctly
("+0.19pp difference") and I mis-read its sign when paraphrasing it.

**Why this matters enough to write down rather than quietly fix.** 0.0032 nats
is **exactly half my 0.0064 bar**. The single prior loss datapoint therefore
points weakly TOWARDS a gain, not away from it, which makes the bar more live
than my prediction implied, not less.

**What does NOT change.** The bar stays at +0.0064 nats, the metric stays mean
eval cross-entropy on stream 5150, the arms are unchanged, and my recorded
prediction of no gain stands. Nothing here was produced by this directory's own
runs; this is a correction to a quoted prior number, made before any of my own
results existed.

Caveat on the prior number itself: `rate_delta` is a different model
(`GatedDeltaLayer` on `mixed_task`, not the shipped `GatedDeltaSSM`), one seed,
scored on stream 12345. It is indicative, not a comparable measurement.

---

## APPENDED 2026-10-18, after 04's AUDIT.md, still before any result

**Status when this was written: no run of mine had finished.** The first pair
of training jobs was killed at roughly step 100 of 800 and every checkpoint was
discarded, so no loss number from this directory existed. Everything below is a
design change made blind, not a response to a result.

### Why

The session that briefed me audited 04 and published
`process/04_volatility_history/AUDIT.md`. Of the five arguments behind 04's
"not worth building", one is **retracted** (the "+0.134 free nonlinear
headroom" compared an already-banked gain against a prospective one), one is
**misapplied** (the gate-shaped MLP reaches the history-augmented 0.9017 too,
so credit assignment does not discriminate between the arms), and the "gate
manufactures its own signal" figure was **4x too large**, having come off a
buggy replay in `signinit_test.py` that used an input-only beta to replay a
memory-gated model.

What survives the audit and still supports my arms: history adds **+0.063 to
+0.085 AUROC** across four model-stream combinations including a held-out
stream, verified as information rather than capacity (five random features
matched to the history block's marginal shapes give **-0.0008**; a
per-sequence constant gives -0.0195); `h_meane` beats `h_cos` on every reading
(+0.034 to +0.036 against +0.012 to +0.018); the near-label oracle ceiling is
**0.9164 / 0.9206 / 0.9231** across three gates, stable enough to look like a
property of the task.

**So "no gain" is no longer a well-supported prior to confirm.** What is left
behind it is one cross-model single-seed null and the state cost. My recorded
prediction stands as my expectation and nothing else; the bar decides.

### Change 1: a fifth arm, `ours+oracle`, the CEILING

```
ours+oracle   [tok, s1, s2, h_cos, h_cosmean, h_ratio, h_meane, h_nvis, o_memb]
```

`o_memb` answers "is the incoming value one this address has shown before?". A
NOISY address redraws from a fixed set so usually yes; a REVISED address moves
to a fresh symbol so usually no. It is close to the class label **by
construction and is not a buildable mechanism.** It is here because it is the
cleanest possible decider: **if even the ceiling buys no loss, the question is
closed for every history mechanism at once**, which is stronger than any
realistic arm can establish. It is never reported as a mechanism.

### Change 2: every gated arm is now 9 wide, so all four are param-identical

`o_memb` becomes a sixth maskable column, so `ours`, `ours+hist`, `ours+rate`
and `ours+oracle` all take 9 inputs and all have **62,250 parameters exactly**.
This is strictly better than the design it replaces, where the oracle arm would
have carried 32 more parameters than its own control. `none` (62,058) and
`ours_pub` (62,058) remain mismatched by construction and are still reported as
such. The 12 runs become 18; `ours_pub` runs last and is dropped if the machine
does not allow it.

### Change 3: a second loss metric, SECONDARY and EXPLORATORY

The task has absorbing states: `succ[s] == s` with probability 1/40 per symbol,
the median sequence sends 28 of its 256 writes to one address, and **44% of all
writes are STABLE repeats where accuracy is already 1.000**. Those positions
dilute any difference in the overall figure.

So alongside the primary I report **loss restricted to positions whose QUERY
token is NOISY or REVISED**, plus a per-class breakdown. The query at position
`t` asks for the successor of `x_t`, so the governing label is `class(x_t)`,
which is a different labelling from the write class `class(x_{t-1})`; both are
computed and kept apart as `cls_q` and `cls_w`.

**Its bar.** No prior seed sd exists for this quantity, so unlike the primary I
cannot fix a number in advance. I fix the **formula** instead, now:

```
bar_hard = 2 x sd_pooled x sqrt(2/3)
```

where `sd_pooled` is the **within-arm** seed sd pooled across all arms, a
nuisance-parameter estimate computed from the spread of seeds within each arm
and never from the contrast between arms. This is the same thing any t-test
does, but it is estimated from this experiment, so:

**the restricted-loss result is EXPLORATORY in all cases** and a gain there
counts only if it reproduces on held-out stream 5151. The PRIMARY verdict of
this directory rests on the overall loss against the fixed +0.0064, unchanged.

### What does NOT change

The primary metric, the +0.0064 bar, the 3 seeds, the published protocol, the
arms `none` / `ours` / `ours+hist` / `ours+rate`, the validity gate, the eval
streams 5150 and 5151, and my recorded prediction. I am not relaxing the bar
to let a gain through: +0.0064 is 7% of the ~0.09 nats the whole memory gate
buys over `none`, so an effect worth a fraction of the headline mechanism still
registers.

---

## APPENDED 2026-10-18 14:00, third amendment, still before any result

**Status: no run of mine has finished.** Seed 0's `none` is at step 500 of 800
and `ours` at step 400. No checkpoint exists, no loss has been scored, and
`score.py` has only ever been run on throwaway 2-step models on a junk stream.

### Why: o_memb is not the ceiling, and I said it was

The briefing session corrected itself. The ~0.92 figure it gave me as "the
task's ceiling, stable across three gates" is a property of its **feature
set**, which was the same in all three measurements, which is exactly why it
looked stable. The true Bayes ceiling, derived analytically from `task.py`
with no model involved, is **0.9449** on stream 777.

So `ours+oracle`, which I added one amendment ago and called "THE CEILING",
reaches 0.92 of an available 0.945 and **understates the upside by ~0.025
AUROC**. The claim "if even the ceiling buys no loss the question is closed"
does not hold for that arm.

### Change: a sixth gated arm, `ours+bayes`, the real ceiling

```
ours+bayes   [tok, s1, s2, h_cos, h_cosmean, h_ratio, h_meane, h_nvis,
              o_memb, b_post]
```

`b_post` is the exact posterior `P(REVISED | the address's value history)`
under a flat prior, computed in `bayes.py` from the generative process alone.
It is a **sufficient statistic for the value history**, so it upper-bounds
every per-address history mechanism, all of which see a lossy function of the
same sequence. It is not buildable: it reads `N_OPT` and `P_REV` off the task
generator. This is the arm that can close the question.

Fed to the gate as `sigmoid(LLR)` rather than the raw LLR. Monotone, so the
AUROC is identical (0.9449 both ways, checked), but it lands in [0, 1]
alongside `o_memb` instead of spanning [-20, 20] and swamping a first layer
whose other columns sit between -1 and 5. No history means exactly 0.5.

**Verified against 04's `audit_bayes_ceiling.py`, which recomputes from
scratch**: AUROC 0.9449 vs its 0.9449; `P(v|NOISY) = 0` on 9.27% of
NOISY/REVISED writes vs its "9 to 10%"; and AUROC by prior writes to the
address 0.6868 / 0.7858 / 0.8896 / 0.9773 / 0.9997 at 1 / 2 / 3-4 / 5-8 / 9+
against its 0.687 / 0.786 / 0.890 / 0.977 / 0.9997.

**Param mismatch, stated now.** `ours+bayes` has 62,282 parameters against the
other gated arms' 62,250: 32 more, from its seventh input column, 0.05%. I did
NOT restart the other arms to match, because 16 runs were already queued and
the briefing session explicitly said not to restart on its account. The
mismatch is reported in the results table. `ours_pub` vs `ours` (192 unused
input weights, 3x the discrepancy) is the empirical calibration of whether a
difference this small moves anything.

**The 16 queued runs are provably unaffected.** `arms.py` is read fresh by
every new training process, so I fingerprinted all six existing arms' weight
initialisation and forward-pass output before the edit and again after:
byte-identical on both. The seventh column is computed only when
`n_hist > 6`.

### Change: loss sliced by prior writes to the query address, EXPLORATORY

Identifiability is steeply visit-count dependent (the table above), and 9 to
10% of writes are PROVABLY REVISED because more than `N_OPT = 4` distinct
values have been seen there. So a gain need not be uniform across writes, and
**a flat null is a different result from a null overall with a gain in the
identifiable slice**.

Reported at bins 0 / 1 / 2 / 3-4 / 5-8 / 9+ prior writes to the query address,
on NOISY+REVISED queries. **No bar, purely descriptive, EXPLORATORY**: these
bins were chosen to match the briefing session's AUROC table, were not
pre-specified when the bar was set, and six bins invite exactly the
multiple-comparison problem the rest of this file avoids. Anything seen here
is a hypothesis for a future run, never a verdict from this one.

### What does NOT change

The primary metric, the +0.0064 bar, the validity gate, 3 seeds, the published
protocol, eval streams 5150 and 5151, and the existing six arms. My recorded
prediction of no gain stands, and it now rests on less than it did, so a gain
is a live outcome and will be reported as one if it happens.

---

## APPENDED 2026-10-18 14:50, fourth amendment. MOTIVATED BY SEED-0 RESULTS

**This one is different from the three above and the difference is the whole
point.** Appendices 1 to 3 were written blind. This one was written after
seeing seed 0 of stream 5150, which means everything it adds is
**EXPLORATORY** and nothing in it may be treated as confirmatory.

It is admissible under this repo's rules for one reason, stated so it can be
checked: **every addition here can only make my headline claim smaller.** The
two new arms are controls for an alternative explanation of a gain I have
already seen. If they fire, the gain is explained away. If they do not, the
headline is unchanged. No bar is relaxed, no metric is swapped, and no arm
that could inflate a gain is added.

### What seed 0 showed, and the problem with it

`ours+hist` beat `ours` by +0.0611 nats, 9.5x the pre-registered bar. But the
gain by prior writes to the query address runs:

```
  0 (fresh)  +0.0707      Bayes identifiability of the class:  n/a
          1  +0.2723                                         0.687
          2  +0.1196                                         0.786
        3-4  +0.0406                                         0.890
        5-8  +0.0159                                         0.977
         9+  -0.0126                                        0.9997
```

The gain is concentrated where the address's class is **least** identifiable
from its history, is slightly **negative** where it is almost perfectly
identifiable, and is positive on **fresh** writes where by construction there
is no history at all. That is backwards for a long-term-volatility story.

A simpler explanation fits it better: `h_nvis = log1p(visit count)` is an
unblanked **per-address visit counter**, and the other four columns being
exactly 0.0 is itself a first-write indicator. "This address is new, do not
overwrite confidently yet" would produce exactly this profile. A visit
counter costs `d_k` of state, one scalar per address, not the `d_k x d_v`
residual store 04 priced the mechanism at, and it has nothing to do with
volatility.

### Two control arms

```
ours+nvis    [tok, s1, s2, 0,0,0,0,h_nvis,0]    62,250, EXACTLY param-matched
ours+fresh   [tok, s1, s2, 0 x 7, o_fresh]      62,314, +64 vs ours, reported
```

`ours+nvis` is the decisive one. If it captures most of the +0.0611, the
finding is "the gate benefits from knowing how many times it has written to
this address", which is cheaper, more parallel-friendly and a different claim
from "volatility history helps". `ours+fresh` separates "how many times" from
"have I ever".

### One correction to how gains are reported

Every gain is measured against `ours`, the masked control. At seed 0 `ours`
(1.5577) is slightly worse than `ours_pub` (1.5544), so masking 6 columns may
handicap the baseline, and if it does, **every gain in this directory is
inflated by exactly that amount**. `score.py` now reports each contrast
against both, with the inflation shown explicitly.

### What does NOT change

The primary metric, the +0.0064 bar, the validity gate, 3 seeds, the
published protocol, eval streams 5150 and 5151, and the eight arms that
preceded this. The PRIMARY verdict is still `ours+hist` and `ours+rate`
against `ours` on overall loss, on 3 seeds, confirmed on 5151.
