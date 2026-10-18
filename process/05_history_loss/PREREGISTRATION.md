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
