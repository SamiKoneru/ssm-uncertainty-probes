# Pre-registration: `ours+cheap`, the cheap-features-only arm

Written **before** the arm exists in `arms.py` and before any run. Nothing in
this file may be edited after the first number lands; corrections go in an
appendix with the original left in place, per the repo's standing rule.

Date: 2026-10-31. Branch: `claude/cheap-history-arm`.

---

## Why this arm exists, and why it is honestly post-hoc

05 ran the five-feature history block (`ours+hist`, +0.0551 nats) and two of
its features **individually**:

```
ours+hist    h_cos, h_cosmean, h_ratio, h_meane, h_nvis    +0.0551   PASS
ours+rate    h_meane alone                                 +0.0060   KILL  (missed the bar by 0.0004)
ours+nvis    h_nvis alone                                  -0.0009   KILL
```

It never ran them **together**. There is no arm in `ARMS` with
`keep = (2, 3, 4)`.

That gap matters because the features split cleanly on deployment cost, and
the split does not match the arm structure:

| feature | what it stores per address | state cost | chunk-freezable |
|---|---|---|---|
| `h_cos` | the previous residual **vector** | `d_k x d_v` | no |
| `h_cosmean` | the running sum of residual **vectors** | `d_k x d_v` | no |
| `h_ratio` | the previous residual **norm** (a scalar) | `d_k` | yes, degrades |
| `h_meane` | the running mean residual **norm** (a scalar) | `d_k` | yes, degrades |
| `h_nvis` | a visit **count** (a scalar) | `d_k` | yes, degrades |

`d_k x d_v` is as large as the memory `S` itself. `d_k` is one number per
address. README.md's cost section names `h_meane` and `h_nvis` as the `d_k`
ones; it omits `h_ratio`, which is also `d_k` because it reads
`p_last.norm(dim=-1)`, a scalar. So the cheap set is the full `(2, 3, 4)`.

**This arm is post-hoc.** It was specified after reading the eval-time
ablation on stream 5150, and it is *not* a control of the `ours+nvis` /
`ours+fresh` kind: those could only explain the headline gain away, whereas
this one could support a positive claim about feasibility. It is therefore
**EXPLORATORY on 5150** and requires confirmation on a stream never used
anywhere in the repo. Both results get reported.

## The question

Do the three cheap, chunk-freezable history features, **taken together**,
reproduce the gain that the full five-feature block gets?

This is the maximal cheap arm: every feature that costs only `d_k`. If it
fails, every cheap subset fails, so one arm settles the whole family.

## Metric

PRIMARY: overall mean cross-entropy over all positions, 6 batches x 32
sequences, mean over training seeds 0, 1, 2. Identical to every other arm in
05, computed by the same `score.py`.

```
gain = loss(ours) - loss(ours+cheap)
```

Positive gain means the cheap block helps. `ours`, not `none`, is the
baseline: same parameter count, same gate width, same FLOPs, only the mask
differs.

SECONDARY, exploratory, no bar: NOISY+REVISED-restricted loss, loss by visit
count, and the per-class mean beta, so the beta collapse that `ours+hist`
showed (0.003 everywhere) can be checked for here too.

## Bars, fixed now

**Bar 1, existence.** `gain >= +0.0064` nats.
The repo's existing pre-registered bar, `2 * 0.0039 * sqrt(2/3)`, unchanged
from `score.py:BAR`. Below this the cheap block is worth nothing together,
the same verdict the two standalone arms already got.

**Bar 2, sufficiency.** `gain >= +0.0276` nats, which is 50% of `ours+hist`'s
+0.0551.
Rationale, stated before seeing anything: the deployment decision is whether
you can delete a `d_k x d_v` state as large as the memory itself and keep the
bulk of the benefit. Half is the honest reading of "the bulk". The 50% is a
judgment call made now, not a number read off a result; it is deliberately
*not* set to the 67% that the ablation's drop-mass would suggest, because
that would be tuning the bar on the data the bar is meant to judge.

## What would kill the idea

`gain < +0.0064`.

Then: the cheap features are worthless alone (already measured) **and**
worthless together, superadditivity does not rescue them, the residual
**direction** features are necessary, and the `d_k x d_v` cost is
unavoidable. A deployable long-term volatility memory does not exist in this
form, and the retraction that motivated this arm would itself be wrong.

## Could this come out the other way? Yes, both ways, with evidence for each

| direction | evidence already in the repo |
|---|---|
| it fails | both cheap features failed standalone: `ours+rate` +0.0060, `ours+nvis` -0.0009 |
| it succeeds | eval-time ablation on the trained `ours+hist` puts `+0.0934` on `h_meane` and `+0.0778` on `h_nvis`, 67% of the single-column drop mass, against `+0.0544` and `+0.0299` for the two direction columns |

These point in opposite directions. That is the reason to run it. The
tension is not a flaw in either measurement: 05's README already states why
they can disagree, "the features are not independent contributions; the gate
co-adapts to the whole block." Neither number is a clean attribution, so
neither can be used to predict this arm.

## Outcome space, decided now

```
gain >= +0.0276            cheap version VIABLE. Next: implement chunk-frozen
                           h_meane / h_nvis / h_ratio and measure the
                           staleness cost against this unfrozen upper bound.

+0.0064 <= gain < +0.0276  REAL BUT MINORITY. The direction features carry
                           most of it; the d_k x d_v cost stays load-bearing
                           and the feasibility question stays open.

gain < +0.0064             KILL, as above.
```

## Protocol, identical to the existing arms

800 steps, batch 24, AdamW lr 2e-3, grad clip 1.0, gate_width 32, threads 2,
seeds 0 / 1 / 2, via the unmodified `run_arm.py`. Parameter count must come
out at 62,250, the same as `ours`, `ours+hist`, `ours+rate` and
`ours+oracle`; `check_cheap.py` asserts this and fails the run if not.

## Streams

```
5150   PRIMARY for comparability with every other arm.  EXPLORATORY here,
       because this arm was specified after reading 5150's ablation.
7771   CONFIRMATION. Verified absent from the entire repo and all three
       sibling worktrees by grep before being chosen; 5151 is already spent
       (logs/score_5151_heldout.log).
```

A pass on 5150 that does not reproduce on 7771 is not a pass.

## Blindness check, and it can fail in both directions

`check_cheap.py` asserts, on an untrained model:

1. the gate's output is **bit-identical** when `h_cos`, `h_cosmean`, `o_memb`,
   `b_post` and `o_fresh` are perturbed, so the arm provably cannot see the
   expensive features;
2. the gate's output **does change** when `h_ratio`, `h_meane` or `h_nvis` are
   perturbed, so the arm provably does see the cheap ones.

Check 2 exists because check 1 alone passes trivially for a model that reads
nothing at all, which is the failure mode that would make the whole arm
meaningless.
