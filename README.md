# ssm-uncertainty-probes

Started from a simple question: *should how hard an SSM writes into a part of
its memory depend on how much that part has been changing lately?* A slot that
keeps flipping between a few values should be written softly; one that has
changed for good should be overwritten. That part works: a write gate that can
see what the memory already holds gives a replicated ~5.5% loss reduction.

Pursuing it raised a second question: *can the same memory signals predict when
the model is about to be wrong, better than output entropy?* The answer is
**no**, for a structural reason.

```
gated_delta_ssm/     THE ARCHITECTURE - clean, runnable, self-contained
process/             how it was arrived at, and three negative results
```

## The architecture

A DeltaNet layer whose write strength `beta` can see what the memory already
holds at the address being written, via two scalars the delta rule computes
anyway. **~5.5% loss reduction, +1.2pp accuracy, zero extra state, ~0.9% FLOPs**,
replicated across 3 seeds. See `gated_delta_ssm/README.md`.

It is **prior art** - Miras 2504.13173, CARVE 2606.27229, and variable-step-size
LMS from the late 1980s. Verified by reading the equations. Treat it as a clean
reference implementation, not a contribution.

## Why the uncertainty question is closed

Cross-entropy is a **proper scoring rule**, so training pays the model directly
to fold every available and useful signal into its posterior, and entropy is a
one-line readout of that. Any state-derived signal reconstructs a correlate of
the same quantity through lossier machinery. A gradient-boosted probe over the
full pre-head state plus every surprise feature reaches 0.8823 test AUROC against
0.8755 for entropy+maxprob (**+0.007**, one model and seed), so on this task the
state adds almost nothing that entropy does not already carry.

Three architectures were tried - unitary/wave, two-timescale accumulators,
delta-rule residual - and all three land in the same place.

## The three lessons that cost the most

1. **Always report a raw-position control.** Phase 01's headline result (0.733 vs
   entropy's 0.424) was a clock: raw position `t` alone scored 0.751.
2. **Always run the gate-off ablation.** Phase 02's signal was 0.577 gated and
   0.526 ungated - mostly manufactured by the loss, since the gate made it a
   learned temperature.
3. **Watch for Simpson's paradox in per-class means.** The key signal separates
   at AUROC **0.834 within change-writes** and **0.353 pooled**. Reading pooled
   means led me to build four fixes for a problem that did not exist.

A fourth, meta: **run the novelty check first.** It ran after the engineering,
and found the mechanism published.
