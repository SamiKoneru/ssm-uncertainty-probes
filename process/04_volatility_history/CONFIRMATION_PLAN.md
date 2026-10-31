# Conclusion reached on stream 777, and what stream 888 has to confirm

Written before stream 888 was touched. Stream 888 was named in
`PREREGISTRATION.md` as held out and has not been looked at.

## The conclusion

**Long-term per-address volatility history is NOT worth building for this gate.**

The information is real but it is the wrong thing to spend new state on:

1. The instantaneous features the shipped gate ALREADY sees separate the two
   hard classes at pooled AUROC **0.8095**. Exact, crosstalk-free, per-address
   history raises that to **0.8929**, a real `+0.0807 +/- 0.0029`. So there is a
   gap, and it is about a quarter of the distance from the current gate to the
   near-label oracle's 0.9206.
2. But a LINEAR read of the same three features gets only **0.6758**, and a
   spline basis over those same three features reaches **0.8112**. So
   `+0.134` of AUROC is sitting unclaimed inside the features the gate already
   has, which is **1.7x what all of history adds**, and it costs zero new state.
3. The strongest single history feature is not the one this session hypothesised.
   `h_cos` (residual direction) adds `+0.0168`; `h_meane` (mean residual norm
   per address, i.e. the rate memory) adds `+0.0361`. The rate memory was built
   in `archive/failed_features/rate_test.py` and produced 5.12% loss against the
   plain gate's 4.93%, inside seed noise.
4. Everything above is an UPPER BOUND measured with exact bookkeeping. Every
   per-key store in this repo lost its signal to superposition crosstalk.
5. Cost: `d_k x d_v` of new state for the residual-direction form, as large as
   the memory itself, written state-dependently. CARVE's chunk-entry fix does
   not rescue it, because `cos(e_t, e_prev)` needs the previous write to the
   SAME address, which can be inside the current chunk.

**Recommended instead:** spend on the nonlinear read, not on new memory. That is
the same answer `process/03` reached from the loss side ("the actual deficit was
gate CAPACITY"), now independently reached from the information side.

## Bars for the confirmation, fixed now

Scored on stream 888, same model `gated_seed0`, no retuning, nothing refitted.

    C1  pooled AUROC, shipped features [tok, s1, s2], tree    >= 0.78
    C2  delta (inst -> inst + history), pooled                >= +0.05
    C3  nonlinearity gap, pooled logit vs tree                >= +0.08
    C4  h_meane adds more than h_cos                          strict inequality
    C5  s1 alone on CHANGE writes                             in [0.75, 0.87]

C1 is set at 0.78 rather than 0.80 on purpose: the claim being confirmed is
"instantaneous already carries most of it", and 0.80 was the pre-registered
premise bar on the PRIMARY stream, which already fired. Re-using 0.80 here
would be testing the premise twice on different data and calling the second
one confirmation. C5 is a window because 777 gave 0.8073 and the recorded
value was 0.8337.

**KILL for the conclusion:** C2 fails (history adds < +0.05, so there is no gap
to argue about and the honest answer becomes a flat negative rather than a
"real but not worth it"), or C3 fails (the nonlinearity story collapses and the
recommendation changes).

## Also running, and NOT part of the confirmation

- `fit_model.py --seed 1`, a second trained model. The premise test landed at
  0.8095, only 1.6 eval-bootstrap SD above its 0.80 bar, and model-seed spread
  is a separate and larger source of variation than eval noise. Whatever seed 1
  says is reported; it cannot retroactively move the seed-0 bar.
- `legacy_0476.py`, the one reproduction that failed its bar.
