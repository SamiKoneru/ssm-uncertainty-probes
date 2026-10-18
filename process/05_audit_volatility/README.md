# 05_audit_volatility - adversarial audit of 04's negative conclusion

**Job: find how `process/04_volatility_history/`'s "not worth building" could be
WRONG.** Not to confirm it. Pre-registered in
[`PREREGISTRATION.md`](PREREGISTRATION.md) before any measurement (commit
`72750c3`), with two sections appended later, each with its reason and its own
bars, and nothing above them edited.

**STATUS: IN PROGRESS.** This file is being written as results land. Sections
marked PENDING have no numbers yet. Nothing here is scored on my held-out
stream 606 until a verdict is committed.

---

## The claims, and where they stand

04's session published its own [`AUDIT.md`](../04_volatility_history/AUDIT.md)
while this audit was running and retracted two of the four claims I was sent to
attack. Its own closing position is now that the evidence "establishes that a
real but bounded information gap exists, and does NOT establish that closing it
is worthless."

| claim | status | by whom |
|---|---|---|
| (a) the gate's own three features separate NOISY/REVISED at pooled 0.80 | PENDING (A1, A7) | - |
| (b) nonlinearity is worth +0.134 vs history's +0.081 | **RETRACTED** | 04, and independently derived here as A8 before their message arrived |
| (c) the gate-shaped MLP reaches 0.8127, so the gap is credit assignment | **correct but non-discriminating** | 04; measured directly here (A1, PENDING) |
| (d) the rate memory carried ~93% of the available separation | **WRONG as stated** | here, A6.1 |

---

## Settled findings

### A6.1 - claim (d)'s "93%" is an arithmetic error. No new data needed.

04 compares `rate_delta`'s logged `c_t` AUROC of 0.3690 against an exact
`h_meane` AUROC of 0.3189 and reports "recovered 0.631 of an available 0.681,
about 93% of the separation on that axis".

That is `(1 - 0.3690) / (1 - 0.3189) = 0.926`, **a ratio of raw AUROCs**.
AUROC's null is 0.5, not 0, so the fraction of SIGNAL recovered is

```
  |0.3690 - 0.5| / |0.3189 - 0.5|  =  0.1310 / 0.1811  =  0.723
```

**72%, not 93%**, and that is before the model-class, gate, stream and mask
differences between the two numbers. Pre-registered bar was "<= 0.80 means (d)
as stated is wrong"; it fires. The inference 04 draws from it - "for this signal
crosstalk was not the problem" - is the thing A6.2 tests directly.

### A2 - the addressing is right, and the assert cited for it proves nothing

`features.py`'s docstring states the addressing claim and, two paragraphs later,
that "`replay` asserts its own recurrence reproduces the shipped forward pass".
**Mutation test: corrupting `addr = tokens[:, t - 1]` to `tokens[:, t]` leaves
that assert PASSING.** The bookkeeping arrays (`last_e`, `sum_e`, `cnt`,
`ndist`, `valset`, `last_val`) are pure side outputs; nothing in them reaches
`S` or the logits, so the assert cannot see them. It constrains the recurrence
and gives **zero coverage of the addressing**.

The shipped addressing is nevertheless **correct**, verified three ways that do
not rely on that assert:

```
A2.1  bookkeeping "address written before" == the task's shifted `seen`
      on every one of 8192 rows                                      PASS
A2.3  an independent per-sequence Python reimplementation of all seven
      history/oracle columns, written from task.py and model.py rather
      than from features.py: max abs diff < 1.5e-6, NaN patterns
      identical                                                      PASS
A2.4  cls_w[t] == class(x_{t-1}) under a symbol->class map rebuilt
      from the generator's own output                                 PASS
```

So 04's numbers are not off by one. This is a verification defect, not a
measurement defect. It matters because `AUDIT.md` credits this assert class with
catching the signinit bug - true for a beta/recurrence bug, which is what that
was, but an addressing or bookkeeping bug sails straight through it. 04's three
separate addressing checks are what actually cover that case.

An anomaly found on the way, **and it was my bug, not theirs**: my first
reference clamped the PRODUCT of the two norms at eps where
`torch.cosine_similarity` clamps each norm separately. That disagreed on 1129 of
3652 rows, all REPEAT writes where `||e_t||` has decayed to 1e-8..1e-5 against
an overall median of 2.5. Those rows are real, and they are why A1 exists: beta
multiplies almost nothing there, yet pooled class-AUROC weights them equally
with high-leverage writes.

### A10 - the ~0.92 "near-label oracle ceiling" is not a ceiling, and the oracle set is not near-label

04's `AUDIT.md` lists this as unchecked: "the true Bayes ceiling given the full
value multiset was not computed, so 0.92 is a measured ceiling for these
features, not a proof." It matters because 04 used ~0.92 to argue part of the
per-class oracle gap is unreachable in principle.

It needs no model. At an address `s` the values written there are exactly the
successors `s` emits, and the generative process at `s` advances only when `s`
is visited, so **that value sequence is a sufficient statistic for
`class(s)`** and the Bayes-optimal score is closed form. With `N = 40` symbols,
`K = 4` option slots, `p = 0.35`, `k` distinct observed values with counts
`c_j`, and `m_j` the unobserved option-slot multiplicities, `r = K - sum m_j`:

```
  L_REV   = (1/N) * prod_{i=2..n} [ (1-p) + p/N ]  or  [ p/N ]
            depending on whether v_i == v_{i-1}

  L_NOISY = sum over {m_j >= 1, sum m_j <= K}
              K!/(prod m_j! r!) * (N-k)^r / N^K * prod_j (m_j/K)^{c_j}

            and exactly 0 once k > K: more than K distinct values cannot come
            out of a K-slot option set
```

**where:** `n` is the number of values written at this address up to and
including the current write (the same information set `o_memb` and `o_ndist`
use); `v_i` is the `i`-th of them; `k` is how many are distinct; `c_j` is the
count of the `j`-th distinct value.

Self-tests, both passed before any AUROC was read:

```
normalisation  sum of L over every value sequence, n=1 and n=2, both classes
               1.000000000000 in all four cases                       PASS
Monte Carlo    direct simulation of the per-address process against the
               likelihoods, every pattern up to n=3, 200k draws
               max discrepancy < 0.003                                PASS
```

Two internal checks also passed: the 10.65% of rows with `k > K` are logically
certain REVISED and their **actual REVISED fraction is exactly 1.0000**; and
adding every other feature to the LLR moves it by **-0.0002**, which is what
sufficiency predicts.

Results, 04's primary row set, 22119 rows on my stream 505:

```
  EXACT BAYES LLR (sufficient statistic)        0.9479
    the same through 04's GBT and split         0.9463 +/- 0.0034
  richer per-address COUNTERS (below)           0.9303 +/- 0.0039
  04's oracle set + visit count                 0.8185 +/- 0.0070
  04's oracle set [o_memb, o_ndist]             0.7826 +/- 0.0075   (04: 0.7847)
    o_memb alone                                0.4847
    o_ndist alone                               0.4562
  excluding the logically-certain k>K rows      0.9337
```

**Two different claims of 04's, two different answers.**

1. **"`o_memb` ... is close to the class label by construction" is decisively
   false.** The pair reaches 0.7826 against the exact statistic's 0.9479, a gap
   of **0.165**, and each feature alone is at chance. 04's own 0.7847
   reproduces almost exactly, so this is not a disagreement about measurement.
2. **"~0.92 is the ceiling" gets a modest correction, not a refutation.** The
   true ceiling is 0.9479, about **+0.025** over 04's 0.9206/0.9236. My
   pre-registered bar needed `>= 0.95` to call it not-a-real-ceiling and
   `<= 0.94` to confirm it, so **this lands in my own AMBIGUOUS band. I set a
   bar this data cannot resolve, which is the same mistake 04 made at 0.80, and
   I am recording it as mis-specified rather than moving it.**

**Independently cross-validated.** 04's session wrote its own derivation
(`04/audit_bayes_ceiling.py`) from `task.py` alone and got 0.9449 on stream 777.
My code on stream 777 gives **0.9449 on 21776 rows** - exact agreement to four
decimals between two independently written implementations. The 0.003 against my
505 figure is stream-to-stream sampling, not a discrepancy: 04's own 777-vs-888
spread is 0.0020 on identical code.

**The part that bears on the decision, and I did not predict it:** the arm that
reaches 0.9303 is nothing but per-address **counters** - fraction of prior
writes whose value had been seen before at that address, fraction of consecutive
repeats, max multiplicity, `ndist`, `nvis`. That is above 04's 0.9236 "oracle +
history", and **none of it needs a `d_k x d_v` residual store**. Scalar counters
per address cost `d_k`, the same as `h_meane`, which 04 already identifies as
the cheap and useful half. So if anything on this axis is worth building it is
counters, not `h_cos`.

### A11.3 - 04's signinit bug is not systemic (confirmed by reading)

Of the files that replay `beta = sigmoid(m.to_b(xc))`, **`signinit_test.py` is
the only one whose model was trained with a memory-conditioned gate**
(`gate="both"`). `s3_test.py`, `address_test.py`, `cosine_test.py` and
`probe_test.py` all train `gate="none"`, for which that replay is correct. 04's
count of "five other files" is four-with-`gate="none"` plus files that train no
gate at all; the substantive claim holds.

**One 04 did not mention, checked here because it feeds its 0.834/0.353:**
`recurrence_test.py` trains `gate="mlp"` but is **safe** - it reconstructs the
gate explicitly (`# gate beta, same as training`) instead of using `to_b`. So
those two figures are not exposed to this bug class.

### A8 - a reasoning defect in (b), recorded independently before 04 retracted it

(b) compares `+0.134` (linear -> nonlinear read of the gate's three features)
against `+0.081` (history's increment). These are increments from DIFFERENT
baselines, and the first is not an investment the shipped architecture can make:
**the gate already IS a `3 -> 32 -> 32 -> 1` GELU MLP**, and 04's own
`gate_capacity.py` shows that exact shape reaches 0.8127 against the tree's
0.8095. The nonlinearity is already banked in the published 5.5%.

Committed as A8 in `72750c3`, before 04's `AUDIT.md` arrived; 04 retracted (b)
on the same grounds. Independent agreement, not a shared assumption.

---

## PENDING

- **A1** the optimal-beta proxy, and **A1.5** whether history identifies the
  class where beta actually has leverage
- **A1b** the counterfactual loss on the frozen model
- **A3** my `j_addrconst` capacity control
- **A4** split-seed error bars and the leakage probe
- **A5** seed stability across four retrained models
- **A6.2** the rate memory on one model and one stream
- **A7** whether "the premise bar is indecisive" was honest
- **A9** 04's linear baseline and its solver overflow warnings
- **A11.1/A11.1b** the replay-bug magnitude
- **A11.2** `gate_capacity.py`'s third row

## Reproducing

```
sh run_all.sh gated_seed0 505          # every measurement for one model/stream
python bayes_ceiling.py --stream 505   # A10, needs no model
python audit_addressing.py             # A2, needs no trained model
```

`04/checkpoints/` is gitignored and absent, so **every model here is retrained
by me** with the published protocol. Bit-exact reproduction of 04's checkpoints
is not expected (CPU reduction order) and any drift is reported.
