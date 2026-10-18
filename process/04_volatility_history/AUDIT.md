# AUDIT of this directory, run after the conclusion was published

Asked for: is any of this biased, inaccurate, buggy, or hand-picked to work in
the scenario being tested? Answer: **the measurement machinery is sound, and two
of the five arguments I built on it do not survive. One headline figure is a bug
artifact I inherited and reproduced instead of diagnosing.**

Scripts: `audit_sanity.py`, `audit_addressing.py`, `audit_trend.py`,
`audit_s2_swap.py`, `audit_ftok.py`, `audit_signinit_bug.py`. Logs alongside.

---

## RETRACTED 1: "the gate manufactures its own signal, 0.61 -> 0.81"

I published that the legacy linear gate's features carry pooled AUROC 0.616
against the shipped gate's 0.810, and called the +0.19 "the strongest single
argument against the volatility memory".

**The 0.616 is a bug artifact.** `archive/failed_features/signinit_test.py:88`
replays the trained model as

    beta = torch.sigmoid(m.to_b(xc))[..., 0]           # INPUT-ONLY beta

but the model it trained is `gate="both"`, whose gate is `gated_model.py:107`

    beta = sigmoid( W x_t + a1*log1p(||e_t||) + a2*prj_t + b )

with a1 = 0.958 and a2 = -1.412 after training. So the `S` that extraction
accumulates is not the trained model's `S`, and the `(s1,s2)` read off it come
from a recurrence that was never trained. `signinit_test.py` has no check that
its replay matches the model's own forward pass, so nothing caught it.

`legacy_0476.py` copied that extraction deliberately, "verbatim in behaviour",
and reproduced 0.4712 against the logged 0.4763. **The reproduction was faithful
INCLUDING the bug, and I read the match as evidence that 0.476 described the
legacy model.** It does not. It describes a recurrence that does not exist.

Same model, same stream, same features, same conditioning, only the replay's
beta differing (`logs/audit_signinit_bug.log`):

```
replay                                           [s1,s2]  [tok,s1,s2]  all five
signinit_test.py's: beta = sigmoid(W x)           0.6161       0.6182    0.6581
correct: beta = sigmoid(W x + a1 s1 + a2 s2 + b)  0.7787       0.7859    0.7910
```

So the corrected comparison across the three gates measured here, on the full
zero-extra-state feature set, stream 777:

```
gate                       loss      pooled AUROC
legacy linear gate="both" 1.5882        0.7910
shipped MLP seed 0        1.5786        0.8122
shipped MLP seed 1        1.5732        0.8370
```

Still monotone in loss, so the effect is real and in the claimed direction, but
it is **+0.046 across the full range of gate quality available here, not
+0.19**. That is roughly a quarter of what I claimed and it is no longer the
strongest argument for anything. Downgraded to a small exploratory trend, n=3.

My corrected replay carries the assert `signinit_test.py` lacked, which is why
this was caught at all.

## RETRACTED 2: "+0.134 of free nonlinear headroom, 1.7x what history offers"

I compared a LINEAR read of the gate's three features (0.676) against a
nonlinear read (0.810) and called the +0.134 unclaimed headroom available for
free, against history's +0.081.

**The shipped gate is already nonlinear.** `gate_capacity.py` shows its own
`3 -> 32 -> 32 -> 1` GELU stack reaches 0.8127, so that +0.134 is already
banked in the published 5.5% and is not available to spend. Comparing a
historical, already-realised gain against a prospective one is apples to
oranges, and it was biased in favour of my conclusion.

## MISAPPLIED: "credit assignment, not information or capacity"

That part is correct as an explanation of the gap to the per-class oracle. But I
used it as an argument against history, and it does not discriminate. I had
truncated my own log when reading it; the third row says:

```
feature set                           width      MLP     tree   linear
shipped gate features [tok, s1, s2]      32   0.8127   0.8095   0.6760
all zero-extra-state                     32   0.8146   0.8122   0.7640
zero-state + exact history               32   0.9017   0.8929   0.8398
```

The gate-shaped MLP reaches the full history-augmented 0.9017 as well, beating
the tree. So the architecture can represent BOTH mappings, and "the gate cannot
learn this from the LM gradient" applies equally to both. It says nothing about
whether history is worth adding.

## What the machinery checks found: no bugs

`audit_sanity.py`:

- **The model reproduces the published result.** Never checked during the
  session, because all measurement ran on streams 777 and 888 and the published
  figure is on 12345. On 12345: seed 0 loss 1.5446 / acc 0.6395, seed 1 1.5401 /
  0.6491, mean 1.5424 / 0.6443 against the published 3-seed 1.5428 / 0.6458.
  Caveat: no input-only baseline was trained here, so the GAIN still cannot be
  computed in this directory, only the absolute loss compared.
- **The +0.08 is information, not capacity.** The session's only control was raw
  position (+0.0030). Adding five RANDOM features with the same marginal shapes
  as the history block (uniform, normal, Cauchy, a count, an exponential) gives
  **-0.0008**, and a per-sequence constant gives **-0.0195**.
- **No NaN-branch leakage.** Zero non-finite values in any history or oracle
  feature on the analysed rows, so the tree cannot branch on missingness.
- **The split is clean.** 86 train / 86 test sequences, overlap 0, rows disjoint,
  class balance 0.468 / 0.494.

`audit_addressing.py`, the highest-consequence possible bug, tested three ways
against known-correct answers rather than by re-reading the code:

- forcing `to_k` and `to_v` to identity on a one-hot embedding and reading the
  argmax of `k_t` and `v_t` off a hand-built token stream: address is the
  PREVIOUS token and value is the CURRENT token, confirmed against the model.
- the vectorised bookkeeping against a deliberately naive single-sequence
  reimplementation written from the spec, on five hand-built cases (all repeats,
  strict alternation, revise-once-then-hold, cycle through a 3-set, never
  repeats an address): identical on `wtype` and `o_memb`.
- `cls_w[t] == class(x_{t-1})` and `cls_q[t] == class(x_t)`, rebuilt from the
  task generator's own per-sequence assignment: 0 mismatches in 1020 writes.

The bookkeeping was then refactored into a shared `features.Bookkeeper` so the
legacy model reuses the identical code instead of a copy, and re-verified
against the same tests.

## The trend claim: attacked with a stated prediction, and it held

The published trend table had three models but only two points in the column
that decides it, because the legacy model's history delta was never measured.
"Every column is monotone" was not a fair description of that. Measured, with
the prediction written down first (`audit_trend.py` prints it before the
numbers): the legacy model is the least separable so it has the most room, so
the claim predicts its delta is the LARGEST.

```
model                         inst   +hist    delta      sd   +oracle
legacy linear gate="both"   0.7910  0.8754  +0.0845  0.0038    0.9164
shipped MLP seed 0          0.8122  0.8929  +0.0807  0.0029    0.9206
shipped MLP seed 1          0.8370  0.9004  +0.0634  0.0030    0.9231
```

Predicted ordering legacy > seed 0 > seed 1, measured +0.0845 > +0.0807 >
+0.0634. **Survives**, n=3, still exploratory, and the spread is small.

Worth more than the trend: **the near-label oracle ceiling is 0.9164 / 0.9206 /
0.9231 across three different gates.** That stability makes ~0.92 a property of
the TASK rather than of any model, which is the one firm bound here.

## Answered in passing: the repo's untested s2 replacement

`model.py:144-146` and `process/03/README.md:217` recommend, untested, replacing
the algebraically redundant `s2` with `||S_{t-1}k_t||`. Measured
(`logs/audit_s2_swap.log`), pooled, stream 777:

```
feature set                             legacy    seed 0    seed 1
[tok, s1, s2]        the shipped one    0.7859    0.8095    0.8320
[tok, s1, |Sk|]      recommended        0.7864    0.8131    0.8351
all five                                0.7910    0.8122    0.8370
```

Worth +0.0005 to +0.0036, inside one bootstrap SD on every model. **The swap is
harmless and not valuable on separability.** It may still be worth making for
the `d_v = 1` degeneracy the algebra note raises, which this does not test.

Also settled: `f_tok` alone is 0.4965 to 0.5133 and adds +0.002 to +0.007 in
company, so the task's per-sequence class randomisation holds and there is no
interaction hiding in the token term. The apparent +0.17 that prompted the check
was the 0.616 bug, not `f_tok`.

## Where the conclusion stands after the audit

Unchanged and verified:

- history adds **+0.063 to +0.085** across four model-stream combinations,
  confirmed on a held-out stream, with a random-feature control at -0.0008
- `h_meane` beats `h_cos` on every reading, so the rate memory is the useful
  half and the residual-direction hypothesis is the weak half
- near-label oracle ceiling ~0.92, stable across three gates
- the anomalies: absorbing states, NOISY 28% repeats, my own Simpson's paradox
  over STABLE repeats, the labelling trap destroying rather than inverting the
  signal, `s3` null, the three-regime beta response

**Weakened:** the recommendation. Of the five reasons published, one is
retracted, one is misapplied, and one is 4x smaller than claimed. What is left
is the prior rate-memory null (a cross-model comparison, one seed) and the
`d_k x d_v` state cost. Both are real, neither is a measurement of loss on this
question.

**So the honest status is: the separability evidence establishes that a real but
bounded information gap exists, and does NOT establish that closing it is
worthless.** "Not worth building" is a judgement about cost against a prior null
result, and I presented it with more support than it has. The loss experiment is
the decider and it was correctly identified as the gap.

## RETRACTED 3: "~0.92 is a property of the task"

This section originally listed the Bayes ceiling as unchecked and said 0.92 was
"a measured ceiling for these features, not a proof". The session auditing this
directory computed it, and `audit_bayes_ceiling.py` then reproduced it here from
an independent analytic derivation off `task.py` alone, with a normalisation
self-check that is exact:

```
                                               stream 777   stream 888
EXACT Bayes AUROC given the value history          0.9449       0.9469
  (the other session reports 0.9479 by its own route)
measured in 04 with the 'near-label oracle'        0.9206       0.9154
measured in 04 with history features               0.8929       0.8830
all five zero-extra-state features                 0.8122       0.8015
```

**So 0.92 is a property of MY FEATURE SET, not of the task.** The true ceiling
for any per-address history mechanism is ~0.945, and I called the 0.92 "the one
firm bound here" and "a property of the TASK" in both this file and the README.
Both wrong. The oracle features are not near-label either: on their own they
reach 0.785, which the other session measures as a 0.165 gap to Bayes.

Consequence, and it runs AGAINST this directory's recommendation: the headroom
available to history is **+0.133** (0.812 to 0.945), not the +0.109 I reported,
and my history feature set captured 0.081 of it, about 61%. The upside I
measured was roughly a fifth too small.

What the exact ceiling also reveals, which nothing here had noticed:
identifiability is steeply dependent on how many times the address has been
written.

```
prior visits        n    Bayes AUROC
           1     3179         0.6868
           2     2741         0.7858
         3-4     4392         0.8896
         5-8     5526         0.9773
         9+      5938         0.9997
```

Early writes are genuinely ambiguous even to Bayes; once an address has 5 or
more prior writes the class is nearly determined. And **9.3 to 10.2% of writes
have `P(v|NOISY) = 0` exactly**, because more than `N_OPT = 4` distinct values
have been seen there, so those are provably REVISED. Verified: every one of them
is in fact REVISED.

That gives the per-write beta question a shape nobody here had stated: a
volatility memory would be useless on an address's first couple of writes and
near-perfect after five, so its value depends entirely on how the task's writes
are distributed over visit counts, which this directory never reported either.

## What this audit did not check
- the "rate memory recovered 93%" comparison still crosses models and streams.
  Settling it needs the `rate_delta` model retrained and measured alongside.
- anything about loss.
