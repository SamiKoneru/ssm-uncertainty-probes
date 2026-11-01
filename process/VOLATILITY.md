# The volatility-history question, start to finish

One question, three directories, two independent audits, eight retractions.
This file is the whole thing as one story, because none of the three READMEs
tells it: each was written while the answer was still moving.

```
04_volatility_history   separability. Asked whether the information is there.
05_audit_volatility     an adversarial audit of 04, 18 pre-registered attacks.
05_history_loss         the loss experiment 04 said would settle it.
```

Code locations are at the bottom. **All three are now merged to main**, see
the note there for how the merge hazard was resolved.

---

## THE ANSWER

**Does long-term per-address volatility history add anything to delta-rule
write strength beyond the instantaneous residual?**

1. **The premise was wrong.** It was argued that no single-step feature could
   separate the two hard classes "in principle", because only the time
   structure distinguishes them. But `S_{t-1}` **is** a compressed history of
   every write to that address, so a gate feature computed from it is not
   single-step at all. The shipped gate's own three features separate the
   classes at **0.80 AUROC**.
2. **Information remains, bounded.** Exact per-address history adds **+0.08
   AUROC** over everything the gate can compute with zero extra state, against
   an exact Bayes ceiling of **0.945**.
3. **It does cut loss, decisively.** **+0.0551 nats**, 8.6x a pre-registered
   bar, 3 seeds, confirmed on a held-out stream. The prediction of "no gain
   beyond seed noise" was wrong.
4. **But two different effects were tangled together**, and only one is
   volatility. The genuine volatility mechanism is worth **+0.039** and needs a
   feature that reads the task generator's parameters, so it is a ceiling and
   not buildable. The larger **+0.055** that the NON-ORACLE features deliver
   comes from somewhere else entirely.
5. **And the thing the investigation was convened to fix is still broken.** The
   1.4pp accuracy gap moves only under the oracle arms. The non-oracle arm
   **loses** 0.4pp of accuracy while gaining loss.

> **"Non-oracle" is not the same as "buildable", and earlier drafts of this
> list said "buildable", which overclaimed.** The +0.055 was measured with
> EXACT per-symbol bookkeeping: one preallocated slot per vocabulary symbol,
> indexed by the literal token id, with zero crosstalk. A real layer never
> sees a token id, has no slot list, and superposes everything, so it can only
> approximate those features lossily. The +0.055 is therefore an UPPER BOUND
> on a mechanism nobody has built, not a measurement of one. See the Crosstalk
> limitation below. The asymmetry matters: this design makes the KILL verdicts
> (e.g. `ours+cheap`, `process/05_history_loss/RESULTS_cheap.md`) STRONG,
> because perfect information buying nothing implies lossy information buys
> nothing, while making every PASS weak.

**Practical answer: do not build a volatility memory.** Not because the
information is absent, and not because it fails to help loss, but because the
part that is genuinely about volatility is small and unreachable, and the part
that is large is not about volatility and is not yet understood.

---

## The task, and why anyone cared

`gated_delta_ssm/task.py`. Each symbol is assigned a class per sequence, so
token identity carries zero information about which policy a key needs:

| class | behaviour | best belief | wants |
|---|---|---|---|
| STABLE | successor never changes | the value | beta irrelevant |
| NOISY | successor redrawn from a fixed 4-option set | the **average** | **low** beta |
| REVISED | successor permanently changes and holds | the **latest** | **high** beta |

NOISY and REVISED both produce a large instantaneous residual `e_t`. A per-class
oracle keeps a **1.4pp accuracy edge** that no instantaneous-feature gate
closed, and an oracle given labels plus features reaches 8.77% loss reduction
against the feature gate's 5.67%. That gap is what the volatility memory was
meant to close.

---

## The decisive tests

| # | test | method | result | reading |
|---|---|---|---|---|
| 1 | can the gate's own features separate the classes? | tree on `[tok, s1, s2]`, by-sequence split, held-out stream | **0.8095 / 0.7983** | the "in principle" premise is false |
| 2 | does exact history add information? | exact out-of-model bookkeeping, paired bootstrap | **+0.063 to +0.085** | real, modest |
| 3 | is that capacity rather than information? | 5 random features matched to the same marginal shapes | **-0.0008** (audit: -0.0364) | information |
| 4 | what is the ceiling? | exact Bayes LLR from the generator, no model | **0.9449 / 0.9469** | two independent derivations, 4 decimals |
| 5 | can the gate's architecture use it? | the shipped `3->32->32->1` trained on labels | **0.8127** inst, **0.9017** +history | architecture is not the limit |
| 6 | **does it cut loss?** | 4 param-matched arms at 62,250, 3 seeds, fresh stream | **+0.0551, 8.6x bar** | **yes** |
| 7 | **is the gain an information effect?** | add a *sufficient statistic* for the history | **+0.0393 < +0.0698** | **no.** A strict superset gains less |
| 8 | is it a per-address visit counter? | the counter alone, and a single "ever written" bit | **-0.0009** and **+0.0032** | no |
| 9 | what did the arms learn? | mean beta, and forcing each arm to its own mean | beta -> **0.002**; forcing costs **+0.2564** | an EMA rate, not a policy |
| 10 | does it close the 1.4pp accuracy gap? | accuracy per arm, 3 seeds | **0.6413 vs 0.6455** | no, it goes backwards |

Test 6's bar was **+0.0064 nats**, derived as `2 x 0.0039 x sqrt(2/3)` from the
repo's own prior seed spread before anything ran. Realized pooled within-arm sd
came out at 0.0036, so the bar was slightly harder to pass than the data
required. Held-out confirmation: **eight arms, two streams, zero
disagreements.**

```
arm              5150 gain   5151 gain   verdict
ours+hist          +0.0551     +0.0563      PASS
ours+oracle        +0.0698     +0.0731      PASS
ours+bayes         +0.0393     +0.0426      PASS
ours+rate          +0.0060     +0.0052      KILL   misses on both
ours+nvis          -0.0009     -0.0027      KILL
ours+fresh         +0.0032     +0.0028      KILL
VALIDITY GATE        +5.42%      +5.60%      PASS
```

---

## The finding worth keeping: two effects, not one

Gain over the control by prior writes to the address, against the exact Bayes
identifiability of the class at that visit count:

```
gain vs ours      0 (fresh)        1        2      3-4      5-8       9+
ours+hist           +0.0826  +0.3131  +0.1402  +0.0587  +0.0163  -0.0294
ours+bayes          -0.0145  +0.1222  +0.0506  +0.0279  +0.0683  +0.0928
Bayes AUROC               -    0.687    0.786    0.890    0.977   0.9997
```

- **`ours+bayes` IS the volatility mechanism, behaving exactly as theorised.**
  Its gain grows with identifiability: negative on fresh writes, +0.0928 once an
  address has nine or more. Worth **+0.0393**. But `b_post` is the exact
  posterior `P(REVISED | value history)`, which needs `N_OPT` and `P_REV` read
  off the generator. A ceiling, not a mechanism.
- **`ours+hist` runs the other way and is the bigger effect.** Five sixths of
  its gain sits at 0 to 2 prior writes and the sign flips negative at 9+. It is
  not identifying the key's class.

Class-AUROC cannot tell these apart, which is why every number in `04` reported
them as one thing.

### What the big effect appears to be

Beta collapses to **~0.002 everywhere**. Since `S_t k = (1-beta) S_{t-1} k +
beta v` is an EMA with rate beta, driving beta toward zero makes the memory a
**uniform running average** over everything ever written to that address:
exactly right for NOISY, exactly wrong for REVISED. The per-class losses follow.

```
arm              acc    STABLE    NOISY   REVISED
ours          0.6455    0.4081   2.8751    2.3705
ours+hist     0.6413    0.3717   2.7094    2.3859   <- REVISED WORSE
ours+oracle   0.6484    0.3684   2.6528    2.3902
```

**The gain is in STABLE and NOISY; REVISED gets worse.** REVISED-versus-NOISY is
the one discrimination the entire hypothesis was built on.

It is not a constant policy: forcing `ours+hist` to its own mean beta costs
+0.2564, and the best constant over a 16-point grid reaches only 1.7517, because
in EMA terms moving between 0.0005 and 0.01 is a 20x swing in update rate. That
check validates itself on the input-only baseline, where forcing the mean costs
**+0.0001**.

The features are also **superadditive**: `h_meane` is worth +0.0060 as a
standalone arm but **+0.0934** to remove from the trained set, a 15x gap.

---

## What is established

All of this survived an 18-attack adversarial audit by a session whose only job
was to break it.

**The machinery.** Addressing and labelling verified three ways plus an
independent reimplementation agreeing to **1.4e-06**; split disjoint with
`|AUROC-0.5| <= 0.0040`; error bars honest (ratio **1.29** against a 2.0 bar);
linear baseline solver-independent at 0.6758 across four solvers; the model
reproduces the published loss (**1.5424** against 1.5428) and was retrained
**bit-exact** by the auditor.

**The measurements.** +0.08 AUROC of history information. The 0.945 Bayes
ceiling. The visit-count identifiability curve (0.687 at one prior write to
0.9997 at nine or more). **9.3 to 10.2% of writes are provably REVISED**,
because more than `N_OPT = 4` distinct values have been seen at that address.

**Facts about the task nobody had recorded.** It has **absorbing states**:
`succ[s] == s` with probability 1/40, so the median sequence sends **28 of its
256 writes to a single address**, the maximum observed is 254, 21% of sequences
send more than 120, and **44% of all writes are STABLE repeats** where accuracy
is already 1.000. And **NOISY is 28% repeats**, not the "~all changes" on
record.

**Corrections to the repo's record.** The 0.834 and 0.353 figures are right and
reproduce. The contradictory 0.476 came from a **replay bug**:
`signinit_test.py:88` replays a `gate="both"` model with the input-only beta,
dropping the a1 and a2 terms it was trained with. **The bug is not systemic**:
five other files with the same pattern train `gate="none"`, for which it is
correct, so `probe_test.py`, this project's headline answer, is unaffected. Also
settled: the labelling trap **destroys** rather than inverts signal (0.807 to
0.520), the token term carries nothing (0.499, so the class randomisation
works), the "amplify on large residual" policy holds in every stratum where beta
is identified, and the repo's untested `s2 -> ||Sk||` swap is worth +0.0005 to
+0.0036, i.e. harmless and not valuable.

---

## The eight retractions

| # | claim | whose | why it died |
|---|---|---|---|
| 1 | "the gate manufactures its own signal, 0.61 to 0.81" | mine | the 0.61 was the `signinit_test.py` replay bug, reproduced rather than diagnosed. Real spread +0.046 |
| 2 | "+0.134 of free nonlinear headroom, 1.7x history" | mine | the shipped gate is already nonlinear and reaches 0.8127; that gain is already banked |
| 3 | "credit assignment, not information or capacity" as an argument against history | mine | does not discriminate: the gate shape reaches the history-augmented 0.9017 too |
| 4 | "~0.92 is a property of the task" | mine | property of my FEATURE SET. Exact Bayes is 0.945 |
| 5 | "the rate memory recovered 93%" | mine | ratio of raw AUROCs where the null is 0.5. Correct arithmetic 0.723, measured 0.513 |
| 6 | "no loss gain beyond seed noise" | mine | +0.0551, 8.6x the bar, confirmed held-out |
| 7 | "the gain is a per-address visit counter" | mine | the counter alone is -0.0009; the fresh bit +0.0032 |
| 8 | "a third of rows have no beta leverage" | the audit's | measured on a randomly initialised model |

The audit's own central attack also failed and was withdrawn: its optimal-beta
proxy does not predict loss (the trained gate scores **RER -0.0359** under it
while being the best policy measured on cross-entropy), and its counterfactual
test is **underpowered 28x**.

---

## The lesson, which is not about volatility

**Every retraction of mine was an inference, not a measurement.** The
measurements survived two independent audits intact. What failed, seven times,
was reasoning from class-AUROC to a decision.

That metric:

- **mis-ranked its own features.** It put `h_meane` first (+0.036) and the
  residual-direction family last (+0.017). On loss, `h_meane` alone is worth
  +0.0060, below the bar, and the direction family carries the gain. Exactly
  reversed.
  **CONFIRMED 2026-10-31** by a dedicated arm rather than by inference.
  `ours+cheap` keeps every `d_k`-cost feature (`h_ratio`, `h_meane`,
  `h_nvis`) and drops both direction features: it scores **-0.0018** on 5150
  and **-0.0005** on fresh stream 7771, KILL on both pre-registered bars,
  against `ours+hist`'s +0.0551 / +0.0578. So the direction family is
  load-bearing, the sentence above stands, and the `d_k x d_v` cost is real.
  `process/05_history_loss/RESULTS_cheap.md`.
- **could not separate the two effects** in the table above, so it reported a
  volatility effect and a non-volatility effect as one number.
- **is not monotone in what it measures.** Adding a sufficient statistic for the
  history *reduced* the loss gain.

A proxy validated only against itself will rank confidently and wrongly. The
only thing that caught it was running the expensive experiment the proxy was
supposed to make unnecessary.

---

## What is still open

- **Why the gain inverts across visit count** between `ours+hist` and
  `ours+bayes`. The two profiles are the most interesting result here and
  neither is explained.
- **Whether the beta collapse is reachable without history at all.** A constant
  is ruled out, but a different parameterisation or schedule reaching the same
  low-beta averaging regime is not. This is the obvious next experiment.
- **Crosstalk.** All the bookkeeping is exact, one slot per symbol. A real layer
  holds it in superposition and every per-key store attempted in this repo died
  of crosstalk. **Every number here is an upper bound** and the gap is untested.
- **The 1.4pp accuracy gap**, the original target, moved by nothing buildable.
- 800 steps only; `validation/convergence.log` has the gated arm still
  descending at 3000. Nothing here is an asymptote.
- One task, one width, one sequence length, one `N_OPT`.

---

## Where the code is

All three merged to main on 2026-10-31, originally from three branches:

```
process/04_volatility_history/    separability + AUDIT.md   (claude/optimistic-easley-09c6ab)
process/05_audit_volatility/      the 18-attack audit       (claude/audit-volatility-05)
process/05_history_loss/          the loss experiment       (claude/sharp-dubinsky-67ebc6)
```

**How the merge hazard was handled.** `claude/audit-volatility-05` was cut from
`claude/optimistic-easley-09c6ab` early and carried a stale
`04_volatility_history/` from before every retraction below, so a naive merge
could have reinstated pre-retraction text. Merging `optimistic-easley` FIRST
makes the audit branch's copy equal to the merge base, so git keeps the newer
one with no conflict. Verified after merging: `04/README.md` and
`04/features.py` are byte-identical to the `optimistic-easley` versions, and
the retraction markers survive. The audit branch's only real change inside `04`
was five added log files, which came through.

`05_audit_volatility/` and `05_history_loss/` share only the `05_` numbering,
not a path, so nothing needed renaming.

Each directory carries its own pre-registration written before any measurement,
its own bars, and its logs. `04/AUDIT.md` records what the audit found against
`04` and what `04` got wrong. Eval streams are disjoint throughout: 12345 is the
repo's reused one, `04` used 777 and 888, the audit used 505, the loss
experiment used 5150 and held out 5151.
