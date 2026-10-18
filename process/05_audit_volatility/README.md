# 05_audit_volatility - adversarial audit of 04's negative conclusion

**Job: find how `process/04_volatility_history/`'s "not worth building" could be
WRONG.** Not to confirm it.

Pre-registered in [`PREREGISTRATION.md`](PREREGISTRATION.md) before any
measurement (`72750c3`), with two sections appended later, each carrying its
reason and its own bars, and nothing above them edited.

**VERDICT IN ONE PARAGRAPH.** 04's measurement machinery is sound and I could
not break it. Its addressing is right, its split is clean, its linear baseline
survives an overflow scare, and every figure of its I recomputed reproduced,
several to four decimals. **Claim (d) is wrong as stated** - "93% recovered" is
a ratio of raw AUROCs, the correct figure is 72%, and measured properly on one
model it is 51%. **Claim (a)'s ~0.92 ceiling is not a ceiling**: the exact Bayes
bound is 0.9466 and 04's "near-label oracle" features reach only 0.783, so they
are nowhere near the label. Claims (b) and (c) were retracted by 04's own
session mid-audit, (b) on grounds I had recorded independently before their
message arrived. **But my own central attack failed.** The optimal-beta proxy I
built to replace class-AUROC is demonstrably misaligned with cross-entropy, and
the counterfactual loss test I built to settle the question is underpowered by
about 28x. On the one pre-registered bar of mine that remained live and
measurable, history delivers 20% of what a perfect class detector is worth, and
that **fires in 04's favour**.

---

## Scoreboard

| # | attack | pre-registered bar | outcome |
|---|---|---|---|
| A1 | does history predict the OPTIMAL BETA better than the class? | `share_hist >= 0.75` / `<= 0.40` | **0.197, FIRES FOR 04** |
| A1 | `R_beta = gain_hist / gain_nonlin` | `>= 1.00` collapse / `<= 0.75` survive | **VOID** - denominator is zero within noise, and its claimant was retracted |
| A1b | counterfactual loss | `g_hist >= 0.0039` / `<= 0.0013` nats | **neither fires, and the test is underpowered by 28x** |
| A2 | addressing + the replay assert | pass/fail | addressing **PASSES** 3 ways; the assert has **zero coverage** of it |
| A3 | is the +0.08 capacity? | `gain_junk >= 0.02` / `<= 0.01` | **-0.0364, capacity REJECTED, fires for 04** |
| A4.1 | are the error bars honest? | `ratio_sd >= 2.0` / `<= 1.5` | **1.29, HONEST, fires for 04** |
| A4.2 | does the split leak? | `\|AUROC-0.5\| <= 0.02` | **<= 0.0040, no leakage, fires for 04** |
| A6.1 | (d)'s "93%" arithmetic | `<= 0.80` means wrong | **0.723, (d) IS WRONG** |
| A6.2 | (d) on one model and stream | `>= 0.80` survive / `<= 0.70` fail | **0.513, (d) FAILS** |
| A7 | was "indecisive" honest? | pooled 95% CI contains 0.80 | **HONEST, fires for 04** (see the pooling trap) |
| A8 | (b)'s framing | argument, no bar | **defect confirmed; 04 retracted it independently** |
| A9 | 04's linear baseline vs its overflow warnings | `>= +0.01` inflated / within `0.01` sound | **SOUND, fires for 04** |
| A10 | is ~0.92 a real ceiling? | `>= 0.95` not-a-ceiling / `<= 0.94` confirmed | **0.9466: AMBIGUOUS by my bar, mis-specified** |
| A10 | is the oracle set near-label? | (part of A10) | **decisively NOT: 0.783 vs 0.947** |
| A11.1 | 04's signinit magnitudes | within 0.03 | **NOT REPRODUCED** (no legacy model trained) |
| A11.1b | the same mechanism on my model | sign and order of magnitude | **CONFIRMED** |
| A11.3 | is the signinit bug systemic? | pass/fail per file | **not systemic, CONFIRMED** |

Every model here was retrained by me (`04/checkpoints/` is gitignored and
absent). **My seed-0 retrain is bit-exact to 04's** at every logged step
(3.8216, 1.7723, 1.5075, 1.7300, 1.6201, 1.6970, 1.7882, 1.5480, 1.6039). My
pre-registration said bit-exact reproduction was not expected and that drift
would be reported; there is none, so the retraining confound is gone and I am
measuring the same model 04 used.

---

## Where my own attack failed, in detail

This is the most important section for anyone deciding what to do next, so it
comes before the findings.

### The optimal-beta proxy is misaligned with cross-entropy

`beta` is a pure mixing coefficient, `S_t k_t = (1-beta) stored + beta v_t`, so
a per-write optimal beta has a closed form against `v_next`, the value the next
write to that address will store:

```
  err_t(beta) = ||beta u_t - w_t||^2 = A_t beta^2 - 2 B_t beta + C_t
  u_t = v_t - stored_t = -e_t     A_t = ||e_t||^2    leverage
  w_t = v_next - stored_t         B_t = <u_t, w_t>   beta*_t = clip(B_t/A_t,0,1)
```

The algebra is verified against a direct simulation of the write (max relative
error 2e-10, argmin within one grid step of 5e-4). Minimising mean `err` is
exactly weighted least squares against `B_t/A_t` with weight `A_t`.

**It does not predict loss.** On 04's primary row set the trained gate's own
beta scores **RER -0.0359**, i.e. WORSE than the best constant beta under my
objective, while on actual cross-entropy that same gate is the best policy
measured, at 1.4648 against the CE-optimal constant's 1.5693. The two objectives
also disagree about the optimal constant: beta-MSE says 0.3659, cross-entropy
says 0.3123.

So the `+0.0440` "credit-assignment gap" this metric reports for claim (c) is an
artifact of the metric, not a measurement of the gate's shortfall. **Claim (c)
cannot be measured this way, and my attempt to is withdrawn.**

Why: the proxy assumes content written at `k = normalize(W_k emb(s))` is read
back at `q = normalize(W_q emb(s))`. Measured alignment is `cos(k_s, q_s) =
0.856` same-symbol against `-0.062` cross-symbol - a good margin, but the gap
plus LayerNorm, the head, and crosstalk from other addresses are enough to
break the ranking.

One thing the proxy does get right: `beta*` has the **largest structural edge**
on both CE (+0.2101) and accuracy (+0.0276) once the confound below is removed.
The target points somewhere real; the scale does not survive.

### A1b's raw result was a mean-beta artifact, and the test is underpowered 28x

Raw A1b, frozen model, beta substituted online, held-out sequences:

```
  beta policy                              CE     acc   mean b   CE gain vs const
  the gate's own beta (shipped)        1.4648  0.6596   0.3123        +0.1447
  best constant beta                   1.6095  0.6673   0.5325        +0.0000
  3 feats, linear                      2.1067  0.6036   0.6683        -0.4973
  3 feats, GBT                         1.5927  0.6596   0.7811        +0.0168
  5 zero-state, GBT                    1.6176  0.6690   0.8047        -0.0082
  5 zero-state + history, GBT          1.5418  0.6690   0.5259        +0.0676
  TRUE CLASS LABEL (oracle)            1.4806  0.6754   0.7103        +0.1289
  per-write beta* (non-causal ceiling) 1.4695  0.7009   0.7130        +0.1400
```

Control passes exactly: replaying with the gate's own beta reproduces the
shipped forward pass to `0.000e+00` on CE, on all ten feature columns, and on
beta.

Read naively this says history is worth **+0.0758 nats**, 19x the repo's seed
sd. **That reading is wrong.** A constant-beta sweep on the same frozen model
shows CE is minimised at `beta = 0.3123`, exactly the gate's mean, and that any
policy whose mean beta sits far from there pays an off-policy penalty:

```
  constant beta   0.05    0.10    0.20   0.3123    0.40   0.5259    0.60
  CE            1.6573  1.6124  1.5751  1.5693  1.5781  1.6074  1.6331
  acc           0.6252  0.6300  0.6414  0.6500  0.6574  0.6665  0.6719
```

Comparing each policy against a constant **at its own mean beta**:

```
  policy                        mean b   CE edge   acc edge
  per-write beta* (non-causal)  0.7130   +0.2101    +0.0276
  TRUE CLASS LABEL (oracle)     0.7103   +0.1990    +0.0021
  3 feats, GBT                  0.7811   +0.1235    -0.0137
  the gate's own beta           0.3123   +0.1045    +0.0096
  5 zero-state, GBT             0.8047   +0.0986    -0.0043
  5 zero-state + history, GBT   0.5259   +0.0656    +0.0025
  best constant beta            0.5325   -0.0021    +0.0008   <- calibration check
  3 feats, linear               0.6683   -0.4327    -0.0697
```

**Adding history LOWERS the structural edge, from +0.0986 to +0.0656**, a change
of **-0.0330** on CE, while on accuracy it raises it by +0.0068. Inconsistent in
sign, and the whole +0.0758 of the naive reading is explained by the history
policy happening to land at mean beta 0.5259 instead of 0.8047.

The off-policy penalty spans **0.1103 nats** across the mean-beta range in play.
The bars were set against the repo's seed sd of 0.0026 to 0.0039. **The
confound is 28x the effect size the test was built to resolve**, so neither bar
firing is uninformative rather than evidence either way. I declared the
off-policy limitation before running and underestimated its magnitude by more
than an order of magnitude.

**Only actual retraining can settle this.** A sibling session
(`claude/sharp-dubinsky-67ebc6`, `process/05_history_loss/`) is running it.

### A1.5 answers 04's question: leverage and identifiability are NOT misaligned

04's session raised the right follow-up: history is near-useless on an
address's first writes and near-perfect after five, so its value depends on how
the writes where **beta has leverage** distribute over visit counts. `A_t =
||e_t||^2` is that leverage, so the two cross directly.

```
   prior    rows  share of     mean   Bayes  hist gain  hist gain
  visits          leverage    ||e||   AUROC   beta RER  x leverage
       1    1433    0.1575   7.8555  0.6875   -0.00149    -0.00023
       2    1223    0.1293   7.7006  0.7804   -0.00027    -0.00003
     3-4    1966    0.2042   7.6374  0.8923   +0.01483    +0.00303
     5-8    2478    0.2521   7.5947  0.9792   +0.01614    +0.00407
      9+    2657    0.2569   7.4603  0.9991   +0.01754    +0.00451
```

The Bayes column reproduces 04's independent figures to within 0.005 on all
five buckets (04: 0.6868 / 0.7858 / 0.8896 / 0.9773 / 0.9997). The
leverage-weighted column sums to 0.01135 against the separately computed
`gain_hist` of 0.0116, which is the internal consistency check.

**Answer: no misalignment.** Leverage is spread almost evenly across visit
counts (0.158 to 0.257), the 5+ buckets carry **50.9%** of it, and **75.6%** of
history's beta value comes from those buckets. History is worthless at one or
two prior writes, as 04 found, but that is only a sixth of the leverage each.
By write type, change writes carry 67.2% of leverage (mean `||e||` 8.59) and
repeats 32.8% (6.49).

### And no feature set predicts the optimal beta well

```
  policy                           RER       CER    % of achievable
  best constant beta           +0.0000   +0.0000
  3 feats, linear              +0.0064   -0.0130      1.9%
  3 feats, GBT                 +0.0082   -0.0415      2.4%
  5 zero-state, GBT            +0.0096   -0.0423      2.9%
  5 zero-state + history, GBT  +0.0212   -0.0044      6.3%
  TRUE CLASS LABEL (oracle)    +0.0684   +0.1598     20.3%
  per-write beta* (cheating)   +0.3361   +1.0000    100.0%
  the trained gate's own beta  -0.0359   -0.3813         -
```

`share_hist = 0.0116 / (0.0684 - 0.0096) = 0.197`, so **history delivers 20% of
what a perfect class detector is worth on the beta target. The pre-registered
bar was `<= 0.40` for "it does not deliver most of it", and it fires in 04's
favour.**

`R_beta` came out at `+6.46` with a bootstrap sd of **121.7** and a 95% interval
of `[-52.1, +49.7]`, because its denominator `gain_nonlin = +0.0018 +/- 0.0022`
is zero within noise. **The point estimate is meaningless and must not be quoted
as "history is 6x nonlinearity".** The bar is void, not fired. Its claimant was
retracted anyway.

---

## Findings that stand

### A6.1 + A6.2 - claim (d) is wrong, but 04's inference about crosstalk is right

04 reports the rate memory "recovered 0.631 of an available 0.681, about 93%".
That is `(1-0.3690)/(1-0.3189)`, **a ratio of raw AUROCs**, and AUROC's null is
0.5, not 0:

```
  |0.3690 - 0.5| / |0.3189 - 0.5| = 0.1310 / 0.1811 = 0.723
```

Measured properly, all three arms on ONE model, ONE stream, ONE mask, ONE
labelling, with the rate memory run as a read-only side channel on the shipped
model so every arm sees the identical `e_t` sequence:

```
  arm                                  AUROC  flipped  |above chance|
  h_meane   exact per-address MEAN    0.3083   0.6917         0.1917
  h_ema     exact per-address EMA     0.3876   0.6124         0.1124
  c_t       superposed EMA (as built) 0.4016   0.5984         0.0984
```

**Recovered: 0.513, not 0.93.** The pre-registered bar (`<= 0.70` fails) fires
on both masks (0.513 correct mask, 0.578 with `rate_test.py`'s own mask).

**The decomposition is the useful part:**

```
  exact MEAN -> exact EMA   (the EMA form)   0.586 recovered   the EMA costs 41%
  exact EMA  -> superposed  (CROSSTALK)      0.875 recovered   crosstalk costs 12%
```

So **04's inference is right and its magnitude is wrong.** Crosstalk really was
not the problem for this signal, which is what 04 concluded from the 93%. But
the built channel carried only half the available separation, and what ate it
was the exponentially-weighted form (`bc ~ 0.198`), not superposition.
**Actionable: build a true running mean (per-address count and sum), not an
EMA.** That is still only `d_k` of state.

### A10 - ~0.92 is not a ceiling, and the "near-label oracle" is not near-label

The per-address value sequence is a **sufficient statistic** for the class (the
process at an address advances only when that address is visited), so the Bayes
bound is closed form, with no model and no classifier. With `N=40`, `K=4`,
`p=0.35`, `k` distinct values with counts `c_j`, `r = K - sum m_j`:

```
  L_REV   = (1/N) prod_{i=2..n} [(1-p) + p/N]  or  [p/N]      per v_i == v_{i-1}
  L_NOISY = sum_{m_j>=1, sum m_j<=K} K!/(prod m_j! r!) (N-k)^r / N^K
                                     * prod_j (m_j/K)^{c_j}
            = 0 exactly once k > K
```

Self-tests, both passed before any AUROC was read: normalisation sums to
`1.000000000000` for `n=1,2` under both classes; a direct Monte Carlo of the
per-address process matches the likelihoods to `< 0.003` on every pattern up to
`n=3`. Two internal checks also passed: the 10.65% of rows with `k > K` are
logically certain REVISED and their **actual REVISED fraction is exactly
1.0000**; and adding every other feature to the LLR moves it by **-0.0002**,
which is what sufficiency predicts.

```
  EXACT BAYES LLR (sufficient statistic)      0.9479   (505)
    same, through 04's GBT and split          0.9463 +/- 0.0034
  richer per-address COUNTERS                 0.9303 +/- 0.0039
  04's oracle set + visit count               0.8185 +/- 0.0070
  04's oracle set [o_memb, o_ndist]           0.7826 +/- 0.0075   (04: 0.7847)
    o_memb alone                              0.4847
    o_ndist alone                             0.4562
  excluding the logically-certain k>K rows     0.9337
```

Stable across four streams: **0.9479 (505), 0.9449 (777), 0.9469 (888), 0.9469
(606)** - mean **0.9466, sd 0.0013**, and **+0.0231** over 04's 0.9236.

**Cross-validated exactly.** 04's session independently derived the same closed
form (`04/audit_bayes_ceiling.py`) and got 0.9449 on 777 and 0.9469 on 888. My
code on those streams gives **0.9449 and 0.9469** - agreement to four decimals
between two separately written implementations.

1. **"`o_memb` ... is close to the class label by construction" is decisively
   false.** The pair is 0.7826 against the exact statistic's 0.9479, a gap of
   **0.165**, and each feature alone is at chance. 04's own 0.7847 reproduces,
   so this is not a measurement dispute.
2. **"~0.92 is the ceiling" gets a correction of +0.023, not a refutation.** My
   bar needed `>= 0.95`, so this lands in my own **AMBIGUOUS** band. I set a bar
   this data cannot resolve, which is the same mistake 04 made at 0.80, and I am
   recording it as mis-specified rather than moving it.

**The part that bears on the decision, which I did not predict:** the arm
reaching 0.9303 is nothing but per-address **counters** - fraction of prior
writes whose value was already seen there, fraction of consecutive repeats, max
multiplicity, `ndist`, `nvis`. That beats 04's 0.9236 "oracle + history" and
**none of it needs a `d_k x d_v` residual store**. Counters cost `d_k`, like
`h_meane`. **If anything on this axis is worth building it is counters, not
`h_cos`.**

### A3 and A4 - capacity, error bars and leakage all survive

My capacity control was sharper than 04's single position feature: five junk
features, two of them sharing the history block's GROUPED structure.

```
  comparison                                        base    with    delta      sd
  5 zero-state -> + 5 HISTORY (04's claim)        0.8105  0.8940  +0.0835  0.0028
  5 zero-state -> + 5 JUNK (capacity control)     0.8105  0.7740  -0.0364  0.0069
  5 zero-state -> + position only (04's control)  0.8105  0.8154  +0.0050  0.0017
  5 zero-state -> + j_addrconst only              0.8105  0.7730  -0.0375  0.0058
  5 zero-state -> + j_seqconst only               0.8105  0.7934  -0.0170  0.0037
  + 5 JUNK -> + 5 JUNK + 5 HISTORY                0.7740  0.8812  +0.1071  0.0057
```

`j_addrconst` is one N(0,1) draw per (sequence, address): **exactly the
per-address grouped structure a volatility memory would have, carrying zero
class information.** It costs -0.0375, and the five junk features together cost
-0.0364, so **capacity is decisively REJECTED** - history's +0.0835 is
information. The last row is the strongest form: history still adds **+0.1071**
on top of junk that has already consumed tree capacity. Every junk feature's
single-feature AUROC is at chance (0.4947 to 0.5477).

`A4.1`: the delta recomputed across 6 independent by-sequence splits has mean
**+0.0836** (min 0.0766, max 0.0885) with an across-split sd of 0.0044 against
a mean reported bootstrap sd of 0.0034. **`ratio_sd = 1.29`, under my 1.5
threshold, so 04's error bars are HONEST** - they slightly understate, as a
test-only bootstrap must, but not materially.

`A4.2`: with the label replaced by a class drawn independently per (sequence,
address), all three feature sets sit at chance (`|AUROC-0.5|` of 0.0040, 0.0034,
0.0040) while the real label through the identical code path gives 0.8940. **No
leakage.**

### A7 - "indecisive" was honest, and pooling is a trap I nearly fell into

My retrained seed-0 model reproduces 04's two figures **exactly**: 777 = 0.8095,
888 = 0.7983, gap +0.0112, all to four decimals. Bit-exact model, bit-exact
measurement.

```
    stream     AUROC   boot sd            95% CI    n rows   vs 0.80
       777    0.8095    0.0058   [0.7980, 0.8205]     21776 straddles
       888    0.7983    0.0067   [0.7848, 0.8108]     21731 straddles
       505    0.8059    0.0064   [0.7931, 0.8180]     22119 straddles
       606    0.8093    0.0056   [0.7983, 0.8204]     21856 straddles

  pooled 777+888   0.8079 +/- 0.0043   [0.7990, 0.8161]   contains 0.80
  pooled all four  0.8135 +/- 0.0029   [0.8078, 0.8194]   EXCLUDES 0.80
```

The second pooled row looks like it convicts 04, and I nearly reported that.
**It does not, and the reason is that pooling changed the estimand.** The
pooled-four AUROC of 0.8135 is HIGHER than every individual stream (0.8095,
0.7983, 0.8059, 0.8093), which pooling cannot do if it were only adding
precision. `measure()` fits the tree on half the POOLED sequences, so pooling
four streams hands the classifier about 4x the training data and it gets better.
0.8135 answers "what does a tree trained on 43k rows reach", not "what is a
sharper estimate of the 10k-row quantity 04 set a bar on".

**At equal training-set size the bar is genuinely straddled**: four streams span
0.7983 to 0.8095 and every single-stream CI contains 0.80. **So 04's
"indecisive" was HONEST**, and its decision not to lean on the premise bar was
correct. `A7b`: the 777-vs-888 gap is 1.9 single-stream bootstrap SDs, so it is
a marginally real difference between streams rather than pure noise, and across
10 split seeds on 777 alone the figure moves 0.8009 to 0.8132.

### A2.5 - the labelling trap, reproduced exactly

```
  variant                                             AUROC        n
  as shipped: label class(x_{t-1}), mask sn_addr     0.8056    11913
  label class(x_t) instead (the known trap)          0.5201     8039
  mask sn_task instead (describes the VALUE token)   0.8480     9653
  pooled over all seen writes, as shipped            0.3521    22119
  pooled, label class(x_t)                           0.4881    22023
```

04 reported the trap as 0.8073 -> 0.5201 on 777. **My 0.5201 matches to four
decimals**, and the pooled 0.3521 matches its 0.353. The trap destroys the
signal rather than inverting it, as 04 said.

### A2 - the addressing is right, and the assert cited for it proves nothing

**Mutation test: corrupting `addr = tokens[:, t-1]` to `tokens[:, t]` leaves
`features.py`'s replay assert PASSING.** The bookkeeping arrays are pure side
outputs; nothing in them reaches `S` or the logits, so the assert constrains the
recurrence and gives **zero coverage of the addressing**.

The addressing is nevertheless correct, three ways that do not use that assert:
`A2.1` bookkeeping "written before" equals the task's shifted `seen` on all 8192
rows; `A2.3` an independent per-sequence reimplementation of all seven
history/oracle columns agrees to `< 1.5e-6` with identical NaN patterns; `A2.4`
`cls_w[t]` matches a symbol-to-class map rebuilt from the generator.

This matters because `AUDIT.md` credits this assert class with catching the
signinit bug - true for a beta/recurrence bug, which is what that was, but an
addressing or bookkeeping bug sails straight through. 04's three separate
addressing checks are what actually cover that case, and they should stay.

### A9 - 04's linear baseline survives its overflow warnings

`04/logs/separability_777.log` PART 5 interleaves `divide by zero`, `overflow`
and `invalid value encountered in matmul` into the rows that produce
"linear 0.676". Checked on the same model and rows:

```
  census: f_tok, f_s1, f_s2, f_nstored, f_nv - 0 inf, 0 nan, max|.| = 10.79
  non-finite reaching the solver after _clean: train 0, test 0
  04's _clean path          0.6695     rank/quantile  0.6584
  _clean + forced finite    0.6695     <- identical, so the warnings change nothing
  all zero-state            0.7636     (04: 0.7640)
```

The warnings are benign iterates inside the solver, not corrupted inputs, and
they **reproduce inside my own independently written linear fit**, so they are a
property of this feature block meeting a linear model rather than anything in
`auroc.py::_clean`. Bar: within 0.01 of 0.676, so **SOUND**. Worth fixing the
noise, not the number.

### A11 - 04's own audit, audited

- **A11.3 the signinit bug is not systemic: CONFIRMED.** Of the files replaying
  `beta = sigmoid(m.to_b(xc))`, `signinit_test.py` is the only one whose model
  was trained with a memory-conditioned gate (`gate="both"`). `s3_test`,
  `address_test`, `cosine_test` and `probe_test` all train `gate="none"`, where
  that replay is correct.
- **One 04 did not mention**, checked because it feeds its 0.834/0.353:
  `recurrence_test.py` trains `gate="mlp"` but is **safe** - it reconstructs the
  gate explicitly (`# gate beta, same as training`) instead of using `to_b`. So
  those two figures are not exposed to this bug class.
- **A11.1b the mechanism: CONFIRMED.** Dropping the gate MLP from the replay of
  my own model moves the trajectory (CE 1.6028 -> 1.7213) and lowers pooled
  separability by -0.027 to -0.038, the same sign as 04's -0.163.
- **A11.1 proper: NOT REPRODUCED.** The exact magnitudes (0.6161 vs 0.7787)
  need an independently trained legacy model, which is another 800-step fit, and
  the machine was carrying two sibling sessions' jobs throughout (load 50 to
  127 on 10 cores). Declared, not quietly dropped.

### A8 - (b)'s framing, recorded before 04 retracted it

(b) compares `+0.134` (linear -> nonlinear read of three features) against
`+0.081` (history's increment). Different baselines, and the first is not an
investment the shipped architecture can make: **the gate already IS a
`3 -> 32 -> 32 -> 1` GELU MLP**, and 04's own `gate_capacity.py` shows that shape
reaches 0.8127 against the tree's 0.8095. Committed as A8 in `72750c3`, before
`AUDIT.md` arrived; 04 retracted (b) on the same grounds. Independent agreement,
not a shared assumption.

---

## What I got wrong

- **Retracted: "a third of the pooled row set has essentially no beta
  leverage, `||e_t||` of 1e-8 to 1e-5 against a median of 2.5."** That was
  measured on a **randomly initialised** model, because `audit_addressing.py`
  deliberately uses one (its bookkeeping is weight-independent), and I carried
  the number across without rechecking it on a trained gate. On the real
  `gated_seed0`: `||e_t||` median **7.2434**, mean 7.6183, **zero** rows below
  1.0; top 10% of rows by leverage carry 0.1534 of it and top 50% carry 0.6409.
  Leverage is only mildly concentrated. Mechanism, after the fact: the trained
  gate writes `beta ~ 0.04` on NOISY and REVISED repeats, so `e` decays like
  `0.96^n` and never reaches the delta rule's zero, while an untrained gate
  drives repeats straight there. **This weakened A1's motivating premise and I
  recorded it before reporting A1's numbers.**
- **My A10 bar at 0.95 was mis-specified**, landing the measurement in its own
  ambiguous band - the same error 04 made at 0.80.
- **My A1b underestimated its own confound by >10x.** I declared the off-policy
  limitation before running; I did not anticipate that it would be 28x the
  effect size.
- A first version of my A2.3 reference clamped the **product** of two norms at
  eps where `torch.cosine_similarity` clamps each norm separately, disagreeing
  on 1129 of 3652 rows. My bug, not 04's, found by the check working as designed.

---

## Verdict on each claim

- **(a)** Survives as a measurement, and its own uncertainty was reported
  honestly. Four streams give 0.7983 to 0.8095 and every single-stream CI
  contains 0.80, so the bar is genuinely straddled and 04's "indecisive" was
  right (A7; pooling appears to resolve it only because it also quadruples the
  classifier's training data). **What does NOT survive is the framing**:
  the headroom above these features is to **0.9466**, not 0.92, so the gap is
  `0.136` rather than `0.109`, and 04's own corrected figure agrees.
- **(b)** Retracted by 04, independently defective per A8. Dead.
- **(c)** Correct as an account of the gap to the per-class oracle, but
  non-discriminating as an argument against history, which 04 conceded.
  **My attempt to measure it directly is withdrawn** - the metric that would
  have measured it is misaligned with loss.
- **(d)** **Wrong as stated.** 72% by correct normalisation, 51% measured on one
  model and stream. Its *inference* about crosstalk survives; the reason the
  mechanism underdelivered is the EMA form.

**So the negative conclusion does not hold in the form 04 published it**, and
04's own revised position - "a real but bounded information gap exists, and this
does NOT establish that closing it is worthless" - is the one the evidence
supports. The bound is larger than 04 thought (0.9466, not 0.92), the cheapest
mechanism that approaches it is per-address **counters** at `d_k`, not a
`d_k x d_v` residual store, and **whether any of it buys loss remains
unmeasured** - my frozen-model attempt cannot resolve effects 28x smaller than
its own confound.

## Reproducing

```
sh run_all.sh gated_seed0 505          # every measurement for one model/stream
python bayes_ceiling.py --stream 505   # A10, needs no model
python audit_addressing.py             # A2, needs no trained model
python audit_cf_controls.py            # A1b's mean-matched constant control
```
