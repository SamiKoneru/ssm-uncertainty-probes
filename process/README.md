# Process

How the architecture in `../gated_delta_ssm/` was arrived at, and the three
questions that were answered negatively on the way. Each phase keeps only the
scripts that establish something; everything else is under `archive/`.

The original question was: **can an SSM's internal state provide an uncertainty
signal that beats output entropy?** The answer is no, for a structural reason,
and the write gate is a spin-off from the machinery built to ask it.

---

## 01_wave_interference - REFUTED (artifact)

Hypothesis: in a norm-preserving (unitary) SSM, the magnitude of the complex
state tracks interference and so signals uncertainty.

    state_magnitude  AUROC 0.733     vs output_entropy 0.424
    raw position t   AUROC 0.751     <- a CLOCK beats the signal
    magnitude, position-controlled   0.489   <- chance

The magnitude correlated 0.94 with position because the normalization was never
implemented, so |h| grew monotonically. Training used length 120 and evaluation
200, and deterministic-step accuracy collapsed 0.99 -> 0.27 past the training
horizon. "High magnitude" meant "past the horizon" meant "the model is broken
here."

    LESSON: always report a raw-position control, and train/eval at one length.

`uncertainty_test.py` (the original), `diagnose.py` (the refutation).

## 02_two_timescale - MECHANISM ONLY, then superseded

`F = a_f F + x`, `S = a_s S + x`, `u = ||F_hat - S_hat||`. Both are diagonal
linear recurrences, so parallel via FFT (verified against a sequential reference
to ~1e-6).

Never validated, because the task could not test it: value-change rate was 88%
on predictable steps and 87% on random ones, inverted on one seed. And u scored
0.577 gated but **0.526 with the gate off** - most of the signal was manufactured
by training, since c = 1/(1+u) multiplying the logits makes u a learned
temperature.

    LESSON: always run the gate-off ablation on any signal that feeds the output.

What does hold: `F_hat - S_hat` is a **difference of two lowpass filters, i.e. a
bandpass**, with an analytic design tool validated to 4e-5:

    B(w) = (1-a_f)/(1 - a_f e^-iw) - (1-a_s)/(1 - a_s e^-iw),    B(0) = 0 always

and **tape placement, not K, controls flatness**: K=2 at tau (0.2, 1024) gives
1.01x spread where the spec's (2, 100) gives 3.82x. My initial "more tapes fixes
it" claim came from a sweep that varied K and the tau range together.

SUPERSEDED: Titans (arXiv:2501.00663) Eq. 9-10 is the same fast/slow
decomposition applied to exactly this quantity - `S_t = eta_t S_{t-1} -
theta_t grad(l)`, which the paper describes as breaking surprise into "past
surprise" and "momentary surprise."

`stability_ssm.py`, `mechanism_test.py` (the bandpass), `k_verdict.py`
(placement).

## 03_uncertainty_probe - the question CLOSED, and the gate found

### A_uncertainty/ - entropy wins, and the bound is measured

`probe_test.py` is the result. A gradient-boosted probe over the full pre-head
state o plus every surprise feature and entropy+maxprob reaches 0.8823 test AUROC
against 0.8755 for entropy+maxprob alone: **+0.007** (one trained model, one seed,
sequence-level split; no standard error computed). An earlier run with the
surprise features only plus entropy+maxprob reached 0.8871 vs 0.8752
(`archive/failed_features/address_test.py`). Either way the probe finds almost
nothing beyond entropy on this task: strong evidence against the family, not a
proof.

Why, structurally: cross-entropy is a **proper scoring rule**, so training pays
the model directly to fold every available and useful signal into its posterior,
and entropy is a one-line readout of that. Every state signal reconstructs a
correlate of the same quantity through lossier machinery.

Getting an honest measurement first required fixing three confounds, each of
which was faking or hiding signal:

    v from the conv   the same value token gave vectors at cosine 0.88, so a
                      repeat write never gave the delta rule's exact zero
    k from the conv   leaked the VALUE token into the ADDRESS, so contested keys
                      drifted more - THIS WAS MOST OF THE APPARENT SIGNAL
    R read at q_t     the write happens at k_t, so u never contained ||e_t||

Fixing the first two took AUROC 0.54 -> 0.736.

Also here: `revision_run.py` (learned beta jumps 0.0022 -> 0.4605 on
non-stationary data), `shift_test.py` (distribution shift: entropy degrades to
chance but u never had signal to lose), `complementarity.py`,
`induction_run.py`.

### B_write_gate/ - the spin-off that worked

`width_seeds.py` is the result: 3 seeds x 5 widths, effect real (sd 0.0006-0.005
vs a 0.09 effect) and saturating at width 32. `oracle_test.py` shows a learned
gate BEATS a per-class oracle on loss. `beta_ablation.py` gives the conservation
law: as beta rises, uncertainty migrates out of the memory's superposition (where
entropy reads it) into the write-surprise stream (where a state signal reads it),
with accuracy flat and loss monotone.

**`interaction_test.py` is the most transferable thing in this repo.** It found a
SIMPSON'S PARADOX that invalidated four of my own proposals:

    s1 separates the two hard key classes at AUROC 0.834 on CHANGE writes
                                       and at AUROC 0.353 POOLED

Both verified on two fresh eval streams in `04_volatility_history` (0.8073 and
0.8121 on change writes, 0.3569 and 0.3589 pooled). Two traps around them:

  - `logs/signinit.log:15` reads `AUROC NOISY vs rest: 0.8345`. That is a
    DIFFERENT contrast and the agreement to three decimals is a coincidence.
  - `archive/failed_features/rate_test.py:11` says "(s1,s2) gets 0.476, i.e.
    chance". That number (`logs/signinit.log:17`) is a 3-way classifier's
    P(REVISED) scored POOLED. **It is correct for the model it was measured
    on**, and reproduces to within 0.005 (`04_volatility_history/legacy_0476.py`
    gets 0.4712 against the logged 0.4763, with all three one-vs-rest AUROCs
    inside 0.002). But it does NOT describe the gate that ships.

    **Separability in (s1,s2) is a property of the TRAINED GATE, not of the
    task.** Both models over the same three eval streams, pooled over all seen
    writes, NOISY vs REVISED (`04_volatility_history/logs/model_vs_task.log`):

        gate                        loss      (s1,s2) pooled AUROC
        legacy linear gate="both"  1.5882     0.604 to 0.616
        shipped MLP 3->32->32->1   1.5428     0.796 to 0.817

    beta shapes `S`, `S` shapes `(s1,s2)`, so a gate with a differentiated write
    policy MANUFACTURES a more separable feature distribution for itself. The
    conclusion signinit drew from 0.476, that the gap to the oracle is
    information-limited, was right about its own model and is wrong about the
    shipped one. **The lever is the gate's policy, which improves its own
    features, not a new memory.**

    Independently of all that, the bar built on 0.476 was read with the wrong
    sign. `logs/rate.log` has `rate_delta c_t AUROC = 0.3690` against that 0.476
    reference, and 0.369 was taken as the weaker number. It is the stronger one:
    AUROC is directional, so |0.369 - 0.5| = 0.131 against |0.476 - 0.5| =
    0.024. **The rate memory cleared its own pass bar by 5x and was recorded as
    failing it.** It still bought no loss, which is the real result; see
    `03_uncertainty_probe/README.md`.

    REVISED = 65% repeats (e~0) + 35% changes to genuinely NOVEL values
    NOISY   = ~all changes, but to values ALREADY in the stored set

The relationship inverts between the pooled and conditioned views. I read pooled
means for several rounds, concluded "the gate learned the policy backwards," and
built `s3`, a sign-initialisation, two rate memories and a recurrence matrix to
fix it. The gate was never backwards. All of that is under
`archive/failed_features/` with the reason each one died.

---

## NOVELTY: the mechanism is prior art

Verified by reading equations from primary sources, after a first attempt
mischaracterised two papers and had to be redone.

    C1  beta gated on residual norm at the write key     TAKEN   P=0.95
    C2  overlap <v, Sk>/||v||^2 as a gate feature        OPEN    P=0.20
    C3  chunk-entry-state parallelism fix                TAKEN   P=0.97
    C4  amplify-on-large-residual policy                 TAKEN   P=0.90

**C1** dies three times: Miras (2504.13173) Eq. 15 at 1.3B scale; AURA
(2606.02775) Algorithm 1; and classically, the delta rule with a matrix state
**is** `d_v` parallel LMS filters, so beta is the LMS step size and step sizes
`sigma(a|e|)`, `1-exp(-a|e|^2)`, `log(1+a e^2)` are late-1980s standard.
**C3** is CARVE (2606.27229), which states the memory-blind premise verbatim.
Its gate reads along the QUERY and never touches the residual, so C1/C2 are not
CARVE's - they are Miras', AURA's, and adaptive filtering's.

The one open question the field has not answered: modern LM work **damps** large
residuals (Miras' Huber, RLA's clipping) while classical adaptive filtering
**amplifies** them, and nobody has let a learned continuous beta in an LM
linear-attention layer report which direction it picks.

MUST CITE if resuming: CARVE 2606.27229 | Miras 2504.13173 | AURA 2606.02775 |
Gated KalmaNet 2511.21016 | SRWM 2202.05780 | GDN-2 2605.22791 | GDN 2412.06464 |
DeltaNet 2406.06484 | OSDN 2605.13473 | Titans 2501.00663 | Kwong & Johnston
VSS-LMS 1992 | Gao & Xie 2001

UNCHECKED, and the biggest remaining risk: the recsys/CTR memory-gating
literature, where "memory coverage" gating is routine.

## A direction that died on inspection

Key geometry. Learned keys looked badly non-orthogonal (cosine 0.68 vs 0.088 for
random vectors), implying a large capacity tax. On measurement the final model's
keys sit at **0.0701, BELOW the random baseline** - the 0.68 came from the
superseded conv-based v1 model. Independently: the theory is KATA (2607.17419,
mutual coherence + Welch bound, naming DeltaNet), the fix is the Mesa/MesaNet
RLS readout, and the phenomenon is the nine-year-old anisotropy/cone effect.
`archive/failed_features/key_geometry.py`.
