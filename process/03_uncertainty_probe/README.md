# STATUS: CLOSED - the mechanism is prior art. Do not build on this.

A verified novelty check (Aug 2026, equations read from primary sources, not
paraphrased) found three of the four claims below are already published. Details
in "NOVELTY VERDICT" at the bottom. The empirical effect here is real and
reproduces across seeds, but it is a reproduction, not a contribution.

# Delta-rule uncertainty, and the write gate that came out of it

Two separate investigations live here. The first failed cleanly; the second is
the actual result.

**Q1: can a delta-rule SSM's write-surprise beat output entropy at predicting
error?  NO**, and the bound is measured: a gradient-boosted probe over every
state feature reaches 0.8871 vs 0.8752 for entropy+maxprob (+0.012, ~2 SE).
Since a trained GBM approximates the best function of those features, that
bounds the whole family.

**Q2: can the write gate `beta` be improved by letting it see the memory?  YES.**
-5.7% loss, +0.6pp accuracy, ZERO extra state. This is the finding.

---

## THE RESULT

Standard DeltaNet computes write strength from the token alone:

    beta_t = sigmoid(w_b . x_t)

That forces ONE global write policy. It cannot write cautiously to a contested
key and decisively to a stale one in the same sequence. Evidence this matters:
learned beta swings **0.0022 -> 0.4605** purely on whether the environment is
stationary.

The change - two scalars the delta rule ALREADY computes while writing:

    stored_t = S_{t-1} k_t                  already computed (it IS half of e_t)
    e_t      = stored_t - v_t               already computed

    s1 = log(1 + ||e_t||)                   surprise magnitude
    s2 = <v_t, stored_t> / ||v_t||^2        overlap: how much do I already have

    beta_t = sigmoid( MLP([ w_b.x_t, s1, s2 ]) )

Learned policy: *overwrite decisively when the value is far from and unlike what
is stored; write softly when it is already there.*

    gate                loss      acc     gate params   extra state   FLOPs
    input-only        1.6329   0.6347            -           -          -
    3->16->1          1.5524   0.6411           81        ZERO        <1%
    3->64->64->1      1.5402   0.6408        4,481        ZERO        <1%
    ---------------------------------------------------------------------
    oracle_class      1.5442   0.6552            3        ZERO   (needs TRUE class labels)

The wide gate **beats the per-class oracle on loss** (5.67% vs 5.42%): it has no
class information at all, but adapts per token where the oracle is stuck with one
constant per class. The oracle keeps a 1.4pp ACCURACY edge no feature gate closed.

Cost detail: `stored_t` is not new work, so we add only two reductions over d_v
and a tiny MLP. Overhead ~ `1/d_k` of the state update: 0.9% at d_k=128, cheaper
as heads widen. Zero extra state at inference.

## THE METHODOLOGICAL FINDING (read this before trusting any per-class number)

**Simpson's paradox destroyed four of my own conclusions.** The signal separating
the two hard key classes lives in `s1` at **AUROC 0.834 - but only among writes
that CHANGED a value.** Pooled with repeat-writes it inverts to 0.353:

    REVISED = 65% repeats (e ~ 0) + 35% changes (e very large, value truly novel)
    NOISY   = ~all changes (e large, but the new value is a MEMBER of the stored
                            set, so e is SMALLER than a fresh one)

    pooled:          NOISY 7.55 > REVISED 6.82     (REVISED dragged down by repeats)
    within changes:  REVISED   > NOISY             (its changes are to novel values)

I read pooled means for several rounds and concluded "the gate learned the policy
backwards." **It never did.** On change writes - the only writes where beta
matters - every gate including the original small MLP had the correct ordering
(NOISY 0.181 < REVISED 0.302).

Everything built to fix that non-existent problem, and all of it wasted:
`s3 = ||stored||/||v||`, negative-sign initialisation, a delta-rule rate memory,
a plain-accumulation rate memory, an overlap rate memory, and a non-overwriting
value matrix for recurrence detection. All null.

**The actual deficit was gate CAPACITY.** 3->16->1 was too small; 3->64->64->1
found the right policy with no new features and no new state (+0.74pp). An
explicit `s1*s2` interaction term helped LESS (+0.30pp) than simply adding width,
so the network can find the interaction, it just needed room.

## What was tried and failed (do not repeat)

- **Accumulate surprise by key**: `R = a R + ||e|| k`, read `|R.q|`. Gets 0.3%
  against the raw statistics' 4.9%. Learned keys sit at cosine **0.68**, so
  interference swamps the clean term - the accumulator destroys more than the
  history adds. Two more auxiliary memories (`E` matrix of residuals, `C` rate
  memory in three flavours) died the same way. **Any per-key store in this
  architecture is a superposition, and superposition crosstalk with correlated
  learned keys kills signals of this size.**
- **Normalised readout** `(R.q)/(Z.q)`: `Z.q` goes negative on 55% of queries.
  Forcing non-negative keys caps selectivity at **2/pi = 0.637 mean cosine,
  independent of dimension** - non-negative vectors occupy one orthant and cannot
  spread. A negativity penalty inherits the identical ceiling; it is geometry.
- **Vector residuals** `E = a E + e k^T`: a key alternating v1/v2 produces
  residuals that are exact negatives and CANCEL. `||e||` accumulates instead.

## Architectural bugs found along the way (each faked or hid signal)

- `v` from the conv -> the same value token yields vectors at cosine 0.88, so
  repeat writes never give the delta rule's exact zero. Route `v` from the raw
  embedding.
- `k` from the conv -> leaks the value token into the ADDRESS, so contested keys
  drift more (0.890 vs 0.957). **This was most of the apparent signal in v1.**
  Route `k` from a hard shift. Fixing both took AUROC 0.54 -> 0.736.
- Reading `R` at the query `q_t` while the write happens at `k_t`. In an
  induction layer those are different tokens, so `u` never contained `||e_t||`.

## Parallelism (unresolved)

`beta_t` now depends on `S_{t-1}`, so the transition operator is not
precomputable and the associative scan breaks. Linearity does NOT rescue this:
with `beta = c^T S d` the update contains `(c^T S d)(S k)`, quadratic in the
state. The MLP costs NOTHING here - it is pointwise, so linear and MLP gates have
identical parallelism characteristics.

    (a) chunk-frozen  compute s1,s2 against the CHUNK-ENTRY state. The WY form is
                      untouched and you are already sequential at chunk
                      boundaries -> likely free, since the chunked delta rule
                      already materialises S_{c-1}K^T. UNTESTED.
    (b) two-pass      exact, 2x within-chunk compute.
    (c) sequential    what every experiment here used. Research only.

Decode is unaffected - it is sequential anyway, so this is ~free at inference.

## Where it transfers

Gated DeltaNet (natural fit; `alpha_t` could be gated the same way), DeltaProduct
(one gated beta per Householder). **NOT Mamba/S6**: no delta rule and no
key-addressed memory, so "what is stored at this address" has no meaning. This is
specific to key-addressed associative memories.

## Status: multi-seed confirmed, then closed on novelty

3 seeds, identical eval stream (`width_seeds.log`):

    config                 loss mean      sd  acc mean      sd   vs base
    input-only baseline       1.6322  0.0018    0.6342  0.0018    0.00%
    gate 3->16->1             1.5534  0.0026    0.6423  0.0017    4.82%
    gate 3->32->32->1         1.5428  0.0039    0.6458  0.0046    5.47%
    gate 3->64->64->1         1.5399  0.0006    0.6453  0.0040    5.65%
    gate 3->128->128->1       1.5396  0.0014    0.6452  0.0033    5.67%

The effect is REAL and reproducible: seed sd 0.0006-0.005 against a 0.09 effect.
It SATURATES at width 32 - accuracy is flat from 32 onward and loss gains only
0.2pp from 32->128 for 10x the parameters. Best config is 3->32->32->1 at 1,621
gate params.

Then the novelty check closed it. See below.

---

# NOVELTY VERDICT (Aug 2026) - CLOSED

Two workflow runs. The FIRST was unreliable: it mischaracterised AURA (a
robotics action-surprise paper, not a memory-residual gate) and Cerruti et al.
(a cross-layer routing study, not a write-gating experiment), and I relayed both
as fact. The SECOND required verbatim quotes from fetched primary text. Its
findings, with equations verified directly:

    C1  beta gated on residual norm ||S_{t-1}k_t - v_t|| at the WRITE KEY
        TAKEN  P=0.95
    C2  beta gated on overlap <v_t, S_{t-1}k_t>/||v_t||^2
        OPEN   P=0.20  (but see the algebra problem below)
    C3  computing gate features against the CHUNK-ENTRY state
        TAKEN  P=0.97
    C4  policy direction: AMPLIFY on large residual
        TAKEN as policy  P=0.90  (the LM-setting RESULT is open)

**C1 dies three times.** Miras (2504.13173) Eq. 15 gives a closed-form effective
step `eta_t * min(1, delta_t/||M(k_t) - v_t||)` at the write key, at 1.3B scale.
AURA (2606.02775) Algorithm 1: `s_t <- normalize||k_t^T W_{t-1} - v_t||^2` fed
through `sigmoid(GateMLP([z_t, o_{t-1}, s_t]))`. And the classical kill: **the
delta rule with a matrix state IS d_v parallel LMS filters sharing input
u = k_t**, so our beta is literally the LMS step size mu(n). Step sizes
`sigma(alpha|e|)`, `1-exp(-alpha|e|^2)` (Gao & Xie 2001) and even
`log(1+alpha e^2)` (ELVSLMS) have been standard since the late 1980s. Our log1p
link is not even a new link function.

**C4** is the founding rationale of that literature: *"when the error is large,
it is desirable to operate with large step size."*

**C3** is CARVE's (2606.27229) central engineering claim, with an O(1/sqrt(L))
staleness proposition we do not have. Its abstract also states our exact
motivating premise verbatim: *"both gates are computed from the incoming token
alone, making the model memory-blind: it decides what to erase without seeing
what it has stored."*

**What CARVE does NOT do** (verified by fetching its full text): its gate reads
along the QUERY, `t_{c,t} = tanh(G_c q_{c,t})` with `G_c = [W_U1,b; W_U1,w]S_{c-1}`;
the residual is absent from the gate; there is no overlap term. So C1/C2 are not
CARVE's. They are Miras', AURA's, and the adaptive-filtering literature's.

## A CONFOUND IN OUR OWN EXPERIMENTS

`s2` is algebraically redundant given `s1` and the two norms:

    s2 = ( ||stored||^2 + ||v||^2 - ||e||^2 ) / ( 2 ||v||^2 )

Given `||stored||` and `||v||`, s2 carries ZERO information beyond s1. The
"nearly additive, so they carry different information" reading in this repo is
WRONG - it is a reparameterisation, and every s1-vs-s2 ablation here is
uninterpretable as run. It also degenerates entirely at d_v = 1. If anyone
revisits this: drop s2 and use `||S_{c-1}k_t||` (slot occupancy at the write
key) as the genuinely independent second feature.

## THE ONE THING STILL OPEN

The modern LM literature DAMPS large residuals (Miras adopts Huber "for reducing
the negative impact of the outlier data"; RLA clips; Lattice retracts). The
classical adaptive-filtering literature AMPLIFIES them. **Nobody has let a
learned continuous beta in an LM linear-attention layer see the residual and
reported which direction it picks.** That the omission is deliberate rather than
accidental is documented: OSDN (2605.13473) explicitly requires a fix that
"does not look at the high-dimensional memory residual", and CARVE names
"memory-blind gating" as an open deficiency.

That is a short empirical paper at 340M+ with a head-to-head against GDN-2 and
Yaad, not an architecture contribution. It is NOT what this repo built.

## MUST-CITE IF ANYONE RESUMES

CARVE 2606.27229 | Miras/Yaad 2504.13173 | AURA 2606.02775 |
Gated KalmaNet 2511.21016 | SRWM 2202.05780 | Recurrent FWP 2106.06295 |
GDN-2 2605.22791 | GDN 2412.06464 | DeltaNet 2406.06484 | OSDN 2605.13473 |
SinkRec 2606.09888 | Lattice 2504.05646 | EntNet 1612.03969 |
Kanerva Machine 1804.01756 | Titans 2501.00663 |
Kwong & Johnston VSS-LMS 1992 | Gao & Xie 2001 | Gollamudi SM-NLMS 1998

## UNCHECKED GAPS

- **recsys / CTR memory-gating literature: UNCHECKED, and the biggest risk.**
  SinkRec came from there and "memory coverage" gating is routine. Could push
  P(C2 taken) from 0.20 to 0.45. Check this FIRST if resuming.
- Neuroscience / RL surprise-modulated plasticity: unchecked, C2 only.
- EntNet's gate equation: one-source, not re-verified.
- Classical IEEE full texts (Kwong 1992, Aboulnasr 1997, Gollamudi 1998):
  paywalled, verified via agreeing secondaries only. Not load-bearing - C1/C4
  are locked by sources read directly.
- ~30 distinct 2026 papers modify the delta-rule write gate. Concurrent
  unpublished work is invisible and the base rate is high.

## CODE MAP

Shared modules at the top of this directory; experiments under the thread that
used them; everything superseded under `archive/`. Moved scripts carry a
sys.path header so any of them runs from where it sits.

    delta_model.py       DeltaLayer. flags: v_from_embed, kq_from_embed,
                         fixed_beta. Thread A.
    gated_model.py       GatedDeltaLayer, 11 gate modes, accumulated across the
                         whole investigation. The CLEAN implementation of the
                         final design is ../../gated_delta_ssm/model.py.
    induction_task.py    in-context recall, STATIONARY
    revision_task.py     NON-stationary. TRAP: revision is latent, so its
                         confidently-wrong cases are IRREDUCIBLE - no signal,
                         not even an oracle, can beat chance on them. Read
                         before reusing.
    mixed_task.py        STABLE / NOISY / REVISED keys wanting OPPOSITE write
                         policies, class randomised per sequence so an
                         input-only gate PROVABLY cannot differentiate

    A_uncertainty/       the original question - CLOSED
      probe_test.py      *** THE RESULT. GBM over every state feature: 0.8871
                         vs 0.8752 for entropy+maxprob. Bounds the family.
      induction_run.py   u 0.808 -> error, 0.948 -> contested; entropy 0.958
      complementarity.py corr(u,entropy)=0.62; entropy+u is WORSE than entropy
      revision_run.py    learned beta jumps 0.0022 -> 0.4605 on non-stationary
      shift_test.py      train/eval volatility mismatch; entropy degrades to
                         chance, u never had signal to lose

    B_write_gate/        the spin-off that worked
      width_seeds.py     *** THE RESULT. 3 seeds x 5 widths. Real (sd
                         0.0006-0.005 vs a 0.09 effect), saturates at width 32.
      interaction_test.py *** THE SIMPSON'S PARADOX. s1 separates the hard
                         classes at AUROC 0.834 on CHANGE writes, 0.353 pooled.
                         Read before trusting any per-class mean here.
      oracle_test.py     per-class ceiling; the learned gate BEATS it on loss
      beta_ablation.py   the conservation law (uncertainty migrates between
                         memory superposition and surprise stream as beta rises)
      mixed_run.py       main gate comparison

    archive/             superseded, kept for the record
      failed_features/   7 dead ends, each with WHY: address_test (R read at q
                         not k; the accumulator destroys the ||e|| it contains),
                         cosine_test, s3_test, rate_test, recurrence_test,
                         signinit_test (gradients push AWAY from the oracle
                         signs), key_geometry (keys are ISOTROPIC at 0.0701,
                         below the 0.0884 random baseline - the 0.68 that
                         motivated a whole direction came from the v1 conv model)
      numpy_demos/       delta_u, keyed_u, decoupled_phi, phi_diag, r_variants.
                         All worked with ORTHOGONAL keys; none survived learned
                         ones. This gap is the story of the whole project.
      beta_align.py      the label-alignment bug that produced the paradox
      firstwrite_test.py beta split by first-write / repeat / change
      kv_task.py, pipeline_test.py, diagnose_pipeline.py, full_diag.py,
      find_regime.py, train_kv.py, train_and_save.py, readout_blend.py
                         the superseded kv_task line (v1 -> v3 confound fixes)

    logs/                29 result logs, one per experiment
    checkpoints/         kv_model{,_v2,_v3}.pt (the v1/v2/v3 fixes),
                         shift_model.pt
