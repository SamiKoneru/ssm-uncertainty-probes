# Gated Delta SSM

A DeltaNet layer whose **write strength can see the memory**.

Standard DeltaNet decides how hard to write from the incoming token alone, which
forces one global write policy. This layer lets `beta` also see what is already
stored at the address being written, using two scalars the delta rule computes
anyway while writing.

```
stored_t = S_{t-1} k_t                    already computed (it IS half of e_t)
e_t      = stored_t - v_t                 the delta rule's residual

s1 = log(1 + ||e_t||)                     surprise:  how wrong was I
s2 = <v_t, stored_t> / ||v_t||^2          overlap:   how much do I already have

beta_t = sigmoid( MLP([ w_beta . x_t , s1 , s2 ]) )
```

Learned policy: **overwrite decisively when the incoming value is far from and
unlike what is stored; write softly when it is already there.**

## Results

3 seeds, identical eval stream, on a task where an input-only gate *provably*
cannot differentiate (each symbol's class is randomised per sequence, so token
identity carries zero information about which policy a key needs):

```
config                 loss mean      sd   acc mean      sd    vs base
input-only baseline       1.6322  0.0018     0.6342  0.0018      0.00%
memory-gated (w=16)       1.5534  0.0026     0.6423  0.0017      4.82%
memory-gated (w=32)       1.5428  0.0039     0.6458  0.0046      5.47%
memory-gated (w=64)       1.5399  0.0006     0.6453  0.0040      5.65%
memory-gated (w=128)      1.5396  0.0014     0.6452  0.0033      5.67%
```

Real and reproducible: seed sd 0.0006-0.005 against a 0.09 effect. **Saturates at
width 32** - accuracy is flat from 32 onward and 32->128 buys 0.2pp of loss for
10x the parameters.

It also **roughly matches a per-class oracle on loss** (5.67% vs 5.42%; the gate
is a 3-seed mean, the oracle a single run): a gate with no class information keeps
pace with one handed ground-truth labels, because the oracle is stuck with one
constant per class while this adapts per token. The oracle still wins on accuracy
(0.655 vs 0.645), and an oracle given the labels plus these features reaches
8.77%, so this gate recovers roughly 60% of the available gain.

## Controls run after the main result

`compare_carve.py` gives every arm the SAME 32-wide MLP head and varies only the
gate's input, so a gain cannot come from gate capacity alone:

```
arm                          gate par  loss mean      sd   vs base
input-only baseline                 0     1.6325  0.0012     0.00%
ours: key-side residual         1,217     1.5426  0.0023     5.51%
CARVE form: query-side          2,433     1.6316  0.0014     0.05%
CARVE form: key-side            2,433     1.6242  0.0015     0.51%
```

`ham_and_position.py` checks two ways the result could be an artifact. A
hand-designed cosine feature through the same MLP head reaches 5.06%, so the
learned residual is not doing anything a cosine cannot mostly recover. And the
accuracy gain is spread across the sequence (+0.008 first half, +0.014 second
half), not concentrated at the end as a burn-in artifact would be. Logs:
`carve_compare.log`, `ham_pos.log`.

## Cost

```
extra state      ZERO      s1, s2 are functions of already-materialised quantities
extra FLOPs      ~0.9%     two reductions over d_v and a divide; scales as 1/d_k
extra params     ~1.6k     at gate_width=32
```

## Parallelism

`beta_t` depends on `S_{t-1}`, so the associative scan breaks. Linearity does not
rescue it: with `beta = c^T S d` the update contains `(c^T S d)(S k)`, quadratic
in the state. The MLP costs nothing here - it is pointwise, so a linear gate has
identical parallelism characteristics.

The fix is granularity: compute `s1, s2` against the **chunk-entry state**
`S_{c-1}`, leaving the WY form untouched, since you are already sequential at
chunk boundaries. **Not implemented here** - `model.py` is the sequential
reference recurrence.

## Use

```bash
python train.py --quick     # 1 seed, 300 steps, ~4 min sanity check
python train.py             # 3 seeds each, ~55 min, reproduces the table above
```

```python
from model import GatedDeltaSSM
m = GatedDeltaSSM(vocab=40, memory_gate=True, gate_width=32)
logits, beta = m(tokens, return_beta=True)
m_base = GatedDeltaSSM(vocab=40, memory_gate=False)   # exact input-only baseline
```

## Two things to know before building on this

**1. THE MECHANISM IS PRIOR ART.** A verified novelty check (equations read from
primary sources) found:

- residual-gated write strength: **Miras** arXiv:2504.13173 Eq. 15 gives
  `eta_t * min(1, delta_t/||M(k_t) - v_t||)` at 1.3B scale; **AURA**
  arXiv:2606.02775 feeds `normalize||k^T W - v||^2` through a sigmoid MLP
- the memory-blind framing and the chunk-entry fix: **CARVE** arXiv:2606.27229,
  which states the premise verbatim and proves an `O(1/sqrt(L))` staleness bound
- and classically: the delta rule with a matrix state **is** `d_v` parallel LMS
  filters, so `beta` is the LMS step size. Step sizes `sigma(a|e|)`,
  `1-exp(-a|e|^2)`, even `log(1+a e^2)` have been standard since the late 1980s.

See `../process/README.md` for the full verdict and the must-cite list.

**2. `s2` IS ALGEBRAICALLY REDUNDANT.** Since `e = Sk - v`:

```
s2 = ( ||stored||^2 + ||v||^2 - ||e||^2 ) / ( 2 ||v||^2 )
```

Conditioned on `||v||`, `(s1, s2)` is a bijection of `(||e||, ||stored||)`. It is
not "surprise plus overlap" - it is residual norm and readout norm in rotated
coordinates, and it degenerates entirely at `d_v = 1`. The recommended (untested)
replacement is `[w_beta . x_t, log(1+||e_t||), ||S_{t-1} k_t||]`. The version
here is the one that was actually measured, so it is what ships.

## Files

```
model.py    the layer. memory_gate=False is the exact input-only baseline.
task.py     STABLE / NOISY / REVISED keys wanting OPPOSITE write policies
train.py    reproduces the result table
```
