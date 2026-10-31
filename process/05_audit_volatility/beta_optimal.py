"""A1. The per-write OPTIMAL beta, in closed form, and how well each feature
set predicts it.

04's claims (a) and (b) are denominated entirely in AUROC on the CLASS LABEL.
04's own pre-registration lists that as a known unmeasured limit: "AUROC on the
class label is a PROXY for 'did the gate choose the right beta'. A gate can be
right without knowing the class: on writes where ||e|| is small, beta multiplies
almost nothing."

That limit is measurable. In `model.py` the write is
`S_t = S_{t-1} - beta_t e_t k_t^T` with `e_t = stored_t - v_t` and unit `k_t`, so

    content at this address after the write = (1 - beta_t) stored_t + beta_t v_t

beta is a pure mixing coefficient. Writing

    stored_t  = S_{t-1} k_t            what the address already holds   [d_v]
    v_t       = W_v emb(x_t)           the value being written          [d_v]
    u_t       = v_t - stored_t = -e_t  the direction beta moves content [d_v]
    v_next                             the value the NEXT write to this
                                       address will store               [d_v]
    w_t       = v_next - stored_t      what the content needs to become [d_v]

the squared error of the stored content against that target is a scalar
quadratic in beta:

    err_t(beta) = || beta u_t - w_t ||^2 = A_t beta^2 - 2 B_t beta + C_t

    A_t = ||u_t||^2 = ||e_t||^2    leverage: how much beta can change anything
    B_t = <u_t, w_t>               how much of the needed fix lies along u_t
    C_t = ||w_t||^2                the error left at beta = 0

    beta*_t = clip(B_t / A_t, 0, 1)

`v_next` is the right target for two reasons that agree: it is the value needed
at the next READ of this address, and it is the value the next WRITE to this
address stores. Both land on the same token index, which is a self-consistency
check on the addressing (see `_check_target_indexing`).

Because `A b^2 - 2B b + C = A (b - B/A)^2 + (C - B^2/A)`, minimising mean `err`
over a feature set is EXACTLY weighted least squares against target `B_t/A_t`
with sample weight `A_t`. No custom loss is needed, and `A_t` supplies the
leverage weighting that pooled class-AUROC throws away.

Two reported scales, both floor-free in the ratio that matters:

    RER(F) = 1 - E[err(beta_F)] / E[err(beta_const)]      pre-registered primary
    CER(F) = 1 - E[A (beta_F - beta*)^2]
                 / E[A (beta_const - beta*)^2]            controllable part only

`C_t - B_t^2/A_t` is irreducible and sits in both the numerator and the
denominator of RER, so it cancels in any DIFFERENCE of RERs. The ratio
`R_beta = gain_hist / gain_nonlin` is therefore identical under RER and CER.
"""
import bisect
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

import _paths                                                  # noqa: F401
HERE = _paths.HERE
FOUR = _paths.FOUR

import features as FE                                          # noqa: E402
import task as T                                               # noqa: E402

INST, HIST = FE.INST, FE.HIST
TGT_CLIP = (-1.0, 2.0)          # fitting target clip; evaluation never uses it


# --------------------------------------------------------------------------
# where the target comes from
# --------------------------------------------------------------------------
def target_index(seq):
    """[B, L+1] tokens -> [B, L] index of the next write to THIS write's address.

    The write at step t addresses a = x_{t-1}. Address a is next READ at
    j = min{ i >= t : x_i = a }, where the model must predict x_{j+1}; and the
    next WRITE to a is at step j+1, storing value(x_{j+1}). Same index either
    way. Returns -1 where there is no such j with j <= L-1.
    """
    B, Lp1 = seq.shape
    L = Lp1 - 1
    out = np.full((B, L), -1, np.int64)
    for b in range(B):
        occ = {}
        for i in range(Lp1):
            occ.setdefault(int(seq[b, i]), []).append(i)
        for t in range(1, L):
            lst = occ[int(seq[b, t - 1])]
            p = bisect.bisect_left(lst, t)
            if p < len(lst) and lst[p] <= L - 1:
                out[b, t] = lst[p] + 1
    return out


@torch.no_grad()
def _e_and_v(m, x):
    """Residuals e_t and the value projection of EVERY token, incl. x_L.

    e_t is recomputed here rather than taken from FE.replay (which does not
    return it); `abc` asserts the two agree through s1 and s2.
    """
    tokens = x[:, :-1]
    B, L = tokens.shape
    emb = m.embed(tokens)
    ctx = F.silu(m.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))
    shifted = F.pad(emb, (0, 0, 1, 0))[:, :L]
    k = F.normalize(m.to_k(shifted), dim=-1)
    v = m.to_v(emb)
    beta_token = m.to_beta(ctx)[..., 0]
    S = emb.new_zeros(B, m.d_v, m.d_k)
    es, st = [], []
    for t in range(L):
        kt, vt = k[:, t], v[:, t]
        stored = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1)
        e = stored - vt
        es.append(e)
        st.append(stored)
        g = beta_token[:, t]
        if m.memory_gate:
            s1 = torch.log1p(e.norm(dim=-1))
            s2 = (vt * stored).sum(-1) / (vt.norm(dim=-1) ** 2 + 1e-6)
            g = m.gate(torch.stack([g, s1, s2], -1))[..., 0]
        beta = torch.sigmoid(g)
        S = S - beta[:, None, None] * e.unsqueeze(-1) * kt.unsqueeze(1)
    e_all = torch.stack(es, 1)
    stored_all = torch.stack(st, 1)
    # v over ALL L+1 tokens, so v_next exists for a target index of L
    v_full = m.to_v(m.embed(x))
    # the identity the algebra below relies on
    bad = (stored_all - (v_full[:, :L] + e_all)).abs().max().item()
    assert bad < 1e-4, f"stored != v + e : {bad:.2e}"
    return e_all.double(), v_full.double()


def abc(m, x, ref):
    """A_t, B_t, C_t and a validity mask, from e_t and the value projections.

    w_t = v_next - stored_t = v_next - v_t - e_t, so no second state pass is
    needed. `ref` is FE.replay's output on the same batch; s1 and s2 are
    re-derived from this e_t and checked against it, which is what ties this
    recomputation to the asserted forward pass.
    """
    e, v_full = _e_and_v(m, x)
    B, L, _ = e.shape
    ne = e.norm(dim=-1)
    # tie e back to the replay that carries the forward-pass assert
    s1 = torch.log1p(ne).numpy()
    assert np.abs(s1 - ref["f_s1"]).max() < 1e-4, "e_t disagrees with FE.replay"

    tgt = target_index(x.numpy())
    valid = tgt >= 0
    ti = np.where(valid, tgt, 0)
    v_next = v_full[torch.arange(B)[:, None], torch.from_numpy(ti)]
    v_t = v_full[:, :L]

    w = v_next - v_t - e
    A = (ne ** 2).numpy()
    Bq = (-(e * (v_next - v_t)).sum(-1)).numpy() + A
    C = (w.norm(dim=-1) ** 2).numpy()
    valid &= A > 0
    return A, Bq, C, valid


def _check_target_indexing(seq, tgt):
    """The target index must be the next occurrence of the address, plus one."""
    B, L = tgt.shape
    rng = np.random.default_rng(0)
    for _ in range(300):
        b = rng.integers(B)
        t = rng.integers(1, L)
        if tgt[b, t] < 0:
            continue
        a = seq[b, t - 1]
        j = tgt[b, t] - 1
        assert seq[b, j] == a, "target is not an occurrence of the address"
        assert not (seq[b, t:j] == a).any(), "an earlier occurrence was skipped"


# --------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------
def collect(m, seed, batches=6, batch=32, junk_rng=None):
    """FE's own columns plus A, B, C, valid, and optional junk controls."""
    ev = np.random.default_rng(seed)
    jr = np.random.default_rng(junk_rng) if junk_rng is not None else None
    acc, off = None, 0
    losses = []
    for bi in range(batches):
        x, cl, sn = T.sample(ev, batch)
        d = FE.replay(m, x)
        A, Bq, C, valid = abc(m, x, d)
        if bi == 0:
            _check_target_indexing(x.numpy(), target_index(x.numpy()))
        L = T.L
        d["A"], d["B"], d["C"] = A, Bq, C
        d["valid"] = valid.astype(np.float64)
        d["cls_w"] = F.pad(cl, (1, 0))[:, :L].numpy().astype(np.int64)
        d["seq"] = np.repeat(np.arange(off, off + batch)[:, None], L, 1)
        d["pos"] = np.repeat(np.arange(L)[None, :], batch, 0).astype(np.float64)
        if jr is not None:
            d["j_gauss"] = jr.normal(size=(batch, L))
            d["j_unif"] = jr.random(size=(batch, L))
            d["j_heavy"] = jr.standard_t(2, size=(batch, L))
            d["j_seqconst"] = np.repeat(
                jr.normal(size=(batch, 1)), L, 1)
            # one draw per (sequence, address): the same grouped structure as a
            # per-address volatility memory, with zero class information
            tabl = jr.normal(size=(batch, T.N_SYM))
            addr = np.concatenate(
                [np.zeros((batch, 1), np.int64), x[:, :L - 1].numpy()], 1)
            d["j_addrconst"] = np.take_along_axis(tabl, addr, 1)
        losses.append(F.cross_entropy(
            torch.from_numpy(d["logits"]).reshape(-1, T.N_SYM),
            x[:, 1:].reshape(-1)).item())
        d.pop("logits")
        off += batch
        if acc is None:
            acc = {k: [v] for k, v in d.items()}
        else:
            for k, v in d.items():
                acc[k].append(v)
    flat = {k: np.concatenate(v).reshape(-1) for k, v in acc.items()}
    flat["valid"] = flat["valid"] > 0.5
    flat["_loss"] = float(np.mean(losses))
    return flat


# --------------------------------------------------------------------------
# policies and scoring
# --------------------------------------------------------------------------
def err(A, Bq, C, beta):
    return A * beta ** 2 - 2 * Bq * beta + C


def beta_star(A, Bq):
    return np.clip(Bq / A, 0.0, 1.0)


class Policy:
    """A fitted map from features to beta, usable offline and online."""

    def __init__(self, kind, feats, seed=0):
        self.kind, self.feats, self.seed = kind, feats, seed

    def fit(self, X, A, Bq):
        tgt = np.clip(Bq / A, *TGT_CLIP)
        if self.kind == "gbm":
            self.med = np.nanmedian(X, 0)
            self.model = HistGradientBoostingRegressor(
                max_iter=200, random_state=self.seed)
            self.model.fit(X, tgt, sample_weight=A)
        elif self.kind == "linear":
            self.med = np.nanmedian(X, 0)
            Z = self._imp(X)
            self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-9
            self.model = LinearRegression()
            self.model.fit((Z - self.mu) / self.sd, tgt, sample_weight=A)
        else:
            raise ValueError(self.kind)
        return self

    def _imp(self, X):
        Z = np.where(np.isfinite(X), X, self.med)
        return np.clip(Z, -1e6, 1e6)

    def predict(self, X):
        if self.kind == "gbm":
            p = self.model.predict(X)
        else:
            p = self.model.predict((self._imp(X) - self.mu) / self.sd)
        return np.clip(p, 0.0, 1.0)


def const_beta(A, Bq):
    """The best single global beta: the 'one global write policy' baseline."""
    return float(np.clip(Bq.sum() / A.sum(), 0.0, 1.0))


def per_class_beta(A, Bq, cls):
    """Best constant beta inside each class: the true-class-label oracle."""
    out = {}
    for c in np.unique(cls):
        s = cls == c
        out[int(c)] = float(np.clip(Bq[s].sum() / A[s].sum(), 0, 1))
    return out


def score(A, Bq, C, beta, b_const, bstar):
    """RER and CER of a beta vector against the best-constant baseline."""
    e_f = err(A, Bq, C, beta).mean()
    e_c = err(A, Bq, C, b_const).mean()
    rer = 1.0 - e_f / e_c
    cf = (A * (beta - bstar) ** 2).mean()
    cc = (A * (b_const - bstar) ** 2).mean()
    cer = 1.0 - cf / cc
    return float(rer), float(cer), float(e_f)


# --------------------------------------------------------------------------
# self-test of the algebra: brute force the quadratic and the argmin
# --------------------------------------------------------------------------
def _self_test(m, seed=505, batch=8):
    """Check err() and beta_star() against a brute-force grid and against a
    direct simulation of the write, with nothing reused from the derivation."""
    x, _, _ = T.sample(np.random.default_rng(seed), batch)
    d = FE.replay(m, x)
    A, Bq, C, valid = abc(m, x, d)
    e, v_full = _e_and_v(m, x)
    tgt = target_index(x.numpy())
    _check_target_indexing(x.numpy(), tgt)

    grid = np.linspace(0.0, 1.0, 2001)
    rng = np.random.default_rng(1)
    worst_err, worst_arg = 0.0, 0.0
    n = 0
    for _ in range(400):
        b = int(rng.integers(batch))
        t = int(rng.integers(1, T.L))
        if not valid[b, t]:
            continue
        n += 1
        stored = (v_full[b, t] + e[b, t]).numpy()
        vt = v_full[b, t].numpy()
        vn = v_full[b, tgt[b, t]].numpy()
        # simulate the write directly, no algebra
        direct = np.array([
            np.sum(((1 - g) * stored + g * vt - vn) ** 2) for g in grid])
        quad = err(A[b, t], Bq[b, t], C[b, t], grid)
        sc = max(direct.max(), 1e-12)
        worst_err = max(worst_err, float(np.abs(direct - quad).max() / sc))
        worst_arg = max(worst_arg, float(abs(
            grid[direct.argmin()] - beta_star(A[b, t], Bq[b, t]))))
    print(f"  self-test on {n} writes")
    print(f"    max relative |err_quadratic - err_simulated|  {worst_err:.3e}"
          f"   (bar < 1e-5)")
    print(f"    max |grid argmin - clip(B/A,0,1)|             {worst_arg:.3e}"
          f"   (bar < 1e-3, grid step 5e-4)")
    assert worst_err < 1e-5 and worst_arg < 1e-3, "algebra self-test FAILED"
    print("    PASS")
