"""
Two-timescale stability memory (spec v1).

    F_t = a_f * F_{t-1} + x_t      a_f small   (fast forgetting)
    S_t = a_s * S_{t-1} + x_t      a_s -> 1    (slow forgetting)

Both are diagonal linear recurrences whose decay is time-invariant and
state-independent, so each is an associative scan.  They are evaluated here as
a causal depthwise convolution with kernel a^j, done by FFT: O(L log L), fully
parallel over the sequence, no sequential loop anywhere in the recurrence.
Nothing nonlinear touches F or S.

Everything nonlinear lives in the readout:

    Fhat = F * gain_f            gain divides out the accumulator's gain
    Shat = S * gain_s
    u    = RMS(Fhat - Shat)      instability
    c    = 1 / (1 + u)           confidence
    y    = Readout([Fhat, Shat]) * c

gain: the spec uses the steady-state gain (1 - a).  That is only correct once
the accumulator has converged, which takes ~1/(1-a) steps: 2 for the fast tape,
100 for the slow one.  Before then Shat systematically undershoots and u decays
with t for reasons that have nothing to do with stability.  bias_correct=True
uses the realized gain (1 - a)/(1 - a^{t+1}) instead, which removes that exactly
and is still just a deterministic per-step scalar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _log_alpha_init(alpha):
    """raw p such that exp(-softplus(p)) == alpha."""
    return torch.log(torch.tensor(1.0 / alpha - 1.0))


def ema_parallel(x, log_alpha):
    """h_t = exp(log_alpha) * h_{t-1} + x_t, as an FFT convolution. x: [B,L,N]."""
    B, L, N = x.shape
    j = torch.arange(L, device=x.device, dtype=x.dtype)
    kernel = torch.exp(log_alpha[None, :] * j[:, None])          # [L,N], a^j
    n = 2 * L
    xf = torch.fft.rfft(x.transpose(1, 2), n=n)                  # [B,N,.]
    kf = torch.fft.rfft(kernel.t().contiguous(), n=n)            # [N,.]
    h = torch.fft.irfft(xf * kf[None], n=n)[..., :L]
    return h.transpose(1, 2)


def ema_sequential(x, log_alpha):
    """Reference implementation of the same recurrence, for verification only."""
    B, L, N = x.shape
    a = torch.exp(log_alpha)
    h = torch.zeros(B, N, device=x.device, dtype=x.dtype)
    out = []
    for t in range(L):
        h = a * h + x[:, t]
        out.append(h)
    return torch.stack(out, 1)


class TwoTimescaleMemory(nn.Module):
    def __init__(self, vocab, d_model=64, n_state=64, n_groups=1,
                 alpha_fast=0.5, alpha_slow=0.99,
                 bias_correct=True, use_gate=True, normalize_u=False):
        super().__init__()
        assert n_state % n_groups == 0
        self.V, self.N, self.G = vocab, n_state, n_groups
        self.bias_correct, self.use_gate, self.normalize_u = bias_correct, use_gate, normalize_u

        self.embed = nn.Embedding(vocab, d_model)
        self.write = nn.Linear(d_model, n_state)                 # x_t = learned projection
        # per-channel decays, alpha = exp(-softplus(p)) in (0,1); state-independent
        self.p_fast = nn.Parameter(_log_alpha_init(alpha_fast) + 0.01 * torch.randn(n_state))
        self.p_slow = nn.Parameter(_log_alpha_init(alpha_slow) + 0.01 * torch.randn(n_state))
        self.readout = nn.Linear(2 * n_state, vocab)

    def alphas(self):
        return torch.exp(-F.softplus(self.p_fast)), torch.exp(-F.softplus(self.p_slow))

    def _gain(self, log_a, L, device):
        """[L,N] gain that puts an accumulator on a unit-weight footing."""
        a1 = 1.0 - torch.exp(log_a)[None, :]                      # (1 - a)
        if not self.bias_correct:
            return a1.expand(L, -1)
        t = torch.arange(L, device=device, dtype=log_a.dtype)[:, None]
        realized = 1.0 - torch.exp(log_a[None, :] * (t + 1.0))    # 1 - a^{t+1}
        return a1 / realized.clamp_min(1e-6)

    def forward(self, tokens):
        """tokens: [B,L] -> logits [B,L,V], instability u [B,L,G]."""
        B, L = tokens.shape
        x = self.write(self.embed(tokens))                        # [B,L,N]
        log_af, log_as = -F.softplus(self.p_fast), -F.softplus(self.p_slow)

        # --- the only recurrent part: two diagonal linear scans -------------
        Fm = ema_parallel(x, log_af)
        Sm = ema_parallel(x, log_as)
        # -------------------------------------------------------------------

        Fh = Fm * self._gain(log_af, L, x.device)[None]
        Sh = Sm * self._gain(log_as, L, x.device)[None]

        d = (Fh - Sh).view(B, L, self.G, self.N // self.G)
        u = d.pow(2).mean(-1).sqrt()                              # [B,L,G]
        if self.normalize_u:
            s = Sh.view(B, L, self.G, -1).pow(2).mean(-1).sqrt()
            u = u / (s + 1e-4)
        c = 1.0 / (1.0 + u)                                       # [B,L,G]

        content = torch.cat([Fh, Sh], -1)
        if not self.use_gate:                                     # ablation: u observed, unused
            return self.readout(content), u
        if self.G == 1:
            return self.readout(content) * c, u                   # spec: gate the output
        g = c.repeat_interleave(self.N // self.G, -1)             # G>1: gate content per group
        return self.readout(torch.cat([Fh * g, Sh * g], -1)), u


# ===========================================================================
# v2: K-timescale filter bank
#
# Two accumulators give exactly one passband, so u responds to churn only at
# scales between 1/(1-a_f) and 1/(1-a_s) and is blind on both sides.  K
# log-spaced tapes give K-1 adjacent bands covering tau_min..tau_max; taking
# the max over bands makes u respond to churn at ANY scale in that range.
#
# Every tape is still a diagonal linear recurrence with state-independent
# decay, so this is K associative scans and nothing nonlinear enters the
# recurrence.  Cost note: all tapes convolve the same x, so only ONE forward
# FFT is needed; the tapes differ only in the kernel spectrum.
# ===========================================================================

def ema_bank(x, log_alpha):
    """x: [B,L,N], log_alpha: [K,N] -> [B,K,L,N]. One shared forward FFT."""
    B, L, N = x.shape
    j = torch.arange(L, device=x.device, dtype=x.dtype)
    kern = torch.exp(log_alpha[:, None, :] * j[None, :, None])        # [K,L,N]
    n = 2 * L
    Xf = torch.fft.rfft(x.transpose(1, 2), n=n)                       # [B,N,.] once
    Kf = torch.fft.rfft(kern.permute(0, 2, 1).contiguous(), n=n)      # [K,N,.]
    H = torch.fft.irfft(Xf[:, None] * Kf[None], n=n)[..., :L]         # [B,K,N,L]
    return H.permute(0, 1, 3, 2)


def bank_gains(log_alpha, L, device, bias_correct=True):
    """[K,L,N] gain putting every tape on a unit-weight footing."""
    a1 = 1.0 - torch.exp(log_alpha)[:, None, :]
    if not bias_correct:
        return a1.expand(-1, L, -1)
    t = torch.arange(L, device=device, dtype=log_alpha.dtype)[None, :, None]
    return a1 / (1.0 - torch.exp(log_alpha[:, None, :] * (t + 1.0))).clamp_min(1e-6)


def multiscale_u(x, log_alpha, n_groups=1, agg="median", bias_correct=True):
    """Instability from adjacent-band differences. x: [B,L,N] -> u: [B,L,G]."""
    B, L, N = x.shape
    H = ema_bank(x, log_alpha) * bank_gains(log_alpha, L, x.device, bias_correct)[None]
    d = (H[:, :-1] - H[:, 1:])                                        # [B,K-1,L,N] bands
    d = d.view(B, -1, L, n_groups, N // n_groups)
    band_u = d.pow(2).mean(-1).sqrt()                                 # [B,K-1,L,G]
    return _agg(band_u, agg)


def _agg(band_u, agg):
    """Collapse [B,K-1,L,G] band responses to [B,L,G].

    max: fires on the single loudest band, so a settled one-time update looks
    identical to persistent churn.  median: a transient step change leaves most
    bands at exactly zero and only the slow tail nonzero, so the median stays
    near zero, while persistent churn has support across all bands.  median is
    the reason to run K > 2 at all.
    """
    if agg == "max":    return band_u.amax(1)
    if agg == "mean":   return band_u.mean(1)
    if agg == "median": return band_u.median(1).values
    raise ValueError(agg)


class MultiScaleStability(nn.Module):
    """Spec v1 readout, but instability comes from a K-tape bank instead of 2."""

    def __init__(self, vocab, d_model=64, n_state=64, n_tapes=8, n_groups=None,
                 tau_min=0.2, tau_max=512.0, agg="median",
                 bias_correct=True, use_gate=True):
        super().__init__()
        # default: one instability value per channel, so u is a field with the
        # same shape as the state.  A scalar (n_groups=1) is only a global
        # temperature and cannot express selective uncertainty.
        n_groups = n_state if n_groups is None else n_groups
        assert n_state % n_groups == 0 and n_tapes >= 2
        self.V, self.N, self.G = vocab, n_state, n_groups
        self.agg, self.bias_correct, self.use_gate = agg, bias_correct, use_gate

        self.embed = nn.Embedding(vocab, d_model)
        self.write = nn.Linear(d_model, n_state)
        taus = torch.logspace(torch.log10(torch.tensor(tau_min)),
                              torch.log10(torch.tensor(tau_max)), n_tapes)
        alphas = torch.exp(-1.0 / taus)                               # tau -> decay
        self.p = nn.Parameter(torch.log(1.0 / alphas - 1.0)[:, None].repeat(1, n_state)
                              + 0.01 * torch.randn(n_tapes, n_state))
        # content stays 2 tapes wide (fastest + slowest) so readout cost is K-independent
        self.readout = nn.Linear(2 * n_state, vocab)

    def log_alpha(self):
        return -F.softplus(self.p)                                    # [K,N]

    def taus(self):
        return -1.0 / self.log_alpha().mean(-1)

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.write(self.embed(tokens))
        la = self.log_alpha()
        H = ema_bank(x, la) * bank_gains(la, L, x.device, self.bias_correct)[None]

        d = (H[:, :-1] - H[:, 1:]).view(B, -1, L, self.G, self.N // self.G)
        band_u = d.pow(2).mean(-1).sqrt()
        u = _agg(band_u, self.agg)                                    # [B,L,G]
        c = 1.0 / (1.0 + u)

        content = torch.cat([H[:, 0], H[:, -1]], -1)                  # fastest, slowest
        if not self.use_gate:
            return self.readout(content), u
        if self.G == 1:
            return self.readout(content) * c, u
        g = c.repeat_interleave(self.N // self.G, -1)
        return self.readout(torch.cat([H[:, 0] * g, H[:, -1] * g], -1)), u
