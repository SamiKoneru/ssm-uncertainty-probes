"""
Experiment for the two-timescale stability memory.

Same process as the earlier test: deterministic runs (predictable) interleaved
with uniformly random steps (genuinely uncertain), ground truth recorded per
step.  Train and eval use the SAME sequence length; the earlier run trained at
120 and evaluated at 200, and the resulting length-extrapolation collapse was
most of what it measured.

Signals compared at predicting per-step error:
    u_t (instability)          the candidate
    output entropy             baseline
    1 - max softmax            baseline
    position t                 TRIVIAL CONTROL -- if this wins, nothing else matters
    u_t, position-controlled   u z-scored within each timestep

Also reported: AUROC against `is_random`, which is the actual per-step ground
truth, and a `use_gate=False` ablation where u is computed but never multiplied
into the logits, separating "u is intrinsically informative" from "u was
trained into a temperature".
"""

import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from stability_ssm import TwoTimescaleMemory

V, LEN, N_STATE = 8, 200, 64
WARM = 5


def sample(rng, det_next, B, L):
    """Vectorized over the batch; identical process to StochasticProcess."""
    seq = np.zeros((B, L), np.int64); isr = np.zeros((B, L), np.int64)
    seq[:, 0] = rng.integers(V, size=B)
    run_left = rng.integers(3, 8, size=B); in_det = np.ones(B, bool)
    for t in range(1, L):
        use_det = in_det & (run_left > 0)
        seq[:, t] = np.where(use_det, det_next[seq[:, t - 1]], rng.integers(V, size=B))
        isr[:, t] = ~use_det
        run_left = np.where(use_det, run_left - 1, run_left)
        in_det = np.where(use_det & (run_left == 0), False, in_det)
        restart = (~use_det) & (rng.random(B) < 0.6)
        in_det = np.where(restart, True, in_det)
        run_left = np.where(restart, rng.integers(3, 8, size=B), run_left)
    return torch.from_numpy(seq), torch.from_numpy(isr)


def run_seed(seed, use_gate, bias_correct=True, steps=1500, batch=64):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    det_next = rng.permutation(V)
    model = TwoTimescaleMemory(V, n_state=N_STATE, use_gate=use_gate,
                              bias_correct=bias_correct)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(steps):
        x, _ = sample(rng, det_next, batch, LEN + 1)
        logits, _ = model(x[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    model.eval()
    with torch.no_grad():
        x, isr = sample(rng, det_next, 400, LEN + 1)
        inp, tgt, rnd = x[:, :-1], x[:, 1:], isr[:, 1:]
        logits, u = model(inp)
        probs = F.softmax(logits, -1)
        wrong = (probs.argmax(-1) != tgt).float()
        ent = -(probs * torch.log(probs + 1e-9)).sum(-1)
        af, asl = model.alphas()

    B, L = wrong.shape
    flat = lambda z: z[:, WARM:].reshape(-1).numpy()
    y, r = flat(wrong), flat(rnd)
    us, e = flat(u[..., 0]), flat(ent)
    mx = flat(1 - probs.max(-1).values)
    pos = flat(torch.arange(L).unsqueeze(0).expand(B, L).float())
    uu = u[:, WARM:, 0].numpy()
    uz = ((uu - uu.mean(0, keepdims=True)) / (uu.std(0, keepdims=True) + 1e-9)).reshape(-1)

    lo = e < np.percentile(e, 33)                       # most-confident third
    cw, cr = lo & (y > .5), lo & (y <= .5)
    sub = np.nan
    if cw.sum() > 10 and cr.sum() > 10:
        sub = roc_auc_score(np.r_[np.ones(cw.sum()), np.zeros(cr.sum())],
                            np.r_[uz[cw], uz[cr]])
    return dict(
        loss=loss.item(), acc=1 - y.mean(),
        acc_det=1 - y[r == 0].mean(), acc_rnd=1 - y[r == 1].mean(),
        a_f=af.mean().item(), a_s=asl.mean().item(),
        err_u=roc_auc_score(y, us), err_ent=roc_auc_score(y, e),
        err_mx=roc_auc_score(y, mx), err_pos=roc_auc_score(y, pos),
        err_uz=roc_auc_score(y, uz),
        rnd_u=roc_auc_score(r, uz), rnd_ent=roc_auc_score(r, e),
        conf_uz=sub, n_cw=int(cw.sum()))


def table(name, rows):
    ks = ["loss", "acc", "acc_det", "acc_rnd", "a_f", "a_s", "err_u", "err_uz",
          "err_ent", "err_mx", "err_pos", "rnd_u", "rnd_ent", "conf_uz"]
    print(f"\n===== {name}  ({len(rows)} seeds, mean +/- std) =====")
    for k in ks:
        v = np.array([r[k] for r in rows], float)
        tag = ""
        if k == "err_uz": tag = "   <-- u, position-controlled"
        if k == "err_pos": tag = "   <-- trivial control"
        if k == "conf_uz": tag = "   <-- key sub-test"
        print(f"  {k:9s} {np.nanmean(v):7.3f} +/- {np.nanstd(v):.3f}{tag}")


if __name__ == "__main__":
    seeds = [0, 1, 2]
    table("GATED (spec: y = Readout(content) * c)", [run_seed(s, True) for s in seeds])
    table("ABLATION use_gate=False (u computed, never used)",
          [run_seed(s, False) for s in seeds])
    table("NO BIAS CORRECTION (spec's steady-state gain, gated)",
          [run_seed(s, True, bias_correct=False) for s in seeds])
