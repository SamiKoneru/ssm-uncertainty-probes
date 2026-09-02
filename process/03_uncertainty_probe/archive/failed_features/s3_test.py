"""
Does s3 = ||stored|| / ||v||  separate NOISY from REVISED, where s1 and s2 fail?

Prediction: the memory at a REVISED key is COMMITTED to one value (reinforced by
a geometric run of ~1/0.35 = 2.9 repeats before it jumps), so ||stored|| is
large.  At a NOISY key it is a BLEND of 4 near-orthogonal options that partially
cancel, so ||stored|| is small.  Both are functions of the PAST only.

Measured on an UNGATED model so beta is uniform and this is an intrinsic
property of the task, not a feedback artifact of the gate.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np, torch, torch.nn.functional as F
import mixed_task as T
from gated_model import GatedDeltaLayer

torch.manual_seed(0); rng = np.random.default_rng(0)
m = GatedDeltaLayer(T.N_SYM, gate="none")
opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
for s in range(800):
    x, _, _ = T.sample(rng, 24)
    lg, _ = m(x[:, :-1])
    loss = F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
print(f"ungated model trained, loss {loss.item():.4f}\n", flush=True)

cols = {n: [] for n in ("ne", "prj", "s3", "nst", "clw", "sn")}
ev = np.random.default_rng(777)
with torch.no_grad():
    for _ in range(4):
        x, cl, sn = T.sample(ev, 32); B, L = x.shape[0], T.L
        emb = m.embed(x[:, :-1])
        xc = F.silu(m.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))
        sh = F.pad(emb, (0, 0, 1, 0))[:, :L]
        k = F.normalize(m.to_k(sh), dim=-1); v = m.to_v(emb)
        beta = torch.sigmoid(m.to_b(xc))[..., 0]
        S = emb.new_zeros(B, 128, 128)
        c = {n: [] for n in ("ne", "prj", "s3", "nst")}
        for t in range(L):
            kt, vt = k[:, t], v[:, t]
            stored = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1)
            e = stored - vt
            c["ne"].append(e.norm(dim=-1))
            c["prj"].append((vt * stored).sum(-1) / (vt.norm(dim=-1) ** 2 + 1e-6))
            c["s3"].append(stored.norm(dim=-1) / (vt.norm(dim=-1) + 1e-6))
            c["nst"].append(stored.norm(dim=-1))
            S = S - beta[:, t, None, None] * e.unsqueeze(-1) * kt.unsqueeze(1)
        for n in c: cols[n].append(torch.stack(c[n], 1))
        cols["clw"].append(F.pad(cl, (1, 0))[:, :L]); cols["sn"].append(sn)
D = {n: torch.cat(v_).numpy() for n, v_ in cols.items()}
clw, sn = D["clw"], D["sn"].astype(bool)

print("by CLASS OF THE WRITE KEY (only keys seen before)")
print(f"{'statistic':30s}{'STABLE':>10}{'NOISY':>10}{'REVISED':>10}{'NOISY-REVISED':>16}")
for st, lab in (("ne", "s1  ||e||"), ("prj", "s2  overlap"),
                ("nst", "    ||stored||"), ("s3", "s3  ||stored||/||v||")):
    a = D[st]; v_ = [a[(clw == i) & sn].mean() for i in range(3)]
    flag = ""
    if st in ("s3", "nst"):
        flag = "  <- want NEGATIVE" if v_[1] < v_[2] else "  <- WRONG SIGN"
    print(f"  {lab:28s}{v_[0]:>10.4f}{v_[1]:>10.4f}{v_[2]:>10.4f}{v_[1]-v_[2]:>+16.4f}{flag}")

# empirical check of the reinforcement story
seq_rep = {1: [], 2: []}
print("\nempirical run length (consecutive occurrences writing the SAME value):")
ev2 = np.random.default_rng(31)
x, cl, sn = T.sample(ev2, 16)
for b in range(16):
    last = {}; run = {}
    for t in range(1, T.L):
        s_ = int(x[b, t - 1]); nxt = int(x[b, t]); c_ = int(cl[b, t - 1])
        if c_ == 0: continue
        if last.get(s_) == nxt: run[s_] = run.get(s_, 1) + 1
        else:
            if s_ in run: seq_rep[c_].append(run[s_])
            run[s_] = 1
        last[s_] = nxt
for c_, nm in ((1, "NOISY"), (2, "REVISED")):
    r = np.array(seq_rep[c_]); print(f"  {nm:8s} mean {r.mean():.3f}  (n={len(r)})")
print("  predicted: NOISY ~1.33 (1/(1-1/4)), REVISED ~2.86 (1/0.35)")
