"""All diagnostics on one checkpoint. usage: full_diag.py <ckpt> <v_from_embed 0|1>"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys, numpy as np, torch, torch.nn.functional as F, kv_task
from sklearn.metrics import roc_auc_score
from delta_model import DeltaLayer

ckpt, vfe = sys.argv[1], bool(int(sys.argv[2]))
kqe = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
V, L = kv_task.configure(32, 32, 100)
m = DeltaLayer(V, d_k=128, d_v=128, v_from_embed=vfe, kq_from_embed=kqe)
m.load_state_dict(torch.load(ckpt)); m.eval()
rng = np.random.default_rng(11)

vcos_s, vcos_d, kcos_s, kcos_d = [], [], [], []
by_write = {}; res_st, res_con = [], []
orc_last, orc_ema, sup_acc, sup_wht, CC = [], [], [], [], []
accs = []

with torch.no_grad():
  for it in range(10):
    seq, msk, con = kv_task.sample(rng, 32); B = seq.shape[0]
    emb = m.embed(seq)
    x = F.silu(m.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))
    if kqe:
        sh = F.pad(emb, (0, 0, 1, 0))[:, :L]
        q = F.normalize(m.to_q(emb), dim=-1); k = F.normalize(m.to_k(sh), dim=-1)
    else:
        q = F.normalize(m.to_q(x), dim=-1); k = F.normalize(m.to_k(x), dim=-1)
    v = m.to_v(emb if vfe else x); beta = torch.sigmoid(m.to_b(x))
    S = x.new_zeros(B, 128, 128); Ra = x.new_zeros(B, 128)
    Cov = torch.eye(128).expand(B, -1, -1).clone() * 1e-3
    E, ua, uw, outs = [], [], [], []
    wq = q[:, 0]
    for t in range(L):
        kt = k[:, t]
        e = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1) - v[:, t]
        S = S - beta[:, t].unsqueeze(-1) * e.unsqueeze(-1) * kt.unsqueeze(1)
        outs.append(torch.bmm(S, q[:, t].unsqueeze(-1)).squeeze(-1))
        n = e.norm(dim=-1, keepdim=True); E.append(n.squeeze(-1))
        Ra = 0.99 * Ra + n * kt
        Cov = 0.99 * Cov + torch.bmm(kt.unsqueeze(-1), kt.unsqueeze(1))
        if t % 3 == 1:
            reg = Cov + 1e-2 * Cov.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None] * torch.eye(128)
            wq = F.normalize(torch.linalg.solve(reg, q[:, t].unsqueeze(-1)).squeeze(-1), dim=-1)
        ua.append((Ra * q[:, t]).sum(-1).abs()); uw.append((Ra * wq).sum(-1).abs())
    E = torch.stack(E, 1).numpy()
    lg = m.head(m.norm(torch.stack(outs, 1))); sl = msk[:, :-1]
    accs.append((lg[:, :-1].argmax(-1) == seq[:, 1:]).float()[sl].mean().item())
    ua, uw = torch.stack(ua, 1), torch.stack(uw, 1)

    pos = torch.arange(L); isval = pos % 3 == 2; iskey = pos % 3 == 1
    op_of = seq[:, (pos // 3) * 3]; kidx = (seq[:, (pos // 3) * 3 + 1] - kv_task.TOK_KEY).numpy()
    vtok = seq[:, (pos // 3) * 3 + 2]
    if it == 0:
        vv = F.normalize(v[:, isval], dim=-1); vt = vtok[:, isval]
        G = torch.einsum('bid,bjd->bij', vv, vv); sm = vt.unsqueeze(2) == vt.unsqueeze(1)
        off = ~torch.eye(vv.shape[1], dtype=bool).unsqueeze(0).expand_as(G)
        vcos_s, vcos_d = G[sm & off].mean().item(), G[~sm & off].mean().item()
        kk = k[:, iskey]; ki = torch.from_numpy(kidx)[:, iskey]
        Gk = torch.einsum('bid,bjd->bij', kk, kk); smk = ki.unsqueeze(2) == ki.unsqueeze(1)
        offk = ~torch.eye(kk.shape[1], dtype=bool).unsqueeze(0).expand_as(Gk)
        kcos_s, kcos_d = Gk[smk & offk].mean().item(), Gk[~smk & offk].mean().item()

    for b in range(B):
        cnt = {}; ema = {}; last = {}
        for o in range(kv_task.N_OPS):
            i = 3 * o; kk_ = int(kidx[b, i]); stable = kk_ < kv_task.N_KEYS // 2
            if op_of[b, i] == 1:
                orc_last.append(last.get(kk_, 0.0)); orc_ema.append(ema.get(kk_, 0.0))
                sup_acc.append(float(ua[b, i + 1])); sup_wht.append(float(uw[b, i + 1]))
                CC.append(float(not stable))
            else:
                r = float(E[b, i + 2]); cnt[kk_] = cnt.get(kk_, 0) + 1
                ema[kk_] = 0.7 * ema.get(kk_, r) + 0.3 * r; last[kk_] = r
                (res_st if stable else res_con).append(r)
                if stable: by_write.setdefault(cnt[kk_], []).append(r)

c = np.array(CC)
print(f"=== {ckpt}   (v_from_embed={vfe}, kq_from_embed={kqe}) ===")
print(f"accuracy {np.mean(accs):.4f}")
print(f"\ncosine, v_t  same value token {vcos_s:.4f}   different {vcos_d:.4f}")
print(f"cosine, k_t  same task key    {kcos_s:.4f}   different {kcos_d:.4f}")
print(f"\nresidual at n-th write of a STABLE key:")
for n in sorted(by_write)[:6]:
    print(f"   #{n}: {np.mean(by_write[n]):.4f}  (n={len(by_write[n])})")
print(f"\nresidual at writes: stable {np.mean(res_st):.4f}  contested {np.mean(res_con):.4f}"
      f"   ratio {np.mean(res_con)/np.mean(res_st):.3f}x")
print(f"\nAUROC (contested key):")
for nme, u in (("ORACLE last residual", orc_last), ("ORACLE ema", orc_ema),
               ("superposition accum", sup_acc), ("superposition +whiten", sup_wht)):
    u = np.array(u); print(f"   {nme:24s} {roc_auc_score(c, u):.3f}"
                           f"   stable {u[c==0].mean():8.4f}  contested {u[c==1].mean():8.4f}")
