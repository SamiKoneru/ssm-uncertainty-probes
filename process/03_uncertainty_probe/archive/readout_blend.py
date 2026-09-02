"""Is the retrieved vector S q a clean value for stable keys and a BLEND for contested?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch, torch.nn.functional as F
import induction_task as T
from delta_model import DeltaLayer

torch.manual_seed(0); rng = np.random.default_rng(0)
m = DeltaLayer(T.N_SYM, d_k=128, d_v=128, v_from_embed=True, kq_from_embed=True)
opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
for s in range(700):
    x, _, _ = T.sample(rng, 24)
    lg, _ = m(x[:, :-1])
    loss = F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
print("final loss", round(loss.item(), 3))

with torch.no_grad():
    x, con, seen = T.sample(rng, 32); L = T.L
    emb = m.embed(x[:, :-1])
    xc = F.silu(m.conv(emb.transpose(1, 2))[..., :L].transpose(1, 2))
    sh = F.pad(emb, (0, 0, 1, 0))[:, :L]
    q = F.normalize(m.to_q(emb), dim=-1); k = F.normalize(m.to_k(sh), dim=-1)
    v = m.to_v(emb); beta = torch.sigmoid(m.to_b(xc))
    B = x.shape[0]; S = emb.new_zeros(B, 128, 128); O = []
    for t in range(L):
        e = torch.bmm(S, k[:, t].unsqueeze(-1)).squeeze(-1) - v[:, t]
        S = S - beta[:, t].unsqueeze(-1) * e.unsqueeze(-1) * k[:, t].unsqueeze(1)
        O.append(torch.bmm(S, q[:, t].unsqueeze(-1)).squeeze(-1))
    O = torch.stack(O, 1)                                     # [B,L,128] retrieved vectors
    Vall = F.normalize(m.to_v(m.embed(torch.arange(T.N_SYM))), dim=-1)   # [40,128]
    cos = F.normalize(O, dim=-1) @ Vall.T                     # [B,L,40]
    top = cos.topk(4, dim=-1).values
    ent_lg = -(F.softmax(m.head(m.norm(O)), -1) * F.log_softmax(m.head(m.norm(O)), -1)).sum(-1)

c = con.bool(); s_ = seen.bool()
sel = lambda z, mask: z[mask].mean().item()
print(f"\nlearned beta: mean {beta.mean():.4f}  (1.0 = exact overwrite, <1 = partial -> blending)")
print(f"\n{'':26s}{'STABLE':>10}{'CONTESTED':>12}")
for nm, z in (("best-match cosine", top[..., 0]), ("2nd best cosine", top[..., 1]),
              ("gap (best - 2nd)", top[..., 0] - top[..., 1]),
              ("||S q||", O.norm(dim=-1)), ("output entropy", ent_lg)):
    print(f"  {nm:24s}{sel(z, s_ & ~c):>10.4f}{sel(z, s_ & c):>12.4f}")
