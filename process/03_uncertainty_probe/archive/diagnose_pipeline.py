"""Where does the pipeline break: at the residual, or at the R readout?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")

import numpy as np, torch, torch.nn.functional as F, kv_task
from delta_model import DeltaLayer

V, L = kv_task.configure(32, 32, 100)
torch.manual_seed(1); rng = np.random.default_rng(1)
m = DeltaLayer(V, d_k=128, d_v=128)
m.load_state_dict(torch.load(os.path.join(_CK, 'kv_model.pt'))); m.eval()

seq, msk, con = kv_task.sample(rng, 32)
B = seq.shape[0]
with torch.no_grad():
    x = m.embed(seq).transpose(1, 2)
    x = F.silu(m.conv(x)[..., :L].transpose(1, 2))
    q = F.normalize(m.to_q(x), dim=-1); k = F.normalize(m.to_k(x), dim=-1)
    v, beta = m.to_v(x), torch.sigmoid(m.to_b(x))
    S = x.new_zeros(B, 128, 128); E = []
    for t in range(L):
        e = torch.bmm(S, k[:, t].unsqueeze(-1)).squeeze(-1) - v[:, t]
        S = S - beta[:, t].unsqueeze(-1) * e.unsqueeze(-1) * k[:, t].unsqueeze(1)
        E.append(e.norm(dim=-1))
    E = torch.stack(E, 1)

pos = torch.arange(L)
is_op, is_key, is_val = pos % 3 == 0, pos % 3 == 1, pos % 3 == 2
op_of = seq[:, pos - (pos % 3)]                       # the op token for each position
is_write = (op_of == 0)
key_idx = seq[:, (pos // 3) * 3 + 1] - kv_task.TOK_KEY
contested = key_idx >= kv_task.N_KEYS // 2

print("mean beta (write gate) by token slot:")
for nm, sel in (("op", is_op), ("key", is_key), ("value", is_val)):
    print(f"   {nm:6s} {beta[:, sel, 0].mean():.4f}")

wv = is_val.unsqueeze(0) & is_write                    # value slot of a WRITE op
print(f"\nresidual ||e_t|| at WRITE-value slots:")
print(f"   stable keys    {E[wv & ~contested].mean():.4f}")
print(f"   contested keys {E[wv &  contested].mean():.4f}")
r = E[wv & contested].mean() / E[wv & ~contested].mean()
print(f"   ratio          {r:.3f}x   <- is the signal there AT ALL?")

kk = k[:, is_key]                                      # key vectors at key slots
ki = key_idx[:, is_key]
G = torch.einsum('bid,bjd->bij', kk, kk)
same_m = ki.unsqueeze(2) == ki.unsqueeze(1)
off = ~torch.eye(kk.shape[1], dtype=bool).unsqueeze(0).expand_as(G)
print(f"\ncosine between learned k vectors at key slots (||k||={kk.norm(dim=-1).mean():.3f}):")
print(f"   SAME task key      {G[same_m & off].mean():.4f}")
print(f"   DIFFERENT task key {G[~same_m & off].mean():.4f}")
print(f"   separation         {(G[same_m & off].mean()-G[~same_m & off].mean()):.4f}  <- addressing aligned?")
