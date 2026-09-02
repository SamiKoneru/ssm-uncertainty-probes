import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
import numpy as np, torch, torch.nn.functional as F, kv_task
from delta_model import DeltaLayer
V, L = kv_task.configure(32, 32, 100)
torch.manual_seed(1); rng = np.random.default_rng(1)
m = DeltaLayer(V, d_k=128, d_v=128, kq_from_embed=True); opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
for s in range(500):
    seq, msk, _ = kv_task.sample(rng, 24)
    lg, _ = m(seq); sl = msk[:, :-1]
    loss = F.cross_entropy(lg[:, :-1][sl], seq[:, 1:][sl])
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
torch.save(m.state_dict(), os.path.join(_CK, "kv_model_v3.pt"))
print("saved, final loss", round(loss.item(), 5))
