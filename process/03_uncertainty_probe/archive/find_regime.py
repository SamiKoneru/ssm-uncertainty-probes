import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F, kv_task
from delta_model import DeltaLayer
for nk, nops in ((24, 90), (32, 100), (48, 120)):
    V, L = kv_task.configure(nk, 32, nops)
    torch.manual_seed(0); rng = np.random.default_rng(0)
    m = DeltaLayer(V, d_k=64, d_v=64); opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    accs = []
    for s in range(900):
        seq, msk, _ = kv_task.sample(rng, 24)
        lg, _ = m(seq); sl = msk[:, :-1]
        loss = F.cross_entropy(lg[:, :-1][sl], seq[:, 1:][sl])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (s + 1) % 300 == 0:
            with torch.no_grad():
                sq, mk, cn = kv_task.sample(rng, 32); lo, _ = m(sq); s2 = mk[:, :-1]
                w = (lo[:, :-1].argmax(-1) != sq[:, 1:]).float()[s2].numpy()
                c = cn[:, :-1][s2].numpy()
            accs.append((s + 1, 1 - w.mean(), 1 - w[c == 0].mean(), 1 - w[c == 1].mean()))
    print(f"{nk} keys, {nops} ops, L={L}:")
    for st, a, ast, ac in accs:
        print(f"   step {st:4d}  acc {a:.3f}   stable {ast:.3f}   contested {ac:.3f}")
