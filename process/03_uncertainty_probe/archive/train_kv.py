import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from delta_model import DeltaLayer
from kv_task import sample, VOCAB, L

def run(seed=0, d_k=64, steps=800, batch=32, lr=2e-3, verbose=False):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = DeltaLayer(VOCAB, d_k=d_k, d_v=d_k)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for s in range(steps):
        seq, msk, _ = sample(rng, batch)
        lg, _ = m(seq)
        # predict token t+1 at scored positions t
        loss = F.cross_entropy(lg[:, :-1][msk[:, :-1]], seq[:, 1:][msk[:, :-1]])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if verbose and (s + 1) % 100 == 0: print(f"   step {s+1:4d} loss {loss.item():.3f}")
    return m, rng

@torch.no_grad()
def evaluate(m, rng, n=8, batch=32):
    U, Y, C, E, X, P = [], [], [], [], [], []
    for _ in range(n):
        seq, msk, con = sample(rng, batch)
        lg, u = m(seq)
        sl = msk[:, :-1]
        pr = F.softmax(lg[:, :-1], -1)
        wrong = (pr.argmax(-1) != seq[:, 1:]).float()
        ent = -(pr * torch.log(pr + 1e-9)).sum(-1)
        pos = torch.arange(L - 1).float().expand(batch, -1)
        U.append(u[:, :-1][sl]); Y.append(wrong[sl]); C.append(con[:, :-1][sl])
        E.append(ent[sl]); X.append((1 - pr.max(-1).values)[sl]); P.append(pos[sl])
    f = lambda z: torch.cat(z).numpy()
    return f(U), f(Y), f(C).astype(float), f(E), f(X), f(P)

if __name__ == "__main__":
    import sys
    if "--quick" in sys.argv:
        m, rng = run(steps=250, verbose=True)
        u, y, c, e, x, p = evaluate(m, rng, n=4)
        print(f"\n  accuracy overall   {1-y.mean():.3f}")
        print(f"  accuracy STABLE    {1-y[c==0].mean():.3f}")
        print(f"  accuracy CONTESTED {1-y[c==1].mean():.3f}   (chance = {1/32:.3f})")
