import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
import mixed_task as T
from gated_model import GatedDeltaLayer

def run(gate, seed=0, steps=800, batch=24):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = GatedDeltaLayer(T.N_SYM, gate=gate)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for s in range(steps):
        x, _, _ = T.sample(rng, batch)
        lg, _ = m(x[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()

    losses, A, B_, C_, W_, S_ = [], [], [], [], [], []
    ev = np.random.default_rng(12345)                       # SAME eval stream for every gate
    with torch.no_grad():
        for _ in range(6):
            x, cl, sn = T.sample(ev, 32)
            lg, beta = m(x[:, :-1])
            losses.append(F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1)).item())
            A.append((lg.argmax(-1) == x[:, 1:]).float()); B_.append(beta)
            C_.append(cl)                                   # class(x_t): governs ACCURACY
            W_.append(F.pad(cl, (1, 0))[:, :T.L])           # class(x_{t-1}): governs BETA
            S_.append(sn)
    f = lambda z: torch.cat(z).reshape(-1).numpy()
    return np.mean(losses), f(A), f(B_), f(C_), f(W_), f(S_).astype(bool), m

if __name__ == "__main__":
    print("beta columns labelled by class of x_{t-1}, the WRITE key\n")
    print(f"{'gate':>9}{'loss':>9}{'acc':>8} | {'acc by class':>26} | {'mean beta by class':>28}")
    print(f"{'':>9}{'':>9}{'':>8} | {'STABLE':>8}{'NOISY':>9}{'REVISED':>9} |"
          f"{'STABLE':>9}{'NOISY':>9}{'REVISED':>10}")
    for gate in ("none", "both", "mlp"):
        loss, acc, bt, cl, cw, sn, m = run(gate)
        a = [acc[(cl == i) & sn].mean() for i in range(3)]
        b = [bt[(cw == i) & sn].mean() for i in range(3)]
        print(f"{gate:>9}{loss:>9.4f}{acc.mean():>8.4f} | {a[0]:>8.3f}{a[1]:>9.3f}{a[2]:>9.3f} |"
              f"{b[0]:>9.4f}{b[1]:>9.4f}{b[2]:>10.4f}", flush=True)
        if gate != "none":
            extra = f", a2 = {m.ga2.item():+.4f}" if gate == "both" else ""
            print(f"{'':>9}(gate a = {m.ga.item():+.4f}, b = {m.gb.item():+.4f}{extra})",
                  flush=True)
    print("\nclass is randomised PER SEQUENCE, so the token carries no class info:")
    print("gate='none' CANNOT vary beta by class. Any spread in its beta columns is noise.")
