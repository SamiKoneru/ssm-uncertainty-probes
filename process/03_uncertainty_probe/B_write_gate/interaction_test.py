"""
The NOISY/REVISED signal exists in s1 but ONLY among CHANGE writes (AUROC 0.834);
pooled with repeats it inverts (0.353). So the gate needs beta's response to s1
to flip on whether s2 says 'repeat'.

  mlp         [base, s1, s2] -> 16 -> 1               (current best)
  mlp_inter   [base, s1, s2, s1*s2] -> 16 -> 1        explicit interaction
  mlp_wide    [base, s1, s2] -> 64 -> 64 -> 1         capacity control:
                                                       can it FIND the interaction?

Diagnostic that matters: beta on CHANGE writes, split by class. The oracle wants
beta_NOISY < beta_REVISED there.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch, torch.nn.functional as F
import mixed_task as T
from gated_model import GatedDeltaLayer

BASE = 1.6329

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
    L_, A, B_, CQ, CW, SN, CH = [], [], [], [], [], [], []
    ev = np.random.default_rng(12345)
    with torch.no_grad():
        for _ in range(6):
            x, cl, sn = T.sample(ev, 32); B, L = x.shape[0], T.L
            lg, beta = m(x[:, :-1])
            L_.append(F.cross_entropy(lg.reshape(-1, T.N_SYM), x[:, 1:].reshape(-1)).item())
            A.append((lg.argmax(-1) == x[:, 1:]).float()); B_.append(beta)
            CQ.append(cl); CW.append(F.pad(cl, (1, 0))[:, :L]); SN.append(sn)
            xs = x.numpy(); chg = np.zeros((B, L), bool)
            for b_ in range(B):
                last = {}
                for t in range(1, L):
                    kk = int(xs[b_, t - 1]); val = int(xs[b_, t])
                    chg[b_, t] = (kk in last) and last[kk] != val
                    last[kk] = val
            CH.append(torch.from_numpy(chg))
    f = lambda z: torch.cat(z).reshape(-1).numpy()
    return (np.mean(L_), f(A), f(B_), f(CQ), f(CW), f(SN).astype(bool), f(CH).astype(bool))

if __name__ == "__main__":
    print(f"{'gate':>11}{'loss':>9}{'acc':>8}{'gain':>8} |{'accREV':>8}"
          f" | beta on CHANGE writes  |  beta overall")
    print(f"{'':>11}{'':>9}{'':>8}{'':>8} |{'':>8}"
          f" |{'STAB':>7}{'NOIS':>7}{'REV':>7} |{'STAB':>7}{'NOIS':>7}{'REV':>7}")
    for gate in ("mlp", "mlp_inter", "mlp_wide"):
        loss, acc, bt, cq, cw, sn, ch = run(gate)
        arev = acc[(cq == 2) & sn].mean()
        bc = [bt[(cw == i) & sn & ch].mean() for i in range(3)]
        bo = [bt[(cw == i) & sn].mean() for i in range(3)]
        flag = "  <- NOIS<REV, correct!" if bc[1] < bc[2] else ""
        print(f"{gate:>11}{loss:>9.4f}{acc.mean():>8.4f}{100*(BASE-loss)/BASE:>7.2f}%"
              f" |{arev:>8.3f} |{bc[0]:>7.3f}{bc[1]:>7.3f}{bc[2]:>7.3f}"
              f" |{bo[0]:>7.3f}{bo[1]:>7.3f}{bo[2]:>7.3f}{flag}", flush=True)
    print(f"\nreference: baseline loss {BASE}, acc 0.6347, accREV 0.565")
    print("oracle_class reached loss 1.5442 (+5.42%), acc 0.6552, accREV 0.662")
