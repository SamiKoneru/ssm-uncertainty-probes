import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
import uncertainty_test as U
torch.manual_seed(0); np.random.seed(0)
proc = U.StochasticProcess(vocab=8, seed=0)
model = U.UnitarySSM(); model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model.pt"))); model.eval()
with torch.no_grad():
    x, isr = U.make_batch(proc, 200, 201)
    inp, tgt = x[:, :-1], x[:, 1:]
    logits, mag = model(inp)
    wrong = (logits.argmax(-1) != tgt).float()
    det = (isr[:, 1:] == 0)
print(" t-range   acc(det steps)   mean|h|")
for lo in range(0, 200, 25):
    sl = slice(lo, lo + 25)
    w, d = wrong[:, sl], det[:, sl]
    print(f"  {lo:3d}-{lo+24:3d}      {1 - w[d].mean():.3f}        {mag[:, sl].mean():.2f}")
