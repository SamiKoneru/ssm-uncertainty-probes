"""Control experiments for the state-magnitude result."""
import numpy as np, torch, torch.nn.functional as F, os
from sklearn.metrics import roc_auc_score
import uncertainty_test as U

torch.manual_seed(0); np.random.seed(0)
proc = U.StochasticProcess(vocab=8, seed=0)
model = U.UnitarySSM(vocab=8, d_model=64, n_state=64)

if os.path.exists("model.pt"):
    model.load_state_dict(torch.load("model.pt"))
else:
    U.train(model, proc, steps=2000, batch=64, length=120)
    torch.save(model.state_dict(), "model.pt")

model.eval()
with torch.no_grad():
    x, is_rand = U.make_batch(proc, 200, 201)
    inp, tgt = x[:, :-1], x[:, 1:]
    tgt_rand = is_rand[:, 1:]                 # was THIS target drawn randomly?
    logits, mag = model(inp)
    probs = F.softmax(logits, -1)
    wrong = (probs.argmax(-1) != tgt).float()
    ent = -(probs * torch.log(probs + 1e-9)).sum(-1)

warm = 5
B, L = wrong.shape
flat = lambda z: z[:, warm:].reshape(-1).numpy()
y, m, e, r = flat(wrong), flat(mag), flat(ent), flat(tgt_rand)
pos = flat(torch.arange(L).unsqueeze(0).expand(B, L).float())

print(f"overall accuracy           : {1-y.mean():.3f}")
print(f"accuracy on DETERMINISTIC  : {1-y[r==0].mean():.3f}   (should be ~1.00)")
print(f"accuracy on RANDOM         : {1-y[r==1].mean():.3f}   (chance = 0.125)")
print(f"mean entropy det / random  : {e[r==0].mean():.3f} / {e[r==1].mean():.3f}  (want det << random)")
print(f"fraction of random steps   : {r.mean():.3f}")

print(f"\ncorr(state_magnitude, position t) = {np.corrcoef(m, pos)[0,1]:.4f}")
print(f"magnitude at t=10 / t=100 / t=195 : "
      f"{mag[:,10].mean():.2f} / {mag[:,100].mean():.2f} / {mag[:,195].mean():.2f}")

print("\n--- AUROC for predicting ERROR ---")
print(f"  state_magnitude                  {roc_auc_score(y, m):.3f}")
print(f"  raw position t (TRIVIAL CONTROL) {roc_auc_score(y, pos):.3f}")

# de-confound: z-score magnitude WITHIN each timestep, so position is removed
mm = mag[:, warm:].numpy()
mz = ((mm - mm.mean(0, keepdims=True)) / (mm.std(0, keepdims=True) + 1e-9)).reshape(-1)
print(f"  magnitude, position-controlled   {roc_auc_score(y, mz):.3f}   <-- the real number")

print("\n--- AUROC for predicting 'this step was RANDOM' (the ground truth) ---")
print(f"  state_magnitude                  {roc_auc_score(r, m):.3f}")
print(f"  magnitude, position-controlled   {roc_auc_score(r, mz):.3f}")
print(f"  output_entropy                   {roc_auc_score(r, e):.3f}")
