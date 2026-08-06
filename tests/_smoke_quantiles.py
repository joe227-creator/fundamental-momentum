import os, numpy as np, torch
os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
torch.set_num_threads(1)
from transformers import TimesFm2_5ModelForPrediction
torch.manual_seed(42); np.random.seed(42)
m=TimesFm2_5ModelForPrediction.from_pretrained("timesfm-2.5-200m-transformers").to(torch.float32).eval()
CL=256
# monotone trend so quantiles should be ordered
x=np.linspace(0,1,CL)+np.random.randn(CL)*0.01
with torch.no_grad():
    o=m(past_values=[torch.tensor(x,dtype=torch.float32)], forecast_context_len=CL, horizon_length=128)
mp=o.mean_predictions[0].numpy()        # (128,)
fp=o.full_predictions[0].numpy()        # (128,10)
print("mean shape",mp.shape,"full shape",fp.shape)
# Is one column == mean?
for q in range(fp.shape[1]):
    print(f"col{q}: mean_over_horizon={fp[:,q].mean():.5f}  match_mean={np.allclose(fp[:,q],mp,atol=1e-5)}  first_step={fp[0,q]:.5f}")
print("mp first 3:", mp[:3])
# Are columns monotone increasing in q at each step? (quantile ordering)
order=np.argsort(fp[0,:])
print("argorder at step0:", order)
print("col means ascending?:", list(np.round(fp.mean(axis=0),5)))
