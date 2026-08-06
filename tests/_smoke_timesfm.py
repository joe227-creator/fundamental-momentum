import time, os
os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
os.environ["OMP_NUM_THREADS"]="4"; os.environ["MKL_NUM_THREADS"]="4"
import numpy as np, torch
torch.set_num_threads(4)
from transformers import TimesFm2_5ModelForPrediction
torch.manual_seed(42); np.random.seed(42)
mdl = TimesFm2_5ModelForPrediction.from_pretrained("timesfm-2.5-200m-transformers").to(torch.float32).eval()
CL=256
def mk(n): return np.cumsum(np.random.randn(CL)*0.01)+0.0002*CL
B=10
pv=[torch.tensor(mk(i),dtype=torch.float32) for i in range(B)]
try:
    t0=time.time()
    with torch.no_grad():
        o=mdl(past_values=pv, forecast_context_len=CL, horizon_length=21)
    print("B=10 ok", round(time.time()-t0,1),"s")
    print("mean", tuple(o.mean_predictions.shape))
    print("full", tuple(o.full_predictions.shape))
except Exception as e:
    print("err", e)
