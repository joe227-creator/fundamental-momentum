import warnings, time
warnings.filterwarnings("ignore")
import numpy as np, torch
from algo_trading import timesfm_experiments as te
from algo_trading.timesfm_experiments import load_context, build_factor_state, eligible_symbols, set_config
from algo_trading.timesfm_engine import _batched_forward, _prepare_context, get_model

cfg, ctx = load_context("config/baseline.json"); set_config(cfg)
state = build_factor_state(ctx, cfg)
# eligible counts per rebalance date (only dates with an assigned split)
counts=[]
for d in state.rebalance_dates:
    if state.split_for_date.get(d) is None: continue
    e=eligible_symbols(state, d)
    counts.append(len(e))
counts=np.array(counts)
print("rebalance dates with split:", len(counts))
print("eligible per date  min/mean/median/max:", counts.min(), round(counts.mean(),1), int(np.median(counts)), counts.max())
print("TOTAL (symbol,date) forecasts needed:", int(counts.sum()))
# batch throughput test
get_model()
CL=256; HOR=21
arrs=[_prepare_context(np.random.randn(180)*0.01, CL, "logret") for _ in range(64)]
for B in [16,24,32,48]:
    try:
        t0=time.time()
        m,f=_batched_forward(arrs[:B], CL, HOR, False, batch=B)
        dt=time.time()-t0
        print(f"batch={B} ok {dt:.1f}s -> {B/dt:.1f} ser/s  mem ok")
    except Exception as e:
        print(f"batch={B} ERR {e}")
