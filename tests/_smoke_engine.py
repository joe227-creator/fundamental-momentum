import numpy as np, pandas as pd, time
from algo_trading.timesfm_engine import forecast_panel, mean_forecast_sum, DOWNSIDE_COL, UPSIDE_COL
np.random.seed(42)
rows=[]
for sym in ["A","B","C"]:
    for d in pd.date_range("2024-01-31", periods=2, freq="ME"):
        lr=np.diff(np.log(np.abs(np.cumsum(np.random.randn(180)*0.01)+10)))  # log-returns
        rows.append({"symbol":sym,"date":d,"series":lr})
req=pd.DataFrame(rows)
t0=time.time()
p=forecast_panel(req, context_len=128, horizon=21, kind="logret", quantiles=True)
print("first call", round(time.time()-t0,1),"s  shape",p.shape, "cols", list(p.columns)[:5],"...")
print("mean_sum:"); print(mean_forecast_sum(p,21))
t0=time.time()
p2=forecast_panel(req, context_len=128, horizon=21, kind="logret", quantiles=True)
print("cached call", round(time.time()-t0,2),"s")
print("identical:", np.allclose(p.values,p2.values))
pq=forecast_panel(req, context_len=128, horizon=21, kind="logret", quantiles=True)
print("downside col0 first row:", pq[[f"q{DOWNSIDE_COL}_h{i}" for i in range(3)]].iloc[0].tolist())
print("median col5 first row:", pq[[f"q5_h{i}" for i in range(3)]].iloc[0].tolist())
print("upside col9 first row:", pq[[f"q{UPSIDE_COL}_h{i}" for i in range(3)]].iloc[0].tolist())
