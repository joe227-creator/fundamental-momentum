"""Per-date TimesFM forecast worker (model-only process).

Usage: python forecast_worker.py YYYY-MM-DD [ctx_len] [horizon] [kind] [field]

Loads the request parquet built by _prepare_requests.py, forecasts every
eligible stock for the SINGLE given date, and appends to the engine's cache
parquet (cache/timesfm/ctx{C}_hor{H}_{kind}_{q}.parquet). One date per process
so a segfault / OOM only loses one date and the driver resumes from cache.
"""
import os, sys, warnings, pickle, time
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from algo_trading.timesfm_engine import forecast_panel, _cache_key, CACHE_ROOT


def main():
    date = pd.Timestamp(sys.argv[1]).normalize()
    ctx_len = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 21
    kind = sys.argv[4] if len(sys.argv) > 4 else "logret"
    field = sys.argv[5] if len(sys.argv) > 5 else "close"
    quantiles = os.environ.get("TFM_QUANTILES", "0") == "1"
    batch = int(os.environ.get("TFM_BATCH", "1"))

    req_path = CACHE_ROOT / f"requests_ctx{ctx_len}_{kind}_{field}.parquet"
    if not req_path.exists():
        print(f"NO REQUESTS FILE {req_path}", flush=True); sys.exit(2)
    req = pd.read_parquet(req_path)
    req["date"] = pd.to_datetime(req["date"]).dt.normalize()
    sub = req[req["date"] == date]
    if sub.empty:
        print(f"DATE {date.date()} not in requests (maybe no eligible stocks)", flush=True); sys.exit(0)
    # already cached?
    cache_path = _cache_key(ctx_len if ctx_len % 32 == 0 else (32*((ctx_len+31)//32)), horizon, kind, quantiles)
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if not isinstance(cached.index, pd.MultiIndex):
            cached = cached.set_index(["symbol", "date"])
        if (date in [d for _, d in cached.index.get_level_values("date").unique()] if False else (cached.index.get_level_values("date") == date).any()):
            have = cached.loc[cached.index.get_level_values("date") == date]
            if len(have) >= len(sub):
                print(f"CACHED {date.date()} ({len(have)} rows)", flush=True); sys.exit(0)

    sub = sub.copy()
    sub["series"] = sub["series"].apply(pickle.loads)  # bytes -> np.ndarray(float32)
    t0 = time.time()
    # process in small groups, persisting after each so a crash loses little
    step = 25
    done = 0
    for i in range(0, len(sub), step):
        chunk = sub.iloc[i:i+step]
        fp = forecast_panel(chunk[["symbol", "date", "series"]], context_len=ctx_len, horizon=horizon, kind=kind, quantiles=quantiles, batch=batch)
        done += len(chunk)
        print(f"  {date.date()} {done}/{len(sub)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"OK {date.date()} rows={done} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
