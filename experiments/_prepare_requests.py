"""Build and persist TimesFM context requests (pandas-only, NO model loaded).

Produces cache/timesfm/requests_ctx{CTX}_{kind}.parquet with columns
[symbol, date, series]. The forecast worker consumes this so the model and the
heavy factor-state never share a process (8GB-RAM Windows constraint).
"""
import sys, time, warnings, pickle
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from algo_trading import timesfm_experiments as te
from algo_trading.timesfm_experiments import load_context, build_factor_state, set_config, build_context_requests
from algo_trading.timesfm_engine import _cache_key, CACHE_ROOT

def main():
    start = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp("2015-01-01")
    ctx_len = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    kind = sys.argv[3] if len(sys.argv) > 3 else "logret"
    field = sys.argv[4] if len(sys.argv) > 4 else "close"

    cfg, ctx = load_context("config/baseline.json"); set_config(cfg)
    state = build_factor_state(ctx, cfg)
    req = build_context_requests(state, ctx, context_len=ctx_len, kind=kind, field=field, eligible_only=True, min_obs=64)
    req = req[req["date"] >= start]
    # store series as lists (parquet-safe via pickle column)
    req_out = req.copy()
    req_out["series"] = req_out["series"].apply(lambda a: pickle.dumps(np.asarray(a, dtype=np.float32)))
    req_out["ctx_len"] = ctx_len; req_out["kind"] = kind; req_out["field"] = field
    path = CACHE_ROOT / f"requests_ctx{ctx_len}_{kind}_{field}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    req_out.to_parquet(path)
    dates = sorted(req["date"].unique())
    print(f"requests: {len(req)} rows, {len(dates)} dates, {req['symbol'].nunique()} symbols -> {path}")
    pd.DataFrame({"date": dates}).to_csv(CACHE_ROOT / f"dates_ctx{ctx_len}_{kind}_{field}.csv", index=False)
    # free heavy state before exit
    del state, ctx
    print("DONE")

if __name__ == "__main__":
    main()
