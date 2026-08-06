import json, time
import pandas as pd, numpy as np
from algo_trading.config import AppConfig
from algo_trading import timesfm_experiments as te
from algo_trading.timesfm_experiments import (
    load_context, build_factor_state, baseline_selection, run_weighted_backtest,
    extended_metrics, subperiod_metrics, benchmark_close, set_config, top_n_equal_weights,
)

t0=time.time()
cfg, ctx = load_context("config/baseline.json", refresh=False)
set_config(cfg)
print("ctx loaded", round(time.time()-t0,1),"s; sessions", len(ctx.prices.sessions), "cols", len(ctx.prices.close.columns))
t0=time.time()
state = build_factor_state(ctx, cfg)
print("factor_state built", round(time.time()-t0,1),"s; rebal_dates", len(state.rebalance_dates), "factors", state.factor_names)
print("weights (split0):", state.weights_by_split[0])
t0=time.time()
sel = baseline_selection(state, cfg)
nonempty = {d:s for d,s in sel.items() if s}
print("baseline selection", len(sel), "dates;", len(nonempty), "non-empty; sample:", list(nonempty.items())[0] if nonempty else None)
# convert to equal-weight top-2 via run_weighted_backtest
wsel = {d: top_n_equal_weights(pd.Series({s:1.0 for s in syms}), cfg.strategy.max_holdings) for d,syms in sel.items()}
# better: re-derive scores. For now equal weight on baseline-selected symbols.
res = run_weighted_backtest(ctx, wsel, cfg, tc_rate=0.001)
print("backtest", round(time.time()-t0,1),"s")
bench = benchmark_close(ctx, "SPY")
m = extended_metrics(res["equity_curve"], res["trade_log"], bench)
print("HARNESS BASELINE (full-turnover, 10bps/side, equal-w top2):")
print(json.dumps({k: (round(v,4) if isinstance(v,float) else v) for k,v in m.items()}, indent=2))
sp = subperiod_metrics(res["equity_curve"], res["trade_log"], bench)
print("subperiods:", {k: round(v.get("sharpe",float("nan")),3) for k,v in sp.items()})
