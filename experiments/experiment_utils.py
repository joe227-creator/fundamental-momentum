"""Shared utilities for the per-experiment runner scripts (no model loaded)."""
import json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from algo_trading import timesfm_experiments as te
from algo_trading.timesfm_experiments import (
    load_context, build_factor_state, set_config, run_weighted_backtest,
    extended_metrics, subperiod_metrics, benchmark_close, load_forecast_panel,
    forecast_coverage, write_outputs, eligible_symbols, score_composite,
    top_n_equal_weights,
)

OUT = Path("artifacts/timesfm")


def setup(start="2015-01-01", config="config/baseline.json"):
    cfg, ctx = load_context(config); set_config(cfg)
    state = build_factor_state(ctx, cfg)
    dates = [pd.Timestamp(d) for d in state.rebalance_dates
             if pd.Timestamp(d) >= pd.Timestamp(start) and state.split_for_date.get(d) is not None]
    bench = benchmark_close(ctx, "SPY")
    return cfg, ctx, state, dates, bench


def backtest(label, wsel, cfg, ctx, bench, start_date=None):
    if start_date is None:
        # Use the earliest rebalance date in the harness (consistent across experiments)
        # to avoid apples-to-oranges baselines from differing first-selection dates.
        start_date = pd.Timestamp("2015-01-02")
    res = run_weighted_backtest(ctx, wsel, cfg, tc_rate=0.001, start_date=start_date)
    m = extended_metrics(res["equity_curve"], res["trade_log"], bench)
    sp = subperiod_metrics(res["equity_curve"], res["trade_log"], bench)
    res["metrics"] = m; res["subperiods"] = sp
    write_outputs(label, res, out_dir=str(OUT))
    return m, sp, res


def panel_or_skip(ctx_len, horizon, kind="logret", quantiles=False, dates=None):
    panel = load_forecast_panel(ctx_len, horizon, kind, quantiles)
    if panel is None:
        print(f"  [skip] no cache for ctx{ctx_len}_hor{horizon}_{kind}_{'q' if quantiles else 'm'}")
        return None
    if dates is not None:
        cov = forecast_coverage(ctx_len, horizon, kind, pd.DatetimeIndex(dates)) if not quantiles else 0.0
        print(f"  cache coverage {cov:.1%}")
    return panel


def save_summary(name, rows):
    pd.DataFrame(rows).to_csv(OUT / f"{name}_summary.csv", index=False)
    (OUT / f"{name}_summary.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")


def fmt(m):
    return f"CAGR={m['cagr']:.4f} Sharpe={m['sharpe']:.4f} MaxDD={m['max_drawdown']:.4f} IR={m['information_ratio']:.4f} Sortino={m['sortino']:.4f} Calmar={m['calmar']:.4f}"
