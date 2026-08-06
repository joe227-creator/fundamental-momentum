"""Year-breakdown analysis for the champion strategy.

Computes per-calendar-year performance metrics using the same
backtest logic as run_champion.py backtest. Does not modify
any core files.
"""
from __future__ import annotations

import json
import sys
import warnings
import os

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")

from algo_trading.timesfm_experiments import (
    load_context, build_factor_state, set_config,
    run_weighted_backtest, extended_metrics,
    benchmark_close, score_composite, top_n_equal_weights,
    eligible_symbols, load_forecast_panel,
)
from algo_trading.timesfm_engine import mean_forecast_sum
from algo_trading.config import AppConfig
from pathlib import Path

CONFIG_PATH = "config/champion.json"
CTX, HOR = 256, 21


def trailing_avg(volume_series, end_session, window):
    s = volume_series.loc[:end_session]
    return s.tail(window).mean()


def main():
    cfg, ctx = load_context(CONFIG_PATH)
    set_config(cfg)

    state = build_factor_state(ctx, cfg)
    sessions = ctx.prices.sessions
    volume = ctx.prices.volume
    bench = benchmark_close(ctx, "SPY")

    raw = json.loads(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    veto = raw.get("champion_veto", {})
    veto_threshold = float(veto.get("veto_threshold", 0.775))
    trailing_window = int(veto.get("trailing_vol_window", 53))
    veto_horizon = int(veto.get("veto_horizon", 21))
    tc_rate = float(veto.get("cost_per_side", 0.0003))
    N = cfg.strategy.max_holdings

    panel = load_forecast_panel(CTX, HOR, kind="volume", quantiles=False)
    pred_vol = mean_forecast_sum(panel, veto_horizon) / veto_horizon
    pred_vol = pred_vol.unstack(level="symbol")

    dates = [
        pd.Timestamp(d) for d in state.rebalance_dates
        if pd.Timestamp(d) >= pd.Timestamp("2015-01-01")
        and state.split_for_date.get(d) is not None
    ]

    wsel = {}
    veto_count = 0
    total_cand = 0
    for d in dates:
        if d not in state.base_mask.index:
            continue
        scores = score_composite(state, d)
        if scores.empty:
            continue
        v = set()
        elig = eligible_symbols(state, d)
        pos = sessions.searchsorted(d, side="left")
        prior_end = sessions[pos - 1] if pos > 0 else d
        for sym in elig:
            tv = (
                trailing_avg(volume[sym], prior_end, trailing_window)
                if sym in volume.columns
                else np.nan
            )
            if not np.isfinite(tv) or tv <= 0:
                continue
            pv = (
                pred_vol.loc[d, sym]
                if (d in pred_vol.index and sym in pred_vol.columns)
                else np.nan
            )
            if not np.isfinite(pv):
                continue
            ratio = pv / tv
            if ratio < veto_threshold:
                v.add(sym)
        total_cand += len(scores)
        veto_count += len(set(scores.index) & v)
        scores = scores.drop(index=[s for s in v if s in scores.index])
        wsel[d] = top_n_equal_weights(scores, N)

    res = run_weighted_backtest(
        ctx, wsel, cfg, tc_rate=tc_rate,
        start_date=pd.Timestamp("2015-01-02"),
    )
    eq = res["equity_curve"]["equity"].dropna()
    trades = res["trade_log"]
    bench_rets = bench.reindex(eq.index).ffill()

    # Full period
    m = extended_metrics(res["equity_curve"], trades, bench)
    print("=" * 100)
    print(f"Full Period:  CAGR={m['cagr']:.4f}  Sharpe={m['sharpe']:.4f}  "
          f"MaxDD={m['max_drawdown']:.4f}  Calmar={m['calmar']:.4f}  "
          f"Sortino={m['sortino']:.4f}  IR={m['information_ratio']:.3f}  "
          f"WinRate={m['win_rate_monthly']:.3f}")
    print("=" * 100)

    # Year by year
    years = sorted(set(d.year for d in eq.index))
    header = (
        f"{'Year':>6}  {'Start':>12}  {'End':>12}  "
        f"{'CAGR':>8}  {'Sharpe':>8}  {'MaxDD':>8}  "
        f"{'Calmar':>8}  {'Sortino':>8}  {'WinRate':>8}  "
        f"{'IR':>8}  {'EndEq':>10}"
    )
    print()
    print(header)
    print("-" * len(header))

    for yr in years:
        sub = eq[eq.index.year == yr]
        if len(sub) < 2:
            continue
        sub_eq = sub.to_frame("equity")
        tm = trades.copy()
        if not tm.empty:
            tm["yr"] = pd.to_datetime(tm["date"]).dt.year
            sub_trades = tm[tm["yr"] == yr].drop(columns=["yr"])
        else:
            sub_trades = trades
        bench_sub = bench_rets[bench_rets.index.year == yr]
        try:
            ym = extended_metrics(sub_eq, sub_trades, bench_sub)
            if not ym:
                continue
            yr_start = sub.index[0].strftime("%Y-%m-%d")
            yr_end = sub.index[-1].strftime("%Y-%m-%d")
            print(
                f"{yr:>6}  {yr_start:>12}  {yr_end:>12}  "
                f"{ym.get('cagr', 0):>8.4f}  {ym.get('sharpe', 0):>8.4f}  "
                f"{ym.get('max_drawdown', 0):>8.4f}  {ym.get('calmar', 0):>8.4f}  "
                f"{ym.get('sortino', 0):>8.4f}  {ym.get('win_rate_monthly', 0):>8.3f}  "
                f"{ym.get('information_ratio', 0):>8.3f}  "
                f"{ym.get('ending_equity', 0):>10.0f}"
            )
        except Exception as e:
            print(f"{yr:>6}  error — {e}")

    print()
    print(f"Veto rate: {veto_count / max(1, total_cand):.1%}")
    print(f"Total sessions in equity curve: {len(eq)}")
    print(f"Total trades: {len(trades) if not trades.empty else 0}")
    print(f"CAGR = {m['cagr']:.4f}  Sharpe = {m['sharpe']:.4f}  "
          f"MaxDD = {m['max_drawdown']:.4f}  Calmar = {m['calmar']:.4f}")


if __name__ == "__main__":
    main()
