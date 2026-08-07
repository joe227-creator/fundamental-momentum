"""Shared harness for the TimesFM momentum experiments.

Reuses the baseline factor library, walk-forward weight fitting, and
rebalance-session selection from ``strategy.py`` so that every experiment is
compared against the *same* universe / eligibility / execution path as the
baseline. Adds:

  * a weighted backtester (the baseline equal-weights; sizing experiments need
    per-position weights) -- does NOT touch ``backtest.py``;
  * extended performance metrics (Sortino, Calmar, turnover, IR vs SPY,
    batting average) per the experiment spec;
  * sub-period robustness splits (pre-COVID / COVID / post-COVID);
  * helpers to build TimesFM context requests and blend tfm scores into the
    baseline factor composite.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .calendar_utils import select_rebalance_sessions
from .config import AppConfig
from .data import MarketContext, load_market_context
from .indicators import build_technical_factors, cross_sectional_rank, shift_for_open_execution
from .strategy import (
    FUNDAMENTAL_FACTOR_NAMES,
    WalkForwardSplit,
    build_factor_library,
    build_walk_forward_splits,
    compute_forward_open_returns,
    fit_indicator_weights,
    _enabled_factor_names,
    _strip_leaky_train_dates,
    _target_holdings,
)

RF_ANNUAL = 0.05
RF_DAILY = RF_ANNUAL / 252.0
TC_PER_SIDE = 0.001  # spec: 0.1% per side

SUBPERIODS = {
    "pre_covid": ("2015-01-01", "2020-02-28"),
    "covid": ("2020-03-01", "2021-12-31"),
    "post_covid": ("2022-01-01", "2100-01-01"),
}

_EXTRA_CACHE = Path("cache/prices")


def load_context(config_path: str = "config/baseline.json", refresh: bool = False) -> tuple[AppConfig, MarketContext]:
    cfg = AppConfig.from_file(config_path)
    ctx = load_market_context(cfg, refresh=refresh)
    return cfg, ctx


def get_extra_close(ticker: str, sessions: pd.DatetimeIndex) -> pd.Series:
    """Fetch (and cache) a daily close series for an ETF not in the IWV panel."""
    _EXTRA_CACHE.mkdir(parents=True, exist_ok=True)
    cache = _EXTRA_CACHE / f"extra_{ticker.lower()}.pkl"
    if cache.exists():
        s = pd.read_pickle(cache)
        return s.reindex(sessions).ffill()
    import yfinance as yf

    df = yf.download(ticker, start=str(sessions.min().date()), end=str(sessions.max().date() + pd.Timedelta(days=1)), progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"Could not download {ticker}")
    # yf returns MultiIndex columns when multiple tickers; normalize
    close = df["Close"] if "Close" in df else df.iloc[:, -1]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    pd.to_pickle(close, cache)
    return close.reindex(sessions).ffill()


def benchmark_close(context: MarketContext, ticker: str = "SPY") -> pd.Series:
    sessions = context.prices.sessions
    if ticker in context.prices.close.columns:
        return context.prices.close[ticker].reindex(sessions)
    return get_extra_close(ticker, sessions)


# ---------------------------------------------------------------------------
# Factor library + walk-forward weights (mirrors strategy.generate_walk_forward_plan
# but exposes the intermediates so experiments can inject/blend factors).
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FactorState:
    all_factors: dict[str, pd.DataFrame]
    ranked_factors: dict[str, pd.DataFrame]
    base_mask: pd.DataFrame
    macro_score: pd.Series
    rebalance_dates: pd.DatetimeIndex
    forward_returns: pd.DataFrame
    factor_names: list[str]
    splits: list[WalkForwardSplit]
    weights_by_split: list[dict[str, float]]  # per-split factor weights
    split_for_date: dict[pd.Timestamp, int]   # rebalance date -> split index


def build_factor_state(context: MarketContext, config: AppConfig) -> FactorState:
    all_factors, base_mask, macro_score = build_factor_library(context, config)
    factor_names = _enabled_factor_names(config, all_factors)
    ranked_factors = {name: cross_sectional_rank(frame) for name, frame in all_factors.items() if name in factor_names}
    rebalance_dates = select_rebalance_sessions(context.prices.sessions, config.strategy.rebalance_frequency)
    forward_returns = compute_forward_open_returns(context.prices.open.loc[:, base_mask.columns], rebalance_dates)
    splits = build_walk_forward_splits(rebalance_dates, config)

    weights_by_split: list[dict[str, float]] = []
    split_for_date: dict[pd.Timestamp, int] = {}
    assigned: set[pd.Timestamp] = set()
    for i, split in enumerate(splits):
        clean_train = _strip_leaky_train_dates(split.train_dates, rebalance_dates, split.test_start)
        w = fit_indicator_weights(
            factor_names=factor_names,
            ranked_factors=ranked_factors,
            forward_returns=forward_returns,
            train_dates=clean_train,
            use_ic_weights=config.walk_forward.use_indicator_ic_weights,
            weighting_scheme=config.walk_forward.indicator_weighting_scheme,
            fundamental_group_weight=config.walk_forward.fundamental_group_weight,
            fundamental_factor_weights=config.walk_forward.fundamental_factor_weights,
        )
        weights_by_split.append(w)
        for d in split.test_dates:
            if d not in assigned:
                split_for_date[pd.Timestamp(d)] = i
                assigned.add(pd.Timestamp(d))
    return FactorState(
        all_factors=all_factors,
        ranked_factors=ranked_factors,
        base_mask=base_mask,
        macro_score=macro_score,
        rebalance_dates=rebalance_dates,
        forward_returns=forward_returns,
        factor_names=factor_names,
        splits=splits,
        weights_by_split=weights_by_split,
        split_for_date=split_for_date,
    )


def baseline_selection(state: FactorState, config: AppConfig) -> dict[pd.Timestamp, list[str]]:
    """Reproduce the baseline top-N selection (sanity check vs CLI)."""
    out: dict[pd.Timestamp, list[str]] = {}
    for d in state.rebalance_dates:
        si = state.split_for_date.get(d)
        if si is None:
            continue
        from .strategy import score_universe_for_date

        scores = score_universe_for_date(
            date=pd.Timestamp(d),
            factor_names=state.factor_names,
            ranked_factors=state.ranked_factors,
            weights=state.weights_by_split[si],
            base_mask=state.base_mask,
            macro_score=state.macro_score,
            config=config,
        )
        out[pd.Timestamp(d)] = scores.index.tolist()
    return out


def eligible_symbols(state: FactorState, date: pd.Timestamp) -> pd.Index:
    if date not in state.base_mask.index:
        return pd.Index([])
    return state.base_mask.loc[date].fillna(False).index[state.base_mask.loc[date].fillna(False).values]


# ---------------------------------------------------------------------------
# TimesFM context builders
# ---------------------------------------------------------------------------


def log_returns(close: pd.DataFrame) -> pd.DataFrame:
    return np.log(close).diff()


def build_context_requests(
    state: FactorState,
    context: MarketContext,
    context_len: int,
    kind: str = "logret",
    field: str = "close",
    eligible_only: bool = True,
    min_obs: int = 64,
) -> pd.DataFrame:
    """Build (symbol, date, series) requests for TimesFM.

    Context window = last `context_len` observations ending the session BEFORE
    the rebalance session (the rebalance trades at the open, so the session's
    own close is not yet known).
    """
    price = getattr(context.prices, field)
    if kind in ("logret", "logvol"):
        series_frame = log_returns(price) if field == "close" else np.log(price.clip(lower=1)).diff()
    else:
        series_frame = price

    rows = []
    sessions = context.prices.sessions
    for d in state.rebalance_dates:
        d = pd.Timestamp(d)
        if d not in state.base_mask.index:
            continue
        si = state.split_for_date.get(d)
        if si is None:
            continue
        if eligible_only:
            syms = eligible_symbols(state, d)
        else:
            syms = price.columns
        # data available at the open of d = up to and including the prior session
        pos = sessions.searchsorted(d, side="left")
        prior_end = sessions[pos - 1] if pos > 0 else d
        window = series_frame.loc[:prior_end]
        if window.empty:
            continue
        avail = window.tail(context_len + 4)
        for sym in syms:
            s = avail[sym].dropna().to_numpy()
            if s.size < min_obs:
                continue
            rows.append({"symbol": sym, "date": d, "series": s[-(context_len + 4):]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Weighted backtester
# ---------------------------------------------------------------------------


def _trade_price(context: MarketContext, session: pd.Timestamp, symbol: str, field: str) -> float:
    frame = getattr(context.prices, field)
    if symbol not in frame.columns:
        return np.nan
    if session in frame.index:
        v = frame.at[session, symbol]
        if pd.notna(v):
            return float(v)
    hist = context.prices.close[symbol].loc[:session].dropna()
    return float(hist.iloc[-1]) if not hist.empty else np.nan


def run_weighted_backtest(
    context: MarketContext,
    weighted_selection: dict[pd.Timestamp, dict[str, float]],
    config: AppConfig,
    tc_rate: float = TC_PER_SIDE,
    initial_capital: float | None = None,
    start_date: pd.Timestamp | str | None = None,
    rebalance_persistence: float = 0.0,
) -> dict[str, Any]:
    """Simulate equity with per-position target weights (fraction of equity).

    weighted_selection: {rebalance_session: {symbol: weight}} with weights
    summing to <= 1.0 (residual stays cash). Rebalances at the session OPEN.
    start_date overrides the context effective_start (use when selections begin
    later than the full data window).
    """
    sessions = context.prices.sessions
    if start_date is not None:
        eff_start = pd.Timestamp(start_date)
    else:
        eff_start = pd.Timestamp(context.metadata.get("effective_start", config.backtest.start_date or sessions.min()))
    sessions = sessions[sessions >= eff_start]
    if config.backtest.end_date is not None:
        sessions = sessions[sessions <= pd.Timestamp(config.backtest.end_date)]
    if sessions.empty:
        raise RuntimeError("No sessions in backtest range.")

    cash = float(initial_capital if initial_capital is not None else config.strategy.initial_capital)
    holdings: dict[str, dict[str, float]] = {}  # symbol -> {shares, entry_date, entry_price, cost_basis}
    eq_rows: list[dict] = []
    trade_rows: list[dict] = []
    sel_rows: list[dict] = []

    for session in sessions:
        ts = pd.Timestamp(session)
        target = weighted_selection.get(ts)
        if target is not None:
            target = {s: float(w) for s, w in target.items() if s in context.prices.close.columns and w > 0}
            # Skip rebalance if target matches current holdings (avoids unnecessary turnover)
            if target and rebalance_persistence <= 0 and set(target.keys()) == set(holdings.keys()):
                target = None
        if target is not None:
            sel_rows.append({"date": ts, "symbols": ",".join(target.keys()), "num_symbols": len(target), "weights": json.dumps(target, default=str)})

            # mark current equity at open prices to size new positions
            open_prices = {s: _trade_price(context, ts, s, "open") for s in list(holdings) + list(target)}

            # close positions not in target (and liquidate to hit target weights)
            eq_open = cash + sum(h["shares"] * open_prices[s] for s, h in holdings.items() if not np.isnan(open_prices.get(s, np.nan)))
            persistent = float(np.clip(rebalance_persistence, 0.0, 1.0))
            if persistent > 0 and holdings and eq_open > 0:
                total_w = sum(target.values())
                if total_w > 1.0 + 1e-9:
                    target = {s: w / total_w for s, w in target.items()}
                current_values = {
                    s: h["shares"] * open_prices[s]
                    for s, h in holdings.items()
                    if not np.isnan(open_prices.get(s, np.nan))
                }
                current_weights = {s: value / eq_open for s, value in current_values.items()}
                symbols = set(current_weights) | set(target)
                desired_weights = {
                    s: persistent * current_weights.get(s, 0.0)
                    + (1.0 - persistent) * target.get(s, 0.0)
                    for s in symbols
                }
                desired_values = {s: eq_open * w for s, w in desired_weights.items()}

                # Sell excess first, then buy deficits. Preserve overlap while
                # making transaction-cost savings measurable.
                for s in list(holdings):
                    px = open_prices.get(s, np.nan)
                    if np.isnan(px) or px <= 0:
                        continue
                    h = holdings[s]
                    current_value = h["shares"] * px
                    excess = max(0.0, current_value - desired_values.get(s, 0.0))
                    if excess <= 0:
                        continue
                    shares = min(h["shares"], excess / px)
                    gross = shares * px
                    fee = gross * tc_rate
                    old_cost = h["cost_basis"]
                    fraction = shares / h["shares"] if h["shares"] else 1.0
                    cash += gross - fee
                    h["shares"] -= shares
                    h["cost_basis"] -= old_cost * fraction
                    trade_rows.append({"date": ts, "action": "SELL", "symbol": s, "shares": shares, "price": px, "gross_value": gross, "fee": fee, "pnl": (gross - fee) - old_cost * fraction, "reason": "persistent_rebalance"})
                    if h["shares"] <= 1e-12:
                        holdings.pop(s)

                for s, desired in desired_values.items():
                    px = open_prices.get(s, np.nan)
                    if np.isnan(px) or px <= 0:
                        continue
                    current = holdings[s]["shares"] * px if s in holdings else 0.0
                    deficit = max(0.0, desired - current)
                    capital = min(deficit, max(0.0, cash / (1.0 + tc_rate)))
                    if capital <= 0:
                        continue
                    fee = capital * tc_rate
                    investable = capital - fee
                    shares = investable / px
                    cash -= capital
                    if s in holdings:
                        holdings[s]["shares"] += shares
                        holdings[s]["cost_basis"] += capital
                    else:
                        holdings[s] = {"shares": shares, "entry_date": ts, "entry_price": px, "cost_basis": capital}
                    trade_rows.append({"date": ts, "action": "BUY", "symbol": s, "shares": shares, "price": px, "gross_value": capital, "fee": fee, "pnl": np.nan, "reason": "persistent_rebalance"})
            else:
                # Sell everything first (original full-turnover path), then buy.
                for s in list(holdings):
                    px = open_prices.get(s, np.nan)
                    if np.isnan(px):
                        continue
                    h = holdings.pop(s)
                    gross = h["shares"] * px
                    fee = gross * tc_rate
                    cash += gross - fee
                    trade_rows.append({"date": ts, "action": "SELL", "symbol": s, "shares": h["shares"], "price": px, "gross_value": gross, "fee": fee, "pnl": (gross - fee) - h["cost_basis"], "reason": "rebalance"})
                if target:
                    total_w = sum(target.values())
                    if total_w > 1.0 + 1e-9:
                        target = {s: w / total_w for s, w in target.items()}
                    for s, w in target.items():
                        px = open_prices.get(s, np.nan)
                        if np.isnan(px) or px <= 0:
                            continue
                        target_capital = cash * float(w)
                        fee = target_capital * tc_rate
                        investable = target_capital - fee
                        if investable <= 0:
                            continue
                        shares = investable / px
                        cash -= target_capital
                        holdings[s] = {"shares": shares, "entry_date": ts, "entry_price": px, "cost_basis": investable + fee}
                        trade_rows.append({"date": ts, "action": "BUY", "symbol": s, "shares": shares, "price": px, "gross_value": target_capital, "fee": fee, "pnl": np.nan, "reason": "rebalance"})

        # mark-to-market at close
        equity = cash
        for s, h in holdings.items():
            px = _trade_price(context, ts, s, "close")
            if not np.isnan(px):
                equity += h["shares"] * px
        eq_rows.append({"date": ts, "equity": equity})

    # final liquidation
    last = pd.Timestamp(sessions[-1])
    for s, h in list(holdings.items()):
        px = _trade_price(context, last, s, "close")
        if np.isnan(px):
            continue
        gross = h["shares"] * px
        fee = gross * tc_rate
        cash += gross - fee
        trade_rows.append({"date": last, "action": "SELL", "symbol": s, "shares": h["shares"], "price": px, "gross_value": gross, "fee": fee, "pnl": (gross - fee) - h["cost_basis"], "reason": "final"})

    eq = pd.DataFrame(eq_rows).drop_duplicates("date", keep="last").set_index("date")
    eq["returns"] = eq["equity"].pct_change()
    trades = pd.DataFrame(trade_rows)
    selections = pd.DataFrame(sel_rows)
    return {"equity_curve": eq, "trade_log": trades, "selection_log": selections}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def extended_metrics(equity_curve: pd.DataFrame, trade_log: pd.DataFrame, benchmark: pd.Series, rf: float = RF_ANNUAL) -> dict[str, float]:
    eq = equity_curve["equity"].dropna()
    if len(eq) < 2:
        return {}
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    days = max(1.0, float((eq.index[-1] - eq.index[0]).days))
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (365.25 / days) - 1.0)
    rets = eq.pct_change().dropna()
    rf_d = rf / 252.0
    excess = rets - rf_d
    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min()) if not dd.empty else 0.0
    sharpe = float(np.sqrt(252) * excess.mean() / excess.std()) if excess.std() else np.nan
    downside = rets[rets < 0]
    sortino = float(np.sqrt(252) * excess.mean() / downside.std()) if (downside.std() and downside.size) else np.nan
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    # monthly win rate
    monthly = eq.resample("ME").last().pct_change().dropna()
    win_rate = float((monthly > 0).mean()) if len(monthly) else np.nan

    # turnover: average monthly buy gross value / starting equity (full-turnover proxy)
    turnover = np.nan
    if not trade_log.empty and "action" in trade_log.columns and "gross_value" in trade_log.columns:
        buys = trade_log[trade_log["action"].eq("BUY")].copy()
        if not buys.empty:
            buys["month"] = pd.to_datetime(buys["date"]).dt.to_period("M")
            monthly_buy = buys.groupby("month")["gross_value"].sum()
            turnover = float(monthly_buy.mean() / eq.iloc[0]) if eq.iloc[0] else np.nan
    # information ratio & batting average vs benchmark
    ir = np.nan
    batting = np.nan
    bench = benchmark.reindex(eq.index).ffill().dropna()
    if len(bench) > 1:
        bench_rets = bench.pct_change().dropna()
        aligned = pd.concat([rets.rename("p"), bench_rets.rename("b")], axis=1).dropna()
        if len(aligned) > 1:
            diff = aligned["p"] - aligned["b"]
            ir = float(np.sqrt(252) * diff.mean() / diff.std()) if diff.std() else np.nan
            monthly_p = eq.resample("ME").last().pct_change()
            monthly_b = bench.reindex(eq.index).ffill().resample("ME").last().pct_change()
            mb = pd.concat([monthly_p.rename("p"), monthly_b.rename("b")], axis=1).dropna()
            batting = float((mb["p"] > mb["b"]).mean()) if len(mb) else np.nan
    return {
        "cagr": cagr,
        "total_return": total_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate_monthly": win_rate,
        "turnover": turnover,
        "information_ratio": ir,
        "batting_avg": batting,
        "ending_equity": float(eq.iloc[-1]),
        "n_months": int(len(monthly)),
    }


def subperiod_metrics(equity_curve: pd.DataFrame, trade_log: pd.DataFrame, benchmark: pd.Series) -> dict[str, dict[str, float]]:
    out = {}
    eq = equity_curve["equity"].dropna()
    for name, (start, end) in SUBPERIODS.items():
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        sub = eq[(eq.index >= s) & (eq.index <= e)]
        if len(sub) < 2:
            out[name] = {}
            continue
        sub_eq = sub.to_frame("equity")
        sub_trades = trade_log[(trade_log["date"] >= s) & (trade_log["date"] <= e)] if not trade_log.empty else trade_log
        out[name] = extended_metrics(sub_eq, sub_trades, benchmark.reindex(sub.index).ffill())
    return out


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------


def top_n_equal_weights(scores: pd.Series, n: int) -> dict[str, float]:
    """scores: symbol->score. Returns top-n equal-weight dict {sym: 1/n}."""
    s = scores.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty or n <= 0:
        return {}
    top = s.sort_values(ascending=False).head(n).index.tolist()
    if not top:
        return {}
    return {sym: 1.0 / len(top) for sym in top}


def score_composite(
    state: FactorState,
    date: pd.Timestamp,
    override: dict[str, pd.Series] | None = None,
    extra_factors: dict[str, tuple[pd.Series, float]] | None = None,
    require_positive: bool = True,
) -> pd.Series:
    """Compute the baseline composite for a date, optionally overriding ranked
    factor values or adding extra factors with explicit weights.

    override: {factor_name: pd.Series of *ranks* (sym->rank)} replacing the
        baseline ranked frame for that factor.
    extra_factors: {name: (rank_series, weight)} added to the composite.
    """
    si = state.split_for_date.get(date)
    if si is None:
        return pd.Series(dtype=float)
    weights = dict(state.weights_by_split[si])
    target_count = _target_holdings(float(state.macro_score.get(date, np.nan)), _cfg_holder.config)
    if target_count == 0:
        return pd.Series(dtype=float)

    composite = pd.Series(0.0, index=state.base_mask.columns, dtype=float)
    for fname in state.factor_names:
        if fname not in weights or fname not in state.ranked_factors:
            continue
        if date not in state.ranked_factors[fname].index:
            continue
        frame = override.get(fname) if (override and fname in override) else state.ranked_factors[fname].loc[date]
        composite = composite.add(frame.fillna(0.0) * weights[fname], fill_value=0.0)
    if extra_factors:
        for name, (ranks, w) in extra_factors.items():
            composite = composite.add(ranks.fillna(0.0) * w, fill_value=0.0)
    eligible = state.base_mask.loc[date].fillna(False)
    composite = composite[eligible].replace([np.inf, -np.inf], np.nan).dropna()
    if require_positive:
        composite = composite[composite > 0.0]
    return composite.sort_values(ascending=False)


class _CfgHolder:
    config: AppConfig = AppConfig()


_cfg_holder = _CfgHolder()


def set_config(config: AppConfig) -> None:
    _cfg_holder.config = config


def load_forecast_panel(ctx_len: int, horizon: int, kind: str = "logret", quantiles: bool = False):
    """Read the cached forecast panel (or None if not yet built)."""
    from .timesfm_engine import _cache_key
    p = _cache_key(ctx_len, horizon, kind, quantiles)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if not isinstance(df.index, pd.MultiIndex):
        df = df.set_index(["symbol", "date"])
    return df.sort_index()


def forecast_coverage(ctx_len: int, horizon: int, kind: str, dates) -> float:
    df = load_forecast_panel(ctx_len, horizon, kind, False)
    if df is None:
        return 0.0
    have = pd.DatetimeIndex(df.index.get_level_values("date").unique())
    return float(len(have.intersection(dates)) / max(1, len(dates)))


def write_outputs(label: str, result: dict, out_dir: str = "artifacts/timesfm") -> None:
    p = Path(out_dir) / label
    p.mkdir(parents=True, exist_ok=True)
    result["equity_curve"].to_csv(p / "equity_curve.csv")
    if not result["trade_log"].empty:
        result["trade_log"].to_csv(p / "trade_log.csv", index=False)
    if not result["selection_log"].empty:
        result["selection_log"].to_csv(p / "selections.csv", index=False)
    (p / "metrics.json").write_text(json.dumps(result.get("metrics", {}), indent=2, default=str), encoding="utf-8")
