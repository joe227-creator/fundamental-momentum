from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AppConfig, ensure_path
from .data import MarketContext, PricePanel


@dataclass(slots=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    selection_log: pd.DataFrame
    metrics: dict[str, Any]


def _trade_price(prices: PricePanel, session: pd.Timestamp, symbol: str, field: str) -> float:
    price_frame = getattr(prices, field)
    if symbol not in price_frame.columns:
        return np.nan
    value = price_frame.at[session, symbol] if session in price_frame.index else np.nan
    if pd.notna(value):
        return float(value)
    close_history = prices.close[symbol].loc[:session].dropna()
    if close_history.empty:
        return np.nan
    return float(close_history.iloc[-1])


def _equity_value(cash: float, holdings: dict[str, dict[str, Any]], prices: PricePanel, session: pd.Timestamp) -> float:
    total = cash
    for symbol, position in holdings.items():
        mark = _trade_price(prices, session, symbol, "close")
        if pd.isna(mark):
            continue
        total += position["shares"] * mark
    return float(total)


def _summarize_metrics(
    equity_curve: pd.DataFrame,
    trade_log: pd.DataFrame,
    benchmark_close: pd.Series | None,
) -> dict[str, Any]:
    equity = equity_curve["equity"]
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    elapsed_days = max(1.0, float((equity.index[-1] - equity.index[0]).days))
    annualized_return = float((equity.iloc[-1] / equity.iloc[0]) ** (365.25 / elapsed_days) - 1.0)
    daily_returns = equity.pct_change().dropna()
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    sharpe = float(np.sqrt(252.0) * daily_returns.mean() / daily_returns.std()) if daily_returns.std() else np.nan

    closed_trades = trade_log[trade_log["action"].eq("SELL")].copy()
    positive_pnl = float(closed_trades[closed_trades["pnl"].gt(0)]["pnl"].sum()) if not closed_trades.empty else 0.0
    negative_pnl = float(closed_trades[closed_trades["pnl"].lt(0)]["pnl"].sum()) if not closed_trades.empty else 0.0
    if negative_pnl < 0.0:
        profit_factor = positive_pnl / abs(negative_pnl)
    elif positive_pnl > 0.0:
        profit_factor = float("inf")
    else:
        profit_factor = np.nan

    win_rate = float(closed_trades["pnl"].gt(0).mean()) if not closed_trades.empty else np.nan

    metrics: dict[str, Any] = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "num_closed_trades": int(len(closed_trades)),
        "ending_equity": float(equity.iloc[-1]),
    }

    if benchmark_close is not None and not benchmark_close.dropna().empty:
        benchmark = benchmark_close.reindex(equity.index).ffill().dropna()
        if len(benchmark) > 1:
            benchmark_total_return = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1.0)
            benchmark_annualized = float((benchmark.iloc[-1] / benchmark.iloc[0]) ** (365.25 / elapsed_days) - 1.0)
            metrics["benchmark_total_return"] = benchmark_total_return
            metrics["benchmark_annualized_return"] = benchmark_annualized
    return metrics


def run_backtest(
    context: MarketContext,
    selection_map: dict[pd.Timestamp, list[str]],
    config: AppConfig,
) -> BacktestResult:
    sessions = context.prices.sessions
    effective_start = pd.Timestamp(context.metadata.get("effective_start", config.backtest.start_date or sessions.min()))
    sessions = sessions[sessions >= effective_start]
    if config.backtest.end_date is not None:
        sessions = sessions[sessions <= pd.Timestamp(config.backtest.end_date)]
    if sessions.empty:
        raise RuntimeError("No trading sessions available within the requested backtest range.")

    price_symbols = context.prices.close.columns
    benchmark_close = (
        context.prices.close[config.backtest.benchmark_ticker] if config.backtest.benchmark_ticker in price_symbols else None
    )
    cash = float(config.strategy.initial_capital)
    holdings: dict[str, dict[str, Any]] = {}
    transaction_cost_rate = config.strategy.transaction_cost_bps / 10_000.0

    equity_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for session in sessions:
        desired_symbols = selection_map.get(pd.Timestamp(session))
        if desired_symbols is not None:
            desired_symbols = [symbol for symbol in desired_symbols if symbol in price_symbols]
            selection_rows.append(
                {
                    "date": session,
                    "symbols": ",".join(desired_symbols),
                    "num_symbols": len(desired_symbols),
                }
            )

            if config.strategy.close_positions_before_new_positions:
                if config.strategy.full_turnover_rebalance:
                    exit_symbols = list(holdings.keys())
                else:
                    exit_symbols = [symbol for symbol in holdings if symbol not in desired_symbols]
                for symbol in exit_symbols:
                    position = holdings.pop(symbol)
                    sell_price = _trade_price(context.prices, session, symbol, "open")
                    if pd.isna(sell_price):
                        holdings[symbol] = position
                        continue
                    gross = position["shares"] * sell_price
                    fee = gross * transaction_cost_rate
                    proceeds = gross - fee
                    cash += proceeds
                    pnl = proceeds - position["cost_basis"]
                    trade_rows.append(
                        {
                            "date": session,
                            "action": "SELL",
                            "symbol": symbol,
                            "shares": position["shares"],
                            "price": sell_price,
                            "gross_value": gross,
                            "fee": fee,
                            "cash_after": cash,
                            "reason": "rebalance_exit",
                            "pnl": pnl,
                        }
                    )

            if desired_symbols:
                if config.strategy.full_turnover_rebalance:
                    capital_per_position = cash / len(desired_symbols)
                else:
                    new_symbols = [symbol for symbol in desired_symbols if symbol not in holdings]
                    capital_per_position = cash / len(new_symbols) if new_symbols else 0.0

                for symbol in desired_symbols:
                    if not config.strategy.full_turnover_rebalance and symbol in holdings:
                        continue
                    buy_price = _trade_price(context.prices, session, symbol, "open")
                    if pd.isna(buy_price) or buy_price <= 0.0:
                        continue
                    fee = capital_per_position * transaction_cost_rate
                    investable = capital_per_position - fee
                    if investable <= 0.0:
                        continue
                    shares = investable / buy_price
                    cash -= capital_per_position
                    holdings[symbol] = {
                        "shares": shares,
                        "entry_date": session,
                        "entry_price": buy_price,
                        "cost_basis": investable + fee,
                    }
                    trade_rows.append(
                        {
                            "date": session,
                            "action": "BUY",
                            "symbol": symbol,
                            "shares": shares,
                            "price": buy_price,
                            "gross_value": investable,
                            "fee": fee,
                            "cash_after": cash,
                            "reason": "rebalance_entry",
                            "pnl": np.nan,
                        }
                    )

        equity_rows.append({"date": session, "equity": _equity_value(cash, holdings, context.prices, session)})

    last_session = pd.Timestamp(sessions[-1])
    for symbol, position in list(holdings.items()):
        sell_price = _trade_price(context.prices, last_session, symbol, "close")
        if pd.isna(sell_price):
            continue
        gross = position["shares"] * sell_price
        fee = gross * transaction_cost_rate
        proceeds = gross - fee
        cash += proceeds
        pnl = proceeds - position["cost_basis"]
        trade_rows.append(
            {
                "date": last_session,
                "action": "SELL",
                "symbol": symbol,
                "shares": position["shares"],
                "price": sell_price,
                "gross_value": gross,
                "fee": fee,
                "cash_after": cash,
                "reason": "final_liquidation",
                "pnl": pnl,
            }
        )

    equity_curve = pd.DataFrame(equity_rows).drop_duplicates(subset="date", keep="last").set_index("date")
    equity_curve["returns"] = equity_curve["equity"].pct_change()
    trade_log = pd.DataFrame(trade_rows)
    selection_log = pd.DataFrame(selection_rows)
    metrics = _summarize_metrics(equity_curve=equity_curve, trade_log=trade_log, benchmark_close=benchmark_close)
    metrics["requested_start"] = context.metadata.get("requested_start")
    metrics["effective_start"] = context.metadata.get("effective_start")
    metrics["requested_end"] = context.metadata.get("requested_end")
    metrics["historical_snapshot_count"] = context.metadata.get("historical_snapshot_count")
    metrics["using_historical_snapshots"] = context.metadata.get("using_historical_snapshots")
    return BacktestResult(equity_curve=equity_curve, trade_log=trade_log, selection_log=selection_log, metrics=metrics)


def write_backtest_outputs(result: BacktestResult, output_dir: str | Path, prefix: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(ensure_path(output_path / f"{prefix}_equity_curve.csv"))
    result.trade_log.to_csv(ensure_path(output_path / f"{prefix}_trade_log.csv"), index=False)
    result.selection_log.to_csv(ensure_path(output_path / f"{prefix}_selections.csv"), index=False)
    metrics_path = ensure_path(output_path / f"{prefix}_metrics.json")
    metrics_path.write_text(json.dumps(result.metrics, indent=2, default=str), encoding="utf-8")