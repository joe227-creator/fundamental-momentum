from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
import warnings

import numpy as np
import pandas as pd
try:
    from scipy.stats import ConstantInputWarning
except Exception:
    ConstantInputWarning = RuntimeWarning

from .calendar_utils import next_or_same_session, select_rebalance_sessions
from .config import AppConfig
from .indicators import build_technical_factors, cross_sectional_rank, shift_for_open_execution

if TYPE_CHECKING:
    from .data import MarketContext


FUNDAMENTAL_FACTOR_NAMES = {
    "book_to_market",
    "earnings_yield",
    "roa",
    "roe",
    "gross_profitability",
    "asset_growth",
    "investment_to_assets",
    "net_issuance",
    "accruals",
    "cash_flow_yield",
    "earnings_surprise",
}


@dataclass(slots=True)
class WalkForwardSplit:
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _enabled_factor_names(config: AppConfig, all_factors: dict[str, pd.DataFrame]) -> list[str]:
    names = []
    for name in all_factors:
        if name == "avg_dollar_volume":
            continue
        if config.indicators.enabled_indicators.get(name, False):
            names.append(name)
    return names


def build_factor_library(context: MarketContext, config: AppConfig) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.Series]:
    enabled_factor_names = {name for name, enabled in config.indicators.enabled_indicators.items() if enabled}
    requested_technical = {"avg_dollar_volume", *(enabled_factor_names & {"sma_crossover", "ema_crossover", "trend_filter", "breakout", "rsi", "cci", "macd", "obv", "bollinger", "ts_momentum", "variable_week_high", "price_action"})}
    if config.strategy.base_filter_requires_trend:
        requested_technical.update({"sma_crossover", "trend_filter"})
    technical = shift_for_open_execution(build_technical_factors(context.prices, config, factor_names=requested_technical))
    fundamentals = shift_for_open_execution(
        {name: frame for name, frame in context.fundamental_factors.items() if name in enabled_factor_names}
    )
    all_factors = {**technical, **fundamentals}
    avg_dollar_volume = technical["avg_dollar_volume"]
    close = context.prices.close
    history_count = close.notna().cumsum()
    membership = context.universe_membership.reindex(index=close.index, columns=close.columns, fill_value=False)

    if config.strategy.base_filter_requires_trend:
        trend_ok = (technical.get("trend_filter", pd.DataFrame(index=close.index, columns=close.columns)) > 0.0) & (
            technical.get("sma_crossover", pd.DataFrame(index=close.index, columns=close.columns)) > 0.0
        )
    else:
        trend_ok = pd.DataFrame(True, index=close.index, columns=close.columns)

    # Decisions are executed at the session open, so any filter that depends on
    # the session's own close or volume must be shifted by one trading day to
    # stay consistent with `shift_for_open_execution` applied to the factors.
    # Membership reflects publicly-known IWV constituents on the session and
    # `trend_ok` is built from `technical`, which is already shifted.
    history_ok = (history_count >= config.universe.min_history_days).shift(1).fillna(False)
    price_ok = (close >= config.universe.min_price).shift(1).fillna(False)
    liquidity_ok = (avg_dollar_volume >= config.universe.min_avg_dollar_volume).shift(1).fillna(False)

    base_mask = (
        membership
        & history_ok
        & price_ok
        & liquidity_ok
        & trend_ok.fillna(False)
    )
    macro_score = context.macro["macro_composite"].shift(1).reindex(close.index)
    return all_factors, base_mask, macro_score


def compute_forward_open_returns(open_prices: pd.DataFrame, rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    open_on_rebalance = open_prices.reindex(rebalance_dates)
    next_open = open_on_rebalance.shift(-1)
    return next_open / open_on_rebalance - 1.0


def build_walk_forward_splits(rebalance_dates: pd.DatetimeIndex, config: AppConfig) -> list[WalkForwardSplit]:
    if rebalance_dates.empty:
        return []

    splits: list[WalkForwardSplit] = []
    step_anchor = rebalance_dates[0] + pd.DateOffset(years=config.walk_forward.train_years)
    cursor = rebalance_dates.searchsorted(step_anchor, side="left")

    while cursor < len(rebalance_dates):
        test_start = pd.Timestamp(rebalance_dates[cursor])
        train_start_boundary = test_start - pd.DateOffset(years=config.walk_forward.train_years)
        test_end_boundary = test_start + pd.DateOffset(months=config.walk_forward.test_months)
        train_dates = rebalance_dates[(rebalance_dates >= train_start_boundary) & (rebalance_dates < test_start)]
        test_dates = rebalance_dates[(rebalance_dates >= test_start) & (rebalance_dates < test_end_boundary)]

        if len(train_dates) >= config.walk_forward.min_train_rebalances and len(test_dates) > 0:
            splits.append(
                WalkForwardSplit(
                    train_dates=train_dates,
                    test_dates=test_dates,
                    train_start=pd.Timestamp(train_dates[0]),
                    train_end=pd.Timestamp(train_dates[-1]),
                    test_start=pd.Timestamp(test_dates[0]),
                    test_end=pd.Timestamp(test_dates[-1]),
                )
            )

        next_anchor = test_start + pd.DateOffset(months=config.walk_forward.step_months)
        next_cursor = rebalance_dates.searchsorted(next_anchor, side="left")
        if next_cursor <= cursor:
            next_cursor = cursor + 1
        cursor = next_cursor
    return splits


def _equal_weight_series(names: list[str]) -> pd.Series:
    if not names:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(names), index=names, dtype=float)


def _normalize_weight_scores(scores: pd.Series) -> pd.Series:
    scores = scores.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if scores.abs().sum() == 0.0:
        return _equal_weight_series(scores.index.tolist())
    return scores / scores.abs().sum()


def _compute_ic_scores(
    factor_names: list[str],
    ranked_factors: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
) -> pd.Series:
    scores: dict[str, float] = {}
    for factor_name in factor_names:
        correlations: list[float] = []
        frame = ranked_factors[factor_name]
        for date in train_dates:
            if date not in forward_returns.index or date not in frame.index:
                continue
            sample = pd.concat(
                [frame.loc[date].rename("factor"), forward_returns.loc[date].rename("forward_return")],
                axis=1,
            ).dropna()
            if len(sample) < 20:
                continue
            correlation = sample["factor"].corr(sample["forward_return"], method="spearman")
            if pd.notna(correlation):
                correlations.append(float(correlation))
        scores[factor_name] = float(np.nanmean(correlations)) if correlations else 0.0
    return pd.Series(scores, dtype=float).reindex(factor_names).fillna(0.0)


def _group_weighted_weights(
    factor_names: list[str],
    ic_scores: pd.Series,
    use_ic_within_groups: bool,
    fundamental_group_weight: float,
    fundamental_factor_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    technical_factors = [name for name in factor_names if name not in FUNDAMENTAL_FACTOR_NAMES]
    fundamental_factors = [name for name in factor_names if name in FUNDAMENTAL_FACTOR_NAMES]

    if not technical_factors:
        technical_share = 0.0
        fundamental_share = 1.0
    elif not fundamental_factors:
        technical_share = 1.0
        fundamental_share = 0.0
    else:
        fundamental_share = float(np.clip(fundamental_group_weight, 0.0, 1.0))
        technical_share = 1.0 - fundamental_share

    weights = pd.Series(0.0, index=factor_names, dtype=float)
    grouped_factors = (
        (technical_factors, technical_share, None),
        (fundamental_factors, fundamental_share, fundamental_factor_weights or {}),
    )
    for names, share, explicit_weights in grouped_factors:
        if not names or share <= 0.0:
            continue
        if explicit_weights:
            local_weights = pd.Series(
                {name: float(explicit_weights.get(name, 0.0)) for name in names},
                dtype=float,
            ).clip(lower=0.0)
            if local_weights.sum() > 0.0:
                local_weights = local_weights / local_weights.sum()
            elif use_ic_within_groups:
                local_weights = _normalize_weight_scores(ic_scores.reindex(names).fillna(0.0))
            else:
                local_weights = _equal_weight_series(names)
        elif use_ic_within_groups:
            local_weights = _normalize_weight_scores(ic_scores.reindex(names).fillna(0.0))
        else:
            local_weights = _equal_weight_series(names)
        weights.loc[names] = local_weights * share
    return weights.to_dict()


def _strip_leaky_train_dates(
    train_dates: pd.DatetimeIndex,
    rebalance_dates: pd.DatetimeIndex,
    decision_session: pd.Timestamp | None,
) -> pd.DatetimeIndex:
    """Drop training rebalance dates whose realized forward open return endpoint
    is at or after the decision session.

    `compute_forward_open_returns` stores at row D the return earned from
    `open[D]` to `open[next_rebalance]`. At decision time (the open of
    `decision_session`), `open[next_rebalance]` is still in the future whenever
    `next_rebalance >= decision_session`, so including those rows in the IC fit
    leaks the very first test-period print into training.
    """
    if decision_session is None or len(train_dates) == 0 or len(rebalance_dates) == 0:
        return train_dates
    decision_ts = pd.Timestamp(decision_session)
    kept: list[pd.Timestamp] = []
    for date in train_dates:
        position = rebalance_dates.searchsorted(pd.Timestamp(date), side="right")
        if position >= len(rebalance_dates):
            # No subsequent rebalance loaded yet: forward return endpoint is
            # unobservable, so this row cannot enter training.
            continue
        next_rebalance = pd.Timestamp(rebalance_dates[position])
        if next_rebalance >= decision_ts:
            continue
        kept.append(pd.Timestamp(date))
    return pd.DatetimeIndex(kept)


def fit_indicator_weights(
    factor_names: list[str],
    ranked_factors: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    use_ic_weights: bool,
    weighting_scheme: str = "legacy",
    fundamental_group_weight: float = 0.25,
    fundamental_factor_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    if not factor_names:
        return {}

    scheme = weighting_scheme.lower()
    if scheme == "legacy" and not use_ic_weights:
        return _equal_weight_series(factor_names).to_dict()

    ic_scores = _compute_ic_scores(
        factor_names=factor_names,
        ranked_factors=ranked_factors,
        forward_returns=forward_returns,
        train_dates=train_dates,
    )

    if scheme == "legacy":
        return _normalize_weight_scores(ic_scores).to_dict()
    if scheme == "grouped_equal":
        return _group_weighted_weights(
            factor_names=factor_names,
            ic_scores=ic_scores,
            use_ic_within_groups=False,
            fundamental_group_weight=fundamental_group_weight,
            fundamental_factor_weights=fundamental_factor_weights,
        )
    if scheme == "grouped_ic":
        return _group_weighted_weights(
            factor_names=factor_names,
            ic_scores=ic_scores,
            use_ic_within_groups=True,
            fundamental_group_weight=fundamental_group_weight,
            fundamental_factor_weights=fundamental_factor_weights,
        )
    raise ValueError(f"Unsupported indicator weighting scheme: {weighting_scheme}")


def _target_holdings(macro_score: float, config: AppConfig) -> int:
    if pd.isna(macro_score):
        return config.strategy.max_holdings
    if macro_score <= config.strategy.macro_flat_threshold and config.strategy.allow_cash_when_risk_off:
        return 0
    if macro_score <= config.strategy.macro_half_risk_threshold:
        return max(1, config.strategy.max_holdings // 2)
    return config.strategy.max_holdings


def score_universe_for_date(
    date: pd.Timestamp,
    factor_names: list[str],
    ranked_factors: dict[str, pd.DataFrame],
    weights: dict[str, float],
    base_mask: pd.DataFrame,
    macro_score: pd.Series,
    config: AppConfig,
) -> pd.Series:
    if date not in base_mask.index:
        return pd.Series(dtype=float)

    target_count = _target_holdings(float(macro_score.get(date, np.nan)), config)
    if target_count == 0:
        return pd.Series(dtype=float)

    composite = pd.Series(0.0, index=base_mask.columns, dtype=float)
    for factor_name in factor_names:
        if factor_name not in weights or factor_name not in ranked_factors:
            continue
        if date not in ranked_factors[factor_name].index:
            continue
        composite = composite.add(ranked_factors[factor_name].loc[date].fillna(0.0) * weights[factor_name], fill_value=0.0)

    eligible = base_mask.loc[date].fillna(False)
    composite = composite[eligible]
    composite = composite.replace([np.inf, -np.inf], np.nan).dropna()
    if config.strategy.require_positive_composite:
        composite = composite[composite > 0.0]
    return composite.sort_values(ascending=False).head(target_count)


def generate_walk_forward_plan(
    context: MarketContext,
    config: AppConfig,
) -> tuple[dict[pd.Timestamp, list[str]], dict[pd.Timestamp, pd.Series], list[dict[str, Any]]]:
    all_factors, base_mask, macro_score = build_factor_library(context, config)
    factor_names = _enabled_factor_names(config, all_factors)
    ranked_factors = {name: cross_sectional_rank(frame) for name, frame in all_factors.items() if name in factor_names}
    rebalance_dates = select_rebalance_sessions(context.prices.sessions, config.strategy.rebalance_frequency)
    forward_returns = compute_forward_open_returns(context.prices.open.loc[:, base_mask.columns], rebalance_dates)
    splits = build_walk_forward_splits(rebalance_dates, config)

    selection_map: dict[pd.Timestamp, list[str]] = {}
    score_map: dict[pd.Timestamp, pd.Series] = {}
    diagnostics: list[dict[str, Any]] = []
    assigned_dates: set[pd.Timestamp] = set()

    for split in splits:
        # Drop the last training observation if its realized forward return
        # endpoint is at or after the test_start open, otherwise the IC fit
        # peeks at the test period.
        clean_train_dates = _strip_leaky_train_dates(
            train_dates=split.train_dates,
            rebalance_dates=rebalance_dates,
            decision_session=split.test_start,
        )
        weights = fit_indicator_weights(
            factor_names=factor_names,
            ranked_factors=ranked_factors,
            forward_returns=forward_returns,
            train_dates=clean_train_dates,
            use_ic_weights=config.walk_forward.use_indicator_ic_weights,
            weighting_scheme=config.walk_forward.indicator_weighting_scheme,
            fundamental_group_weight=config.walk_forward.fundamental_group_weight,
            fundamental_factor_weights=config.walk_forward.fundamental_factor_weights,
        )
        usable_test_dates = [date for date in split.test_dates if date not in assigned_dates]
        for date in usable_test_dates:
            score_series = score_universe_for_date(
                date=pd.Timestamp(date),
                factor_names=factor_names,
                ranked_factors=ranked_factors,
                weights=weights,
                base_mask=base_mask,
                macro_score=macro_score,
                config=config,
            )
            selection_map[pd.Timestamp(date)] = score_series.index.tolist()
            score_map[pd.Timestamp(date)] = score_series
            assigned_dates.add(pd.Timestamp(date))
        diagnostics.append(
            {
                "train_start": split.train_start,
                "train_end": split.train_end,
                "test_start": split.test_start,
                "test_end": split.test_end,
                "weights": weights,
                "test_dates": len(usable_test_dates),
            }
        )

    return selection_map, score_map, diagnostics


def screen_live_session(
    context: MarketContext,
    config: AppConfig,
    as_of: str | pd.Timestamp,
) -> tuple[pd.Timestamp, pd.DataFrame, dict[str, float]]:
    try:
        session = next_or_same_session(as_of, context.prices.sessions)
    except ValueError:
        session = pd.Timestamp(context.prices.sessions[-1])
    all_factors, base_mask, macro_score = build_factor_library(context, config)
    factor_names = _enabled_factor_names(config, all_factors)
    ranked_factors = {name: cross_sectional_rank(frame) for name, frame in all_factors.items() if name in factor_names}
    rebalance_dates = select_rebalance_sessions(context.prices.sessions, config.strategy.rebalance_frequency)
    train_window_start = session - pd.DateOffset(years=config.walk_forward.train_years)
    train_dates = rebalance_dates[(rebalance_dates >= train_window_start) & (rebalance_dates < session)]
    # Same look-ahead guard as the walk-forward planner: drop training rows
    # whose realized forward return endpoint is at or after today's session.
    train_dates = _strip_leaky_train_dates(
        train_dates=train_dates,
        rebalance_dates=rebalance_dates,
        decision_session=session,
    )
    forward_returns = compute_forward_open_returns(context.prices.open.loc[:, base_mask.columns], rebalance_dates)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConstantInputWarning)
        weights = fit_indicator_weights(
            factor_names=factor_names,
            ranked_factors=ranked_factors,
            forward_returns=forward_returns,
            train_dates=train_dates,
            use_ic_weights=config.walk_forward.use_indicator_ic_weights,
            weighting_scheme=config.walk_forward.indicator_weighting_scheme,
            fundamental_group_weight=config.walk_forward.fundamental_group_weight,
            fundamental_factor_weights=config.walk_forward.fundamental_factor_weights,
        )
    scores = score_universe_for_date(
        date=session,
        factor_names=factor_names,
        ranked_factors=ranked_factors,
        weights=weights,
        base_mask=base_mask,
        macro_score=macro_score,
        config=config,
    )
    if scores.empty:
        output = pd.DataFrame(columns=["symbol", "composite_score", "name", "sector", "etf_weight_pct", "macro_composite"])
    else:
        metadata = context.holdings.set_index("ticker").reindex(scores.index)
        output = pd.DataFrame(
            {
                "symbol": scores.index,
                "composite_score": scores.values,
                "name": metadata["name"].values,
                "sector": metadata["sector"].values,
                "etf_weight_pct": metadata["weight_pct"].values,
                "macro_composite": float(macro_score.get(session, np.nan)),
            }
        )
    return session, output.reset_index(drop=True), weights