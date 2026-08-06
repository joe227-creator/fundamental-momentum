from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AppConfig
from .data import PricePanel


def _resample_rule(timeframe: str) -> str:
    timeframe = timeframe.lower()
    if timeframe == "daily":
        return "D"
    if timeframe == "weekly":
        return "W-FRI"
    if timeframe == "monthly":
        return "ME"
    raise ValueError(f"Unsupported signal timeframe: {timeframe}")


def resample_price_panel(prices: PricePanel, timeframe: str) -> PricePanel:
    timeframe = timeframe.lower()
    if timeframe == "daily":
        return prices

    rule = _resample_rule(timeframe)

    def _resample(frame: pd.DataFrame, how: str) -> pd.DataFrame:
        if how == "first":
            return frame.resample(rule).first()
        if how == "max":
            return frame.resample(rule).max()
        if how == "min":
            return frame.resample(rule).min()
        if how == "last":
            return frame.resample(rule).last()
        if how == "sum":
            return frame.resample(rule).sum()
        raise ValueError(f"Unsupported resample method: {how}")

    resampled = PricePanel(
        open=_resample(prices.open, "first"),
        high=_resample(prices.high, "max"),
        low=_resample(prices.low, "min"),
        close=_resample(prices.close, "last"),
        volume=_resample(prices.volume, "sum"),
    )
    return resampled


def _align_to_sessions(frame: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    aligned_index = frame.index.union(sessions).sort_values()
    return frame.reindex(aligned_index).ffill().reindex(sessions)


def sma(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=max(2, window // 2)).mean()


def ema(frame: pd.DataFrame, span: int) -> pd.DataFrame:
    return frame.ewm(span=span, adjust=False, min_periods=max(2, span // 2)).mean()


def rsi(close: pd.DataFrame, period: int) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def cci(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, period: int) -> pd.DataFrame:
    typical_price = (high + low + close) / 3.0
    moving_average = typical_price.rolling(period, min_periods=max(2, period // 2)).mean()
    mean_deviation = (typical_price - moving_average).abs().rolling(period, min_periods=max(2, period // 2)).mean()
    return (typical_price - moving_average) / (0.015 * mean_deviation.replace(0.0, np.nan))


def macd_histogram(close: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
    fast_line = ema(close, fast)
    slow_line = ema(close, slow)
    macd_line = fast_line - slow_line
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line


def obv(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    direction = np.sign(close.diff()).fillna(0.0)
    signed_volume = direction * volume.fillna(0.0)
    return signed_volume.cumsum()


def bollinger_position(close: pd.DataFrame, window: int, num_std: float) -> pd.DataFrame:
    middle = sma(close, window)
    std = close.rolling(window, min_periods=max(2, window // 2)).std()
    return (close - middle) / (num_std * std.replace(0.0, np.nan))


def time_series_momentum(close: pd.DataFrame, lookback_months: int, skip_months: int) -> pd.DataFrame:
    monthly_close = close.resample("ME").last()
    recent = monthly_close.shift(skip_months)
    anchor = monthly_close.shift(lookback_months)
    signal = recent / anchor - 1.0
    return signal.reindex(close.index, method="ffill")


def variable_week_high(close: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    weekly_close = close.resample("W-FRI").last()
    signals = []
    for window in windows:
        rolling_high = weekly_close.rolling(window, min_periods=max(2, window // 2)).max()
        signals.append(weekly_close / rolling_high - 1.0)
    combined = sum(signals) / len(signals)
    return combined.reindex(close.index, method="ffill")


def price_action_score(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, weeks: int) -> pd.DataFrame:
    weekly_close = close.resample("W-FRI").last()
    weekly_high = high.resample("W-FRI").max()
    weekly_low = low.resample("W-FRI").min()
    up_weeks = (weekly_close.diff() > 0).astype(float).rolling(weeks, min_periods=max(2, weeks // 2)).mean()
    higher_highs = (weekly_high.diff() > 0).astype(float).rolling(weeks, min_periods=max(2, weeks // 2)).mean()
    higher_lows = (weekly_low.diff() > 0).astype(float).rolling(weeks, min_periods=max(2, weeks // 2)).mean()
    combined = (up_weeks + higher_highs + higher_lows) / 3.0
    return combined.reindex(close.index, method="ffill")


def average_dollar_volume(close: pd.DataFrame, volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return (close * volume).rolling(window, min_periods=max(2, window // 2)).mean()


def cross_sectional_rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True) - 0.5


def shift_for_open_execution(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {name: frame.shift(1) for name, frame in frames.items()}


def build_technical_factors(
    prices: PricePanel,
    config: AppConfig,
    factor_names: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    signal_prices = resample_price_panel(prices, config.strategy.signal_timeframe)
    sessions = prices.close.index
    indicator_config = config.indicators
    requested = factor_names or {
        "sma_crossover",
        "ema_crossover",
        "trend_filter",
        "breakout",
        "rsi",
        "cci",
        "macd",
        "obv",
        "bollinger",
        "ts_momentum",
        "variable_week_high",
        "price_action",
        "avg_dollar_volume",
    }

    factors: dict[str, pd.DataFrame] = {}

    sma_fast = sma_slow = sma_trend = None
    if {"sma_crossover", "trend_filter"} & requested:
        sma_fast = sma(signal_prices.close, indicator_config.sma_fast)
        sma_slow = sma(signal_prices.close, indicator_config.sma_slow)
    if "trend_filter" in requested:
        sma_trend = sma(signal_prices.close, indicator_config.sma_trend)
    ema_fast = ema_slow = None
    if "ema_crossover" in requested:
        ema_fast = ema(signal_prices.close, indicator_config.ema_fast)
        ema_slow = ema(signal_prices.close, indicator_config.ema_slow)
    if "breakout" in requested:
        breakout_window = max(5, indicator_config.breakout_weeks)
        breakout = signal_prices.close / signal_prices.high.rolling(
            breakout_window,
            min_periods=max(2, breakout_window // 2),
        ).max() - 1.0
        factors["breakout"] = _align_to_sessions(breakout, sessions)
    if "rsi" in requested:
        rsi_value = rsi(signal_prices.close, indicator_config.rsi_period)
        factors["rsi"] = _align_to_sessions((rsi_value - 50.0) / 50.0, sessions)
    if "cci" in requested:
        cci_value = cci(signal_prices.high, signal_prices.low, signal_prices.close, indicator_config.cci_period)
        factors["cci"] = _align_to_sessions(cci_value / 200.0, sessions)
    if "macd" in requested:
        macd_value = macd_histogram(
            signal_prices.close,
            indicator_config.macd_fast,
            indicator_config.macd_slow,
            indicator_config.macd_signal,
        )
        factors["macd"] = _align_to_sessions(macd_value / signal_prices.close.replace(0.0, np.nan), sessions)
    if "obv" in requested:
        obv_value = obv(signal_prices.close, signal_prices.volume)
        obv_trend = (obv_value - ema(obv_value, indicator_config.obv_ema)) / ema(
            obv_value.abs() + 1.0,
            indicator_config.obv_ema,
        )
        factors["obv"] = _align_to_sessions(obv_trend, sessions)
    if "bollinger" in requested:
        bollinger_value = bollinger_position(
            signal_prices.close,
            indicator_config.bollinger_window,
            indicator_config.bollinger_std,
        )
        factors["bollinger"] = _align_to_sessions(bollinger_value, sessions)
    if "ts_momentum" in requested:
        ts_mom = time_series_momentum(
            signal_prices.close,
            indicator_config.ts_mom_lookback_months,
            indicator_config.ts_mom_skip_months,
        )
        factors["ts_momentum"] = _align_to_sessions(ts_mom, sessions)
    if "variable_week_high" in requested:
        variable_high = variable_week_high(signal_prices.close, indicator_config.variable_week_high_windows)
        factors["variable_week_high"] = _align_to_sessions(variable_high, sessions)
    if "price_action" in requested:
        price_action = price_action_score(
            signal_prices.high,
            signal_prices.low,
            signal_prices.close,
            indicator_config.price_action_weeks,
        )
        factors["price_action"] = _align_to_sessions(price_action - 0.5, sessions)
    if "sma_crossover" in requested and sma_fast is not None and sma_slow is not None:
        factors["sma_crossover"] = _align_to_sessions((sma_fast - sma_slow) / sma_slow.replace(0.0, np.nan), sessions)
    if "ema_crossover" in requested and ema_fast is not None and ema_slow is not None:
        factors["ema_crossover"] = _align_to_sessions((ema_fast - ema_slow) / ema_slow.replace(0.0, np.nan), sessions)
    if "trend_filter" in requested and sma_trend is not None:
        factors["trend_filter"] = _align_to_sessions(
            (signal_prices.close - sma_trend) / sma_trend.replace(0.0, np.nan),
            sessions,
        )
    if "avg_dollar_volume" in requested:
        factors["avg_dollar_volume"] = average_dollar_volume(prices.close, prices.volume)

    return factors
