from __future__ import annotations

from datetime import date, datetime

import pandas as pd


def normalize_timestamp(value: str | date | datetime | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).normalize().tz_localize(None)


def get_trading_sessions(start: str | date | datetime, end: str | date | datetime) -> pd.DatetimeIndex:
    import pandas_market_calendars as mcal

    calendar = mcal.get_calendar("NYSE")
    schedule = calendar.schedule(start_date=normalize_timestamp(start), end_date=normalize_timestamp(end))
    sessions = pd.DatetimeIndex(schedule.index)
    return sessions.tz_localize(None).normalize()


def first_trading_days_of_weeks(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    frame = pd.DataFrame({"session": sessions})
    iso = frame["session"].dt.isocalendar()
    weekly = frame.groupby([iso["year"], iso["week"]], sort=True)["session"].min()
    return pd.DatetimeIndex(weekly.to_list())


def first_trading_days_of_months(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    frame = pd.DataFrame({"session": sessions})
    monthly = frame.groupby(frame["session"].dt.to_period("M"), sort=True)["session"].min()
    return pd.DatetimeIndex(monthly.to_list())


def select_rebalance_sessions(sessions: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    frequency = frequency.lower()
    if frequency == "weekly":
        return first_trading_days_of_weeks(sessions)
    if frequency == "biweekly":
        weekly = first_trading_days_of_weeks(sessions)
        return weekly[::2]
    if frequency == "monthly":
        return first_trading_days_of_months(sessions)
    raise ValueError(f"Unsupported rebalance frequency: {frequency}")


def is_rebalance_session(
    target: str | date | datetime | pd.Timestamp,
    sessions: pd.DatetimeIndex,
    frequency: str,
) -> bool:
    target_ts = normalize_timestamp(target)
    rebalance_sessions = set(select_rebalance_sessions(sessions, frequency))
    return target_ts in rebalance_sessions


def next_rebalance_session(
    target: str | date | datetime | pd.Timestamp,
    sessions: pd.DatetimeIndex,
    frequency: str,
) -> pd.Timestamp:
    target_ts = normalize_timestamp(target)
    rebalance_sessions = select_rebalance_sessions(sessions, frequency)
    position = rebalance_sessions.searchsorted(target_ts, side="left")
    if position >= len(rebalance_sessions):
        raise ValueError("Target is later than the last available rebalance session.")
    return pd.Timestamp(rebalance_sessions[position])


def previous_session(target: str | date | datetime | pd.Timestamp, sessions: pd.DatetimeIndex) -> pd.Timestamp:
    target_ts = normalize_timestamp(target)
    position = sessions.searchsorted(target_ts, side="left") - 1
    if position < 0:
        raise ValueError("Target is earlier than the first available trading session.")
    return pd.Timestamp(sessions[position])


def next_or_same_session(target: str | date | datetime | pd.Timestamp, sessions: pd.DatetimeIndex) -> pd.Timestamp:
    target_ts = normalize_timestamp(target)
    position = sessions.searchsorted(target_ts, side="left")
    if position >= len(sessions):
        raise ValueError("Target is later than the last available trading session.")
    return pd.Timestamp(sessions[position])


def is_first_trading_day_of_week(target: str | date | datetime | pd.Timestamp, sessions: pd.DatetimeIndex) -> bool:
    target_ts = normalize_timestamp(target)
    weekly = set(first_trading_days_of_weeks(sessions))
    return target_ts in weekly