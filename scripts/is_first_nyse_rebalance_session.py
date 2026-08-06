"""Calendar guard for the champion live runner.

The champion rebalances monthly (``rebalance_frequency: "monthly"``), so the
guard returns exit 0 only when the target date is the first NYSE trading day of
its month, and exit 2 otherwise. Used by ``run_champion_live.ps1`` to skip
non-rebalance days so the monthly rebalance is never missed.

Usage:
    python is_first_nyse_rebalance_session.py YYYY-MM-DD
"""
from __future__ import annotations

import sys

import pandas as pd
import pandas_market_calendars as mcal


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: is_first_nyse_rebalance_session.py YYYY-MM-DD")
        return 1

    target = pd.Timestamp(sys.argv[1]).normalize()
    if target.tzinfo is not None:
        target = target.tz_localize(None)

    month_start = pd.Timestamp(target.year, target.month, 1)
    month_end = month_start + pd.offsets.MonthEnd(0)
    calendar = mcal.get_calendar("NYSE")
    schedule = calendar.schedule(start_date=month_start.date(), end_date=month_end.date())
    sessions = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()

    if len(sessions) == 0:
        print(f"{target.date()} is not an NYSE trading day.")
        return 2

    first_session = pd.Timestamp(sessions[0]).normalize()
    if target not in sessions:
        print(f"{target.date()} is not an NYSE trading day; first of month is {first_session.date()}.")
        return 2
    if target != first_session:
        print(f"{target.date()} is not the first NYSE trading day of the month; first is {first_session.date()}.")
        return 2

    print(f"{target.date()} is the first NYSE trading day of the month.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
