"""Calendar guard for the champion live runner.

The champion rebalances monthly (``rebalance_frequency: "monthly"``), so the
guard returns exit 0 only when the candidate session is the first NYSE trading
day of its month, and exit 2 otherwise. Used by ``run_champion_live.ps1`` to skip
non-rebalance days so the monthly rebalance is never missed.

The guard always prints ``SESSION=YYYY-MM-DD`` (the candidate session) as its
first stdout line so the launcher can parse it and pass it to ``run_champion.py
--as-of``.

Two modes:

* Explicit (manual / test / ``-AsOf`` override):
    python is_first_nyse_rebalance_session.py YYYY-MM-DD
  The candidate session is the supplied date (must be an NYSE trading day).

* Auto (scheduled run, no argument): the candidate is the **last completed NYSE
  session** -- the most recent session whose 16:00 ET close has passed, computed
  from the current UTC time in America/New_York. This makes the guard
  timezone-correct: a 21:00 SGT trigger (= 08:00/09:00 ET the same calendar day,
  before the US open) evaluates the most recent *completed* US session, not the
  SGT-local or not-yet-traded ET date. In auto
  mode, the guard returns the month's first NYSE session (``SESSION=...``) once
  that first session has completed, so later days of the same month can catch up
  a missed first-day run if the state file has not been written yet.

The "has this month already been processed?" catch-up check is intentionally NOT
in this guard -- it lives in ``run_champion_live.ps1`` via a state file so the
guard stays pure and unit-testable.
"""
from __future__ import annotations

import sys

import pandas as pd
import pandas_market_calendars as mcal


def _last_completed_nyse_session(now_utc: pd.Timestamp | None = None) -> pd.Timestamp | None:
    """Most recent NYSE session whose 16:00 ET close has passed, as a naive date."""
    if now_utc is None:
        now_utc = pd.Timestamp.now(tz="UTC")
    now_et = now_utc.tz_convert("America/New_York")
    cal = mcal.get_calendar("NYSE")
    start = (now_et - pd.Timedelta(days=14)).normalize()
    sched = cal.schedule(start_date=start.date(), end_date=now_et.date())
    if sched.empty:
        return None
    completed = sched[sched["market_close"] <= now_et]
    if completed.empty:
        return None
    last_open = completed.index[-1]
    return pd.Timestamp(last_open).tz_localize(None).normalize()


def _first_nyse_session_of_month(target: pd.Timestamp) -> pd.DatetimeIndex:
    month_start = pd.Timestamp(target.year, target.month, 1)
    month_end = month_start + pd.offsets.MonthEnd(0)
    cal = mcal.get_calendar("NYSE")
    schedule = cal.schedule(start_date=month_start.date(), end_date=month_end.date())
    return pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()


def main() -> int:
    if len(sys.argv) not in (1, 2, 3):
        print("Usage: is_first_nyse_rebalance_session.py [YYYY-MM-DD] [--now-utc ISO_TIMESTAMP]")
        return 1

    auto_mode = len(sys.argv) == 1 or (len(sys.argv) == 3 and sys.argv[1] == "--now-utc")
    if len(sys.argv) == 2:
        target = pd.Timestamp(sys.argv[1]).normalize()
        if target.tzinfo is not None:
            target = target.tz_localize(None)
    elif auto_mode:
        now_utc = pd.Timestamp(sys.argv[2]) if len(sys.argv) == 3 else None
        candidate = _last_completed_nyse_session(now_utc=now_utc)
        if candidate is None:
            print("No completed NYSE session found in the last 14 days.")
            return 2
        target = candidate
    else:
        print("Usage: is_first_nyse_rebalance_session.py [YYYY-MM-DD] [--now-utc ISO_TIMESTAMP]")
        return 1

    sessions = _first_nyse_session_of_month(target)
    if len(sessions) == 0:
        print(f"SESSION={target.strftime('%Y-%m-%d')}")
        print(f"{target.date()} is not an NYSE trading day.")
        return 2

    first_session = pd.Timestamp(sessions[0]).normalize()

    if auto_mode:
        print(f"SESSION={first_session.strftime('%Y-%m-%d')}")
        print(f"CANDIDATE_SESSION={target.strftime('%Y-%m-%d')}")
        if target < first_session:
            print(f"{target.date()} is before the first NYSE trading day of the month ({first_session.date()}).")
            return 2
        print(f"{first_session.date()} is the first NYSE trading day of the month; latest completed session is {target.date()}.")
        return 0

    print(f"SESSION={target.strftime('%Y-%m-%d')}")
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
