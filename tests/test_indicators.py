import numpy as np
import pandas as pd

from algo_trading.indicators import _align_to_sessions, cross_sectional_rank, time_series_momentum


def test_time_series_momentum_is_positive_for_uptrend() -> None:
    index = pd.date_range("2020-01-01", periods=500, freq="B")
    close = pd.DataFrame({"AAA": np.linspace(10.0, 50.0, len(index))}, index=index)
    signal = time_series_momentum(close, lookback_months=12, skip_months=1)
    assert signal.iloc[-1, 0] > 0.0


def test_cross_sectional_rank_is_centered() -> None:
    frame = pd.DataFrame(
        [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
        columns=["A", "B", "C"],
    )
    ranked = cross_sectional_rank(frame)
    assert np.isclose(ranked.loc[frame.index[0], "A"], -1 / 6)
    assert np.isclose(ranked.loc[frame.index[0], "C"], 0.5)


def test_align_to_sessions_preserves_non_session_period_end_values() -> None:
    signal = pd.DataFrame(
        {"AAA": [42.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-03-31")]),
    )
    sessions = pd.DatetimeIndex([pd.Timestamp("2024-03-29"), pd.Timestamp("2024-04-01")])

    aligned = _align_to_sessions(signal, sessions)

    assert np.isnan(aligned.loc[pd.Timestamp("2024-03-29"), "AAA"])
    assert aligned.loc[pd.Timestamp("2024-04-01"), "AAA"] == 42.0
