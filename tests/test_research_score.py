import numpy as np
import pandas as pd
import pytest

from research_score import complete_window_returns, research_score


def test_complete_windows_exclude_incomplete_tail() -> None:
    index = pd.date_range("2020-01-01", periods=253, freq="D")
    equity = pd.Series(np.arange(1.0, 254.0), index=index)

    windows = complete_window_returns(equity, window_sessions=126)

    assert len(windows) == 2
    assert windows["sessions"].tolist() == [126, 126]
    assert len(equity) - 2 * 126 == 1


def test_research_score_applies_fixed_penalties() -> None:
    result = research_score(
        mean_window_return=0.10,
        maximum_dd=-0.60,
        sharpe=0.50,
        turnover=0.30,
        baseline_turnover=0.10,
        win_rate=0.75,
    )

    expected = 0.10 + 0.20 * (0.10 / 0.60) + 0.10 * 0.75 - 0.35 * 0.10 - 0.15 * 0.30 - 0.10 * 0.20
    assert result["research_score"] == pytest.approx(expected)
    assert result["drawdown_penalty"] == pytest.approx(0.035)
    assert result["sharpe_penalty"] == pytest.approx(0.045)
    assert result["turnover_penalty"] == pytest.approx(0.02)
