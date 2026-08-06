import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from algo_trading.config import AppConfig
from algo_trading.data import MacroDataUnavailableError, build_fundamental_factor_frames, load_macro_data
from algo_trading.env_utils import load_environment
from algo_trading.strategy import build_factor_library


def test_load_environment_reads_github_env_without_overwriting_existing_value(tmp_path: Path, monkeypatch) -> None:
    env_dir = tmp_path / ".github"
    env_dir.mkdir(parents=True)
    env_file = env_dir / ".env"
    env_file.write_text("FRED_API_KEY=from-file\nTEST_VALUE=hello\n", encoding="utf-8")

    monkeypatch.delenv("TEST_VALUE", raising=False)
    monkeypatch.setenv("FRED_API_KEY", "already-set")

    loaded = load_environment(start_dir=tmp_path)

    assert loaded == env_file
    assert os.environ["FRED_API_KEY"] == "already-set"
    assert os.environ["TEST_VALUE"] == "hello"


def test_build_factor_library_requires_membership_and_listing_history() -> None:
    index = pd.date_range("2024-01-01", periods=80, freq="B")
    close = pd.DataFrame(
        {
            "AAA": np.linspace(20.0, 40.0, len(index)),
            "BBB": np.r_[np.repeat(np.nan, 65), np.linspace(15.0, 18.0, 15)],
        },
        index=index,
    )
    volume = pd.DataFrame(5_000_000.0, index=index, columns=close.columns)
    membership = pd.DataFrame(True, index=index, columns=close.columns)
    membership.loc[:, "BBB"] = False
    membership.loc[index[-15:], "BBB"] = True
    macro = pd.DataFrame({"macro_composite": 0.0}, index=index)
    prices = SimpleNamespace(open=close, high=close, low=close, close=close, volume=volume)
    context = SimpleNamespace(
        prices=prices,
        fundamental_factors={},
        universe_membership=membership,
        macro=macro,
    )

    config = AppConfig()
    config.strategy.base_filter_requires_trend = False
    config.universe.min_history_days = 20

    _factors, base_mask, _macro_score = build_factor_library(context, config)

    assert not bool(base_mask.loc[index[10], "AAA"])
    assert bool(base_mask.loc[index[40], "AAA"])
    assert not bool(base_mask.loc[index[-1], "BBB"])


def test_load_macro_data_uses_cached_series_without_fred_key(tmp_path: Path, monkeypatch) -> None:
    sessions = pd.date_range("2024-01-02", periods=10, freq="B")
    cache_dir = tmp_path / "cache" / "fred"
    cache_dir.mkdir(parents=True)
    cache_index = pd.date_range("2023-12-01", periods=40, freq="B")
    pd.DataFrame({"VIXCLS": np.linspace(10.0, 30.0, len(cache_index))}, index=cache_index).to_csv(
        cache_dir / "VIXCLS.csv"
    )

    monkeypatch.delenv("FRED_API_KEY", raising=False)

    config = AppConfig()
    config.macro.zscore_window = 5
    config.macro.enabled_series = {series_id: series_id == "VIXCLS" for series_id in config.macro.enabled_series}

    macro = load_macro_data(sessions=sessions, config=config, cache_dir=tmp_path / "cache")

    assert list(macro.index) == list(sessions)
    assert "VIXCLS" in macro.columns
    assert "macro_composite" in macro.columns


def test_load_macro_data_lags_monthly_series_by_publication_delay(tmp_path: Path, monkeypatch) -> None:
    """A monthly FRED value must not be usable until ~45 days after its
    observation date (publication lag). Without the lag the macro risk gate
    would peek at unreleased monthly data."""
    sessions = pd.bdate_range("2024-01-02", "2024-06-30")
    cache_dir = tmp_path / "cache" / "fred"
    cache_dir.mkdir(parents=True)
    obs_index = pd.date_range("2024-01-01", "2024-05-01", freq="MS")
    pd.DataFrame({"CFNAI": np.linspace(0.1, 0.5, len(obs_index))}, index=obs_index).to_csv(
        cache_dir / "CFNAI.csv"
    )

    monkeypatch.delenv("FRED_API_KEY", raising=False)

    config = AppConfig()
    config.macro.zscore_window = 2
    config.macro.enabled_series = {sid: (sid == "CFNAI") for sid in config.macro.enabled_series}

    macro = load_macro_data(sessions=sessions, config=config, cache_dir=tmp_path / "cache")

    # Jan 1 observation is not released until ~Feb 15 (45-day lag). At a session
    # before that release date the standardized monthly value must be NaN.
    assert np.isnan(macro["CFNAI"].loc[pd.Timestamp("2024-01-15")])
    # After the Feb release (Mar 17 -> first session Mar 18) + 2-period z-score
    # warmup, the value is available and non-NaN.
    assert pd.notna(macro["CFNAI"].loc[pd.Timestamp("2024-03-18")])


def test_load_macro_data_raises_structured_error_without_fred_key_or_cache(tmp_path: Path, monkeypatch) -> None:
    sessions = pd.date_range("2024-01-02", periods=5, freq="B")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    config = AppConfig()
    config.macro.enabled_series = {series_id: series_id == "VIXCLS" for series_id in config.macro.enabled_series}

    with pytest.raises(MacroDataUnavailableError) as excinfo:
        load_macro_data(sessions=sessions, config=config, cache_dir=tmp_path / "cache")

    assert excinfo.value.cache_dir == tmp_path / "cache" / "fred"
    assert excinfo.value.missing_series == ["VIXCLS"]


def test_fundamental_factors_preserve_weekend_available_dates() -> None:
    sessions = pd.DatetimeIndex([pd.Timestamp("2024-03-29"), pd.Timestamp("2024-04-01")])
    close = pd.DataFrame({"AAA": [10.0, 11.0]}, index=sessions)
    statement_date = pd.Timestamp("2024-03-30")
    bundles = {
        "AAA": {
            "info": {"sharesOutstanding": 100.0},
            "quarterly_balance_sheet": pd.DataFrame(
                [[1000.0], [500.0]],
                index=["Total Assets", "Stockholders Equity"],
                columns=[statement_date],
            ),
        }
    }

    factors = build_fundamental_factor_frames(
        close=close,
        bundles=bundles,
        statement_lag_days=1,
        factor_names=["book_to_market"],
    )

    assert np.isnan(factors["book_to_market"].loc[pd.Timestamp("2024-03-29"), "AAA"])
    assert factors["book_to_market"].loc[pd.Timestamp("2024-04-01"), "AAA"] == 0.5
