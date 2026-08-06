import pandas as pd

from algo_trading.config import AppConfig
from algo_trading.strategy import _group_weighted_weights, build_walk_forward_splits


def test_walk_forward_splits_require_train_window() -> None:
    rebalance_dates = pd.date_range("2018-01-02", periods=72, freq="MS")
    config = AppConfig()
    config.walk_forward.train_years = 2
    config.walk_forward.test_months = 6
    config.walk_forward.step_months = 6
    config.walk_forward.min_train_rebalances = 12

    splits = build_walk_forward_splits(rebalance_dates, config)

    assert splits
    first = splits[0]
    assert len(first.train_dates) >= 12
    assert first.train_end < first.test_start
    assert len(first.test_dates) > 0


def test_group_weighted_weights_respect_explicit_fundamental_split() -> None:
    weights = _group_weighted_weights(
        factor_names=["sma_crossover", "trend_filter", "asset_growth", "earnings_yield"],
        ic_scores=pd.Series(0.0, index=["sma_crossover", "trend_filter", "asset_growth", "earnings_yield"]),
        use_ic_within_groups=False,
        fundamental_group_weight=0.075,
        fundamental_factor_weights={"asset_growth": 2.0, "earnings_yield": 1.0},
    )

    assert abs(weights["sma_crossover"] - 0.4625) < 1e-9
    assert abs(weights["trend_filter"] - 0.4625) < 1e-9
    assert abs(weights["asset_growth"] - 0.05) < 1e-9
    assert abs(weights["earnings_yield"] - 0.025) < 1e-9