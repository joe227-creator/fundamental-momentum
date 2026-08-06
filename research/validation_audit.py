"""Multi-seed, regime, sensitivity, and leakage audit for frozen baseline."""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import overfit_harness as oh
from algo_trading.timesfm_experiments import eligible_symbols, load_forecast_panel

from research_score import _evaluate_result, _load_reference, _prepare_strategy, _select


REGIMES = {
    "bull_2016_2017": ("2016-01-01", "2017-12-31"),
    "bear_2018": ("2018-01-01", "2018-12-31"),
    "covid_crash": ("2020-02-19", "2020-03-23"),
    "covid_recovery": ("2020-03-24", "2020-12-31"),
    "bear_2022": ("2022-01-01", "2022-12-31"),
    "bull_2023_2024": ("2023-01-01", "2024-12-31"),
    "weak_2025": ("2025-01-01", "2025-12-31"),
}


def _regime_metrics(curve: pd.DataFrame) -> pd.DataFrame:
    equity = curve["equity"].dropna()
    rows: list[dict[str, Any]] = []
    for name, (start, end) in REGIMES.items():
        sub = equity[(equity.index >= pd.Timestamp(start)) & (equity.index <= pd.Timestamp(end))]
        if len(sub) < 2:
            rows.append({"regime": name, "sessions": len(sub), "return": np.nan, "max_drawdown": np.nan, "sharpe": np.nan})
            continue
        returns = sub.pct_change().dropna()
        drawdown = sub / sub.cummax() - 1.0
        sharpe = np.sqrt(252.0) * (returns - 0.05 / 252.0).mean() / returns.std() if returns.std() else np.nan
        rows.append({
            "regime": name,
            "sessions": len(sub),
            "return": float(sub.iloc[-1] / sub.iloc[0] - 1.0),
            "max_drawdown": float(drawdown.min()),
            "sharpe": float(sharpe),
        })
    return pd.DataFrame(rows)


def _sensitivity(prepared: dict[str, Any], baseline_turnover: float) -> pd.DataFrame:
    changes = [
        ("weight_tilt_minus", "weight_tilt", max(0.0, prepared["weight_tilt"] - 0.02)),
        ("weight_tilt_plus", "weight_tilt", prepared["weight_tilt"] + 0.02),
        ("conf_vol_pow_minus", "conf_tilt_vol_pow", max(1.0, prepared["conf_tilt_pow"] - 1.0)),
        ("conf_vol_pow_plus", "conf_tilt_vol_pow", prepared["conf_tilt_pow"] + 1.0),
        ("vol_lookback_14", "vol_lookback", 14),
        ("vol_lookback_42", "vol_lookback", 42),
        ("veto_threshold_030", "veto_threshold", 0.30),
        ("veto_threshold_040", "veto_threshold", 0.40),
    ]
    original = {
        "weight_tilt": prepared["weight_tilt"],
        "conf_tilt_pow": prepared["conf_tilt_pow"],
        "vol_prior_frame": prepared["vol_prior_frame"],
        "veto_mask": prepared["veto_mask"],
    }
    rows: list[dict[str, Any]] = []
    try:
        for label, key, value in changes:
            if key == "weight_tilt":
                prepared["weight_tilt"] = float(value)
            elif key == "conf_tilt_vol_pow":
                prepared["conf_tilt_pow"] = float(value)
            elif key == "vol_lookback":
                frame = prepared["ctx"].prices.close.pct_change().rolling(int(value), min_periods=10).std().reindex(prepared["prior_ends"])
                frame.index = pd.DatetimeIndex(prepared["dates"])
                prepared["vol_prior_frame"] = frame
            elif key == "veto_threshold":
                prepared["veto_mask"], _, _, _ = oh.build_veto_mask(
                    prepared["state"], prepared["ctx"], prepared["predicted"], float(value), prepared["trailing_window"]
                )
            metrics, _, _, _ = _evaluate_result(prepared, _select(prepared), baseline_turnover)
            rows.append({"label": label, "parameter": key, "value": value, **metrics})
            gc.collect()
    finally:
        prepared["weight_tilt"] = original["weight_tilt"]
        prepared["conf_tilt_pow"] = original["conf_tilt_pow"]
        prepared["vol_prior_frame"] = original["vol_prior_frame"]
        prepared["veto_mask"] = original["veto_mask"]
    return pd.DataFrame(rows)


def _leakage_checks(prepared: dict[str, Any]) -> dict[str, Any]:
    state = prepared["state"]
    overlap = [i for i, split in enumerate(state.splits) if split.train_end >= split.test_start]
    lookahead = [str(date.date()) for date, prior in zip(prepared["dates"], prepared["prior_ends"]) if not prior < date]
    qpanel = load_forecast_panel(oh.CTX, oh.HOR, kind="volume", quantiles=True)
    cache_by_date: dict[pd.Timestamp, set[str]] = {}
    if qpanel is not None:
        for symbol, date in qpanel.index:
            cache_by_date.setdefault(pd.Timestamp(date), set()).add(str(symbol))
    eligible_pairs = missing_pairs = 0
    for date in prepared["dates"]:
        eligible = eligible_symbols(state, date)
        eligible_pairs += len(eligible)
        missing_pairs += sum(str(symbol) not in cache_by_date.get(date, set()) for symbol in eligible)
    return {
        "walk_forward_overlap_splits": overlap,
        "prior_end_lookahead_dates": lookahead,
        "eligible_forecast_pairs": eligible_pairs,
        "missing_eligible_forecasts": missing_pairs,
        "all_checks_pass": not overlap and not lookahead and missing_pairs == 0,
        "historical_snapshot_count": prepared["ctx"].metadata.get("historical_snapshot_count"),
        "data_notes": prepared["ctx"].metadata.get("notes", []),
    }


def run_validation(config_path: str, params_path: str, reference_path: str, artifact_dir: str, validation_config_path: str) -> int:
    settings = json.loads(Path(validation_config_path).read_text(encoding="utf-8"))
    n_seeds, base_seed, sigma = int(settings.get("n_seeds", 10)), int(settings.get("seed", 42)), float(settings.get("perturb_sigma", 0.03))
    baseline_turnover = _load_reference(Path(reference_path))
    if baseline_turnover is None:
        raise RuntimeError(f"Missing immutable baseline reference: {reference_path}")
    prepared = _prepare_strategy(config_path, params_path)
    base_metrics, curve, windows, trades = _evaluate_result(prepared, _select(prepared), baseline_turnover)
    seed_results: list[dict[str, Any]] = []
    for index in range(n_seeds):
        metrics, _, _, _ = _evaluate_result(prepared, _select(prepared, seed=base_seed + 1000 + index, sigma=sigma), baseline_turnover)
        metrics["seed"] = base_seed + 1000 + index
        seed_results.append(metrics)
        gc.collect()
    scores = np.array([row["research_score"] for row in seed_results])
    returns = np.array([row["mean_rolling_6m_return"] for row in seed_results])
    seed_summary = {
        "n_seeds": n_seeds,
        "score_mean": float(scores.mean()), "score_median": float(np.median(scores)), "score_std": float(scores.std(ddof=0)),
        "score_min": float(scores.min()), "score_max": float(scores.max()),
        "return_mean": float(returns.mean()), "return_median": float(np.median(returns)), "return_std": float(returns.std(ddof=0)),
        "return_min": float(returns.min()), "return_max": float(returns.max()),
        "return_std_over_mean": float(returns.std(ddof=0) / abs(returns.mean())),
        "instability_warning": bool(returns.std(ddof=0) > 0.05 * abs(returns.mean())),
    }
    regimes, sensitivity, leakage = _regime_metrics(curve), _sensitivity(prepared, baseline_turnover), _leakage_checks(prepared)
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    curve.to_csv(root / "equity_drawdown_curve.csv")
    windows.to_csv(root / "rolling_126_session_windows.csv", index=False)
    trades.to_csv(root / "trade_log.csv", index=False)
    regimes.to_csv(root / "regime_metrics.csv", index=False)
    sensitivity.to_csv(root / "parameter_sensitivity.csv", index=False)
    (root / "baseline_metrics.json").write_text(json.dumps(base_metrics, indent=2), encoding="utf-8")
    (root / "seed_results.json").write_text(json.dumps(seed_results, indent=2), encoding="utf-8")
    (root / "seed_summary.json").write_text(json.dumps(seed_summary, indent=2), encoding="utf-8")
    (root / "leakage_checks.json").write_text(json.dumps(leakage, indent=2, default=str), encoding="utf-8")
    output = {
        **{f"baseline_{key}": value for key, value in base_metrics.items() if isinstance(value, (int, float))},
        "seed_score_mean": seed_summary["score_mean"], "seed_score_median": seed_summary["score_median"],
        "seed_score_std": seed_summary["score_std"], "seed_score_min": seed_summary["score_min"], "seed_score_max": seed_summary["score_max"],
        "seed_return_mean": seed_summary["return_mean"], "seed_return_std": seed_summary["return_std"],
        "seed_return_std_over_mean": seed_summary["return_std_over_mean"], "seed_instability_warning": int(seed_summary["instability_warning"]),
        "leakage_checks_pass": int(leakage["all_checks_pass"]), "missing_eligible_forecasts": leakage["missing_eligible_forecasts"],
    }
    for key, value in output.items():
        print(f"METRIC {key}={value}")
    print(f"ARTIFACT_DIR {artifact_dir}")
    return 0
