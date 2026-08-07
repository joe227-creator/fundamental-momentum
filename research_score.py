"""Fixed Research-score evaluator for the OpenResearch experiment tree.

The strategy path is shared with ``overfit_harness.py``. This module adds the
user-defined objective, immutable baseline turnover reference, and text
artifacts needed to compare every branch under one evaluation contract.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")

import overfit_harness as oh
from algo_trading.config import AppConfig
from algo_trading.timesfm_experiments import RF_ANNUAL, load_forecast_panel
from algo_trading.timesfm_engine import mean_forecast_sum


WINDOW_SESSIONS = 126
START_DATE = pd.Timestamp("2015-01-02")
DEFAULT_REFERENCE = Path("research/baseline_reference.json")


def complete_window_returns(
    equity: pd.Series, window_sessions: int = WINDOW_SESSIONS
) -> pd.DataFrame:
    """Return one row per complete non-overlapping equity window."""
    values = equity.dropna().sort_index()
    count = len(values) // window_sessions
    rows: list[dict[str, Any]] = []
    for number in range(count):
        chunk = values.iloc[number * window_sessions : (number + 1) * window_sessions]
        rows.append(
            {
                "window": number + 1,
                "start_date": pd.Timestamp(chunk.index[0]).date().isoformat(),
                "end_date": pd.Timestamp(chunk.index[-1]).date().isoformat(),
                "sessions": len(chunk),
                "return": float(chunk.iloc[-1] / chunk.iloc[0] - 1.0),
            }
        )
    return pd.DataFrame(rows)


def maximum_drawdown(equity: pd.Series) -> tuple[float, pd.Series]:
    values = equity.dropna().sort_index()
    drawdown = values / values.cummax() - 1.0
    return (float(drawdown.min()) if not drawdown.empty else np.nan), drawdown


def annualized_sharpe(equity: pd.Series, rf_annual: float = RF_ANNUAL) -> float:
    values = equity.dropna().sort_index()
    returns = values.pct_change().dropna()
    if len(returns) < 2 or not np.isfinite(returns.std()) or returns.std() == 0:
        return np.nan
    excess = returns - rf_annual / 252.0
    return float(np.sqrt(252.0) * excess.mean() / returns.std())


def turnover_measure(equity: pd.Series, trade_log: pd.DataFrame) -> float:
    """Match existing extended_metrics turnover: mean monthly buys / initial equity."""
    if equity.empty or trade_log.empty:
        return 0.0
    buys = trade_log[trade_log.get("action", pd.Series(dtype=str)).eq("BUY")].copy()
    if buys.empty:
        return 0.0
    if "gross_value" in buys.columns:
        gross = pd.to_numeric(buys["gross_value"], errors="coerce")
    else:
        gross = pd.to_numeric(buys.get("shares"), errors="coerce") * pd.to_numeric(
            buys.get("price"), errors="coerce"
        )
    buys["gross_value"] = gross
    buys["month"] = pd.to_datetime(buys["date"]).dt.to_period("M")
    monthly_buy = buys.groupby("month")["gross_value"].sum().dropna()
    if monthly_buy.empty or float(equity.iloc[0]) == 0:
        return 0.0
    return float(monthly_buy.mean() / float(equity.iloc[0]))


def research_score(
    mean_window_return: float,
    maximum_dd: float,
    sharpe: float,
    turnover: float,
    baseline_turnover: float,
    win_rate: float,
) -> dict[str, float]:
    """Apply fixed Research-score formula without parameter changes."""
    values = [
        mean_window_return,
        maximum_dd,
        sharpe,
        turnover,
        baseline_turnover,
        win_rate,
    ]
    if not all(np.isfinite(value) for value in values):
        raise ValueError(f"Non-finite score input: {values}")
    return_on_risk = mean_window_return / max(abs(maximum_dd), 1e-6)
    drawdown_penalty = 0.35 * max(0.0, -0.50 - maximum_dd)
    sharpe_penalty = 0.15 * max(0.0, 0.80 - sharpe)
    turnover_penalty = 0.10 * max(0.0, turnover - baseline_turnover)
    score = (
        mean_window_return
        + 0.20 * return_on_risk
        + 0.10 * win_rate
        - drawdown_penalty
        - sharpe_penalty
        - turnover_penalty
    )
    return {
        "research_score": float(score),
        "mean_rolling_6m_return": float(mean_window_return),
        "return_on_risk": float(return_on_risk),
        "win_rate": float(win_rate),
        "maximum_drawdown": float(maximum_dd),
        "sharpe": float(sharpe),
        "turnover": float(turnover),
        "baseline_turnover": float(baseline_turnover),
        "drawdown_penalty": float(drawdown_penalty),
        "sharpe_penalty": float(sharpe_penalty),
        "turnover_penalty": float(turnover_penalty),
    }


def _load_reference(path: Path) -> float | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    value = float(raw["baseline_turnover"])
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"Invalid baseline turnover in {path}: {value}")
    return value


def _load_predictions(params: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    context_len = int(params.get("forecast_context_len", 256))
    horizon = int(params.get("veto_horizon", 21))
    use_quantile = bool(params.get("use_quantile_veto", False))
    if use_quantile:
        panel = load_forecast_panel(context_len, oh.HOR, kind="volume", quantiles=True)
        if panel is None:
            raise RuntimeError("Missing quantile volume forecast cache")
        column = int(params.get("quantile_col", 0))
        columns = [f"q{column}_h{h}" for h in range(horizon)]
        predicted = panel[columns].sum(axis=1) / horizon
    else:
        panel = load_forecast_panel(context_len, oh.HOR, kind="volume", quantiles=False)
        if panel is None:
            raise RuntimeError("Missing volume forecast cache")
        predicted = mean_forecast_sum(panel, horizon) / horizon
    return predicted.unstack(level="symbol"), context_len


def _prepare_strategy(config_path: str, params_path: str) -> dict[str, Any]:
    params = json.loads(Path(params_path).read_text(encoding="utf-8"))
    oh.CTX = int(params.get("forecast_context_len", 256))
    cfg, ctx = oh.load_context(config_path)
    cfg = oh.apply_params_to_config(cfg, params)
    oh.set_config(cfg)
    state = oh.build_factor_state(ctx, cfg)
    benchmark = oh.benchmark_close(ctx, "SPY")
    predicted, _ = _load_predictions(params)
    threshold = float(params.get("veto_threshold", 0.775))
    trailing_window = int(params.get("trailing_vol_window", 53))
    veto_mask, base_scores, prior_ends, dates = oh.build_veto_mask(
        state,
        ctx,
        predicted,
        threshold,
        trailing_window,
        use_relative_veto=bool(params.get("use_relative_veto", False)),
        relative_veto_pct=float(params.get("relative_veto_pct", 0.05)),
    )
    consensus_blend = float(np.clip(params.get("consensus_blend", 0.0), 0.0, 1.0))
    if consensus_blend > 0.0:
        consensus_scores: dict[pd.Timestamp, pd.Series] = {}
        for date in dates:
            if date not in base_scores:
                continue
            factor_ranks = [
                state.ranked_factors[name].loc[date].rename(name)
                for name in state.factor_names
                if name in state.ranked_factors and date in state.ranked_factors[name].index
            ]
            if not factor_ranks:
                continue
            median_rank = pd.concat(factor_ranks, axis=1).median(axis=1).fillna(0.0)
            base = base_scores[date].reindex(median_rank.index).fillna(0.0)
            blended = (1.0 - consensus_blend) * base + consensus_blend * median_rank
            eligible = state.base_mask.loc[date].fillna(False)
            blended = blended[eligible].replace([np.inf, -np.inf], np.nan).dropna()
            if bool(params.get("require_positive_composite", True)):
                blended = blended[blended > 0.0]
            consensus_scores[date] = blended.sort_values(ascending=False)
        base_scores = consensus_scores

    # Optional return-forecast tilt, retained for isolated ablations already
    # supported by the existing harness.
    alpha_weight = float(params.get("tfm_alpha_weight", 0.0))
    if alpha_weight > 0.0:
        panel = load_forecast_panel(oh.CTX, oh.HOR, kind="logret", quantiles=False)
        if panel is None:
            raise RuntimeError("Missing log-return forecast cache")
        alpha = panel[[f"m{h}" for h in range(oh.HOR)]].sum(axis=1).unstack(level="symbol")
        for date in list(base_scores):
            if date not in alpha.index:
                continue
            values = alpha.loc[date].dropna()
            std = float(values.std())
            if values.empty or std == 0:
                continue
            z = ((values - float(values.mean())) / std).reindex(base_scores[date].index).fillna(0.0)
            base_scores[date] = base_scores[date] + alpha_weight * z

    vol_scaled = bool(params.get("vol_scaled_weights", False))
    vol_prior_frame = None
    if vol_scaled:
        lookback = int(params.get("vol_lookback", 63))
        ret_std = ctx.prices.close.pct_change().rolling(lookback, min_periods=10).std()
        vol_prior_frame = ret_std.reindex(prior_ends)
        vol_prior_frame.index = pd.DatetimeIndex(dates)

    corr_threshold = float(params.get("corr_threshold", 0.0))
    rets_panel = ctx.prices.close.pct_change() if corr_threshold > 0 else None
    corr_prior_map = dict(zip(dates, prior_ends)) if corr_threshold > 0 else None
    corr_sessions = ctx.prices.sessions if corr_threshold > 0 else None

    median_gap = None
    vol_factor_map = None
    conf_tilt = bool(params.get("conf_tilt", False))
    if conf_tilt:
        gaps = []
        for scores in base_scores.values():
            clean = scores.replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean) >= 2:
                top = clean.sort_values(ascending=False).head(2)
                gaps.append(abs(float(top.iloc[0]) - float(top.iloc[1])))
        median_gap = float(np.median(gaps)) if gaps else None
        if bool(params.get("conf_tilt_vol", False)):
            vol_window = int(params.get("conf_tilt_vol_window", 21))
            vol_cap = float(params.get("conf_tilt_vol_cap", 2.0))
            vol_power = float(params.get("conf_tilt_vol_pow", 1.0))
            benchmark_vol = benchmark.pct_change().rolling(vol_window, min_periods=5).std()
            prior_by_date = dict(zip(dates, prior_ends))
            observed = [
                float(benchmark_vol.loc[end])
                for end in prior_ends
                if end in benchmark_vol.index
                and np.isfinite(benchmark_vol.loc[end])
                and benchmark_vol.loc[end] > 0
            ]
            median_vol = float(np.median(observed)) if observed else 1.0
            vol_factor_map = {}
            for date in dates:
                end = prior_by_date.get(date)
                factor = 1.0
                if end in benchmark_vol.index and median_vol > 0:
                    value = float(benchmark_vol.loc[end])
                    if np.isfinite(value) and value > 0:
                        factor = min(vol_cap, (value / median_vol) ** vol_power)
                vol_factor_map[date] = factor

    regime_scale = None
    if bool(params.get("use_regime_cash", False)):
        sma_window = int(params.get("regime_cash_sma", params.get("sma_trend", 201)))
        cash_fraction = float(params.get("regime_cash_fraction", 0.5))
        sma = benchmark.rolling(sma_window, min_periods=10).mean()
        regime_scale = {}
        for date, end in zip(dates, prior_ends):
            regime_scale[date] = (
                cash_fraction
                if end in sma.index
                and np.isfinite(benchmark.get(end, np.nan))
                and np.isfinite(sma.get(end, np.nan))
                and benchmark.loc[end] < sma.loc[end]
                else 1.0
            )

    return {
        "params": params,
        "cfg": cfg,
        "ctx": ctx,
        "state": state,
        "benchmark": benchmark,
        "veto_mask": veto_mask,
        "base_scores": base_scores,
        "N": int(params["max_holdings"]),
        "vol_prior_frame": vol_prior_frame,
        "vol_scaled": vol_scaled,
        "corr_threshold": corr_threshold,
        "rets_panel": rets_panel,
        "corr_prior_map": corr_prior_map,
        "corr_sessions": corr_sessions,
        "weight_tilt": float(params.get("weight_tilt", 0.0)),
        "conf_tilt": conf_tilt,
        "median_gap": median_gap,
        "conf_tilt_pow": float(params.get("conf_tilt_pow", 1.0)),
        "conf_tilt_vol": bool(params.get("conf_tilt_vol", False)),
        "vol_factor_map": vol_factor_map,
        "consensus_blend": consensus_blend,
        "regime_scale": regime_scale,
        "prior_ends": prior_ends,
        "dates": dates,
    }


def _select(prepared: dict[str, Any], seed: int | None = None, sigma: float = 0.0):
    rng = np.random.default_rng(seed) if sigma > 0 and seed is not None else None
    selected = oh.apply_veto_and_select(
        prepared["base_scores"],
        prepared["veto_mask"],
        prepared["N"],
        rng=rng,
        perturb_sigma=sigma,
        vol_prior_frame=prepared["vol_prior_frame"],
        vol_scaled=prepared["vol_scaled"],
        corr_threshold=prepared["corr_threshold"],
        rets_panel=prepared["rets_panel"],
        corr_prior_map=prepared["corr_prior_map"],
        corr_sessions=prepared["corr_sessions"],
        weight_tilt=prepared["weight_tilt"],
        conf_tilt=prepared["conf_tilt"],
        median_gap=prepared["median_gap"],
        conf_tilt_pow=prepared["conf_tilt_pow"],
        conf_tilt_vol=prepared["conf_tilt_vol"],
        vol_factor_map=prepared["vol_factor_map"],
    )
    if prepared["regime_scale"] is not None:
        selected = {
            date: {symbol: weight * prepared["regime_scale"].get(date, 1.0) for symbol, weight in weights.items()}
            for date, weights in selected.items()
        }
    return selected


def _evaluate_result(prepared: dict[str, Any], selection: dict[pd.Timestamp, dict[str, float]], baseline_turnover: float) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    params = prepared["params"]
    result = oh.run_weighted_backtest(
        prepared["ctx"],
        selection,
        prepared["cfg"],
        tc_rate=float(params.get("cost_per_side", 0.0003)),
        start_date=START_DATE,
        rebalance_persistence=float(params.get("rebalance_persistence", 0.0)),
    )
    equity = result["equity_curve"]["equity"].dropna()
    if equity.empty or not equity.index.is_monotonic_increasing or (equity <= 0).any():
        raise ValueError("Invalid equity curve")
    windows = complete_window_returns(equity)
    if windows.empty or len(windows) < 2:
        raise ValueError("At least two complete 126-session windows required")
    maximum_dd, drawdown = maximum_drawdown(equity)
    sharpe = annualized_sharpe(equity)
    turnover = turnover_measure(equity, result["trade_log"])
    mean_return = float(windows["return"].mean())
    win_rate = float((windows["return"] > 0).mean())
    metrics = research_score(mean_return, maximum_dd, sharpe, turnover, baseline_turnover, win_rate)
    metrics.update(
        {
            "n_windows": int(len(windows)),
            "window_return_median": float(windows["return"].median()),
            "window_return_min": float(windows["return"].min()),
            "window_return_max": float(windows["return"].max()),
            "window_return_std": float(windows["return"].std(ddof=0)),
            "ending_equity": float(equity.iloc[-1]),
            "start_date": equity.index[0].date().isoformat(),
            "end_date": equity.index[-1].date().isoformat(),
        }
    )
    curve = result["equity_curve"].copy()
    curve["drawdown"] = drawdown.reindex(curve.index)
    return metrics, curve, windows, result["trade_log"]


def _write_artifacts(
    artifact_dir: Path,
    metrics: dict[str, Any],
    curve: pd.DataFrame,
    windows: pd.DataFrame,
    trades: pd.DataFrame,
    seed_results: list[dict[str, Any]],
    cfg: AppConfig,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(artifact_dir / "equity_drawdown_curve.csv")
    windows.to_csv(artifact_dir / "rolling_126_session_windows.csv", index=False)
    trades.to_csv(artifact_dir / "trade_log.csv", index=False)
    (artifact_dir / "seed_results.json").write_text(json.dumps(seed_results, indent=2), encoding="utf-8")
    (artifact_dir / "resolved_config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str), encoding="utf-8")
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/research.json")
    parser.add_argument("--params", default="research_params.json")
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--artifact-dir", default=".openresearch/artifacts")
    parser.add_argument("--perturb-runs", type=int, default=5)
    parser.add_argument("--perturb-sigma", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-baseline-reference", action="store_true")
    args = parser.parse_args()

    optuna_config = Path("research/optuna_config.json")
    if optuna_config.exists():
        from research.optuna_runner import run_optuna

        return run_optuna(
            config_path=args.config,
            params_path=args.params,
            reference_path=args.reference,
            artifact_dir=args.artifact_dir,
            optuna_config_path=str(optuna_config),
            perturb_runs=args.perturb_runs,
            perturb_sigma=args.perturb_sigma,
            seed=args.seed,
        )

    prepared = _prepare_strategy(args.config, args.params)
    base_selection = _select(prepared)
    provisional_turnover = 0.0
    if _load_reference(Path(args.reference)) is None:
        if not args.bootstrap_baseline_reference:
            raise RuntimeError(f"Missing immutable baseline reference: {args.reference}")
        provisional, _, _, _ = _evaluate_result(prepared, base_selection, 0.0)
        provisional_turnover = provisional["turnover"]
        reference_path = Path(args.reference)
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(
            json.dumps(
                {
                    "baseline_turnover": provisional_turnover,
                    "window_sessions": WINDOW_SESSIONS,
                    "turnover_definition": "mean monthly BUY gross value divided by initial equity",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    baseline_turnover = _load_reference(Path(args.reference))
    if baseline_turnover is None:
        baseline_turnover = provisional_turnover

    metrics, curve, windows, trades = _evaluate_result(prepared, base_selection, baseline_turnover)
    seed_results: list[dict[str, Any]] = []
    for index in range(max(0, int(args.perturb_runs))):
        seed = int(args.seed) + 1000 + index
        selected = _select(prepared, seed=seed, sigma=float(args.perturb_sigma))
        seed_metrics, _, _, _ = _evaluate_result(prepared, selected, baseline_turnover)
        seed_metrics["seed"] = seed
        seed_results.append(seed_metrics)

    if seed_results:
        scores = np.array([row["research_score"] for row in seed_results], dtype=float)
        metrics.update(
            {
                "perturb_score_mean": float(scores.mean()),
                "perturb_score_std": float(scores.std(ddof=0)),
                "perturb_score_min": float(scores.min()),
                "perturb_score_max": float(scores.max()),
            }
        )
    else:
        metrics.update(
            {
                "perturb_score_mean": np.nan,
                "perturb_score_std": np.nan,
                "perturb_score_min": np.nan,
                "perturb_score_max": np.nan,
            }
        )

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    metrics["commit"] = commit
    metrics["config_path"] = args.config
    metrics["params_path"] = args.params
    _write_artifacts(Path(args.artifact_dir), metrics, curve, windows, trades, seed_results, prepared["cfg"])

    for key in [
        "research_score",
        "mean_rolling_6m_return",
        "return_on_risk",
        "win_rate",
        "maximum_drawdown",
        "sharpe",
        "turnover",
        "baseline_turnover",
        "drawdown_penalty",
        "sharpe_penalty",
        "turnover_penalty",
        "n_windows",
        "window_return_median",
        "window_return_min",
        "window_return_std",
        "perturb_score_mean",
        "perturb_score_std",
        "perturb_score_min",
        "perturb_score_max",
    ]:
        value = metrics[key]
        print(f"METRIC {key}={value}")
    print(f"ARTIFACT_DIR {args.artifact_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
