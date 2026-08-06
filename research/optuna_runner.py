"""Direction-specific Optuna wrapper for fixed Research-score experiments."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import optuna

from research_score import (
    _evaluate_result,
    _load_reference,
    _prepare_strategy,
    _select,
    _write_artifacts,
)


def _suggest(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    kind = spec["type"]
    if kind == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
    if kind == "float":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(f"Unsupported Optuna parameter type: {kind}")


def _evaluate_params(
    config_path: str,
    base_params: dict[str, Any],
    params: dict[str, Any],
    baseline_turnover: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    merged = dict(base_params)
    merged.update(params)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
        json.dump(merged, handle)
        trial_path = handle.name
    try:
        prepared = _prepare_strategy(config_path, trial_path)
        selection = _select(prepared)
        metrics, _, _, _ = _evaluate_result(prepared, selection, baseline_turnover)
        return metrics, prepared
    finally:
        Path(trial_path).unlink(missing_ok=True)


def run_optuna(
    config_path: str,
    params_path: str,
    reference_path: str,
    artifact_dir: str,
    optuna_config_path: str,
    perturb_runs: int,
    perturb_sigma: float,
    seed: int,
) -> int:
    direction = json.loads(Path(optuna_config_path).read_text(encoding="utf-8"))
    base_params = json.loads(Path(params_path).read_text(encoding="utf-8"))
    fixed = dict(direction.get("fixed", {}))
    search = dict(direction["search"])
    n_trials = int(direction["n_trials"])
    study_seed = int(direction.get("seed", seed))
    baseline_turnover = _load_reference(Path(reference_path))
    if baseline_turnover is None:
        raise RuntimeError(f"Missing immutable baseline reference: {reference_path}")

    artifact_root = Path(artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    study_name = str(direction.get("study_name", direction["name"]))
    storage_path = artifact_root / f"optuna_{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=False,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=study_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
    )

    def objective(trial: optuna.Trial) -> float:
        params = dict(fixed)
        for name, spec in search.items():
            params[name] = _suggest(trial, name, spec)
        metrics, _ = _evaluate_params(config_path, base_params, params, baseline_turnover)
        trial.set_user_attr("maximum_drawdown", metrics["maximum_drawdown"])
        trial.set_user_attr("turnover", metrics["turnover"])
        trial.set_user_attr("win_rate", metrics["win_rate"])
        trial.report(metrics["research_score"], step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return metrics["research_score"]

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    best_params = dict(fixed)
    best_params.update(study.best_params)
    best_metrics, best_prepared = _evaluate_params(
        config_path, base_params, best_params, baseline_turnover
    )

    seed_results: list[dict[str, Any]] = []
    for index in range(max(0, int(perturb_runs))):
        selection = _select(
            best_prepared, seed=seed + 1000 + index, sigma=perturb_sigma
        )
        metrics, _, _, _ = _evaluate_result(
            best_prepared, selection, baseline_turnover
        )
        metrics["seed"] = seed + 1000 + index
        seed_results.append(metrics)

    if seed_results:
        scores = np.array([row["research_score"] for row in seed_results], dtype=float)
        best_metrics.update(
            {
                "perturb_score_mean": float(scores.mean()),
                "perturb_score_std": float(scores.std(ddof=0)),
                "perturb_score_min": float(scores.min()),
                "perturb_score_max": float(scores.max()),
            }
        )
    best_metrics.update(
        {
            "optuna_direction": direction["name"],
            "optuna_n_trials": n_trials,
            "optuna_completed_trials": len(study.trials),
            "optuna_best_trial": study.best_trial.number,
        }
    )

    best_path = artifact_root / "optuna_best_params.json"
    best_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    study.trials_dataframe().to_csv(artifact_root / "optuna_trials.csv", index=False)
    (artifact_root / "optuna_config.json").write_text(
        json.dumps(direction, indent=2), encoding="utf-8"
    )
    # Recompute confirmed best artifacts without changing the selected params.
    best_selection = _select(best_prepared)
    best_metrics, curve, windows, trades = _evaluate_result(
        best_prepared, best_selection, baseline_turnover
    )
    best_metrics.update(
        {
            "optuna_direction": direction["name"],
            "optuna_n_trials": n_trials,
            "optuna_completed_trials": len(study.trials),
            "optuna_best_trial": study.best_trial.number,
            "perturb_score_mean": float(np.mean([row["research_score"] for row in seed_results])) if seed_results else np.nan,
            "perturb_score_std": float(np.std([row["research_score"] for row in seed_results])) if seed_results else np.nan,
            "perturb_score_min": float(np.min([row["research_score"] for row in seed_results])) if seed_results else np.nan,
            "perturb_score_max": float(np.max([row["research_score"] for row in seed_results])) if seed_results else np.nan,
        }
    )
    _write_artifacts(artifact_root, best_metrics, curve, windows, trades, seed_results, best_prepared["cfg"])
    for key, value in best_metrics.items():
        if key.startswith("optuna_") or key.startswith("perturb_") or key in {
            "research_score",
            "mean_rolling_6m_return",
            "return_on_risk",
            "win_rate",
            "maximum_drawdown",
            "sharpe",
            "turnover",
            "baseline_turnover",
            "n_windows",
            "window_return_median",
            "window_return_min",
            "window_return_std",
        }:
            print(f"METRIC {key}={value}")
    print(f"ARTIFACT_DIR {artifact_dir}")
    return 0
