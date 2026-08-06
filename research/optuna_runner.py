"""Run one fixed-score Optuna direction."""
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


def _evaluate(
    config_path: str,
    base_params: dict[str, Any],
    overrides: dict[str, Any],
    baseline_turnover: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    merged = dict(base_params)
    merged.update(overrides)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
        json.dump(merged, handle)
        trial_path = handle.name
    try:
        prepared = _prepare_strategy(config_path, trial_path)
        metrics, _, _, _ = _evaluate_result(
            prepared, _select(prepared), baseline_turnover
        )
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
    baseline_turnover = _load_reference(Path(reference_path))
    if baseline_turnover is None:
        raise RuntimeError(f"Missing immutable baseline reference: {reference_path}")

    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=str(direction.get("study_name", direction["name"])),
        storage=f"sqlite:///{root / ('optuna_' + direction['name'] + '.db')}",
        load_if_exists=False,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(direction.get("seed", seed))),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
    )

    def objective(trial: optuna.Trial) -> float:
        params = dict(fixed)
        for name, spec in search.items():
            params[name] = _suggest(trial, name, spec)
        metrics, _ = _evaluate(config_path, base_params, params, baseline_turnover)
        trial.set_user_attr("maximum_drawdown", metrics["maximum_drawdown"])
        trial.set_user_attr("turnover", metrics["turnover"])
        trial.set_user_attr("win_rate", metrics["win_rate"])
        trial.report(metrics["research_score"], step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return metrics["research_score"]

    study.optimize(objective, n_trials=int(direction["n_trials"]), gc_after_trial=True)
    best_params = dict(fixed)
    best_params.update(study.best_params)
    best_metrics, prepared = _evaluate(
        config_path, base_params, best_params, baseline_turnover
    )
    seed_results: list[dict[str, Any]] = []
    for index in range(max(0, int(perturb_runs))):
        metrics, _, _, _ = _evaluate_result(
            prepared,
            _select(prepared, seed=seed + 1000 + index, sigma=perturb_sigma),
            baseline_turnover,
        )
        metrics["seed"] = seed + 1000 + index
        seed_results.append(metrics)

    scores = np.array([row["research_score"] for row in seed_results], dtype=float)
    best_metrics.update(
        {
            "optuna_direction": direction["name"],
            "optuna_n_trials": int(direction["n_trials"]),
            "optuna_completed_trials": len(study.trials),
            "optuna_best_trial": study.best_trial.number,
            "perturb_score_mean": float(scores.mean()) if scores.size else np.nan,
            "perturb_score_std": float(scores.std(ddof=0)) if scores.size else np.nan,
            "perturb_score_min": float(scores.min()) if scores.size else np.nan,
            "perturb_score_max": float(scores.max()) if scores.size else np.nan,
        }
    )
    best_selection = _select(prepared)
    _, curve, windows, trades = _evaluate_result(
        prepared, best_selection, baseline_turnover
    )
    (root / "optuna_best_params.json").write_text(
        json.dumps(best_params, indent=2), encoding="utf-8"
    )
    study.trials_dataframe().to_csv(root / "optuna_trials.csv", index=False)
    _write_artifacts(root, best_metrics, curve, windows, trades, seed_results, prepared["cfg"])
    for key in (
        "research_score",
        "mean_rolling_6m_return",
        "return_on_risk",
        "win_rate",
        "maximum_drawdown",
        "sharpe",
        "turnover",
        "baseline_turnover",
        "n_windows",
        "perturb_score_mean",
        "perturb_score_std",
        "perturb_score_min",
        "perturb_score_max",
        "optuna_direction",
        "optuna_n_trials",
        "optuna_completed_trials",
        "optuna_best_trial",
    ):
        print(f"METRIC {key}={best_metrics[key]}")
    print(f"ARTIFACT_DIR {artifact_dir}")
    return 0
