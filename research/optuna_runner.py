"""Run one fixed-score Optuna direction."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import optuna

from research_score import _evaluate_result, _load_reference, _prepare_strategy, _select, _write_artifacts


def _suggest(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    if spec["type"] == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
    return trial.suggest_float(name, float(spec["low"]), float(spec["high"]))


def _evaluate(config_path, base_params, overrides, baseline_turnover):
    params = dict(base_params)
    params.update(overrides)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
        json.dump(params, handle)
        trial_path = handle.name
    try:
        prepared = _prepare_strategy(config_path, trial_path)
        metrics, _, _, _ = _evaluate_result(prepared, _select(prepared), baseline_turnover)
        return metrics, prepared
    finally:
        Path(trial_path).unlink(missing_ok=True)


def run_optuna(config_path, params_path, reference_path, artifact_dir,
               optuna_config_path, perturb_runs, perturb_sigma, seed):
    direction = json.loads(Path(optuna_config_path).read_text(encoding="utf-8"))
    base_params = json.loads(Path(params_path).read_text(encoding="utf-8"))
    baseline_turnover = _load_reference(Path(reference_path))
    if baseline_turnover is None:
        raise RuntimeError(f"Missing immutable baseline reference: {reference_path}")
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=direction["study_name"],
        storage=f"sqlite:///{root / 'optuna_sector_concentration_cap.db'}",
        load_if_exists=False,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(direction.get("seed", seed))),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
    )

    def objective(trial):
        values = {name: _suggest(trial, name, spec) for name, spec in direction["search"].items()}
        metrics, _ = _evaluate(config_path, base_params, values, baseline_turnover)
        trial.report(metrics["research_score"], step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return metrics["research_score"]

    study.optimize(objective, n_trials=int(direction["n_trials"]), gc_after_trial=True)
    best_params = dict(study.best_params)
    best_metrics, prepared = _evaluate(config_path, base_params, best_params, baseline_turnover)
    seeds = []
    for i in range(max(0, int(perturb_runs))):
        metrics, _, _, _ = _evaluate_result(prepared, _select(prepared, seed=seed + 1000 + i, sigma=perturb_sigma), baseline_turnover)
        metrics["seed"] = seed + 1000 + i
        seeds.append(metrics)
    scores = np.array([row["research_score"] for row in seeds], dtype=float)
    best_metrics.update({
        "optuna_direction": direction["name"],
        "optuna_n_trials": int(direction["n_trials"]),
        "optuna_completed_trials": len(study.trials),
        "optuna_best_trial": study.best_trial.number,
        "perturb_score_mean": float(scores.mean()), "perturb_score_std": float(scores.std(ddof=0)),
        "perturb_score_min": float(scores.min()), "perturb_score_max": float(scores.max()),
    })
    _, curve, windows, trades = _evaluate_result(prepared, _select(prepared), baseline_turnover)
    (root / "optuna_best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    study.trials_dataframe().to_csv(root / "optuna_trials.csv", index=False)
    _write_artifacts(root, best_metrics, curve, windows, trades, seeds, prepared["cfg"])
    for key in ("research_score", "mean_rolling_6m_return", "return_on_risk", "win_rate", "maximum_drawdown", "sharpe", "turnover", "perturb_score_mean", "perturb_score_std", "perturb_score_min", "perturb_score_max"):
        print(f"METRIC {key}={best_metrics[key]}")
    print(f"ARTIFACT_DIR {artifact_dir}")
    return 0
