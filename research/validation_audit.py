"""Validate fixed asymmetric persistence candidate."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from research_score import _evaluate_result, _load_reference, _prepare_strategy, _select, _write_artifacts


BASELINE_SCORE = 0.6149878551980299


def run_validation(config_path: str, params_path: str, reference_path: str,
                   artifact_dir: str, validation_config_path: str) -> int:
    validation = json.loads(Path(validation_config_path).read_text(encoding="utf-8"))
    params = json.loads(Path(params_path).read_text(encoding="utf-8"))
    params.update(validation.get("fixed", {}))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
        json.dump(params, handle)
        trial_path = handle.name
    try:
        prepared = _prepare_strategy(config_path, trial_path)
    finally:
        Path(trial_path).unlink(missing_ok=True)
    baseline_turnover = _load_reference(Path(reference_path))
    if baseline_turnover is None:
        raise RuntimeError(f"Missing immutable baseline reference: {reference_path}")
    base_metrics, curve, windows, trades = _evaluate_result(prepared, _select(prepared), baseline_turnover)
    results: list[dict[str, Any]] = []
    for index in range(int(validation.get("n_seeds", 10))):
        seed = int(validation.get("seed_start", 1042)) + index
        metrics, _, _, _ = _evaluate_result(
            prepared,
            _select(prepared, seed=seed, sigma=float(validation.get("perturb_sigma", 0.03))),
            baseline_turnover,
        )
        metrics["seed"] = seed
        results.append(metrics)
    scores = np.array([row["research_score"] for row in results], dtype=float)
    returns = np.array([row["mean_rolling_6m_return"] for row in results], dtype=float)
    metrics = dict(base_metrics)
    metrics.update({
        "validation_seed_score_mean": float(scores.mean()),
        "validation_seed_score_median": float(np.median(scores)),
        "validation_seed_score_std": float(scores.std(ddof=0)),
        "validation_seed_score_min": float(scores.min()),
        "validation_seed_score_max": float(scores.max()),
        "validation_seed_return_std_over_mean": float(returns.std(ddof=0) / max(abs(returns.mean()), 1e-9)),
        "validation_baseline_score": BASELINE_SCORE,
        "validation_score_mean_beats_baseline": int(scores.mean() > BASELINE_SCORE),
        "validation_min_beats_baseline": int(scores.min() > BASELINE_SCORE),
    })
    root = Path(artifact_dir)
    _write_artifacts(root, metrics, curve, windows, trades, results, prepared["cfg"])
    for key in ("research_score", "validation_seed_score_mean", "validation_seed_score_median",
                "validation_seed_score_std", "validation_seed_score_min", "validation_seed_score_max",
                "validation_seed_return_std_over_mean", "validation_baseline_score",
                "validation_score_mean_beats_baseline", "validation_min_beats_baseline"):
        print(f"METRIC {key}={metrics[key]}")
    print(f"ARTIFACT_DIR {artifact_dir}")
    return 0
