from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd

from .backtest import run_backtest, write_backtest_outputs
from .config import AppConfig
from .data import MarketContext
from .strategy import generate_walk_forward_plan


RESULT_COLUMNS = [
    "experiment_id",
    "timestamp",
    "status",
    "score",
    "annualized_return",
    "profit_factor",
    "max_drawdown",
    "total_return",
    "num_closed_trades",
    "config_hash",
    "description",
]

UNIQUE_CANDIDATE_SAMPLE_LIMIT = 256


def objective_score(metrics: dict[str, Any], config: AppConfig) -> float:
    annualized_return = float(metrics.get("annualized_return", np.nan))
    profit_factor = float(metrics.get("profit_factor", np.nan))
    drawdown = abs(float(metrics.get("max_drawdown", 0.0)))
    if not np.isfinite(annualized_return):
        annualized_return = -1.0
    if not np.isfinite(profit_factor):
        profit_factor = 0.0

    score = (
        annualized_return * config.research.annualized_return_weight
        + min(profit_factor, 5.0) * config.research.profit_factor_weight
        - drawdown * config.research.drawdown_weight
    )
    if drawdown > config.research.max_drawdown_limit:
        score -= 1000.0 * (drawdown - config.research.max_drawdown_limit)
    return float(score)


def load_research_space(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _set_nested(data: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    current = data
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _get_nested(data: dict[str, Any], dotted_path: str) -> Any:
    current: Any = data
    for part in dotted_path.split("."):
        current = current[part]
    return current


def mutate_config(base_config: AppConfig, mutations: dict[str, Any]) -> AppConfig:
    payload = deepcopy(asdict(base_config))
    for path, value in mutations.items():
        _set_nested(payload, path, value)
    return AppConfig.from_dict(payload)


def sample_mutations(space: dict[str, Any], rng: random.Random, base_config: AppConfig | None = None) -> dict[str, Any]:
    parameters = space.get("parameters", {})
    if not parameters:
        return {}
    keys = list(parameters)
    min_changes = int(space.get("min_changes", 1))
    max_changes = int(space.get("max_changes", min(4, len(keys))))
    num_changes = rng.randint(min_changes, min(max_changes, len(keys)))
    chosen = rng.sample(keys, num_changes)
    base_payload = asdict(base_config) if base_config is not None else None
    mutations: dict[str, Any] = {}
    for path in chosen:
        options = list(parameters[path])
        if base_payload is not None:
            current_value = _get_nested(base_payload, path)
            different_options = [option for option in options if option != current_value]
            if different_options:
                options = different_options
        mutations[path] = rng.choice(options)
    return mutations


def describe_mutations(mutations: dict[str, Any]) -> str:
    if not mutations:
        return "baseline"
    parts = [f"{path}={value}" for path, value in sorted(mutations.items())]
    return "; ".join(parts)


def describe_config_differences(
    reference_config: AppConfig,
    candidate_config: AppConfig,
    tracked_paths: list[str],
) -> str:
    reference_payload = asdict(reference_config)
    candidate_payload = asdict(candidate_config)
    parts = []
    for path in sorted(tracked_paths):
        if _get_nested(reference_payload, path) == _get_nested(candidate_payload, path):
            continue
        parts.append(f"{path}={_get_nested(candidate_payload, path)}")
    return "; ".join(parts) if parts else "baseline"


def _ensure_result_file(path: Path) -> None:
    if not path.exists():
        path.write_text("\t".join(RESULT_COLUMNS) + "\n", encoding="utf-8")


def _next_experiment_index(path: str | Path) -> int:
    target = Path(path)
    if not target.exists():
        return 0
    lines = [line for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) <= 1:
        return 0
    last_id = lines[-1].split("\t", 1)[0]
    if last_id.startswith("exp") and last_id[3:].isdigit():
        return int(last_id[3:]) + 1
    return len(lines) - 1


def _load_existing_results(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    if not target.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.read_csv(target, sep="\t")


def _load_resume_state(
    result_path: str | Path,
    output_dir: str | Path,
    base_config: AppConfig,
) -> tuple[set[str], float, AppConfig | None, bool]:
    existing_results = _load_existing_results(result_path)
    if existing_results.empty:
        return set(), -np.inf, None, False

    seen_hashes = {
        str(config_hash)
        for config_hash in existing_results.get("config_hash", pd.Series(dtype=str)).dropna().astype(str)
        if config_hash
    }
    existing_results = existing_results.copy()
    existing_results["score"] = pd.to_numeric(existing_results.get("score"), errors="coerce")
    existing_results["max_drawdown"] = pd.to_numeric(existing_results.get("max_drawdown"), errors="coerce")
    eligible = existing_results.loc[
        existing_results["max_drawdown"].abs().le(base_config.research.max_drawdown_limit)
    ]
    if eligible.empty:
        return seen_hashes, -np.inf, None, True

    best_score = float(eligible["score"].max())
    baseline_matches_best = eligible.loc[
        eligible.get("config_hash", pd.Series(dtype=str)).astype(str).eq(base_config.config_hash())
        & eligible["score"].sub(best_score).abs().le(1e-9)
    ]
    if not baseline_matches_best.empty:
        return seen_hashes, best_score, base_config, True

    best_row = eligible.sort_values("score", ascending=False).iloc[0]
    best_score = float(best_row["score"])
    best_experiment_id = str(best_row["experiment_id"])
    best_config_path = Path(output_dir) / "configs" / f"{best_experiment_id}.json"
    best_config: AppConfig | None = None
    if best_config_path.exists():
        best_config = AppConfig.from_file(best_config_path)
    else:
        best_saved_path = Path(output_dir) / "best_config.json"
        if best_saved_path.exists():
            best_config = AppConfig.from_file(best_saved_path)
        elif str(best_row.get("config_hash", "")) == base_config.config_hash():
            best_config = base_config

    return seen_hashes, best_score, best_config, True


def _sample_unique_candidate(
    research_space: dict[str, Any],
    rng: random.Random,
    anchor_config: AppConfig,
    seen_hashes: set[str],
) -> tuple[dict[str, Any], AppConfig] | None:
    for _ in range(UNIQUE_CANDIDATE_SAMPLE_LIMIT):
        mutations = sample_mutations(research_space, rng, base_config=anchor_config)
        candidate = mutate_config(anchor_config, mutations)
        if candidate.config_hash() in seen_hashes:
            continue
        return mutations, candidate
    return None


def append_result(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    _ensure_result_file(target)
    ordered = [row.get(column, "") for column in RESULT_COLUMNS]
    rendered = "\t".join(str(value) for value in ordered)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(rendered + "\n")


def append_finding(path: str | Path, experiment_id: str, status: str, description: str, metrics: dict[str, Any], reason: str) -> None:
    target = Path(path)
    if not target.exists():
        target.write_text("# Research Findings\n\n", encoding="utf-8")
    lines = [
        f"## {experiment_id} - {status.upper()}",
        f"Description: {description}",
        (
            "Metrics: annualized_return="
            f"{metrics.get('annualized_return', np.nan):.4f}, "
            f"profit_factor={metrics.get('profit_factor', np.nan):.4f}, "
            f"max_drawdown={metrics.get('max_drawdown', np.nan):.4f}, "
            f"total_return={metrics.get('total_return', np.nan):.4f}"
        ),
        f"Reason: {reason}",
        "",
    ]
    with target.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def evaluate_experiment(
    context: MarketContext,
    config: AppConfig,
) -> tuple[
    dict[str, Any],
    dict[pd.Timestamp, list[str]],
    dict[pd.Timestamp, pd.Series],
    list[dict[str, Any]],
    Any,
]:
    selection_map, score_map, diagnostics = generate_walk_forward_plan(context, config)
    backtest = run_backtest(context, selection_map, config)
    metrics = dict(backtest.metrics)
    metrics["score"] = objective_score(metrics, config)
    return metrics, selection_map, score_map, diagnostics, backtest


def run_autoresearch(
    context: MarketContext,
    base_config: AppConfig,
    research_space_path: str | Path,
    result_path: str | Path,
    finding_path: str | Path,
    output_dir: str | Path,
    iterations: int | None = None,
) -> pd.DataFrame:
    research_space = load_research_space(research_space_path)
    tracked_paths = list(research_space.get("parameters", {}))
    total_iterations = iterations or base_config.research.iterations
    no_improvement_limit = max(1, base_config.research.consecutive_no_improvement_limit)
    rng = random.Random(base_config.research.random_seed)
    output_root = Path(output_dir)
    configs_dir = output_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes, best_score, best_config, has_existing_results = _load_resume_state(
        result_path=result_path,
        output_dir=output_root,
        base_config=base_config,
    )
    consecutive_no_improvement = 0
    candidate_count = 0
    results: list[dict[str, Any]] = []
    experiment_index = _next_experiment_index(result_path)
    index = 0
    while True:
        if index == 0 and not has_existing_results:
            mutations = {}
            candidate = base_config
        else:
            if candidate_count >= total_iterations or consecutive_no_improvement >= no_improvement_limit:
                break
            anchor_config = best_config or base_config
            sampled = _sample_unique_candidate(research_space, rng, anchor_config, seen_hashes)
            if sampled is None:
                break
            mutations, candidate = sampled
            candidate_count += 1

        experiment_id = f"exp{experiment_index:04d}"
        description = describe_config_differences(base_config, candidate, tracked_paths)
        config_path = configs_dir / f"{experiment_id}.json"
        candidate.write(config_path)

        metrics, _selection_map, _score_map, diagnostics, backtest = evaluate_experiment(context, candidate)
        score = float(metrics["score"])
        drawdown = abs(float(metrics.get("max_drawdown", 0.0)))
        if drawdown > candidate.research.max_drawdown_limit:
            status = "discard"
            reason = (
                f"Discarded because max drawdown {drawdown:.2%} breached the "
                f"{candidate.research.max_drawdown_limit:.0%} limit."
            )
        elif score > best_score:
            status = "keep"
            best_score = score
            best_config = candidate
            if index > 0:
                consecutive_no_improvement = 0
            reason = f"Kept because score improved to {score:.4f} with acceptable drawdown."
        else:
            status = "discard"
            if index > 0:
                consecutive_no_improvement += 1
            reason = f"Discarded because score {score:.4f} did not beat the current best {best_score:.4f}."

        artifact_prefix = f"{experiment_id}_{status}"
        write_backtest_outputs(backtest, output_root / "runs", artifact_prefix)
        diagnostics_path = output_root / "runs" / f"{artifact_prefix}_walk_forward.json"
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")

        row = {
            "experiment_id": experiment_id,
            "timestamp": pd.Timestamp.now("UTC").isoformat(),
            "status": status,
            "score": f"{score:.6f}",
            "annualized_return": f"{metrics.get('annualized_return', np.nan):.6f}",
            "profit_factor": f"{metrics.get('profit_factor', np.nan):.6f}",
            "max_drawdown": f"{metrics.get('max_drawdown', np.nan):.6f}",
            "total_return": f"{metrics.get('total_return', np.nan):.6f}",
            "num_closed_trades": int(metrics.get("num_closed_trades", 0)),
            "config_hash": candidate.config_hash(),
            "description": description,
        }
        append_result(result_path, row)
        append_finding(finding_path, experiment_id, status, description, metrics, reason)
        results.append(row)
        seen_hashes.add(candidate.config_hash())
        index += 1
        experiment_index += 1

    if best_config is not None:
        best_config.write(output_root / "best_config.json")

    return pd.DataFrame(results, columns=RESULT_COLUMNS)