from __future__ import annotations

import os
from pathlib import Path


def _candidate_env_paths(start_dir: str | Path | None = None, explicit_path: str | Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(path: str | Path | None) -> None:
        if path is None:
            return
        resolved = Path(path).expanduser()
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)

    add(explicit_path)
    add(os.getenv("ALGO_TRADING_ENV_FILE"))

    base = Path.cwd() if start_dir is None else Path(start_dir)
    for directory in [base, *base.parents]:
        add(directory / ".env")
        add(directory / ".github" / ".env")

    return candidates


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return key, value


def load_environment(start_dir: str | Path | None = None, explicit_path: str | Path | None = None) -> Path | None:
    first_loaded: Path | None = None
    for path in _candidate_env_paths(start_dir=start_dir, explicit_path=explicit_path):
        if not path.exists() or not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_env_line(line)
            if parsed is None:
                continue
            key, value = parsed
            os.environ.setdefault(key, value)
        if first_loaded is None:
            first_loaded = path
    return first_loaded