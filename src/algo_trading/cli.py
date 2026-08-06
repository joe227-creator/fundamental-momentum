from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .backtest import run_backtest, write_backtest_outputs
from .calendar_utils import is_rebalance_session, next_rebalance_session
from .config import AppConfig
from .data import (
    FUNDAMENTAL_FACTOR_NAMES,
    MacroDataUnavailableError,
    import_historical_holdings_directory,
    load_market_context,
    write_holdings_snapshot,
)
from .env_utils import load_environment
from .research import run_autoresearch
from .strategy import generate_walk_forward_plan, screen_live_session


def _resolve_config_path(config_arg: str) -> Path:
    candidate = Path(config_arg)
    if candidate.exists():
        return candidate
    fallback = Path("config") / config_arg
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Config file not found: {config_arg}")


class UserFacingCliError(RuntimeError):
    pass


def _dedupe_messages(messages: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for message in messages:
        clean = message.strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        deduped.append(clean)
    return deduped


def _friendly_screen_note(note: str) -> str:
    if note == "No historical holdings snapshots were found; falling back to current IWV holdings introduces survivorship bias.":
        return (
            "Historical IWV snapshots are not archived yet, so older history uses today's holdings. "
            "That makes long-run backtest context less reliable."
        )
    if note.startswith("Historical holdings snapshots begin on ") and "remain survivorship-biased." in note:
        return note.replace(
            "so earlier dates use current IWV holdings fallback and remain survivorship-biased.",
            "so earlier history still falls back to today's holdings and is less reliable.",
        )
    if note.startswith("Effective study start moved from "):
        remainder = note.removeprefix("Effective study start moved from ")
        _requested_start, _, tail = remainder.partition(" to ")
        effective_start, _, _ = tail.partition(" based on available benchmark and universe history.")
        if effective_start:
            return f"Usable benchmark/universe history begins on {effective_start}, so earlier data is not used."
    return note


def _collect_screen_notes(metadata_notes: list[str], warning_messages: list[str]) -> list[str]:
    cached_fred_series: list[str] = []
    extra_messages: list[str] = []
    prefix = "Using cached FRED series "
    suffix = " without refresh because FRED_API_KEY is not set."
    for message in warning_messages:
        if message.startswith(prefix) and message.endswith(suffix):
            cached_fred_series.append(message.removeprefix(prefix).removesuffix(suffix))
            continue
        extra_messages.append(message)
    if cached_fred_series:
        extra_messages.append(
            "Using cached FRED macro data because FRED_API_KEY is not set. "
            f"Cached enabled series: {', '.join(sorted(set(cached_fred_series)))}."
        )
    return _dedupe_messages([_friendly_screen_note(message) for message in [*metadata_notes, *extra_messages]])


def _build_live_setup_error(error: MacroDataUnavailableError) -> str:
    project_root = Path.cwd()
    env_path = project_root / ".env"
    github_env_path = project_root / ".github" / ".env"
    missing_preview = ", ".join(error.missing_series[:6]) if error.missing_series else "unknown"
    if len(error.missing_series) > 6:
        missing_preview = f"{missing_preview}, ..."
    return "\n".join(
        [
            "Live screen setup is incomplete.",
            "",
            "The baseline live strategy uses FRED macro data as a risk filter. The required macro files are not cached yet and no FRED_API_KEY was found.",
            "",
            "What to do next:",
            f"1. Create {env_path} or {github_env_path}",
            "2. Add a line like: FRED_API_KEY=your_key_here",
            '3. Or, for this PowerShell session only, run: $env:FRED_API_KEY = "your_key_here"',
            "4. Re-run the same screen command.",
            "",
            f"Macro cache folder: {error.cache_dir}",
            f"Missing cached series: {missing_preview}",
            "",
            "The script stops here on purpose so it does not guess the macro regime for live trading.",
        ]
    )


def _load_screen_context(config: AppConfig, refresh_data: bool):
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        try:
            context = load_market_context(config, refresh=refresh_data, refresh_holdings=True)
        except MacroDataUnavailableError as error:
            raise UserFacingCliError(_build_live_setup_error(error)) from None
    notes = _collect_screen_notes(
        metadata_notes=list(context.metadata.get("notes", [])),
        warning_messages=[str(warning.message) for warning in caught_warnings],
    )
    return context, notes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IWV constituent algo trading research harness")
    parser.add_argument("--config", default="config/baseline.json", help="Path to the JSON strategy config")
    parser.add_argument("--env-file", help="Optional .env file to load before running")

    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest = subparsers.add_parser("backtest", help="Run the walk-forward backtest")
    backtest.add_argument("--refresh-data", action="store_true", help="Ignore caches and re-download market data")
    backtest.add_argument("--output-dir", default="artifacts/backtests", help="Directory for backtest artifacts")

    screen = subparsers.add_parser("screen", help="Run the weekly live screen before market open")
    screen.add_argument(
        "--as-of",
        default=str(pd.Timestamp.today().normalize().date()),
        help="Trading date to evaluate; defaults to today's system date",
    )
    screen.add_argument("--refresh-data", action="store_true", help="Ignore caches and re-download market data")
    screen.add_argument("--current-positions", help="Optional CSV with columns symbol and shares")
    screen.add_argument("--output-dir", default="artifacts/live", help="Directory for live screen artifacts")

    research = subparsers.add_parser("research", help="Run the autoresearch-style experiment loop")
    research.add_argument("--research-space", default="config/research_space.json", help="JSON mutation space")
    research.add_argument("--iterations", type=int, help="Override the configured number of experiments")
    research.add_argument("--refresh-data", action="store_true", help="Ignore caches and re-download market data")
    research.add_argument("--output-dir", default="artifacts/research", help="Directory for research artifacts")
    research.add_argument("--result-path", default="result.tsv", help="TSV file for experiment results")
    research.add_argument("--finding-path", default="research finding.txt", help="Text file for experiment findings")

    snapshot = subparsers.add_parser("snapshot-holdings", help="Archive the current IWV holdings for future point-in-time studies")
    snapshot.add_argument("--as-of", default=str(pd.Timestamp.today().normalize().date()), help="Snapshot date label")
    snapshot.add_argument("--refresh-data", action="store_true", help="Ignore caches and re-download holdings data")

    bulk_import = subparsers.add_parser("import-holdings-history", help="Bulk import dated historical IWV holdings CSVs into the local snapshot archive")
    bulk_import.add_argument("--source-dir", required=True, help="Directory containing dated holdings CSV files")
    bulk_import.add_argument("--pattern", default="*.csv", help="Glob pattern to match historical holdings files")
    bulk_import.add_argument("--overwrite", action="store_true", help="Overwrite existing dated snapshot files in the archive")

    return parser


def _load_current_positions(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["symbol", "shares"])
    frame = pd.read_csv(path)
    frame.columns = [column.strip().lower() for column in frame.columns]
    if "symbol" not in frame.columns:
        raise ValueError("Current positions CSV must include a symbol column.")
    if "shares" not in frame.columns:
        frame["shares"] = 0.0
    frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce").fillna(0.0)
    frame = frame.loc[frame["symbol"].ne("")]
    frame = frame.groupby("symbol", as_index=False, sort=True)["shares"].sum()
    frame = frame.loc[frame["shares"].ne(0.0)].reset_index(drop=True)
    return frame[["symbol", "shares"]]


def _build_rebalance_plan(
    screen_df: pd.DataFrame,
    current_positions: pd.DataFrame,
    rebalance_due: bool,
    config: AppConfig,
    symbol_names: dict[str, object] | None = None,
) -> pd.DataFrame:
    desired_symbols = screen_df["symbol"].astype(str).tolist() if not screen_df.empty else []
    current_symbols = current_positions["symbol"].tolist()
    current_position_map = current_positions.set_index("symbol")["shares"].to_dict() if not current_positions.empty else {}
    screen_rows = screen_df.set_index("symbol") if not screen_df.empty else pd.DataFrame()
    name_by_symbol = dict(symbol_names or {})
    if "name" in screen_rows.columns:
        name_by_symbol.update(screen_rows["name"].dropna().to_dict())
    target_weight = 1.0 / len(desired_symbols) if desired_symbols else 0.0
    rank_by_symbol = {symbol: rank for rank, symbol in enumerate(desired_symbols, start=1)}
    plan_columns = ["priority", "action", "symbol", "name", "shares", "target_weight", "target_rank", "composite_score", "note"]
    rows = []

    if not rebalance_due:
        for symbol in current_symbols:
            rows.append(
                {
                    "priority": 0,
                    "action": "HOLD",
                    "symbol": symbol,
                    "name": name_by_symbol.get(symbol, np.nan),
                    "shares": float(current_position_map.get(symbol, 0.0)),
                    "target_weight": np.nan,
                    "target_rank": rank_by_symbol.get(symbol),
                    "composite_score": float(screen_rows.at[symbol, "composite_score"]) if symbol in screen_rows.index else np.nan,
                    "note": "No scheduled rebalance for this session.",
                }
            )
        return pd.DataFrame(rows, columns=plan_columns)

    desired_set = set(desired_symbols)
    current_set = set(current_symbols)
    if config.strategy.full_turnover_rebalance:
        sell_symbols = current_symbols
        hold_symbols: list[str] = []
        buy_symbols = desired_symbols
    else:
        sell_symbols = [symbol for symbol in current_symbols if symbol not in desired_set]
        hold_symbols = [symbol for symbol in desired_symbols if symbol in current_set]
        buy_symbols = [symbol for symbol in desired_symbols if symbol not in current_set]

    for symbol in sell_symbols:
        rows.append(
            {
                "priority": 1,
                "action": "SELL",
                "symbol": symbol,
                "name": name_by_symbol.get(symbol, np.nan),
                "shares": float(current_position_map.get(symbol, 0.0)),
                "target_weight": 0.0,
                "target_rank": np.nan,
                "composite_score": float(screen_rows.at[symbol, "composite_score"]) if symbol in screen_rows.index else np.nan,
                "note": "Current holding is outside the target portfolio.",
            }
        )
    for symbol in hold_symbols:
        rows.append(
            {
                "priority": 2,
                "action": "HOLD",
                "symbol": symbol,
                "name": name_by_symbol.get(symbol, np.nan),
                "shares": float(current_position_map.get(symbol, 0.0)),
                "target_weight": target_weight,
                "target_rank": rank_by_symbol.get(symbol),
                "composite_score": float(screen_rows.at[symbol, "composite_score"]),
                "note": "Current holding remains in the target portfolio.",
            }
        )
    allocation = target_weight
    for symbol in buy_symbols:
        rows.append(
            {
                "priority": 3,
                "action": "BUY",
                "symbol": symbol,
                "name": name_by_symbol.get(symbol, np.nan),
                "shares": np.nan,
                "target_weight": allocation,
                "target_rank": rank_by_symbol.get(symbol),
                "composite_score": float(screen_rows.at[symbol, "composite_score"]),
                "note": "Open after scheduled sells complete.",
            }
        )
    return pd.DataFrame(rows, columns=plan_columns)


def _live_trade_log_columns() -> list[str]:
    return [
        "date",
        "rebalance_due",
        "priority",
        "action",
        "symbol",
        "name",
        "shares",
        "target_weight",
        "target_rank",
        "composite_score",
        "note",
    ]


def _run_backtest(config: AppConfig, refresh_data: bool, output_dir: str) -> None:
    context = load_market_context(config, refresh=refresh_data)
    selection_map, _score_map, diagnostics = generate_walk_forward_plan(context, config)
    result = run_backtest(context, selection_map, config)
    write_backtest_outputs(result, output_dir, "backtest")
    diagnostics_path = Path(output_dir) / "backtest_walk_forward.json"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")
    if context.metadata.get("notes"):
        print("Study notes:")
        for note in context.metadata["notes"]:
            print(f"- {note}")
    print(json.dumps(result.metrics, indent=2, default=str))


def _run_screen(config: AppConfig, as_of: str, refresh_data: bool, current_positions_path: str | None, output_dir: str) -> None:
    context, screen_notes = _load_screen_context(config, refresh_data=refresh_data)
    session, screen_df, weights = screen_live_session(context, config, as_of)
    current_positions = _load_current_positions(current_positions_path)
    requested_session = pd.Timestamp(as_of).normalize()
    latest_loaded_session = pd.Timestamp(context.prices.sessions.max()).normalize()
    if requested_session > latest_loaded_session:
        import pandas_market_calendars as mcal
        calendar = mcal.get_calendar("NYSE")
        schedule = calendar.schedule(start_date=latest_loaded_session.date(), end_date=requested_session.date())
        future_sessions = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
        extended_sessions = context.prices.sessions.union(future_sessions)
    else:
        extended_sessions = context.prices.sessions
    rebalance_target = requested_session if requested_session > latest_loaded_session else session
    rebalance_due = is_rebalance_session(rebalance_target, extended_sessions, config.strategy.rebalance_frequency)
    try:
        next_rebalance = next_rebalance_session(rebalance_target, extended_sessions, config.strategy.rebalance_frequency)
    except ValueError:
        next_rebalance = pd.NaT
    holdings = getattr(context, "holdings", pd.DataFrame())
    holdings_names = {}
    if {"ticker", "name"}.issubset(holdings.columns):
        holdings_names = holdings.set_index("ticker")["name"].to_dict()
    plan = _build_rebalance_plan(screen_df, current_positions, rebalance_due, config, symbol_names=holdings_names)
    orders = plan.loc[plan["action"].isin(["SELL", "BUY"])].reset_index(drop=True)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = session.strftime("%Y%m%d")
    screen_path = output_root / f"screen_{stamp}.csv"
    plan_path = output_root / f"rebalance_plan_{stamp}.csv"
    orders_path = output_root / f"orders_{stamp}.csv"
    weights_path = output_root / f"weights_{stamp}.json"
    trade_log_path = output_root / "live_trade_log.csv"
    session_log_path = output_root / "live_session_log.csv"

    screen_df.to_csv(screen_path, index=False)
    plan.to_csv(plan_path, index=False)
    orders_path.write_text(orders.to_csv(index=False), encoding="utf-8")
    weights_path.write_text(json.dumps(weights, indent=2, default=str), encoding="utf-8")

    session_summary = pd.DataFrame(
        [
            {
                "date": session,
                "rebalance_due": rebalance_due,
                "rebalance_frequency": config.strategy.rebalance_frequency,
                "next_rebalance_date": next_rebalance,
                "current_positions": len(current_positions),
                "target_positions": len(screen_df),
                "sell_orders": int((plan["action"] == "SELL").sum()) if not plan.empty else 0,
                "buy_orders": int((plan["action"] == "BUY").sum()) if not plan.empty else 0,
                "hold_positions": int((plan["action"] == "HOLD").sum()) if not plan.empty else 0,
                "macro_composite": float(screen_df["macro_composite"].iloc[0]) if not screen_df.empty else np.nan,
            }
        ]
    )
    if session_log_path.exists():
        existing_sessions = pd.read_csv(session_log_path)
        session_summary = pd.concat([existing_sessions, session_summary], ignore_index=True)
    session_summary.to_csv(session_log_path, index=False)

    if not trade_log_path.exists():
        pd.DataFrame(columns=_live_trade_log_columns()).to_csv(trade_log_path, index=False)

    if not orders.empty:
        orders_with_date = orders.copy()
        orders_with_date.insert(0, "date", session)
        orders_with_date.insert(1, "rebalance_due", rebalance_due)
        if trade_log_path.exists():
            existing = pd.read_csv(trade_log_path)
            combined = pd.concat([existing, orders_with_date], ignore_index=True)
        else:
            combined = orders_with_date
        combined.reindex(columns=_live_trade_log_columns()).to_csv(trade_log_path, index=False)

    print(f"Screen session: {session.date()}")
    if requested_session > latest_loaded_session:
        print(
            f"Requested date {requested_session.date()} is later than the latest loaded market session; using {session.date()} instead."
        )
    print(f"Rebalance due: {'yes' if rebalance_due else 'no'}")
    print(f"Next scheduled rebalance: {next_rebalance.date() if pd.notna(next_rebalance) else 'unavailable in loaded history'}")
    if screen_notes:
        print("Background notes:")
        for note in screen_notes:
            print(f"- {note}")
    print(screen_df.to_string(index=False))
    print(f"Saved screen to {screen_path}")
    print(f"Saved rebalance plan to {plan_path}")
    print(f"Saved orders to {orders_path}")
    if not rebalance_due:
        print("No trade instructions were generated because this session is outside the configured rebalance cadence.")
    elif orders.empty:
        print("No trade instructions were generated because current holdings already match the target portfolio.")


def _run_research(
    config: AppConfig,
    research_space: str,
    iterations: int | None,
    refresh_data: bool,
    output_dir: str,
    result_path: str,
    finding_path: str,
) -> None:
    research_payload = json.loads(Path(research_space).read_text(encoding="utf-8"))
    requested_fundamental_factors = []
    for path, values in research_payload.get("parameters", {}).items():
        prefix = "indicators.enabled_indicators."
        if not path.startswith(prefix):
            continue
        factor_name = path.removeprefix(prefix)
        if factor_name not in FUNDAMENTAL_FACTOR_NAMES:
            continue
        if True in values:
            requested_fundamental_factors.append(factor_name)

    context = load_market_context(
        config,
        refresh=refresh_data,
        requested_fundamental_factors=requested_fundamental_factors,
    )
    if context.metadata.get("notes"):
        for note in context.metadata["notes"]:
            print(f"Note: {note}")
    results = run_autoresearch(
        context=context,
        base_config=config,
        research_space_path=research_space,
        result_path=result_path,
        finding_path=finding_path,
        output_dir=output_dir,
        iterations=iterations,
    )
    print(results.to_string(index=False))


def _run_snapshot_holdings(config: AppConfig, as_of: str, refresh_data: bool) -> None:
    path = write_holdings_snapshot(config=config, as_of=as_of, refresh=refresh_data)
    print(f"Saved holdings snapshot to {path}")


def _run_import_holdings_history(config: AppConfig, source_dir: str, pattern: str, overwrite: bool) -> None:
    imported = import_historical_holdings_directory(
        config=config,
        source_dir=source_dir,
        pattern=pattern,
        overwrite=overwrite,
    )
    print(f"Imported {len(imported)} holdings snapshots")
    for path in imported:
        print(path)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    loaded_env = load_environment(explicit_path=args.env_file)
    config = AppConfig.from_file(_resolve_config_path(args.config))

    if loaded_env is not None:
        print(f"Loaded environment from {loaded_env}")

    try:
        if args.command == "backtest":
            _run_backtest(config, refresh_data=args.refresh_data, output_dir=args.output_dir)
        elif args.command == "screen":
            _run_screen(
                config,
                as_of=args.as_of,
                refresh_data=args.refresh_data,
                current_positions_path=args.current_positions,
                output_dir=args.output_dir,
            )
        elif args.command == "research":
            _run_research(
                config,
                research_space=args.research_space,
                iterations=args.iterations,
                refresh_data=args.refresh_data,
                output_dir=args.output_dir,
                result_path=args.result_path,
                finding_path=args.finding_path,
            )
        elif args.command == "snapshot-holdings":
            _run_snapshot_holdings(config, as_of=args.as_of, refresh_data=args.refresh_data)
        elif args.command == "import-holdings-history":
            _run_import_holdings_history(config, source_dir=args.source_dir, pattern=args.pattern, overwrite=args.overwrite)
        else:
            raise ValueError(f"Unsupported command: {args.command}")
    except UserFacingCliError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
