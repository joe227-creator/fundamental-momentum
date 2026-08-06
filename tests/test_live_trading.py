import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import algo_trading.cli as cli_module
from algo_trading.calendar_utils import is_rebalance_session, next_rebalance_session
from algo_trading.cli import _build_rebalance_plan, _load_current_positions
from algo_trading.config import AppConfig


def test_load_current_positions_aggregates_duplicate_symbols(tmp_path) -> None:
    positions_path = tmp_path / "positions.csv"
    positions_path.write_text(
        "symbol,shares\nAAPL,10\naapl,5\nMSFT,0\n,7\n",
        encoding="utf-8",
    )

    frame = _load_current_positions(str(positions_path))

    assert frame.to_dict(orient="records") == [{"symbol": "AAPL", "shares": 15.0}]


def test_build_rebalance_plan_keeps_existing_and_sizes_replacement_from_final_portfolio() -> None:
    config = AppConfig()
    config.strategy.full_turnover_rebalance = False
    screen_df = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "name": ["Alpha", "Beta"],
            "composite_score": [1.2, 0.9],
        }
    )
    current_positions = pd.DataFrame(
        {
            "symbol": ["AAA", "CCC"],
            "shares": [10.0, 4.0],
        }
    )

    plan = _build_rebalance_plan(screen_df, current_positions, rebalance_due=True, config=config)

    assert plan["action"].tolist() == ["SELL", "HOLD", "BUY"]
    assert plan.loc[plan["action"].eq("SELL"), "symbol"].item() == "CCC"
    assert plan.loc[plan["action"].eq("HOLD"), "name"].item() == "Alpha"
    assert plan.loc[plan["action"].eq("BUY"), "name"].item() == "Beta"
    assert plan.loc[plan["action"].eq("HOLD"), "target_weight"].item() == 0.5
    assert plan.loc[plan["action"].eq("BUY"), "target_weight"].item() == 0.5
    assert plan.loc[plan["action"].eq("BUY"), "target_rank"].item() == 2


def test_build_rebalance_plan_holds_existing_positions_when_rebalance_not_due() -> None:
    config = AppConfig()
    config.strategy.full_turnover_rebalance = False
    screen_df = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "composite_score": [1.2, 0.9],
        }
    )
    current_positions = pd.DataFrame(
        {
            "symbol": ["AAA", "CCC"],
            "shares": [10.0, 4.0],
        }
    )

    plan = _build_rebalance_plan(screen_df, current_positions, rebalance_due=False, config=config)

    assert set(plan["action"]) == {"HOLD"}
    assert len(plan) == 2
    assert plan["note"].str.contains("No scheduled rebalance").all()


def test_rebalance_calendar_helpers_cover_monthly_sessions() -> None:
    sessions = pd.bdate_range("2026-04-01", "2026-05-31")

    assert is_rebalance_session(pd.Timestamp("2026-04-01"), sessions, "monthly")
    assert not is_rebalance_session(pd.Timestamp("2026-04-13"), sessions, "monthly")
    assert next_rebalance_session(pd.Timestamp("2026-04-13"), sessions, "monthly") == pd.Timestamp("2026-04-01") + pd.offsets.MonthBegin(1)


def test_load_screen_context_rewrites_internal_notes(monkeypatch) -> None:
    raw_note = "No historical holdings snapshots were found; falling back to current IWV holdings introduces survivorship bias."
    call_args: dict[str, object] = {}

    def fake_load_market_context(config: AppConfig, refresh: bool, refresh_holdings: bool):
        import warnings

        call_args["refresh"] = refresh
        call_args["refresh_holdings"] = refresh_holdings
        warnings.warn(raw_note, stacklevel=2)
        return SimpleNamespace(metadata={"notes": [raw_note]})

    monkeypatch.setattr(cli_module, "load_market_context", fake_load_market_context)

    _context, notes = cli_module._load_screen_context(AppConfig(), refresh_data=False)

    assert call_args == {"refresh": False, "refresh_holdings": True}
    assert notes == [
        "Historical IWV snapshots are not archived yet, so older history uses today's holdings. That makes long-run backtest context less reliable."
    ]


def test_main_exits_cleanly_for_user_facing_live_errors(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(AppConfig().to_dict()), encoding="utf-8")

    def fake_run_screen(*args, **kwargs) -> None:
        raise cli_module.UserFacingCliError("Live screen setup is incomplete.")

    monkeypatch.setattr(cli_module, "_run_screen", fake_run_screen)
    monkeypatch.setattr(
        sys,
        "argv",
        ["algo_trading.cli", "--config", str(config_path), "screen", "--as-of", "2026-04-13"],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli_module.main()

    assert str(excinfo.value) == "Live screen setup is incomplete."


def test_main_screen_defaults_to_today_when_as_of_is_omitted(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(AppConfig().to_dict()), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run_screen(config: AppConfig, as_of: str, refresh_data: bool, current_positions_path: str | None, output_dir: str) -> None:
        captured["as_of"] = as_of
        captured["refresh_data"] = refresh_data
        captured["current_positions_path"] = current_positions_path
        captured["output_dir"] = output_dir

    monkeypatch.setattr(cli_module, "_run_screen", fake_run_screen)
    monkeypatch.setattr(
        sys,
        "argv",
        ["algo_trading.cli", "--config", str(config_path), "screen"],
    )

    cli_module.main()

    assert captured == {
        "as_of": str(pd.Timestamp.today().normalize().date()),
        "refresh_data": False,
        "current_positions_path": None,
        "output_dir": "artifacts/live",
    }


def test_run_screen_writes_live_artifacts_and_creates_empty_trade_log(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "live"
    config = AppConfig()
    config.strategy.rebalance_frequency = "monthly"
    sessions = pd.DatetimeIndex([pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-10")])
    context = SimpleNamespace(prices=SimpleNamespace(sessions=sessions), metadata={})
    screen_df = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "composite_score": [1.2, 0.9],
            "name": ["Alpha", "Beta"],
            "sector": ["Tech", "Health"],
            "etf_weight_pct": [0.2, 0.1],
            "macro_composite": [0.5, 0.5],
        }
    )

    monkeypatch.setattr(cli_module, "_load_screen_context", lambda config, refresh_data: (context, []))
    monkeypatch.setattr(
        cli_module,
        "screen_live_session",
        lambda context, config, as_of: (pd.Timestamp("2026-04-10"), screen_df, {"ts_momentum": 1.0}),
    )

    cli_module._run_screen(
        config,
        as_of="2026-04-10",
        refresh_data=False,
        current_positions_path=None,
        output_dir=str(output_dir),
    )

    assert (output_dir / "screen_20260410.csv").exists()
    assert (output_dir / "rebalance_plan_20260410.csv").exists()
    assert (output_dir / "orders_20260410.csv").exists()
    assert (output_dir / "weights_20260410.json").exists()
    assert (output_dir / "live_session_log.csv").exists()
    assert (output_dir / "live_trade_log.csv").exists()

    trade_log = pd.read_csv(output_dir / "live_trade_log.csv")
    assert list(trade_log.columns) == [
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
    assert trade_log.empty

    session_log = pd.read_csv(output_dir / "live_session_log.csv")
    assert session_log["rebalance_due"].tolist() == [False]


def test_run_screen_appends_trade_log_when_orders_exist(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "live"
    config = AppConfig()
    config.strategy.rebalance_frequency = "monthly"
    positions_path = tmp_path / "positions.csv"
    positions_path.write_text("symbol,shares\nCCC,4\n", encoding="utf-8")

    sessions = pd.DatetimeIndex([pd.Timestamp("2026-04-01"), pd.Timestamp("2026-05-01")])
    context = SimpleNamespace(
        prices=SimpleNamespace(sessions=sessions),
        holdings=pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC"],
                "name": ["Alpha", "Beta", "Gamma"],
            }
        ),
        metadata={},
    )
    screen_df = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "composite_score": [1.2, 0.9],
            "name": ["Alpha", "Beta"],
            "sector": ["Tech", "Health"],
            "etf_weight_pct": [0.2, 0.1],
            "macro_composite": [0.5, 0.5],
        }
    )

    monkeypatch.setattr(cli_module, "_load_screen_context", lambda config, refresh_data: (context, []))
    monkeypatch.setattr(
        cli_module,
        "screen_live_session",
        lambda context, config, as_of: (pd.Timestamp(as_of), screen_df, {"ts_momentum": 1.0}),
    )

    for as_of in ["2026-04-01", "2026-05-01"]:
        cli_module._run_screen(
            config,
            as_of=as_of,
            refresh_data=False,
            current_positions_path=str(positions_path),
            output_dir=str(output_dir),
        )

    trade_log = pd.read_csv(output_dir / "live_trade_log.csv")
    assert len(trade_log) == 6
    assert set(trade_log["action"]) == {"SELL", "BUY"}
    assert trade_log.loc[trade_log["symbol"].eq("CCC"), "name"].eq("Gamma").all()
    trade_dates = pd.to_datetime(trade_log["date"]).dt.strftime("%Y-%m-%d").tolist()
    assert trade_dates.count("2026-04-01") == 3
    assert trade_dates.count("2026-05-01") == 3

    session_log = pd.read_csv(output_dir / "live_session_log.csv")
    assert len(session_log) == 2


def test_run_live_launcher_uses_repo_venv(tmp_path: Path) -> None:
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        pytest.skip("PowerShell is not available")

    repo_root = Path(__file__).resolve().parents[1]
    launcher = repo_root / "scripts" / "run_champion_live.ps1"

    # Use a non-rebalance date so the calendar guard short-circuits and the
    # launcher exits cleanly without needing FRED data or TimesFM forecasts.
    result = subprocess.run(
        [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(launcher), "-AsOf", "2026-04-13"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    combined_output = f"{result.stdout}\n{result.stderr}"
    assert "ModuleNotFoundError" not in combined_output
    assert "ConstantInputWarning" not in combined_output
    assert "Champion live signal check started" in combined_output
    assert result.returncode == 0


def test_calendar_guard_flags_first_session_of_month() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    guard = repo_root / "scripts" / "is_first_nyse_rebalance_session.py"
    python = sys.executable

    # 2026-04-01 is the first NYSE trading day of April 2026.
    result = subprocess.run([python, str(guard), "2026-04-01"], cwd=repo_root, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    # The guard must announce the candidate session so the launcher can parse it.
    assert "SESSION=2026-04-01" in result.stdout

    # 2026-04-13 is a trading day but not the first of the month.
    result = subprocess.run([python, str(guard), "2026-04-13"], cwd=repo_root, capture_output=True, text=True, check=False)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "SESSION=2026-04-13" in result.stdout

    # 2026-04-04 is a Saturday (not a trading day).
    result = subprocess.run([python, str(guard), "2026-04-04"], cwd=repo_root, capture_output=True, text=True, check=False)
    assert result.returncode == 2, result.stdout + result.stderr


def test_calendar_guard_auto_mode_returns_month_first_session_for_catchup() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    guard = repo_root / "scripts" / "is_first_nyse_rebalance_session.py"
    python = sys.executable

    # 2026-07-03 01:00 UTC = 2026-07-02 21:00 ET, after the July 2 close.
    # If July was not processed yet, the launcher must catch up the July 1
    # first-of-month rebalance rather than exit 2 because July 2 itself is not
    # first-of-month.
    result = subprocess.run(
        [python, str(guard), "--now-utc", "2026-07-03T01:00:00+00:00"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SESSION=2026-07-01" in result.stdout
    assert "CANDIDATE_SESSION=2026-07-02" in result.stdout


def test_launcher_skips_when_month_already_processed(tmp_path: Path) -> None:
    """Catch-up: a state file recording the month makes the launcher skip the screen."""
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        pytest.skip("PowerShell is not available")

    repo_root = Path(__file__).resolve().parents[1]
    launcher = repo_root / "scripts" / "run_champion_live.ps1"
    state_file = repo_root / "artifacts" / "live" / "last_rebalance_processed.txt"
    lock_file = repo_root / "artifacts" / "live" / ".run.lock"

    # Pre-record April so a 2026-04-01 run is treated as already done.
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text("2026-04", encoding="ascii")
    try:
        result = subprocess.run(
            [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(launcher), "-AsOf", "2026-04-01"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        combined = f"{result.stdout}\n{result.stderr}"
        assert result.returncode == 0, combined
        assert "already processed" in combined
        # The screen must NOT have run (no forecast refresh, no python run_champion.py).
        assert "Refreshing TimesFM forecasts" not in combined
        assert "Run ended." in combined
        assert not lock_file.exists()  # lock always cleaned up
    finally:
        if state_file.exists():
            state_file.unlink()
        if lock_file.exists():
            lock_file.unlink()
