from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha1
import io
import os
from pathlib import Path
import pickle
import re
import time
from typing import Any, Iterable, Iterator
import warnings

from fredapi import Fred
import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .config import AppConfig
from .env_utils import load_environment

FRED_SERIES_METADATA: dict[str, dict[str, Any]] = {
    "CFNAIDIFF": {
        "name": "Chicago Fed National Activity Index Diffusion Index",
        "frequency": "monthly",
        "inverse": False,
    },
    "EMVMACROBUS": {
        "name": "Macro Volatility Tracker: Business Investment And Sentiment",
        "frequency": "monthly",
        "inverse": True,
    },
    "VIXCLS": {"name": "CBOE VIX", "frequency": "daily", "inverse": True},
    "VXVCLS": {"name": "CBOE 3-Month Volatility Index", "frequency": "daily", "inverse": True},
    "GVZCLS": {"name": "CBOE Gold Volatility Index", "frequency": "daily", "inverse": True},
    "OVXCLS": {"name": "CBOE Crude Oil Volatility Index", "frequency": "daily", "inverse": True},
    "DGS10": {"name": "10-Year Treasury Rate", "frequency": "daily", "inverse": True},
    "T10Y3M": {"name": "10Y-3M Treasury Spread", "frequency": "daily", "inverse": False},
    "FEDFUNDS": {"name": "Fed Funds Rate", "frequency": "monthly", "inverse": True},
    "STLFSI4": {"name": "St. Louis Fed Stress Index", "frequency": "weekly", "inverse": True},
    "NFCI": {"name": "Chicago Fed National Financial Conditions Index", "frequency": "weekly", "inverse": True},
    "UMCSENT": {"name": "U. Michigan Sentiment", "frequency": "monthly", "inverse": False},
    "CFNAI": {"name": "Chicago Fed National Activity Index", "frequency": "monthly", "inverse": False},
    "DTWEXBGS": {"name": "Broad Dollar Index", "frequency": "daily", "inverse": True},
}

# Point-in-time publication lag (calendar days) applied to each FRED series
# before forward-filling onto the session grid. Daily series publish same-day
# (the +1-session shift in ``build_factor_library`` covers next-session use);
# weekly and monthly series publish with a multi-day / multi-week delay that
# the single-session shift cannot cover. Without this lag the macro risk gate
# would use unreleased monthly/weekly values (look-ahead bias).
PUBLICATION_LAG_DAYS: dict[str, int] = {
    "daily": 0,
    "weekly": 10,
    "monthly": 45,
}

_COMMON_TICKER_FIXUPS = {
    "BRKB": "BRK-B",
    "BFB": "BF-B",
}

_INCOME_LABELS = {
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "gross_profit": ["Gross Profit"],
}

_BALANCE_LABELS = {
    "assets": ["Total Assets"],
    "equity": ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"],
}

_CASHFLOW_LABELS = {
    "operating_cash_flow": [
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
        "Net Cash Provided By Operating Activities",
    ],
    "capex": ["Capital Expenditure", "Capital Expenditure Reported"],
    "issuance": ["Issuance Of Capital Stock", "Issuance Of Common Stock"],
    "repurchase": ["Repurchase Of Capital Stock", "Repurchase Of Common Stock"],
}

FUNDAMENTAL_FACTOR_NAMES = (
    "book_to_market",
    "earnings_yield",
    "roa",
    "roe",
    "gross_profitability",
    "asset_growth",
    "investment_to_assets",
    "net_issuance",
    "accruals",
    "cash_flow_yield",
    "earnings_surprise",
)


@dataclass(slots=True)
class PricePanel:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    @property
    def sessions(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)

    def subset(self, symbols: Iterable[str]) -> "PricePanel":
        symbol_list = [symbol for symbol in symbols if symbol in self.close.columns]
        return PricePanel(
            open=self.open.loc[:, symbol_list],
            high=self.high.loc[:, symbol_list],
            low=self.low.loc[:, symbol_list],
            close=self.close.loc[:, symbol_list],
            volume=self.volume.loc[:, symbol_list],
        )


@dataclass(slots=True)
class MarketContext:
    holdings: pd.DataFrame
    prices: PricePanel
    macro: pd.DataFrame
    fundamental_factors: dict[str, pd.DataFrame]
    universe_membership: pd.DataFrame
    metadata: dict[str, Any]


class MacroDataUnavailableError(RuntimeError):
    def __init__(self, cache_dir: Path, missing_series: list[str]) -> None:
        self.cache_dir = cache_dir
        self.missing_series = missing_series
        preview = ", ".join(missing_series[:4]) if missing_series else "unknown"
        suffix = ", ..." if len(missing_series) > 4 else ""
        super().__init__(
            "FRED_API_KEY is required to fetch macro data from FRED when cached data is unavailable. "
            f"Missing cached series: {preview}{suffix}"
        )


def project_cache_dir(root: str | Path | None = None) -> Path:
    base = Path.cwd() if root is None else Path(root)
    cache_dir = base if base.name.lower() == "cache" else base / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _is_fresh(path: Path, max_age_days: int) -> bool:
    if not path.exists():
        return False
    age = pd.Timestamp.now("UTC") - pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
    return age <= pd.Timedelta(days=max_age_days)


def _chunked(values: list[str], size: int) -> Iterator[list[str]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _normalize_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper().replace(".", "-").replace("/", "-")
    return _COMMON_TICKER_FIXUPS.get(cleaned, cleaned)


def _normalize_holdings_frame(holdings: pd.DataFrame, universe_limit: int | None) -> pd.DataFrame:
    clean = holdings.rename(columns=lambda value: str(value).strip().lower().replace(" ", "_"))
    if "asset_class" in clean.columns:
        clean = clean[clean["asset_class"].eq("Equity")].copy()
    if "ticker" not in clean.columns:
        raise ValueError("Holdings frame must include a ticker column.")

    clean["ticker"] = clean["ticker"].fillna("").map(_normalize_symbol)
    clean = clean[clean["ticker"].ne("") & clean["ticker"].ne("-")]

    if "weight_(%)" in clean.columns and "weight_pct" not in clean.columns:
        clean = clean.rename(columns={"weight_(%)": "weight_pct"})
    if "weight_pct" not in clean.columns:
        clean["weight_pct"] = np.nan
    clean["weight_pct"] = pd.to_numeric(clean["weight_pct"], errors="coerce")

    if "name" not in clean.columns:
        clean["name"] = clean["ticker"]
    if "sector" not in clean.columns:
        clean["sector"] = np.nan

    clean = clean[["ticker", "name", "sector", "weight_pct"]]
    clean = clean.sort_values("weight_pct", ascending=False, na_position="last")
    clean = clean.drop_duplicates("ticker")
    if universe_limit is not None:
        clean = clean.head(universe_limit)
    return clean.reset_index(drop=True)


def _resolve_holdings_dir(config: AppConfig) -> Path | None:
    if config.universe.historical_holdings_dir is None:
        return None
    directory = Path(config.universe.historical_holdings_dir)
    if not directory.is_absolute():
        directory = Path.cwd() / directory
    return directory


def _resolve_backtest_window(config: AppConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp(config.backtest.end_date or pd.Timestamp.today().normalize()).normalize()
    if config.backtest.start_date is not None:
        start = pd.Timestamp(config.backtest.start_date).normalize()
    elif config.backtest.lookback_years is not None:
        start = (end - pd.DateOffset(years=config.backtest.lookback_years)).normalize()
    else:
        start = pd.Timestamp("1996-01-01")
    return start, end


def _parse_snapshot_date(path: Path) -> pd.Timestamp | None:
    match = re.search(r"(\d{4}-\d{2}-\d{2}|\d{8})", path.stem)
    if not match:
        return None
    token = match.group(1)
    if "-" in token:
        parsed = pd.to_datetime(token, format="%Y-%m-%d", errors="coerce")
    else:
        parsed = pd.to_datetime(token, format="%Y%m%d", errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def load_historical_holdings_snapshots(current_holdings: pd.DataFrame, config: AppConfig) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    if not config.universe.use_historical_snapshots:
        return []
    directory = _resolve_holdings_dir(config)
    if directory is None or not directory.exists():
        return []

    snapshots: list[tuple[pd.Timestamp, pd.DataFrame]] = []
    for path in sorted(directory.glob("*.csv")):
        snapshot_date = _parse_snapshot_date(path)
        if snapshot_date is None:
            continue
        frame = _normalize_holdings_frame(pd.read_csv(path), config.universe.universe_limit)
        if frame.empty:
            continue
        snapshots.append((snapshot_date, frame))

    if not snapshots:
        return []

    today = pd.Timestamp.today().normalize()
    if snapshots[-1][0] < today:
        snapshots.append((today, current_holdings.copy()))
    return snapshots


def write_holdings_snapshot(
    config: AppConfig,
    as_of: str | pd.Timestamp | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> Path:
    directory = _resolve_holdings_dir(config)
    if directory is None:
        raise RuntimeError("historical_holdings_dir must be set to archive holdings snapshots.")
    directory.mkdir(parents=True, exist_ok=True)
    holdings = fetch_iwv_holdings(config=config, cache_dir=cache_dir, refresh=refresh, apply_limit=False)
    stamp = pd.Timestamp(as_of or pd.Timestamp.today().normalize()).normalize().strftime("%Y-%m-%d")
    path = directory / f"{stamp}.csv"
    holdings.to_csv(path, index=False)
    return path


def import_historical_holdings_directory(
    config: AppConfig,
    source_dir: str | Path,
    pattern: str = "*.csv",
    overwrite: bool = False,
) -> list[Path]:
    source_root = Path(source_dir)
    if not source_root.exists() or not source_root.is_dir():
        raise RuntimeError(f"Historical holdings source directory not found: {source_root}")

    target_dir = _resolve_holdings_dir(config)
    if target_dir is None:
        raise RuntimeError("historical_holdings_dir must be set to import holdings snapshots.")
    target_dir.mkdir(parents=True, exist_ok=True)

    imported: list[Path] = []
    for source_path in sorted(source_root.rglob(pattern)):
        snapshot_date = _parse_snapshot_date(source_path)
        if snapshot_date is None:
            continue
        normalized = _normalize_holdings_frame(pd.read_csv(source_path), universe_limit=None)
        if normalized.empty:
            continue
        destination = target_dir / f"{snapshot_date.strftime('%Y-%m-%d')}.csv"
        if destination.exists() and not overwrite:
            continue
        normalized.to_csv(destination, index=False)
        imported.append(destination)
    return imported


def _combine_holdings_metadata(current_holdings: pd.DataFrame, snapshots: list[tuple[pd.Timestamp, pd.DataFrame]]) -> pd.DataFrame:
    frames = [current_holdings.assign(snapshot_date=pd.Timestamp.today().normalize())]
    frames.extend(frame.assign(snapshot_date=snapshot_date) for snapshot_date, frame in snapshots)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("snapshot_date")
    combined = combined.drop_duplicates("ticker", keep="last")
    return combined.drop(columns=["snapshot_date"]).reset_index(drop=True)


def _build_universe_membership(
    sessions: pd.DatetimeIndex,
    symbols: list[str],
    current_holdings: pd.DataFrame,
    snapshots: list[tuple[pd.Timestamp, pd.DataFrame]],
    config: AppConfig,
) -> tuple[pd.DataFrame, list[str]]:
    membership = pd.DataFrame(False, index=sessions, columns=symbols)
    notes: list[str] = []

    if snapshots:
        ordered = sorted(snapshots, key=lambda item: item[0])
        first_snapshot = ordered[0][0]
        if sessions.min() < first_snapshot and config.universe.allow_current_holdings_fallback:
            fallback_members = [symbol for symbol in current_holdings["ticker"] if symbol in membership.columns]
            pre_snapshot_mask = sessions < first_snapshot
            if fallback_members and pre_snapshot_mask.any():
                membership.loc[pre_snapshot_mask, fallback_members] = True
            note = (
                f"Historical holdings snapshots begin on {first_snapshot.date()}, so earlier dates use current IWV holdings fallback and remain survivorship-biased."
            )
            notes.append(note)
            warnings.warn(note, stacklevel=2)
        for index, (snapshot_date, frame) in enumerate(ordered):
            next_date = ordered[index + 1][0] if index + 1 < len(ordered) else sessions.max() + pd.Timedelta(days=1)
            mask = (sessions >= snapshot_date) & (sessions < next_date)
            if not mask.any():
                continue
            members = [symbol for symbol in frame["ticker"] if symbol in membership.columns]
            if members:
                membership.loc[mask, members] = True
        if sessions.min() < first_snapshot and not config.universe.allow_current_holdings_fallback:
            notes.append(
                f"Historical holdings snapshots begin on {first_snapshot.date()}, so earlier dates are excluded to avoid look-ahead bias."
            )
    elif config.universe.allow_current_holdings_fallback:
        members = [symbol for symbol in current_holdings["ticker"] if symbol in membership.columns]
        if members:
            membership.loc[:, members] = True
        note = "No historical holdings snapshots were found; falling back to current IWV holdings introduces survivorship bias."
        notes.append(note)
        warnings.warn(note, stacklevel=2)
    else:
        raise RuntimeError("No historical holdings snapshots were found and current-holdings fallback is disabled.")

    return membership, notes


def _frame_with_datetime_columns(frame: Any) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    clean = frame.copy()
    clean.columns = pd.to_datetime(clean.columns, errors="coerce")
    valid_columns = [column for column in clean.columns if pd.notna(column)]
    if not valid_columns:
        return pd.DataFrame()
    clean = clean.loc[:, valid_columns]
    clean.columns = pd.DatetimeIndex(clean.columns).tz_localize(None)
    clean = clean.sort_index(axis=1)
    return clean


def _series_from_labels(frame: pd.DataFrame, labels: list[str]) -> pd.Series:
    for label in labels:
        if label in frame.index:
            series = pd.to_numeric(frame.loc[label], errors="coerce")
            series.index = pd.DatetimeIndex(series.index).tz_localize(None)
            return series.sort_index()
    return pd.Series(index=pd.DatetimeIndex([]), dtype=float)


def _align_events_to_sessions(frame: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    aligned_index = frame.index.union(sessions).sort_values()
    return frame.reindex(aligned_index).ffill().reindex(sessions)


def _latest_value(quarterly: pd.DataFrame, annual: pd.DataFrame, labels: list[str], cutoff: pd.Timestamp) -> float:
    quarterly_series = _series_from_labels(quarterly, labels)
    quarterly_series = quarterly_series.loc[quarterly_series.index <= cutoff].dropna()
    if not quarterly_series.empty:
        return float(quarterly_series.iloc[-1])
    annual_series = _series_from_labels(annual, labels)
    annual_series = annual_series.loc[annual_series.index <= cutoff].dropna()
    if not annual_series.empty:
        return float(annual_series.iloc[-1])
    return np.nan


def _ttm_value(quarterly: pd.DataFrame, annual: pd.DataFrame, labels: list[str], cutoff: pd.Timestamp) -> float:
    quarterly_series = _series_from_labels(quarterly, labels)
    quarterly_series = quarterly_series.loc[quarterly_series.index <= cutoff].dropna()
    if len(quarterly_series) >= 4:
        return float(quarterly_series.iloc[-4:].sum())
    annual_series = _series_from_labels(annual, labels)
    annual_series = annual_series.loc[annual_series.index <= cutoff].dropna()
    if not annual_series.empty:
        return float(annual_series.iloc[-1])
    return np.nan


def _year_over_year_change(quarterly: pd.DataFrame, annual: pd.DataFrame, labels: list[str], cutoff: pd.Timestamp) -> float:
    quarterly_series = _series_from_labels(quarterly, labels)
    quarterly_series = quarterly_series.loc[quarterly_series.index <= cutoff].dropna()
    if len(quarterly_series) >= 5:
        current = quarterly_series.iloc[-1]
        prior = quarterly_series.iloc[-5]
        if prior:
            return float(current / prior - 1.0)
    annual_series = _series_from_labels(annual, labels)
    annual_series = annual_series.loc[annual_series.index <= cutoff].dropna()
    if len(annual_series) >= 2:
        current = annual_series.iloc[-1]
        prior = annual_series.iloc[-2]
        if prior:
            return float(current / prior - 1.0)
    return np.nan


def _first_float(mapping: dict[str, Any], keys: list[str]) -> float:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return np.nan


def fetch_iwv_holdings(
    config: AppConfig,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    apply_limit: bool = True,
) -> pd.DataFrame:
    cache_root = project_cache_dir(cache_dir)
    holdings_path = cache_root / "iwv_holdings.csv"
    if not refresh and _is_fresh(holdings_path, config.universe.holdings_cache_days):
        holdings = pd.read_csv(holdings_path)
    else:
        try:
            response = requests.get(config.universe.holdings_url, timeout=30)
            response.raise_for_status()
            text = response.text.replace("\ufeff", "")
            lines = text.splitlines()
            header_row = next(index for index, line in enumerate(lines) if line.startswith("Ticker,"))
            trailing_blank = next(
                (index for index, line in enumerate(lines[header_row + 1 :], start=header_row + 1) if not line.strip()),
                len(lines),
            )
            payload = "\n".join(lines[header_row:trailing_blank])
            holdings = pd.read_csv(io.StringIO(payload))
            holdings.to_csv(holdings_path, index=False)
        except Exception:
            if holdings_path.exists():
                holdings = pd.read_csv(holdings_path)
            else:
                raise
    limit = config.universe.universe_limit if apply_limit else None
    return _normalize_holdings_frame(holdings, universe_limit=limit)


# Wikipedia "List of S&P 500 companies" — the standard free, programmatic
# source for the current S&P 500 constituents. Used when etf_ticker == "SPY"
# (the iShares IVV/IWV CSV endpoints bot-block programmatic fetches). Same
# survivorship caveat as the IWV fallback: this is the *current* constituent
# list, so backtests on it are optimistic vs a tradable point-in-time universe.
WIKIPEDIA_SP500_URL = (
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
)


def fetch_sp500_holdings(
    config: AppConfig,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    apply_limit: bool = True,
) -> pd.DataFrame:
    """Fetch the current S&P 500 constituents from Wikipedia and cache them.

    Refreshes `cache/sp500_holdings.csv` when older than `holdings_cache_days`
    (default 7), so each run sees a fresh constituent list as S&P 500 rebalances.
    Falls back to the cached file if the live fetch fails. Symbols are normalized
    via `_normalize_symbol` (BRK.B -> BRK-B) so they match yfinance price data.
    """
    from bs4 import BeautifulSoup  # lazy import: bs4 is optional for non-SPY use

    cache_root = project_cache_dir(cache_dir)
    holdings_path = cache_root / "sp500_holdings.csv"
    if not refresh and _is_fresh(holdings_path, config.universe.holdings_cache_days):
        holdings = pd.read_csv(holdings_path)
    else:
        try:
            response = requests.get(
                WIKIPEDIA_SP500_URL,
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", {"class": "wikitable"})
            if table is None:
                raise RuntimeError("Wikipedia S&P 500 wikitable not found")
            rows = table.find_all("tr")
            header = [cell.get_text(strip=True) for cell in rows[0].find_all(["th", "td"])]
            records = []
            for tr in rows[1:]:
                cells = [cell.get_text(strip=True) for cell in tr.find_all(["td", "th"])]
                if cells and cells[0] and len(cells) >= len(header):
                    records.append(cells[: len(header)])
            holdings = pd.DataFrame(records, columns=header)
            # Normalize to the ticker/name/sector schema expected downstream.
            holdings = holdings.rename(
                columns={
                    "Symbol": "ticker",
                    "Security": "name",
                    "GICSSector": "sector",
                    "GICS Sector": "sector",
                }
            )
            holdings.to_csv(holdings_path, index=False)
        except Exception:
            if holdings_path.exists():
                holdings = pd.read_csv(holdings_path)
            else:
                raise
    limit = config.universe.universe_limit if apply_limit else None
    return _normalize_holdings_frame(holdings, universe_limit=limit)


def _download_batch(batch: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    raw = yf.download(
        tickers=batch,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        repair=True,
        progress=False,
        group_by="ticker",
        threads=True,
        # yfinance has no default HTTP timeout; without this a stalled connection
        # on a cache-miss download hangs indefinitely (no error, no completion).
        timeout=60,
    )
    frames: dict[str, pd.DataFrame] = {}
    if raw.empty:
        return frames

    if isinstance(raw.columns, pd.MultiIndex):
        for symbol in batch:
            if symbol not in raw.columns.get_level_values(0):
                continue
            frame = raw[symbol].copy()
            frames[symbol] = frame
    else:
        frames[batch[0]] = raw.copy()

    normalized: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        frame = frame.rename(columns=lambda value: str(value).strip().title())
        required = [column for column in ["Open", "High", "Low", "Close", "Volume"] if column in frame.columns]
        if len(required) < 5:
            continue
        subset = frame[required].copy()
        subset.index = pd.DatetimeIndex(subset.index).tz_localize(None).normalize()
        subset = subset.sort_index()
        normalized[symbol] = subset
    return normalized


def _fetch_fred_series_with_retries(
    fred: Fred,
    series_id: str,
    start: str,
    end: str,
    attempts: int = 3,
) -> pd.Series:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fred.get_series(series_id, observation_start=start, observation_end=end)
        except Exception as error:
            last_error = error
            if attempt < attempts:
                time.sleep(1.0 * attempt)
    raise RuntimeError(f"Unable to fetch FRED series {series_id}: {last_error}") from last_error


def download_price_panel(
    tickers: list[str],
    start: str,
    end: str,
    config: AppConfig,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> PricePanel:
    cache_root = project_cache_dir(cache_dir)
    price_dir = cache_root / "prices"
    price_dir.mkdir(parents=True, exist_ok=True)
    key = sha1("|".join([*sorted(tickers), start, end]).encode("utf-8")).hexdigest()[:16]
    cache_path = price_dir / f"panel_{key}.pkl"

    if not refresh and _is_fresh(cache_path, config.backtest.price_cache_days):
        return pickle.loads(cache_path.read_bytes())

    symbol_frames: dict[str, pd.DataFrame] = {}
    for batch in _chunked(tickers, 100):
        symbol_frames.update(_download_batch(batch, start=start, end=end))

    if not symbol_frames:
        raise RuntimeError("No price data could be downloaded from yfinance.")

    # Partial-download guard: a mostly-failed yfinance fetch (e.g. a transient
    # outage mid-batch) would otherwise silently shrink the universe and produce
    # a wrong "latest session". Warn on any failures and abort if a large share
    # of the universe is missing so the caller does not act on a tiny universe.
    failed = [t for t in tickers if t not in symbol_frames]
    failure_rate = len(failed) / max(1, len(tickers))
    if failed:
        preview = ", ".join(failed[:10]) + ("…" if len(failed) > 10 else "")
        warnings.warn(
            f"yfinance returned no data for {len(failed)}/{len(tickers)} tickers "
            f"({failure_rate:.0%}); missing include {preview}.",
            stacklevel=2,
        )
        if failure_rate > 0.20:
            raise RuntimeError(
                f"Partial price download: {failure_rate:.0%} of tickers failed (>20% threshold). "
                f"Re-run with --refresh-data or check connectivity. Missing: {preview}"
            )

    open_df = pd.concat({symbol: frame["Open"] for symbol, frame in symbol_frames.items()}, axis=1).sort_index(axis=1)
    high_df = pd.concat({symbol: frame["High"] for symbol, frame in symbol_frames.items()}, axis=1).sort_index(axis=1)
    low_df = pd.concat({symbol: frame["Low"] for symbol, frame in symbol_frames.items()}, axis=1).sort_index(axis=1)
    close_df = pd.concat({symbol: frame["Close"] for symbol, frame in symbol_frames.items()}, axis=1).sort_index(axis=1)
    volume_df = pd.concat({symbol: frame["Volume"] for symbol, frame in symbol_frames.items()}, axis=1).sort_index(axis=1)

    panel = PricePanel(open=open_df, high=high_df, low=low_df, close=close_df, volume=volume_df)
    cache_path.write_bytes(pickle.dumps(panel))
    return panel


def load_macro_data(
    sessions: pd.DatetimeIndex,
    config: AppConfig,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    cache_root = project_cache_dir(cache_dir)
    fred_dir = cache_root / "fred"
    fred_dir.mkdir(parents=True, exist_ok=True)

    if "FRED_API_KEY" not in os.environ:
        load_environment()
    api_key = os.getenv("FRED_API_KEY")
    fred = Fred(api_key=api_key) if api_key else None
    macro: dict[str, pd.Series] = {}
    enabled_series = [series_id for series_id in FRED_SERIES_METADATA if config.macro.enabled_series.get(series_id, False)]
    start = str((sessions.min() - pd.Timedelta(days=400)).date())
    end = str(sessions.max().date())

    for series_id in enabled_series:
        meta = FRED_SERIES_METADATA[series_id]
        cache_path = fred_dir / f"{series_id}.csv"
        cache_is_fresh = not refresh and _is_fresh(cache_path, config.backtest.fred_cache_days)
        use_cached_series = cache_is_fresh or (cache_path.exists() and fred is None)
        if use_cached_series:
            if not cache_is_fresh and fred is None:
                warnings.warn(
                    f"Using cached FRED series {series_id} without refresh because FRED_API_KEY is not set.",
                    stacklevel=2,
                )
            frame = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            series = frame.iloc[:, 0]
        else:
            if fred is None:
                missing_series = [
                    enabled_series_id
                    for enabled_series_id in enabled_series
                    if not (fred_dir / f"{enabled_series_id}.csv").exists()
                ]
                raise MacroDataUnavailableError(
                    cache_dir=fred_dir,
                    missing_series=missing_series or [series_id],
                )
            try:
                series = _fetch_fred_series_with_retries(fred=fred, series_id=series_id, start=start, end=end)
            except RuntimeError as error:
                warnings.warn(str(error), stacklevel=2)
                continue
            series = pd.to_numeric(series, errors="coerce")
            frame = pd.DataFrame({series_id: series})
            frame.to_csv(cache_path)

        series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
        publication_lag_days = PUBLICATION_LAG_DAYS.get(meta["frequency"], 0)
        if publication_lag_days > 0:
            # Shift the observation date forward by the publication delay so a
            # value is only available on/after its release date. ``reindex``
            # with ffill then yields NaN for sessions before release.
            series.index = series.index + pd.Timedelta(days=publication_lag_days)
        aligned = series.reindex(sessions, method="ffill")
        if meta["inverse"]:
            aligned = -aligned
        min_periods = min(config.macro.zscore_window, max(30, config.macro.zscore_window // 4))
        rolling_mean = aligned.rolling(config.macro.zscore_window, min_periods=min_periods).mean()
        rolling_std = aligned.rolling(config.macro.zscore_window, min_periods=min_periods).std()
        standardized = (aligned - rolling_mean) / rolling_std.replace(0.0, np.nan)
        macro[series_id] = standardized

    macro_frame = pd.DataFrame(macro, index=sessions).sort_index()
    if macro_frame.empty:
        macro_frame = pd.DataFrame(index=sessions)
        macro_frame["macro_composite"] = 0.0
        return macro_frame
    macro_frame["macro_composite"] = macro_frame.mean(axis=1, skipna=True)
    return macro_frame


def _download_fundamental_bundle(symbol: str) -> dict[str, Any]:
    ticker = yf.Ticker(symbol)
    info: dict[str, Any] = {}
    try:
        raw_info = ticker.info
        if isinstance(raw_info, dict):
            info = raw_info
    except Exception:
        info = {}

    bundle: dict[str, Any] = {
        "info": info,
        "income_stmt": _frame_with_datetime_columns(ticker.income_stmt),
        "quarterly_income_stmt": _frame_with_datetime_columns(ticker.quarterly_income_stmt),
        "balance_sheet": _frame_with_datetime_columns(ticker.balance_sheet),
        "quarterly_balance_sheet": _frame_with_datetime_columns(ticker.quarterly_balance_sheet),
        "cashflow": _frame_with_datetime_columns(ticker.cashflow),
        "quarterly_cashflow": _frame_with_datetime_columns(ticker.quarterly_cashflow),
    }
    earnings_history = pd.DataFrame()
    try:
        earnings_history = ticker.earnings_history
    except Exception:
        try:
            earnings_history = ticker.get_earnings_history()
        except Exception:
            earnings_history = pd.DataFrame()
    bundle["earnings_history"] = earnings_history
    return bundle


def fetch_fundamental_bundles(
    symbols: list[str],
    config: AppConfig,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> dict[str, dict[str, Any]]:
    cache_root = project_cache_dir(cache_dir)
    fundamentals_dir = cache_root / "fundamentals"
    fundamentals_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    to_download: list[str] = []
    for symbol in symbols:
        path = fundamentals_dir / f"{symbol}.pkl"
        if not refresh and _is_fresh(path, config.universe.fundamentals_cache_days):
            results[symbol] = pickle.loads(path.read_bytes())
        else:
            to_download.append(symbol)

    max_workers = min(6, max(1, len(to_download)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_fundamental_bundle, symbol): symbol for symbol in to_download}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                bundle = future.result()
            except Exception:
                bundle = {
                    "info": {},
                    "income_stmt": pd.DataFrame(),
                    "quarterly_income_stmt": pd.DataFrame(),
                    "balance_sheet": pd.DataFrame(),
                    "quarterly_balance_sheet": pd.DataFrame(),
                    "cashflow": pd.DataFrame(),
                    "quarterly_cashflow": pd.DataFrame(),
                    "earnings_history": pd.DataFrame(),
                }
            results[symbol] = bundle
            (fundamentals_dir / f"{symbol}.pkl").write_bytes(pickle.dumps(bundle))
    return results


def _earnings_surprise_series(frame: Any) -> pd.Series:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)

    data = frame.copy()
    if "date" in data.columns:
        dates = pd.to_datetime(data["date"], errors="coerce")
    elif "Earnings Date" in data.columns:
        dates = pd.to_datetime(data["Earnings Date"], errors="coerce")
    else:
        dates = pd.to_datetime(data.index, errors="coerce")

    surprise_column = next((column for column in data.columns if "surprise" in str(column).lower()), None)
    if surprise_column is None:
        return pd.Series(index=pd.DatetimeIndex([]), dtype=float)

    surprise = pd.to_numeric(data[surprise_column], errors="coerce")
    series = pd.Series(surprise.values, index=pd.DatetimeIndex(dates).tz_localize(None))
    series = series[series.index.notna()].sort_index()
    return series


def _build_symbol_fundamentals(
    symbol: str,
    close_index: pd.DatetimeIndex,
    close_series: pd.Series,
    bundle: dict[str, Any],
    lag_days: int,
) -> pd.DataFrame:
    quarterly_income = _frame_with_datetime_columns(bundle.get("quarterly_income_stmt"))
    annual_income = _frame_with_datetime_columns(bundle.get("income_stmt"))
    quarterly_balance = _frame_with_datetime_columns(bundle.get("quarterly_balance_sheet"))
    annual_balance = _frame_with_datetime_columns(bundle.get("balance_sheet"))
    quarterly_cashflow = _frame_with_datetime_columns(bundle.get("quarterly_cashflow"))
    annual_cashflow = _frame_with_datetime_columns(bundle.get("cashflow"))
    info = bundle.get("info", {}) if isinstance(bundle.get("info", {}), dict) else {}
    earnings_surprise = _earnings_surprise_series(bundle.get("earnings_history"))

    statement_dates = sorted(
        {
            *quarterly_income.columns.tolist(),
            *annual_income.columns.tolist(),
            *quarterly_balance.columns.tolist(),
            *annual_balance.columns.tolist(),
            *quarterly_cashflow.columns.tolist(),
            *annual_cashflow.columns.tolist(),
        }
    )
    if not statement_dates:
        return pd.DataFrame(index=close_index)

    shares_outstanding = _first_float(info, ["sharesOutstanding", "impliedSharesOutstanding"])
    records: list[dict[str, float | pd.Timestamp]] = []
    for statement_date in statement_dates:
        available_date = pd.Timestamp(statement_date) + pd.Timedelta(days=lag_days)
        latest_close = close_series.loc[:available_date].dropna()
        if latest_close.empty:
            continue
        price = float(latest_close.iloc[-1])
        assets = _latest_value(quarterly_balance, annual_balance, _BALANCE_LABELS["assets"], statement_date)
        equity = _latest_value(quarterly_balance, annual_balance, _BALANCE_LABELS["equity"], statement_date)
        net_income_ttm = _ttm_value(quarterly_income, annual_income, _INCOME_LABELS["net_income"], statement_date)
        gross_profit_ttm = _ttm_value(quarterly_income, annual_income, _INCOME_LABELS["gross_profit"], statement_date)
        operating_cash_flow_ttm = _ttm_value(
            quarterly_cashflow,
            annual_cashflow,
            _CASHFLOW_LABELS["operating_cash_flow"],
            statement_date,
        )
        capex_ttm = _ttm_value(quarterly_cashflow, annual_cashflow, _CASHFLOW_LABELS["capex"], statement_date)
        issuance_ttm = _ttm_value(quarterly_cashflow, annual_cashflow, _CASHFLOW_LABELS["issuance"], statement_date)
        repurchase_ttm = _ttm_value(quarterly_cashflow, annual_cashflow, _CASHFLOW_LABELS["repurchase"], statement_date)
        asset_growth = _year_over_year_change(quarterly_balance, annual_balance, _BALANCE_LABELS["assets"], statement_date)

        market_cap = price * shares_outstanding if np.isfinite(shares_outstanding) else np.nan
        recent_surprise = earnings_surprise.loc[earnings_surprise.index <= statement_date].dropna()
        last_surprise = float(recent_surprise.iloc[-1]) if not recent_surprise.empty else np.nan

        records.append(
            {
                "date": available_date,
                "book_to_market": equity / market_cap if market_cap and np.isfinite(equity) else np.nan,
                "earnings_yield": net_income_ttm / market_cap if market_cap else np.nan,
                "roa": net_income_ttm / assets if assets else np.nan,
                "roe": net_income_ttm / equity if equity else np.nan,
                "gross_profitability": gross_profit_ttm / assets if assets else np.nan,
                "asset_growth": -asset_growth if np.isfinite(asset_growth) else np.nan,
                "investment_to_assets": -(abs(capex_ttm) / assets) if assets else np.nan,
                "net_issuance": -((issuance_ttm + repurchase_ttm) / assets) if assets else np.nan,
                "accruals": -((net_income_ttm - operating_cash_flow_ttm) / assets) if assets else np.nan,
                "cash_flow_yield": operating_cash_flow_ttm / market_cap if market_cap else np.nan,
                "earnings_surprise": last_surprise,
            }
        )

    if not records:
        return pd.DataFrame(index=close_index)

    frame = pd.DataFrame.from_records(records).drop_duplicates(subset="date", keep="last")
    frame = frame.set_index("date").sort_index()
    frame = _align_events_to_sessions(frame, close_index)
    return frame


def build_fundamental_factor_frames(
    close: pd.DataFrame,
    bundles: dict[str, dict[str, Any]],
    statement_lag_days: int,
    factor_names: list[str] | set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    requested_factor_names = list(factor_names) if factor_names is not None else [
        "book_to_market",
        "earnings_yield",
        "roa",
        "roe",
        "gross_profitability",
        "asset_growth",
        "investment_to_assets",
        "net_issuance",
        "accruals",
        "cash_flow_yield",
        "earnings_surprise",
    ]
    if not requested_factor_names:
        return {}
    factors = {name: pd.DataFrame(index=close.index, columns=close.columns, dtype=float) for name in requested_factor_names}

    for symbol in close.columns:
        bundle = bundles.get(symbol, {})
        per_symbol = _build_symbol_fundamentals(
            symbol=symbol,
            close_index=pd.DatetimeIndex(close.index),
            close_series=close[symbol],
            bundle=bundle,
            lag_days=statement_lag_days,
        )
        if per_symbol.empty:
            continue
        for factor_name in requested_factor_names:
            if factor_name in per_symbol.columns:
                factors[factor_name][symbol] = per_symbol[factor_name]
    return factors


def load_market_context(
    config: AppConfig,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    refresh_holdings: bool | None = None,
    requested_fundamental_factors: list[str] | set[str] | tuple[str, ...] | None = None,
) -> MarketContext:
    current_holdings = (
        fetch_sp500_holdings(config=config, cache_dir=cache_dir,
                            refresh=refresh if refresh_holdings is None else refresh_holdings)
        if str(config.universe.etf_ticker).upper() == "SPY"
        else fetch_iwv_holdings(config=config, cache_dir=cache_dir,
                               refresh=refresh if refresh_holdings is None else refresh_holdings)
    )
    snapshots = load_historical_holdings_snapshots(current_holdings=current_holdings, config=config)
    holdings_metadata = _combine_holdings_metadata(current_holdings=current_holdings, snapshots=snapshots)

    requested_start, requested_end = _resolve_backtest_window(config)
    warmup_years = config.walk_forward.train_years + 2
    download_start = str((requested_start - pd.DateOffset(years=warmup_years)).date())

    tickers = holdings_metadata["ticker"].tolist()
    benchmark = config.backtest.benchmark_ticker
    ordered_tickers = list(dict.fromkeys([*tickers, benchmark]))
    prices = download_price_panel(
        tickers=ordered_tickers,
        start=download_start,
        end=str(requested_end.date()),
        config=config,
        cache_dir=cache_dir,
        refresh=refresh,
    )

    valid_symbols = []
    for symbol in tickers:
        if symbol not in prices.close.columns:
            continue
        close_series = prices.close[symbol].dropna()
        if len(close_series) < config.universe.min_history_days:
            continue
        valid_symbols.append(symbol)

    filtered_holdings = holdings_metadata[holdings_metadata["ticker"].isin(valid_symbols)].copy()
    filtered_prices = prices.subset([*valid_symbols, benchmark])
    universe_membership, notes = _build_universe_membership(
        sessions=filtered_prices.sessions,
        symbols=valid_symbols,
        current_holdings=current_holdings,
        snapshots=snapshots,
        config=config,
    )

    benchmark_start = None
    if benchmark in filtered_prices.close.columns:
        benchmark_history = filtered_prices.close[benchmark].dropna()
        if not benchmark_history.empty:
            benchmark_start = pd.Timestamp(benchmark_history.index[0])
    membership_dates = universe_membership.index[universe_membership.any(axis=1)]
    membership_start = pd.Timestamp(membership_dates[0]) if len(membership_dates) > 0 else None
    effective_start_candidates = [requested_start]
    if benchmark_start is not None:
        effective_start_candidates.append(benchmark_start)
    if membership_start is not None:
        effective_start_candidates.append(membership_start)
    effective_start = max(effective_start_candidates)

    if effective_start > requested_start:
        note = f"Effective study start moved from {requested_start.date()} to {effective_start.date()} based on available benchmark and universe history."
        notes.append(note)
        warnings.warn(note, stacklevel=2)

    macro = load_macro_data(filtered_prices.sessions, config=config, cache_dir=cache_dir, refresh=refresh)
    bundles = fetch_fundamental_bundles(valid_symbols, config=config, cache_dir=cache_dir, refresh=refresh)
    requested_factor_set = {
        factor_name
        for factor_name in (requested_fundamental_factors or [])
        if factor_name in FUNDAMENTAL_FACTOR_NAMES
    }
    enabled_fundamental_factors = [
        factor_name
        for factor_name in FUNDAMENTAL_FACTOR_NAMES
        if config.indicators.enabled_indicators.get(factor_name, False) or factor_name in requested_factor_set
    ]
    fundamental_factors = build_fundamental_factor_frames(
        close=filtered_prices.close.loc[:, valid_symbols],
        bundles=bundles,
        statement_lag_days=config.universe.statement_lag_days,
        factor_names=enabled_fundamental_factors,
    )
    return MarketContext(
        holdings=filtered_holdings.reset_index(drop=True),
        prices=filtered_prices,
        macro=macro,
        fundamental_factors=fundamental_factors,
        universe_membership=universe_membership,
        metadata={
            "requested_start": str(requested_start.date()),
            "requested_end": str(requested_end.date()),
            "effective_start": str(effective_start.date()),
            "using_historical_snapshots": bool(snapshots),
            "historical_snapshot_count": len(snapshots),
            "notes": notes,
        },
    )
