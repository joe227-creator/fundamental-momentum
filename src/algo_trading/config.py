from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha1
import json
from pathlib import Path
from typing import Any

DEFAULT_IWV_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239714/"
    "ishares-russell-3000-etf/1467271812596.ajax?"
    "fileType=csv&fileName=IWV_holdings&dataType=fund"
)


@dataclass(slots=True)
class UniverseConfig:
    etf_ticker: str = "IWV"
    holdings_url: str = DEFAULT_IWV_HOLDINGS_URL
    universe_limit: int | None = 500
    historical_holdings_dir: str | None = "data/holdings_history"
    use_historical_snapshots: bool = True
    allow_current_holdings_fallback: bool = True
    min_price: float = 5.0
    min_avg_dollar_volume: float = 20_000_000.0
    min_history_days: int = 260
    statement_lag_days: int = 45
    fundamentals_cache_days: int = 30
    holdings_cache_days: int = 7


@dataclass(slots=True)
class IndicatorConfig:
    sma_fast: int = 20
    sma_slow: int = 50
    sma_trend: int = 200
    ema_fast: int = 21
    ema_slow: int = 63
    breakout_weeks: int = 26
    variable_week_high_windows: list[int] = field(default_factory=lambda: [13, 26, 52])
    ts_mom_lookback_months: int = 12
    ts_mom_skip_months: int = 1
    rsi_period: int = 14
    cci_period: int = 20
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    obv_ema: int = 20
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    price_action_weeks: int = 8
    enabled_indicators: dict[str, bool] = field(
        default_factory=lambda: {
            "sma_crossover": True,
            "ema_crossover": True,
            "trend_filter": True,
            "breakout": True,
            "rsi": True,
            "cci": True,
            "macd": True,
            "obv": True,
            "bollinger": True,
            "ts_momentum": True,
            "variable_week_high": True,
            "price_action": True,
            "book_to_market": True,
            "earnings_yield": True,
            "roa": True,
            "roe": True,
            "gross_profitability": True,
            "asset_growth": True,
            "investment_to_assets": True,
            "net_issuance": True,
            "accruals": True,
            "cash_flow_yield": True,
            "earnings_surprise": True,
        }
    )


@dataclass(slots=True)
class StrategyConfig:
    signal_timeframe: str = "daily"
    rebalance_frequency: str = "weekly"
    max_holdings: int = 5
    close_positions_before_new_positions: bool = True
    transaction_cost_bps: float = 5.0
    initial_capital: float = 100_000.0
    full_turnover_rebalance: bool = True
    require_positive_composite: bool = True
    allow_cash_when_risk_off: bool = True
    macro_half_risk_threshold: float = -0.5
    macro_flat_threshold: float = -1.0
    base_filter_requires_trend: bool = True


@dataclass(slots=True)
class BacktestConfig:
    start_date: str | None = None
    lookback_years: int | None = 30
    end_date: str | None = None
    benchmark_ticker: str = "IWV"
    price_cache_days: int = 3
    fred_cache_days: int = 7


@dataclass(slots=True)
class WalkForwardConfig:
    train_years: int = 5
    test_months: int = 12
    step_months: int = 6
    min_train_rebalances: int = 26
    use_indicator_ic_weights: bool = True
    indicator_weighting_scheme: str = "legacy"
    fundamental_group_weight: float = 0.25
    fundamental_factor_weights: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ResearchConfig:
    iterations: int = 20
    random_seed: int = 42
    max_drawdown_limit: float = 0.50
    annualized_return_weight: float = 100.0
    profit_factor_weight: float = 10.0
    drawdown_weight: float = 50.0
    consecutive_no_improvement_limit: int = 20
    keep_top_n: int = 5


@dataclass(slots=True)
class MacroConfig:
    zscore_window: int = 252
    enabled_series: dict[str, bool] = field(
        default_factory=lambda: {
            "CFNAIDIFF": True,
            "EMVMACROBUS": True,
            "VIXCLS": True,
            "VXVCLS": True,
            "GVZCLS": True,
            "OVXCLS": True,
            "DGS10": True,
            "T10Y3M": True,
            "FEDFUNDS": True,
            "STLFSI4": True,
            "NFCI": True,
            "UMCSENT": True,
            "CFNAI": True,
            "DTWEXBGS": True,
        }
    )


@dataclass(slots=True)
class AppConfig:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    macro: MacroConfig = field(default_factory=MacroConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AppConfig":
        return cls(
            universe=UniverseConfig(**raw.get("universe", {})),
            indicators=IndicatorConfig(**raw.get("indicators", {})),
            strategy=StrategyConfig(**raw.get("strategy", {})),
            backtest=BacktestConfig(**raw.get("backtest", {})),
            walk_forward=WalkForwardConfig(**raw.get("walk_forward", {})),
            research=ResearchConfig(**raw.get("research", {})),
            macro=MacroConfig(**raw.get("macro", {})),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "AppConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha1(payload.encode("utf-8")).hexdigest()[:12]

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def ensure_path(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
