# SPY / S&P 500 Rotation — Champion Strategy Harness

> **UPDATE (2026-06-30): Final SPY champion deployed (iter83, max_holdings=2).**
> The autoresearch concluded after 85 runs (30 keep / 55 discard). The live
> champion is now the **S&P 500 (SPY) top-500 momentum-rotation** strategy tuned
> to maximize `worst_year_sharpe` (the minimum annual Sharpe across 2015-2025).
> The old IWV 1-stock deployment is **superseded** — `run_champion.py` now uses
> `config/research.json` + `research_params.json` (the iter83 champion).
>
> **Result:** `worst_year_sharpe = 0.596` (baseline 0.034, +1656%), **all 11 years
> positive**, sharpe 1.45, cagr 0.80, robust 0.950. Verified not-overfit (OOS
> walk-forward 0 negative folds, min +0.730; perturbation guard +0.020 < 0.10).
> Reproduced bit-for-bit via `python run_champion.py backtest` (0.5964).
>
> **What changed this session:**
> 1. `run_champion.py` rewired to the iter83 champion — `backtest` delegates to
>    `overfit_harness.run_benchmark` (bit-for-bit); `live` reuses the harness
>    `build_veto_mask` + `_size_weights` helpers (P20 quantile veto + conviction
>    tilt + vol-adjusted sizing, vol_pow=4).
> 2. **S&P 500 holdings now auto-update**: `src/algo_trading/data.py` gained
>    `fetch_sp500_holdings` (scrapes Wikipedia's *List of S&P 500 companies* via
>    requests + BeautifulSoup, cached 7 days). `load_market_context` uses it when
>    `etf_ticker == "SPY"`, so each run sees the latest constituents as the index
>    rebalances (no more stale manual snapshot).
> 3. **TimesFM coverage auto-updates with holdings**: `generate_spy_forecasts.py`
>    only forecasts missing `(symbol, date)` pairs (idempotent, resumable). The
>    scheduled task (`run_champion_live.ps1`) now runs it before each monthly live
>    screen, so forecast coverage tracks the auto-updating S&P 500 holdings — no
>    manual forecast regeneration needed. It exits early (no GPU model load) when
>    the cache already covers the universe; the live runner still warns if a
>    session lacks coverage.
> 4. **No data leakage re-confirmed**: `audit_pit_primary.py` → 0 look-ahead
>    (prior_end < d for all 132 dates), 0 missing forecasts in 2015-2025, 0
>    under-vetoed selections. Wikipedia fetch = current constituents (survivorship
>    caveat, NOT a leak — leaks are about signal/forecast timing, which is PIT-safe).
>
> **Honest caveat (unchanged):** the universe is current S&P 500 survivors (~503
> + IWV-inherited extras = 581), no delisted names. Backtest returns are
> **optimistic vs a tradable PIT universe** — survivorship bias is NOT eliminated.
> The 1-ticker swap from the Wikipedia refresh (CAG→HONA) did not shift the floor.

## How New Data Is Handled (No Human Intervention Needed)

**Automatic data refresh:** The model pulls fresh price, macro, holdings, and fundamental data via yfinance and FRED whenever local caches expire:
- **Price data**: cached 3 days (`price_cache_days: 3`)
- **FRED macro data**: cached 7 days (`fred_cache_days: 7`)
- **S&P 500 holdings**: cached 7 days (`holdings_cache_days: 7`) — fetched from Wikipedia's *List of S&P 500 companies* (`fetch_sp500_holdings` in `src/algo_trading/data.py`) when `etf_ticker == "SPY"`, so the universe auto-updates as the index rebalances
- **Fundamentals**: cached 30 days (`fundamentals_cache_days: 30`)
- **TimesFM volume forecasts**: pre-computed cache (`cache/ctx256_hor21_volume_q.parquet`) — **auto-refreshed by the scheduled task** on each monthly rebalance day (`generate_spy_forecasts.py --quantiles` runs before the live screen, forecasts any new symbols, exits early if the cache covers the universe). Also runnable manually: `python generate_spy_forecasts.py --quantiles` (idempotent; only forecasts missing `(symbol, date)` pairs).

The scheduled task runs automatically Mon–Fri at 9 PM SGT (before US market open). No human intervention is required for routine data fetching or strategy updates.

**Rolling window approach (not expanding):** The walk-forward training window is a **fixed 5-year rolling window** (`train_years: 5`). At each step, the model trains on the most recent 5 years of rebalance sessions, then tests on the next 12 months. The window advances by 12 months per step (`step_months: 12`). This means old data beyond 5 years is dropped — the model does not accumulate an ever-growing training set.

**Walk-forward mechanics:**
1. Training window: fixed 5 years of rebalance sessions immediately preceding the test period
2. Test period: 12 months following training
3. Step: 12-month advance per split
4. Indicator weights are re-fitted from scratch each split using Spearman IC on the training window
5. Leakage guard: `_strip_leaky_train_dates` drops any training row whose forward-return endpoint falls into the test period

**How new data enters the system as time passes:**
- Each live run calls `load_market_context()`, which re-downloads stale data
- `build_factor_state()` re-computes all factors on the full price panel
- `generate_walk_forward_plan()` or `score_composite()` selects the appropriate pre-computed walk-forward split for the current date
- For live screening (`screen_live_session`), a fresh training window is constructed on-the-fly: `session - 5 years` of rebalance dates → fit IC weights → score universe

This project builds a weekly rotation strategy from the holdings of iShares Russell 3000 ETF IWV (live champion) and — for the active research target — the S&P 500 (SPY) universe, ranks the universe with technical and fundamental factors, and evaluates parameter combinations with walk-forward validation.

## Champion Strategy (Current Best — IWV Live Deployment)

> The IWV champion is the **live trading** strategy. The **active research target**
> is SPY/S&P 500 (see [SPY Research Target](#spy-research-target-full-veto)).

The champion strategy was discovered through 262 autoresearch iterations. It uses a concentrated 1-stock portfolio with TimesFM volume veto and a skip-when-unchanged turnover optimization.

**Backtest result (2015-01-02 to present, updated 2026-06-28 after the macro publication-lag fix):**

| Metric | Value |
|--------|-------|
| CAGR | 117.94% |
| Sharpe | 1.516 |
| Max Drawdown | -67.05% |
| Calmar | 1.759 |
| Sortino | 2.197 |
| Veto Rate | 17.4% |
| Win Rate (Monthly) | 67.2% |
| Information Ratio | 1.498 |

_Prior values (CAGR 124.48%, MaxDD -60.63%) used unreleased monthly/weekly FRED values in the macro risk gate. The point-in-time fix reduced CAGR and deepened the drawdown, confirming the macro gate was previously look-ahead-flattered. These numbers remain survivorship-biased because `data/holdings_history/` has no historical snapshots (see [Code Review](#code-review-2026-06-28) → Leakage Caveat 2)._

**Year-by-year breakdown (post macro-PIT fix, 2026-06-28):**

| Year | Start | End | CAGR | Sharpe | MaxDD | Calmar | Sortino | WinRate | IR | EndEq |
|------|-------|-----|------|--------|-------|--------|---------|---------|----|-------|
| 2015 | 2015-01-02 | 2015-12-31 | 16.1% | 0.464 | -21.8% | 0.739 | 0.682 | 0.636 | 0.623 | 115,094 |
| 2016 | 2016-01-04 | 2016-12-30 | 173.3% | 2.294 | -29.7% | 5.846 | 3.357 | 1.000 | 2.256 | 304,377 |
| 2017 | 2017-01-03 | 2017-12-29 | 107.6% | 1.707 | -19.0% | 5.658 | 2.723 | 0.727 | 1.443 | 611,703 |
| 2018 | 2018-01-02 | 2018-12-31 | -46.9% | -1.083 | -55.2% | -0.850 | -1.187 | 0.455 | -1.015 | 342,438 |
| 2019 | 2019-01-02 | 2019-12-31 | 29.7% | 0.815 | -22.0% | 1.346 | 1.193 | 0.545 | 0.112 | 433,670 |
| 2020 | 2020-01-02 | 2020-12-31 | 239.5% | 1.803 | -60.6% | 3.950 | 2.393 | 0.636 | 1.846 | 1,569,507 |
| 2021 | 2021-01-04 | 2021-12-31 | 106.0% | 1.544 | -25.0% | 4.247 | 2.781 | 0.636 | 1.231 | 3,315,363 |
| 2022 | 2022-01-03 | 2022-12-30 | 3.5% | 0.226 | -34.0% | 0.102 | 0.350 | 0.455 | 0.804 | 3,574,445 |
| 2023 | 2023-01-03 | 2023-12-29 | 199.6% | 2.203 | -18.8% | 10.631 | 5.535 | 0.636 | 1.930 | 10,046,270 |
| 2024 | 2024-01-02 | 2024-12-31 | 406.1% | 2.408 | -30.7% | 13.247 | 4.140 | 0.636 | 2.321 | 48,865,685 |
| 2025 | 2025-01-02 | 2025-12-31 | 357.7% | 2.115 | -55.5% | 6.442 | 2.976 | 0.909 | 2.143 | 218,668,844 |
| 2026 | 2026-01-02 | 2026-06-26 | 992.5% | 3.425 | -21.4% | 46.352 | 5.561 | 0.800 | 3.507 | 759,860,020 |

### Champion Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_holdings` | 1 | Single-stock concentration |
| `sma_fast/slow/trend` | 3/50/201 | Moving average parameters |
| `ts_mom_lookback/skip` | 9/0 | Time-series momentum (months) |
| `fundamental_group_weight` | 0.03 | Fundamental factor group weight |
| `asset_growth_weight` | 20.0 | Asset growth factor weight |
| `earnings_yield_weight` | 3.0 | Earnings yield factor weight |
| `macro_half_risk/flat` | -0.5/-1.0 | Macro risk-off thresholds |
| `min_avg_dollar_volume` | $11M | Liquidity floor |
| `min_price` | $0.50 | Price floor |
| `veto_threshold` | 0.775 | TFM predicted vol / trailing vol ratio |
| `trailing_vol_window` | 53 | Sessions for trailing volume average |
| `veto_horizon` | 21 | TFM forecast horizon (sessions) |
| `cost_per_side` | 3 bps | Transaction cost per side |

### Plain-English Summary

The champion script answers this question:

> Based on the strategy rules, what stocks should I hold now, and do I need to trade today?

It does **not** place trades for you. It only creates files that tell you what the strategy would do.

**The most important rule**: run the live screen weekly for monitoring, but only trade when the output says `Rebalance due: yes`. If it says `no`, do not trade from that run.

### What The Champion Script Does

When you run `run_champion.py live`, it:

1. Loads the champion strategy settings from `config/champion.json`.
2. Downloads or reads the current IWV stock universe from yfinance.
3. Loads the TimesFM volume forecast for the veto.
4. Ranks the universe using technical factors (SMA crossover, trend filter, time-series momentum) and fundamental factors (asset growth, earnings yield) under grouped-equal weighting.
5. Applies the TFM volume veto: any stock whose predicted volume / trailing average volume is below 0.775 is removed from consideration.
6. Selects the top-1 remaining stock by composite score.
7. Checks whether this session is a scheduled rebalance session.
8. Compares the target stock against your current holdings (if `positions.csv` is provided).
9. Writes a screen CSV and prints a summary to the console.

### Code Changes

1. **Skip-when-unchanged optimization** (`src/algo_trading/timesfm_experiments.py`): When the target stock matches the current holding, the backtest skips the sell+rebuy cycle. This eliminates unnecessary turnover costs. The optimization is in `run_weighted_backtest` and applies to both backtest and live trade modes.

2. **Champion config** (`config/champion.json`): Self-contained config that bakes in all winning parameters. The `champion_veto` section holds the TFM veto parameters used by `run_champion.py`.

3. **Champion runner** (`run_champion.py`): Unified script for backtest verification and live screening with TFM volume veto.

4. **Champion live launcher** (`scripts/run_champion_live.ps1`): PowerShell wrapper with NYSE calendar guard for scheduled task execution.

### Data Leakage Audit

The **price / technical / fundamental** factor pipeline is leak-free: `shift_for_open_execution` shifts all factors by 1 session; `history_ok`, `price_ok`, `liquidity_ok` use `.shift(1)`; the TFM volume veto uses `prior_end = sessions[pos - 1]`; the TFM forecast context ends at the session before rebalance; `_strip_leaky_train_dates` removes walk-forward training rows whose forward return reaches into the test period.

Two **material** leakage issues remain — (1) macro publication-lag look-ahead and (2) full-universe survivorship bias from the empty `data/holdings_history/` directory. These are documented with evidence and fixes in the [Code Review (2026-06-28)](#code-review-2026-06-28) section ("Leakage Caveats"). Read that section before trusting backtest numbers.

## SPY Research Target (Full Veto) — Active Autoresearch

The active autoresearch target was **switched from IWV to SPY/S&P 500 on 2026-06-29**
per user direction, to test whether the IWV-tuned champion generalizes and to
optimize fresh on the large-cap universe. The TimesFM volume veto now has
**100% forecast coverage** (all 581 universe symbols).

### Setup changes
- `config/research.json`: `etf_ticker` and `benchmark_ticker` set to `SPY`;
  `universe_limit=500` (top-500 S&P 500).
- **S&P 500 constituents**: fetched from Wikipedia (iShares IVV bot-protection
  blocks programmatic CSV). Same **survivorship caveat** as IWV (current
  survivors, no delisted names) — returns are OPTIMISTIC vs a tradable PIT
  universe. Not eliminated.
- **Full TimesFM veto**: the 380 missing SPY symbols' volume forecasts were
  GPU-regenerated via `generate_spy_forecasts.py` (torch+transformers installed;
  float32 inputs to avoid float16 volume overflow; 49,373 forecasts, 0 NaN,
  100% coverage). The original 200 IWV symbols' CPU forecasts are preserved.
- IWV holdings backup: `cache/iwv_holdings.csv.iwv_backup`.

### SPY baseline (full-veto champion iter21 params on S&P 500)

| Metric | Value |
|--------|-------|
| worst_year_sharpe (PRIMARY) | +0.034 (2022 floor, barely positive) |
| Sharpe | 1.11 |
| CAGR | 0.47 |
| Max Drawdown | -0.38 |
| Robust Score | 0.69 |
| Veto Rate | 20.7% (vs IWV 8.5% — full veto is 2.5× more active) |
| perturb_worst_std | 0.19 |
| perturb_mean_worst_year | -0.31 (perturbed floor NEGATIVE) |

Year Sharpes: 2015=0.68 2016=1.06 2017=0.89 **2018=0.13** 2019=1.51 2020=0.97
2021=0.94 **2022=0.03** 2023=1.33 2024=2.24 2025=1.01. Weak years: **2022 (0.03)**
and **2018 (0.13)**. The IWV-tuned champion does NOT transfer cleanly to full-veto
SPY (floor 0.31 → 0.03) — there is real optimization headroom.

### Key SPY-specific finding
The full veto **over-vetoes** (20.7%) good 2022 names, making the +0.03 floor a
**fragile knife-edge**: every veto-param nudge (threshold ±0.02, trailing window 34)
and max_holdings 3 crashed 2022 negative. The veto_threshold=0.80 is a sharp local
peak. The principled fix is the **quantile P20 veto** (Phase 3 below) — veto on
downside-liquidity, not mean → fewer, smarter vetoes.

### New TSFM research directions (from `New directions.txt`)
A 5-phase TSFM-in-finance upgrade roadmap (see `.auto/ideas.md` for full detail):
1. **Phase 3 — Quantile P20 veto** (low effort/risk, moderate impact): veto on the
   P20 (downside) volume forecast instead of the mean. Directly targets the
   over-vetoing root cause. ★Try first.
2. **Phase 4 — Volatility gate** (medium effort, high impact): a second risk gate
   on forecasted volatility z-score.
3. **Phase 1 — Kronos backbone** (low effort/risk, high impact): finance-native
   TSFM drop-in replacement for TimesFM.
4. **Phase 2 — TimesFM-ICF in-context** (very low effort, moderate impact).
5. **Phase 5 — Prune-then-finetune adapter** (high effort, very high impact).

Critical caveats (Chronos study): TSFM signals must stay COARSE gates (as now), not
high-turnover alpha — even Sharpe>3 gross collapses negative net after 3bps. The
3bps cost + skip-when-unchanged already mitigate this.

## Key Files (Do Not Remove)

These files are the backbone of the champion strategy. Removing any of them breaks live trading, the scheduled task, or reproducibility.

### Immutable — Strategy Logic

| File | Role | Critical Because |
|------|------|------------------|
| `src/algo_trading/timesfm_experiments.py` | Backtest engine with skip-when-unchanged | Line 294: skip-rebalance check avoids unnecessary turnover. Also houses `run_weighted_backtest`, `build_factor_state`, `score_composite`, `build_context_requests`, and the TFM context builder. **This file is shared by `autoresearch.py` and `run_champion.py`.** |
| `src/algo_trading/strategy.py` | Factor library + walk-forward weights | `build_factor_library`, `fit_indicator_weights`, `_group_weighted_weights`, `score_universe_for_date`, `screen_live_session`. All factor computation and live-screen logic. |
| `src/algo_trading/indicators.py` | Technical indicator computiation + `shift_for_open_execution` | Line 138: `.shift(1)` enforces no-look-ahead. All SMA/EMA/TS-momentum/RIS/CCI/MACD/OBV/Bollinger/price-action indicators. |
| `src/algo_trading/config.py` | Dataclasses defining every tunable parameter | `AppConfig`, `StrategyConfig`, `IndicatorConfig`, `WalkForwardConfig`, `UniverseConfig`, `MacroConfig`, `BacktestConfig`. If slots or field names change, all config files must be updated. |
| `src/algo_trading/data.py` | Data loading + universe construction | `load_market_context` (builds the full price/factor/macro context), `fetch_sp500_holdings` (Wikipedia S&P 500 scrape, cached 7 days — auto-updates the SPY universe), `fetch_iwv_holdings` (legacy IWV). The universe auto-refreshes each run. |
| `src/algo_trading/calendar_utils.py` | Rebalance calendar (`select_rebalance_sessions`, `is_rebalance_session`, `next_rebalance_session`) | Determines when the live screen emits trade orders. Supports "weekly", "biweekly", "monthly". |

### Immutable — Configs

| File | Role | Critical Because |
|------|------|------------------|
| `config/research.json` | **The live champion config** — SPY universe (limit 500), benchmark SPY, monthly rebalance | `run_champion.py` reads this (via `overfit_harness.apply_params_to_config`). The iter83 champion parameters are overlaid from `research_params.json`. |
| `research_params.json` | **The iter83 champion parameters** — max_holdings=2, ts_mom=10/0, sma=3/50/210, P20 quantile veto (threshold 0.35, ctx=256, horizon=21), vol_scaled_weights (lookback 21), conviction tilt (weight_tilt=0.17, conf_tilt, conf_tilt_pow=50, vol-adjusted, vol_pow=4), fund_weight=0.02, macro -0.5/-1.0, min_volume 11M, cost 3bps | Applied on top of `config/research.json` by `overfit_harness.apply_params_to_config`. This is the single source of truth for the champion tuning. |
| `config/champion.json` | **Legacy IWV champion config** (superseded 2026-06-30) | The old 1-stock IWV deployment. Kept as a historical artifact — `run_champion.py` no longer reads it. Do not delete (baseline reference). |
| `config/baseline.json` | The pre-champion baseline config | Used by `autoresearch.py` during research loops. |

### Immutable — Entry Points

| File | Role | Critical Because |
|------|------|------------------|
| `run_champion.py` | **Champion strategy runner** (backtest + live) | `backtest` mode delegates to `overfit_harness.run_benchmark` (perturb off) — byte-for-bit reproduction of the iter83 champion (worst_year_sharpe=0.596). `live` mode reuses the harness `build_veto_mask` + `_size_weights` helpers to apply the P20 quantile veto + conviction-tilt + vol-adjusted sizing for the current session. Reads `config/research.json` + `research_params.json`. |
| `overfit_harness.py` | **The champion backtest engine** | `apply_params_to_config`, `build_veto_mask` (P20 quantile veto), `_size_weights` (conviction tilt + vol-adjusted + vol_pow), `run_benchmark`. Shared by `run_champion.py` (backtest + live) and the autoresearch loop. The iter83 champion logic lives here. |
| `generate_spy_forecasts.py` | **TimesFM forecast generator** (GPU) | Generates/refreshes the P20 quantile forecast cache for the SPY universe. Only forecasts missing `(symbol, date)` pairs (idempotent, resumable, checkpoints every 8000). Run after a holdings refresh to cover new symbols: `python generate_spy_forecasts.py --quantiles`. |
| `scripts/run_champion_live.ps1` | **Scheduled-task launcher** | Executed by Windows Task Scheduler Mon–Fri at 9 PM SGT. Contains the NYSE calendar guard (first-trading-day-of-month check). Resolves Python interpreter. Writes run logs to `artifacts/live/champion_run.log`. |
| `scripts/is_first_nyse_rebalance_session.py` | NYSE calendar guard | Checks if a date is the first NYSE trading day of the month (matches `rebalance_frequency: "monthly"`). Used by `run_champion_live.ps1` to skip non-rebalance days. |
| `autoresearch.py` | Research loop entrypoint | The engine that discovered the champion. **Must be preserved so future research can re-run from a known state.** |

### Immutable — TFM Forecast Stack

| File | Role | Critical Because |
|------|------|------------------|
| `src/algo_trading/timesfm_engine.py` | TimesFM model loading and inference | `load_forecast_panel`, `mean_forecast_sum`. The volume veto and any TFM alpha blend depend on this. |
| `experiments/forecast_worker.py` | Batch forecast worker | Generates the pre-computed TFM forecast caches that `load_forecast_panel` reads. Without cached forecasts, the veto cannot run. |
| `experiments/run_forecast_driver.sh` | Forecast cache generation driver | Queue → worker pipeline that produces the volume/logret forecast panels. **If cache is stale or missing, run this to regenerate.** |

### Removable / Replaceable

These files are safe to modify, move, or delete — they do not affect the champion strategy:

- `experiments/run_walkforward.py` — archival alpha-blend experiment (Exp 1), not champion
- `_smoke_*.py` (in `tests/`) — smoke tests (diagnostic, not required for trading)
- `tests/` — test suite (diagnostic, not required for trading)
- `run_live.ps1` — dead code reference (file does not exist; the live launcher is `scripts/run_champion_live.ps1`)
- `result.tsv`, `research finding.txt` — research logs (already removed)
- Old `Weekly Script` directory variants on Desktop — replaced by this repo's current location

**Rule of thumb**: everything under `src/algo_trading/`, `config/`, `experiments/forecast_worker.py`, `experiments/run_forecast_driver.sh`, `run_champion.py`, `scripts/run_champion_live.ps1`, and `scripts/is_first_nyse_rebalance_session.py` is load-bearing. Everything else can be recreated from scratch if needed.

## Setup

1. Install Python 3.11+.
2. Create and activate a virtual environment.
3. Install dependencies with `pip install -e .`.
4. Either place your API keys in `.env` or `.github/.env`, or pass an explicit env file path with `--env-file`.

Example on PowerShell:

```powershell
$env:FRED_API_KEY = "your-key"
pip install -e .
```

## Commands

### Champion Strategy (Recommended)

Run the champion backtest to verify reproducibility:

```powershell
python run_champion.py backtest
```

Run the champion live screen for today:

```powershell
python run_champion.py live
```

Run the champion live screen for a specific date:

```powershell
python run_champion.py live --as-of 2026-06-23
```

Run the champion live screen with current positions:

```powershell
python run_champion.py live --current-positions positions.csv
```

Force-refresh cached data (prices, FRED, fundamentals):

```powershell
python run_champion.py live --refresh-data
```

The live screen outputs:
- `artifacts/live/champion_screen_YYYYMMDD.csv` — ranked screen with TFM veto applied
- Console output with rebalance status, vetoed stocks, and BUY/HOLD actions
- A `RESULT` line with key fields for log parsing (see below)

#### Current Holdings File (`positions.csv`)

Use this file if you already own stocks and want the script to compare them against the target:

```csv
symbol,shares
WDC,50
LITE,35
```

- `symbol` is the stock ticker (uppercase).
- `shares` is how many shares you currently own.
- If you own the same ticker on multiple rows, the script combines them.
- If you do not currently own anything, skip `--current-positions`.
- A header-only file (`symbol,shares` with no data rows) also works.

#### What To Look For In The Output

After running `run_champion.py live`, look at the terminal:

```text
Champion Screen (iter83, max_holdings=2) — 2025-12-01
Rebalance due: yes
Next rebalance: 2025-12-01
Veto coverage: ON (P20 quantile)
Vetoed stocks: 28 (ATO, BSX, ...)

 rank symbol            name                 sector weight composite_score action
    1    WDC Western Digital Information Technology 0.6814        0.370790    BUY
    2   ECHO        EchoStar Communication Services 0.3186        0.353016    BUY
RESULT session=2025-12-01 rebalance=yes action=BUY symbol=WDC vetoed=28 veto_cov=on runtime=114.7s
```

- **`Rebalance due: yes`** — review the screen and consider placing trades.
- **`Rebalance due: no`** — monitoring only, do not trade from this run.
- **`Veto coverage: ON (P20 quantile)`** — the TimesFM downside-quantile veto is active. If it says `OFF — run generate_spy_forecasts.py --quantiles`, the live session has no forecast coverage; re-run the forecast generator first, then re-screen.
- **`Vetoed stocks`** — how many stocks the P20 volume veto removed (~3% of candidates).
- **`weight` column** — the conviction-tilt + vol-adjusted position weights (not equal-weight; the top-1 gets more weight when high-confidence).
- **`action` column** — `BUY` (open this position), `HOLD` (keep this position).
- **`RESULT` line** — grep-friendly single-line summary logged to `champion_run.log`. Key fields: `session`, `rebalance` (yes/no), `action`, `symbol` (top-1), `vetoed` count, `veto_cov` (on/off), `runtime`.
- **Two ranks** are shown because `max_holdings=2` — the strategy holds 2 stocks with conviction-tilted weights.

### Scheduled Task (Live Trading)

The champion strategy runs automatically via Windows Task Scheduler:

- **Task name**: `Fundamental Momentum Champion`
- **Schedule**: Weekly, Monday–Friday at 9:00 PM SGT (before US market open at 9:30 PM SGT)
- **Calendar guard**: Only runs on the first NYSE trading day of the month (skips non-rebalance days and holidays)
- **Script**: `scripts/run_champion_live.ps1`

#### Execution Flow

1. Task Scheduler triggers `run_champion_live.ps1` Mon–Fri at 9 PM SGT
2. PowerShell resolves Python, creates `champion_run.log` if missing
3. **Calendar guard** (`is_first_nyse_rebalance_session.py`) checks if today is the first NYSE trading day of the month
   - If **NO** → logs `Not the first NYSE trading day of the month. No action needed.` and exits (exit 0)
   - If **YES** → continues to the forecast refresh
4. **TimesFM forecast auto-refresh** (`generate_spy_forecasts.py --quantiles`) — runs on the rebalance day so coverage tracks the auto-updating S&P 500 holdings:
   - Loads the (Wikipedia-refreshed) holdings and the forecast cache
   - Finds universe symbols missing from the cache and forecasts only those `(symbol, date)` pairs on GPU
   - **Exits early without loading the model if the cache already covers the universe** (the common no-op case, ~60s)
   - Non-fatal on failure (no GPU / errors): logs and continues — the live screen runs with `veto_cov=off` + a warning
5. **Champion screen** (`run_champion.py live`) runs the full analysis:
   - Checks if today is a **monthly** rebalance session (first trading day of the month)
   - Prints `Rebalance due: yes` or `Rebalance due: no` and `Veto coverage: ON/OFF`
   - Outputs a `RESULT` line with key fields
6. All stdout and stderr is captured and timestamped into `champion_run.log`

#### Why This Means

- **Signal computation** uses weekly price resampling (`signal_timeframe: "weekly"`) for factor calculation
- **Trading decisions** happen **monthly** (`rebalance_frequency: "monthly"`) — only on the first trading day of each month
- **The calendar guard** blocks execution on all days except the first NYSE trading day of the month (monthly), so the screen runs once a month on the rebalance day
- The `Rebalance due: yes/no` line tells you which case you're in (should always say `yes` on a monthly rebalance day unless macro is risk-off or no stocks pass)

To recreate the scheduled task (run in an elevated PowerShell — this removes any stale task from the old repo location and registers the champion at the current path):

```powershell
# 1. Remove the OLD task (it points to the previous "Weekly Script" location which no longer exists)
Unregister-ScheduledTask -TaskName 'Fundamental Momentum Champion' -Confirm:$false -ErrorAction SilentlyContinue

# 2. Register the NEW champion task at the current repo path
$repo = 'C:\Users\User\Desktop\Fundamental Momentum Test'
$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument ('-NoProfile -ExecutionPolicy Bypass -File "' + $repo + '\scripts\run_champion_live.ps1" -CurrentPositions positions.csv') `
    -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 9pm
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName 'Fundamental Momentum Champion' -Action $action -Trigger $trigger -Settings $settings -Force
```

To manually trigger the scheduled task:

```powershell
Start-ScheduledTask -TaskName 'Fundamental Momentum Champion'
```

**Verify the trigger days** (the live log once showed a Saturday execution — confirm Mon-Fri only):

```powershell
$task = Get-ScheduledTask -TaskName 'Fundamental Momentum Champion'
$task.Triggers | Format-List DaysOfWeek, Enabled
```

If Saturday or Sunday appear in `DaysOfWeek`, re-run step 1 (Unregister) then step 2 (Register) above to repair.

### Legacy CLI Commands

Run a walk-forward backtest with the baseline config:

```powershell
python -m algo_trading.cli --config config/baseline.json backtest
```

Run the live screen (baseline, no TFM veto):

```powershell
python -m algo_trading.cli --config config/baseline.json screen
```

Run the autoresearch experiment loop:

```powershell
python -m algo_trading.cli --config config/baseline.json research --research-space config/research_space.json --iterations 20
```

Archive IWV holdings snapshots:

```powershell
python -m algo_trading.cli --config config/baseline.json snapshot-holdings
```

## Outputs

### Champion Live Outputs (`artifacts/live/`)

These files are written by `run_champion.py live` or the scheduled task (`run_champion_live.ps1`):

- **`champion_screen_YYYYMMDD.csv`** — The ranked stock list for that run. Contains columns: `rank`, `symbol`, `name`, `sector`, `composite_score`, `action` (BUY/HOLD), and `vetoed_symbols`. Use it to see which stocks rank strongest after the TFM volume veto.

- **`champion_run.log`** — Append-only run log written by the scheduled task. Every execution is timestamped. Contains full champion screen output, calendar guard results, and a `RESULT` line per run. Check this first if the task ran but you didn't see the output.
  
  Grep for `RESULT` to get a compact summary of each run:
  ```text
  RESULT session=2026-06-22 rebalance=yes action=BUY symbol=NVDA vetoed=3 runtime=12.5s
  RESULT session=2026-06-15 rebalance=no action=HOLD symbol=NVDA vetoed=2 runtime=11.2s
  ```

### Baseline CLI Outputs

- **`artifacts/backtests/`** — Baseline backtest equity curves, trade logs, and metrics (from `python -m algo_trading.cli backtest`).
- **`artifacts/research/`** — Autoresearch experiment artifacts.

### Shared Outputs

- **`data/holdings_history/`** — Dated IWV holdings snapshots for point-in-time universe studies.
- **`result.tsv`** — One row per autoresearch experiment.
- **`research finding.txt`** — Autoresearch findings log.

## Long-Term Maintenance

### Weekly Operations

#### Detailed Weekly Checklist

1. Open the project folder in PowerShell:
   cd "C:\Users\User\Desktop\Fundamental Momentum Test"
2. Update `positions.csv` if your holdings changed (see format above).
3. The **scheduled task runs automatically** Monday–Friday at 9 PM SGT. To run manually:
   ```powershell
   python run_champion.py live --current-positions positions.csv
   ```
4. Check the output for **`Rebalance due: yes`** (terminal or `artifacts/live/champion_run.log`).
5. If it says **`no`**, stop. Do not trade from that run.
6. If it says **`yes`**, open `artifacts/live/champion_screen_YYYYMMDD.csv`.
7. Place **SELL** orders first (if any current holdings are being replaced).
8. Place **BUY** orders only after sells are complete. The `weight` column gives the target portfolio weight for each of the 2 holdings (conviction-tilted, not equal-weight).
9. After orders fill, update `positions.csv` for the next run.

You can also grep the log for the `RESULT` line to quickly check recent runs:

```powershell
Select-String -Path artifacts/live/champion_run.log -Pattern "^RESULT"
```

#### Real-Use Policy

1. Run weekly for monitoring even on non-rebalance weeks.
2. Trade only on scheduled rebalance runs (`Rebalance due: yes`).
3. Start with paper trading or very small size until you are comfortable with the strategy behavior.
4. Keep `config/research.json` and `research_params.json` unchanged unless you intentionally review and approve a strategy change after new research. (The legacy `config/champion.json` is no longer used by the live runner.)
5. Keep your own broker confirmations outside this project if you need compliance records.
6. The script does not connect to your broker — you must place trades yourself.

### Data Refresh

- **Price data**: Cached for 3 days (`backtest.price_cache_days`). Force refresh with `--refresh-data` or by deleting `cache/prices/`.
- **FRED macro data**: Cached for 7 days (`backtest.fred_cache_days`). Requires `FRED_API_KEY` for first download.
- **Fundamentals**: Cached for 30 days (`universe.fundamentals_cache_days`).
- **S&P 500 holdings**: Cached for 7 days (`universe.holdings_cache_days`) in `cache/sp500_holdings.csv`. Auto-refreshed from Wikipedia by `fetch_sp500_holdings` each run when stale. Force refresh by deleting the cache file or running with `--refresh-data`.
- **TimesFM volume forecasts**: Pre-computed in `cache/ctx256_hor21_volume_q.parquet`. **Auto-refreshed by the scheduled task** on each monthly rebalance day (`run_champion_live.ps1` runs `generate_spy_forecasts.py --quantiles` before the live screen, so coverage tracks the auto-updating S&P 500 holdings). It only forecasts missing `(symbol, date)` pairs on GPU (idempotent, resumable; exits early with no model load if the cache covers the universe). For manual runs, invoke it directly: `python generate_spy_forecasts.py --quantiles`. Without coverage, new symbols are un-vetoed (the live runner warns `Veto coverage: OFF`).

#### FRED Setup Failure

If the live screen prints:

```text
Live screen setup is incomplete
```

it means the script needs FRED macro data but cannot find cached data or a FRED API key.

**Fix:**
1. Create `.env` in the project folder.
2. Add: `FRED_API_KEY=your_key_here`
3. Run the same command again.

The script stops here because the macro composite is used as a risk-off gate in the strategy. Skipping it would change the strategy behavior. The macro data is cached after the first successful download, so this is a one-time setup step per machine.

### Performance Monitoring

- **Re-run the champion backtest** monthly to verify the strategy still reproduces `worst_year_sharpe ≈ 0.596` (all 11 years 2015-2025 positive; survivorship-biased):
  ```powershell
  python run_champion.py backtest
  ```
- **Compare per-year Sharpes** — the floor is 2015 (0.596). If any year drops negative, investigate regime change or forecast coverage gaps.
- **Monitor veto rate** — should stay around 3% (P20 downside-quantile gate). A sudden change suggests TimesFM forecast degradation or a universe drift; re-run `generate_spy_forecasts.py --quantiles`.
- **Monitor forecast coverage** — the live `RESULT` line reports `veto_cov=on/off`. If `off`, the session has no forecast coverage; re-run the forecast generator.

### Config Updates

- **Champion config**: `config/research.json` — the SPY universe/backtest settings.
- **Champion parameters**: `research_params.json` — the iter83 champion tuning (max_holdings=2, P20 veto, conviction tilt, vol-adjusted, vol_pow=4). This is the single source of truth for the champion; `run_champion.py` reads it via `overfit_harness.apply_params_to_config`.
- **Legacy IWV config**: `config/champion.json` — the old 1-stock IWV deployment (superseded; kept as a historical artifact).
- **Autoresearch params**: `autoresearch_params.json` — used by `autoresearch.py` for research iterations.
- **To run new research**: Edit `research_params.json` (or `autoresearch_params.json`), run `python overfit_harness.py --config config/research.json --params research_params.json` (or the autoresearch loop), log results, and if a new champion is found, update `research_params.json`.

### Troubleshooting

- **TFM forecast cache missing / `Veto coverage: OFF`**: The champion backtest and live veto require the pre-computed TimesFM P20 quantile forecast cache (`cache/ctx256_hor21_volume_q.parquet`). If `load_forecast_panel` returns None or the live screen reports `veto_cov=off`, regenerate it: `python generate_spy_forecasts.py --quantiles` (GPU, ~15 min; idempotent — only forecasts missing `(symbol, date)` pairs). Run this after any S&P 500 holdings refresh to cover new symbols.
- **FRED API key missing**: The live screen needs cached FRED data. Set `FRED_API_KEY` in `.env` or run the screen once with `--refresh-data`.
- **`fetch_sp500_holdings` fails (Wikipedia unreachable)**: The function falls back to the cached `cache/sp500_holdings.csv` if the live fetch fails. If no cache exists, ensure network access or temporarily restore a prior `cache/sp500_holdings.csv`.
- **Live screen slow (~100-115s)**: The live mode builds the champion setup (median_gap, vol_factor) over all 2015-present rebalance dates to match the backtest sizing exactly. This is expected for a monthly run.
- **Python not found**: The PowerShell launcher checks `.venv\Scripts\python.exe`, then `ALGO_TRADING_PYTHON` env var, then system Python 3.12, then `python` on PATH.
- **Scheduled task not running**: Check Task Scheduler → `Fundamental Momentum Champion` → History tab for errors.

## Strategy Coverage

Technical factors in the baseline engine:

- Simple and exponential moving-average crossovers
- Long-trend filter
- Trading-range breakout and variable-week-high momentum
- RSI, CCI, MACD histogram, OBV trend, Bollinger band position
- Time-series momentum (champion uses 9-month lookback, 0-month skip)
- Weekly price-action persistence

Fundamental factors in the baseline engine:

- Book-to-market, earnings yield, ROA, ROE
- Gross profitability-to-assets, asset growth, investment-to-assets
- Net issuance, accruals, cash-flow yield, earnings surprise

The champion uses `asset_growth` (weight=20.0) and `earnings_yield` (weight=3.0) with `fundamental_group_weight=0.03` under `grouped_equal` weighting.

Macro regime inputs from FRED: CFNAIDIFF, EMVMACROBUS, VIXCLS, VXVCLS, GVZCLS, OVXCLS, DGS10, T10Y3M, FEDFUNDS, STLFSI4, NFCI, UMCSENT, CFNAI, DTWEXBGS.

## Code Review (2026-06-28)

A systematic review of the data pipeline covering data leakage, model structure, backtest/live path consistency, and scheduled-task correctness. Full audit details below.

### Data Leakage Audit: ⚠️ CONDITIONAL PASS (2 material caveats)

Price/technical/fundamental factor paths are leak-free, but the macro path and the universe membership path have material issues (detailed below the table):

| Checkpoint | Location | Guard | Verdict |
|------------|----------|-------|---------|
| Factor shift | `indicators.py:138-139` | `shift_for_open_execution` applies `.shift(1)` to ALL technical + fundamental factors in `build_factor_library` | ✅ Factors from session T's close available at T+1's open |
| Base mask filters | `strategy.py:77-82` | `history_ok`, `price_ok`, `liquidity_ok` all use `.shift(1).fillna(False)` | ✅ Filters use prior-session data only |
| Macro score | `strategy.py` `build_factor_library` | `macro_composite.shift(1)` | ⚠️ Covers ~1 session only — insufficient for monthly/weekly FRED publication lag (see Caveat 1) |
| Membership | `strategy.py` `build_factor_library` / `data.py` `_build_universe_membership` | Point-in-time IWV snapshots with `allow_current_holdings_fallback` | ⚠️ Snapshot dir is empty → today's constituents used for ALL dates (see Caveat 2) |
| Walk-forward splits | `strategy.py:107-147` | Training window = most recent 5 years STRICTLY before test_start; test = 12 months after | ✅ No train/test overlap |
| Leaky train dates | `strategy.py:187-212` | `_strip_leaky_train_dates` drops training rows whose forward-return endpoint reaches into the test period | ✅ Called in `generate_walk_forward_plan`, `screen_live_session`, AND `build_factor_state` |
| TFM veto trailing volume | `timesfm_experiments.py` / `run_champion.py` | `prior_end = sessions[pos - 1]` — trailing average uses data up to the session BEFORE rebalance | ✅ No look-ahead into rebalance session data |
| TFM forecast context | `timesfm_experiments.py:232-254` | `build_context_requests` sets context window end at `prior_end` (session before rebalance) | ✅ Forecasts don't see the rebalance session |
| Forward returns | `strategy.py:100-103` | `compute_forward_open_returns` uses `open` prices and `.shift(-1)` — used only for IC fitting, never for scoring | ✅ Forward returns computed correctly for training |
| Fundamental factors | `data.py` | Statement data shifted by `statement_lag_days` (60 for champion); only close prices at or before `available_date` used | ✅ Point-in-time approximation with reporting lag |
| FRED macro data | `data.py` `load_macro_data` | FRED series `reindex(sessions, method="ffill")` by **observation date**, no publication lag; `.shift(1)` in `build_factor_library` | ⚠️ Daily series OK; monthly/weekly series used 3-6 weeks early (see Caveat 1) |

### Leakage Caveats (material — revise the prior PASS verdict)

**Caveat 1 — Macro publication-lag look-ahead (FIXED 2026-06-28).** `load_macro_data` previously reindexed each FRED series onto the session grid with `method="ffill"` keyed on the series **observation date**, with only a single `.shift(1)` downstream — adequate for daily series but not for monthly/weekly series that publish 3-6 weeks late. **Fix:** `data.py` now shifts each series forward by a per-frequency publication lag (`PUBLICATION_LAG_DAYS`: daily=0, weekly=10, monthly=45 calendar days) before reindex/ffill, so a value is only available on/after its release date; the existing `.shift(1)` in `build_factor_library` still covers open-execution timing. Verified by `test_load_macro_data_lags_monthly_series_by_publication_delay`. Daily series (VIX, yields, etc.) are unaffected; the monthly/weekly macro risk gate is now point-in-time. Residual: the 45-day lag is a conservative blanket for monthly series; per-series exact release calendars could tighten it further.

**Caveat 2 — Universe survivorship bias (partially mitigated 2026-06-28).** Both configs set `use_historical_snapshots: true` and `historical_holdings_dir: "data/holdings_history"`. Until today that directory was empty, so with `allow_current_holdings_fallback: true` the **entire** backtest used today's IWV constituents for every session. **Mitigation:** a snapshot of the current IWV holdings was archived to `data/holdings_history/2026-06-28.csv` via `cli snapshot-holdings`, so the **live** path is point-in-time from today forward. The **historical backtest** (2015→today) still falls back to today's constituents for all sessions before the first snapshot because no historical IWV snapshots are available to backfill; this overstates backtest returns relative to a tradable universe. **Fix (data, not code):** backfill `data/holdings_history/` with monthly IWV snapshots (e.g. via a paid historical-holdings dataset imported with `cli import-holdings-history`), or set `allow_current_holdings_fallback: false` to exclude unarchived periods (this would shrink the backtest to the snapshot era). Going forward, run `cli snapshot-holdings` on the first trading day of each month so the archive grows.

**Minor — annual 10-K lag.** `statement_lag_days=60` is conservative for quarterly 10-Qs (~40-45d) but slightly aggressive for annual 10-Ks (large accelerated filers 60d, others 90d). A fixed 60d lag for annual statements is a minor optimistic assumption; it applies identically to backtest and live so it is not a backtest/live inconsistency.

### Model Structure: ⚠️ PASS (with observations)

**Two parallel scoring functions exist — different call sites, same core logic:**

| Function | File | Used By | Notes |
|----------|------|---------|-------|
| `score_universe_for_date` | `strategy.py` | Legacy CLI, `generate_walk_forward_plan`, `screen_live_session` | Receives `weights`, `factor_names`, `base_mask`, `macro_score`, `config` as explicit parameters |
| `score_composite` | `timesfm_experiments.py` | `run_champion.py`, `autoresearch.py` | Gets weights from `state.weights_by_split[si]`; supports `override` and `extra_factors`; uses `_cfg_holder.config` (set via `set_config`) |

Both functions:
- Sum `ranked_factor * weight` for each enabled factor
- Apply `base_mask` eligibility filter
- Call `_target_holdings(macro_score, config)` for risk-off gating
- Filter by `require_positive_composite` and `head(N)`

⚠️ **Observation**: These are near-duplicate implementations that could diverge. The champion path uses `score_composite` exclusively, which is the more featureful version (TFM alpha blend, override support). The legacy CLI uses `score_universe_for_date`. Not a bug today, but refactoring to a single `score_universe` function shared by both would eliminate the risk of future divergence. **This is not blocking — both paths produce identical results for the champion config since no override or extra_factors are used in production.**

**Factor weight computation verified:**
- Champion uses `weighting_scheme: "grouped_equal"` with `fundamental_group_weight: 0.03`
- Technical factors get 97% of weight, fundamentals get 3% (split internally: `asset_growth`=20, `earnings_yield`=3, normalized to 20/23 and 3/23 respectively)
- `use_indicator_ic_weights: false` means IC scores are computed but not used for within-group weighting

**Enabled factors (champion config):**
- Technical: `sma_crossover`, `ema_crossover`, `trend_filter`, `ts_momentum`
- Fundamental: `earnings_yield`, `asset_growth`
- All other factors disabled (`breakout`, `rsi`, `cci`, `macd`, `obv`, `bollinger`, `variable_week_high`, `price_action`, `book_to_market`, `roa`, `roe`, `gross_profitability`, `investment_to_assets`, `net_issuance`, `accruals`, `cash_flow_yield`, `earnings_surprise`)

**Champion config parameter consistency:**
- `signal_timeframe: "weekly"` → `resample_price_panel` resamples to W-FRI; factors then forward-filled to daily sessions
- `rebalance_frequency: "monthly"` → `select_rebalance_sessions` returns first trading day of each month
- `sma_fast/slow/trend: 3/50/201` → SMA crossover + trend filter
- `ts_mom_lookback/skip: 9/0` → 9-month momentum with no skip
- `macro_half_risk/flat: -0.5/-1.0` → defensive gating thresholds
- `veto_threshold: 0.775` vs `trailing_vol_window: 53` → TFM volume veto ratio
- All parameters self-consistent and correctly wired through `AppConfig.from_file`

### Model Input / Output, Structure & Parameters

**TimesFM is a veto model, not the alpha model.** The champion's stock selection is the factor composite (technical + fundamental ranks). TimesFM-2.5-200m is used **only** for the volume veto: input = 256-session raw daily volume series ending at the session before rebalance (`build_context_requests`, `prior_end = sessions[pos-1]` — no look-ahead); output = 21-step mean forecast → `mean_forecast_sum / 21` = predicted average daily volume → `pred_vol / trailing_53_avg_vol < 0.775` vetoes the stock. `quantiles=False` for the champion (only the mean column is used). Patch length 32, context rounded to a multiple of 32 (256 OK), native horizon 128 sliced to 21.

**Champion factor weights are fixed across all walk-forward splits.** `indicator_weighting_scheme: "grouped_equal"` + `use_indicator_ic_weights: false` + explicit `fundamental_factor_weights {asset_growth: 20, earnings_yield: 3}` + equal-weighted technicals means `fit_indicator_weights` returns identical weights for every split. `build_walk_forward_splits` / `split_for_date` therefore map every rebalance date to the same weights — the walk-forward machinery is decorative for the champion (it still drives `min_train_rebalances` eligibility and the `_strip_leaky_train_dates` guard). Net weights: technical 0.97/4 ≈ 0.2425 each (`sma_crossover`, `ema_crossover`, `trend_filter`, `ts_momentum`); fundamental 0.03 split 20/23 ≈ 0.0261 (`asset_growth`) and 3/23 ≈ 0.0039 (`earnings_yield`).

**Half-risk band is dead at `max_holdings=1`.** `_target_holdings` returns `max(1, max_holdings // 2)` in the half-risk band; for `max_holdings=1` that is `max(1, 0) = 1` (full risk). So `macro_half_risk_threshold=-0.5` has no effect for the champion — only the flat gate (`macro_flat_threshold=-1.0` → 0 holdings) is active. Consistent between backtest and live (both call `_target_holdings` via `score_composite`), so not a consistency bug, just a degenerate edge of N=1.

**`score_composite` reads config from a module global.** `score_composite` uses `_cfg_holder.config` (set by `set_config(cfg)`) for `_target_holdings`, while `score_universe_for_date` takes `config` as an explicit argument. `run_champion.py` and `autoresearch.py` always call `set_config` first, so this is correct today, but fragile — calling `score_composite` without `set_config` silently uses the default `AppConfig` (max_holdings=5, thresholds -0.5/-1.0).

**`positions.csv` vs champion `max_holdings=1`.** `positions.csv` currently lists two holdings (LITE, WDC) while the champion targets a single position. The file is either stale from an earlier `max_holdings=2` config or manually maintained. `run_champion.py live --current-positions positions.csv` will mark only the top-1 as HOLD/BUY and the other as absent, over-trading relative to the file. Reconcile before the next live run.

**`champion_veto` is not modeled by `AppConfig`.** The `champion_veto` block in `config/champion.json` is read by `run_champion._load_veto_params` straight from the raw JSON; `AppConfig.from_dict` silently ignores it. So any CLI path that builds `AppConfig` (e.g. `cli backtest --config champion.json`) will **not** apply the veto — see the two-engine note below.

### Backtest vs Live Consistency: ⚠️ PASS within the champion path; two-engine divergence vs the CLI

**Within `run_champion.py` (and `autoresearch.py`) the path is consistent.** `run_backtest()` and `run_live()` share the same core:

```
load_context(CONFIG_PATH)  →  build_factor_state(ctx, cfg)  →  score_composite(state, session)
    →  TFM volume veto (same trailing_avg, same pred_vol/trailing_vol ratio check, same prior_end)
    →  top_n_equal_weights(scores, N=1)  →  run_weighted_backtest(ctx, wsel, cfg)  [backtest only]
```

`run_champion.py backtest` and `autoresearch.py` are equivalent: `autoresearch.py` loads `config/baseline.json` then applies `autoresearch_params.json`, whose values are identical to `config/champion.json` + its `champion_veto` block (max_holdings=1, ts_mom=9/0, sma=3/50/201, fund_weight=0.03, asset_growth=20, earnings_yield=3, macro=-0.5/-1.0, min_volume=11M, min_price=0.50, veto 0.775/53/21, cost_per_side=0.0003). Both use `run_weighted_backtest` with `start_date=2015-01-02` and `tc_rate=0.0003`. So the champion backtest is reproducible from `autoresearch.py`. ✅

**Skip-when-unchanged** (`run_weighted_backtest`): if the target symbol set equals the current holdings set, the rebalance is skipped (no sell+rebuy). With `max_holdings=1` this avoids turnover when the top stock is unchanged. ✅

⚠️ **Two divergent backtest engines + two scoring entry points.** The CLI harness and the champion harness are NOT the same strategy:

| Path | Scoring | Backtest engine | Veto | Sizing |
|------|---------|-----------------|------|--------|
| `run_champion.py backtest` / `autoresearch.py` | `score_composite` | `timesfm_experiments.run_weighted_backtest` | ✅ TFM volume veto | equal-weight of cash, full-turnover-with-skip |
| `run_champion.py live` | `score_composite` | (screen only) | ✅ TFM volume veto | top-1 → orders |
| `cli backtest` (`algo_trading.cli`) | `generate_walk_forward_plan` → `score_universe_for_date` | `backtest.run_backtest` | ❌ no veto | equal-dollar, keep-overlap |
| `cli screen` (`algo_trading.cli`) | `screen_live_session` → `score_universe_for_date` | (screen only) | ❌ no veto | top-N → rebalance plan |

`run_weighted_backtest` sells everything and re-buys target weights each rebalance (unless the target set is unchanged) and sizes by `cash * weight`. `backtest.run_backtest` keeps overlap positions at their existing cost basis and deploys cash to new names (`cash / len(new_symbols)`), sizing by equal-dollar. For `max_holdings=1` the two engines produce the same trades; for `max_holdings≥2` (e.g. `config/baseline.json`) they diverge meaningfully. **Practical consequence:** `python -m algo_trading.cli --config config/champion.json backtest` does **not** reproduce `python run_champion.py backtest` — the CLI run omits the veto and uses a different sizing/turnover engine. Use `run_champion.py backtest` (or `autoresearch.py`) to reproduce the champion. The CLI path is the baseline/research harness, not the champion harness.

### Scheduled Task Flow: ✅ FIXED (was weekly guard vs monthly rebalance mismatch)

| Step | Component | Logic |
|------|-----------|-------|
| 1 | Task Scheduler triggers Mon-Fri 9PM | `scripts/run_champion_live.ps1` |
| 2 | Calendar guard | `is_first_nyse_rebalance_session.py` returns exit 2 (no action) unless the target is the **first NYSE trading day of the month** (matches `rebalance_frequency: "monthly"`) |
| 3 | Champion screen | `run_champion.py live` runs factor computation + TFM veto + scoring |
| 4 | Rebalance decision | `is_rebalance_session(session, sessions, "monthly")` — true on the first trading day of the month |
| 5 | Output | CSV + `RESULT` log line with `rebalance=yes/no` |

**Fix (2026-06-28):** the guard was changed from first-trading-day-of-the-week to first-trading-day-of-the-month and renamed `is_first_nyse_week_session.py` → `is_first_nyse_rebalance_session.py`; `run_champion_live.ps1` references and log messages updated. Guard and rebalance gate now agree, so every monthly rebalance is screened live. Verified: `is_first_nyse_rebalance_session.py 2026-04-01`→exit 0, `2026-04-13`→exit 2, `2026-04-04` (Sat)→exit 2, `2026-04-03` (Good Friday)→exit 2. **Action item:** correct the Windows Task Scheduler trigger to weekdays only (the prior log showed a Saturday trigger), and run `cli snapshot-holdings` on the first of each month to grow the holdings archive.

### Unused Parameter in Champion Config

`full_turnover_rebalance: false` is set in `config/champion.json` but `run_weighted_backtest` (the only backtester used by the champion path) does not reference `full_turnover_rebalance` — it always does full turnover to target weights. The `full_turnover_rebalance` flag is only used by the legacy `run_backtest` in `backtest.py`. This is harmless but could cause confusion. Consider removing or documenting.

### Review Summary

| Dimension | Status | Issues |
|-----------|--------|--------|
| Data leakage — price/technical/fundamental factors | ✅ PASS | Shift + walk-forward guards correct |
| Data leakage — macro risk gate | ✅ FIXED | Monthly/weekly FRED series now lagged by publication delay (Caveat 1) |
| Data leakage — universe membership | ⚠️ MITIGATED | Current IWV snapshot archived; historical backtest still survivorship-biased (Caveat 2) |
| Model input/output | ✅ PASS | TimesFM veto input/output PIT-correct; it is a veto, not the alpha model |
| Model structure & params | ⚠️ OBSERVATIONS | Fixed weights across splits; half-risk dead at N=1; `score_composite` uses a config global; `champion_veto` outside `AppConfig`; `positions.csv` has 2 holdings vs `max_holdings=1` |
| Backtest vs live (champion path) | ✅ PASS | `run_champion` + `autoresearch` consistent and reproducible |
| Backtest vs live (CLI vs champion) | ⚠️ DIVERGENT | Two engines + two scoring entry points; CLI omits veto and sizes differently |
| Scheduled run | ✅ FIXED | Guard changed to first-of-month; matches monthly rebalance (2026-06-28) |
| Test suite | ✅ PASS | 22/22 pass; stale `run_live.ps1` references updated to `run_champion_live.ps1` + guard test |

**Bottom line:** The price/factor pipeline is leak-free; the macro risk gate is now point-in-time (publication-lag fix); the live schedule guard now matches the monthly rebalance; the test suite is green (22/22); and a current IWV snapshot is archived for PIT going forward. The champion backtest↔live↔autoresearch path is internally consistent and reproducible. **Remaining data limitation:** the historical backtest is still survivorship-biased because no historical IWV snapshots are available to backfill — backtest CAGR/Sharpe should still be treated as optimistic until `data/holdings_history/` is backfilled with monthly snapshots. The CLI harness remains a different strategy from the champion (no veto, different sizing) — use `run_champion.py backtest` to reproduce the champion.

## Notes

- **`signal_timeframe="weekly"` vs `rebalance_frequency="monthly"`**: These are independent settings. `signal_timeframe` controls how price data is resampled for factor computation (weekly bars for SMA, RSI, etc. — see `indicators.py:147`). `rebalance_frequency` controls which trading days are eligible for trade decisions (first trading day of each month — see `calendar_utils.py:42`). The strategy monitors on the first NYSE day of each week (via the `run_champion_live.ps1` calendar guard), but only trades on monthly rebalance sessions. The skip-when-unchanged optimization in `run_weighted_backtest` makes the rebalance frequency irrelevant for turnover — positions only change when the top stock changes.
- The `universe_limit=200` matches the baseline config. The `min_price=0.50` and `min_avg_dollar_volume=11M` are applied as filters in `build_factor_library`, not during data download.
- The `statement_lag_days=60` matches the baseline. This parameter affects fundamental factor computation in `load_market_context` and must match the baseline for reproducibility.
- Transaction costs are 3 bps per side (6 bps round trip), realistic for liquid stocks with $11M+ daily volume.
- Historical fundamentals are approximated from yfinance statement history with a 60-day reporting lag.
- **The script does not connect to your broker** — you must place trades yourself through your brokerage platform.
- **Universe survivorship bias is partially mitigated.** `data/holdings_history/2026-06-28.csv` is archived, so the live path is point-in-time from today forward. The historical backtest still uses today's IWV constituents for pre-snapshot dates because no historical snapshots are available to backfill. Backfill `data/holdings_history/` with monthly snapshots to remove this overstatement (see Code Review → Leakage Caveat 2).
- **Fundamental data comes from yfinance** and is not a licensed point-in-time dataset. Statement dates are shifted by a 60-day reporting lag (`statement_lag_days`) to approximate filing availability, but this is still an approximation.
- **The weekly calendar guard and the monthly rebalance are now aligned (fixed 2026-06-28).** `is_first_nyse_rebalance_session.py` now gates on the first trading day of the month, matching `rebalance_frequency: "monthly"`, so every monthly rebalance is screened live. Do not trade from a run that says `Rebalance due: no`. Run `cli snapshot-holdings` on the first of each month to grow the holdings archive.
- **Max drawdown can exceed 60%** — the champion strategy has a -67.05% historical drawdown (post macro-PIT fix, 2026-06-28). Size your positions accordingly and consider the real-use policy above.
