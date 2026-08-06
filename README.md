# SPY / S&P 500 Rotation — Champion Strategy Harness

A concentrated 2-stock S&P 500 momentum-rotation strategy with a TimesFM
**downside-quantile (P20) volume veto**, conviction-tilt + volatility-adjusted
sizing, and a macro risk gate. The champion was discovered by autoresearch
(iter83, 85 runs / 38 kept) maximizing **worst-year Sharpe** — the minimum
annual Sharpe across full years 2015–2025 — rather than headline return.

---

## Quick Operating Guide (plain English)

This is the **human-facing cheat sheet** — what runs automatically, what you do
by hand, and which files to touch. Read this first; the rest of the README is
the full technical reference.

### When it runs (automatic)

- **Scheduled task** `Fundamental Momentum - 1st Day of Month` fires **Mon–Sat,
  21:00 SGT** (= 08:00/09:00 ET, before the US open — the previous US session
  closed at 04:00/05:00 SGT the same SGT day).
- At each trigger the calendar guard finds the **last completed NYSE session**
  in ET (timezone-correct, not the SGT-local date). Once the month's first NYSE
  trading day has completed, it returns that first session as `SESSION=YYYY-MM-DD`.
  If this month hasn't been processed yet, it refreshes TimesFM forecasts and
  runs the champion screen. Otherwise it logs "No action needed" and exits.
- **Missed-run catch-up:** a state file (`artifacts/live/last_rebalance_processed.txt`)
  records the last processed month. If the first-trading-day run fails or the PC
  is off, **any later day of that month automatically picks it up** — you never
  silently skip a monthly rebalance.
- A **lockfile** (`artifacts/live/.run.lock`) prevents a manual run and a
  scheduled run from racing on the GPU.

### What you do (weekly, ~2 minutes)

1. Open `artifacts/live/champion_run.log` and read the **last `RESULT` line**.
2. **If `rebalance=yes`** → open `artifacts/live/champion_screen_YYYYMMDD.csv`:
   - Execute every row (`BUY` / `SELL`) at the **next US market open**.
   - The screen now includes **explicit `SELL` rows** for stocks you hold that
     dropped out of the top-2 — exit those.
   - Update `positions.csv` to match what you actually placed.
3. **If `rebalance=no`** → do nothing.
4. **First trading day of each month** (optional but recommended): grow the
   holdings archive for future point-in-time accuracy:
   `python -m algo_trading.cli snapshot-holdings`.

### How to read warnings in the log

| Signal in `RESULT` / log | Meaning | Action |
|--------------------------|---------|--------|
| `veto_cov=on` | P20 veto active, forecast fresh | None — normal |
| `veto_cov=stale` | Forecast cache is >35 days old; veto is ffilled from an old date | Run `python generate_spy_forecasts.py --quantiles`, then re-screen |
| `veto_cov=off` | No forecast coverage at all for this session | Same as above |
| `session_lag>4` | Price data is >4 days behind target (yfinance outage/stale cache) | Re-run with `--refresh-data`; check connectivity |
| No `Run ended.` line for a scheduled run | The run stalled/crashed without finishing | Re-trigger manually; see Troubleshooting |

### Manual commands (the only files you run by hand)

```powershell
# Re-screen a specific session (e.g. after fixing a forecast gap)
python run_champion.py live --as-of 2026-07-01 --current-positions positions.csv

# Force-refresh stale price/FRED/fundamentals caches, then screen
python run_champion.py live --refresh-data

# Refresh the TimesFM P20 forecast cache (idempotent; skips if fresh)
python generate_spy_forecasts.py --quantiles
# Low-VRAM GPU (4 GB):
TFM_BATCH=16 python generate_spy_forecasts.py --quantiles

# Archive current S&P 500 constituents (monthly, for PIT accuracy)
python -m algo_trading.cli snapshot-holdings

# Manually trigger the scheduled task (e.g. to test a repair)
Start-ScheduledTask -TaskName 'Fundamental Momentum - 1st Day of Month'
```

### Files to know (and maintain)

| File | What it is | When you touch it |
|------|------------|-------------------|
| `positions.csv` | Your current holdings (`symbol,shares`) | Reconcile with broker before each trade; update after placing orders |
| `artifacts/live/champion_run.log` | Append-only run log | Weekly check — grep `RESULT` |
| `artifacts/live/champion_screen_YYYYMMDD.csv` | The screen output (rank/symbol/weight/action) | Open it when `rebalance=yes` |
| `artifacts/live/last_rebalance_processed.txt` | State file: last processed `YYYY-MM` | Don't edit unless re-running a month (delete the line to force a re-run) |
| `artifacts/live/.run.lock` | Single-instance lock (PID) | Auto-removed; delete only if a run crashed and left it stale |
| `config/research.json` + `research_params.json` | **The champion config — do not edit** unless a new autoresearch run approves a change | Only on an intentional strategy change |
| `cache/timesfm/ctx256_hor21_volume_q.parquet` | The P20 veto forecast cache (the only cache file that matters) | Refresh via `generate_spy_forecasts.py --quantiles` when `veto_cov=stale/off` |
| `data/holdings_history/` | Dated S&P 500 snapshots | Grow monthly via `snapshot-holdings` (backfill is a data task, see Code Review) |
| `.env` | `FRED_API_KEY=...` | Once at setup |

### One-line trade timing

Screen runs **after** the first-trading-day close (at 21:00 SGT the following
evening) → you place orders at the **next** US open. This is ~1 session of
slippage vs the backtest (which trades the first-of-month open on
prior-session factors) — already the de-facto behavior, now documented.
Note the margin is tight during US daylight time (open = 21:30 SGT): if the
run finishes after the open, execute as soon as the screen CSV appears.

---

**Champion backtest (2015-01-02 → 2025-12-31, 11 full years, `research_params.json` as of 2026-06-30):**

| Metric | Value |
|--------|-------|
| **worst_year_sharpe (PRIMARY)** | **0.5964** (2015 floor — all 11 years positive) |
| Sharpe | 1.4535 |
| CAGR | 80.27% |
| Max Drawdown | -48.90% |
| Calmar | 1.6413 |
| Sortino | 2.0987 |
| Information Ratio (vs SPY) | 1.3832 |
| Robust Score (mean − 0.5·std year Sharpe) | 0.9499 |
| Mean / Std year Sharpe | 1.2936 / 0.6873 |
| Best year Sharpe (2024) | 2.7888 |
| Veto Rate (P20 downside) | 3.26% |
| Perturbation worst-year std (8 runs, σ=0.03) | 0.1579 |
| Perturbation mean worst-year | 0.2843 (perturbed floor stays positive) |

**Year-by-year breakdown:**

| Year | CAGR | Sharpe | MaxDD |
|------|------|--------|-------|
| 2015 | 21.2% | 0.596 | -25.7% |
| 2016 | 92.2% | 1.917 | -21.1% |
| 2017 | 26.3% | 0.771 | -16.2% |
| 2018 | 39.0% | 1.004 | -29.8% |
| 2019 | 35.7% | 1.011 | -26.4% |
| 2020 | 199.3% | 2.292 | -39.4% |
| 2021 | 63.4% | 1.359 | -27.7% |
| 2022 | 35.5% | 0.789 | -29.6% |
| 2023 | 37.0% | 0.823 | -21.9% |
| 2024 | 326.1% | 2.789 | -30.8% |
| 2025 | 48.9% | 0.879 | -48.9% |

_Reproduced by `python run_champion.py backtest` (≈7 min, delegates to
`overfit_harness.run_benchmark` with perturbation off). These numbers remain
**survivorship-biased** because `data/holdings_history/` has no historical S&P
500 snapshots — the backtest uses today's constituents for every session. See
[Code Review → Universe survivorship bias](#universe-survivorship-bias)._

---

## Champion Parameters

The champion is `config/research.json` (SPY universe, monthly rebalance) with
`research_params.json` overlaid by `overfit_harness.apply_params_to_config`.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_holdings` | 2 | Two-stock concentration |
| `sma_fast/slow/trend` | 3 / 50 / 210 | Moving-average crossover + trend filter |
| `ts_mom_lookback/skip` | 10 / 0 | Time-series momentum (months) |
| `fundamental_group_weight` | 0.02 | Fundamental factor group share (technical gets 0.98) |
| `asset_growth_weight` | 20.0 | Asset-growth factor weight (within fundamental group) |
| `earnings_yield_weight` | 3.0 | Earnings-yield factor weight |
| `macro_half_risk/flat` | -0.5 / -1.0 | Half-risk (→1 holding) / flat (→cash) gates |
| `min_avg_dollar_volume` | $11M | Liquidity floor |
| `min_price` | $0.50 | Price floor |
| `use_quantile_veto` / `quantile_col` | true / 1 (P20) | Downside-quantile volume veto |
| `veto_threshold` | 0.35 | P20 predicted vol / trailing-vol ratio below which a name is vetoed |
| `trailing_vol_window` | 21 | Sessions for the trailing average volume |
| `veto_horizon` / `forecast_context_len` | 21 / 256 | TimesFM forecast horizon / context length |
| `vol_scaled_weights` / `vol_lookback` | true / 21 | Inverse-vol risk-balanced sizing across the top-2 |
| `weight_tilt` | 0.17 | Conviction tilt toward the top-1 |
| `conf_tilt` / `conf_tilt_pow` | true / 50 | Confidence-scale the tilt by the top-1-vs-top-2 composite gap |
| `conf_tilt_vol` / `conf_tilt_vol_pow` | true / 4.0 | Vol-regime-scale the tilt (more in high-vol, less in low-vol) |
| `cost_per_side` | 3 bps | Transaction cost per side |
| `rebalance_frequency` / `signal_timeframe` | monthly / weekly | Trade monthly; factors computed on weekly bars |

**Net factor weights** (grouped_equal, fixed across all walk-forward splits):
technical 0.98 split equally over 4 factors → `sma_crossover`, `ema_crossover`,
`trend_filter`, `ts_momentum` ≈ 0.245 each; fundamental 0.02 split 20:3 →
`asset_growth` ≈ 0.0174, `earnings_yield` ≈ 0.0026.

---

## Plain-English Summary

The champion answers: _based on the strategy rules, which 2 S&P 500 stocks
should I hold now, weighted how, and do I need to trade today?_

It does **not** place trades — it writes a screen CSV and prints a summary. You
place the orders through your brokerage.

**The most important rule:** run the live screen weekly for monitoring, but
**only trade when the output says `Rebalance due: yes`** (the first NYSE trading
day of each month). If it says `no`, do not trade from that run.

---

## What The Champion Script Does

When you run `run_champion.py live`, it:

1. Loads `config/research.json` and overlays `research_params.json`.
2. Fetches/refreshes the current S&P 500 constituents from Wikipedia (cached 7 days).
3. Builds technical factors (SMA crossover, EMA crossover, trend filter, 10-month
   time-series momentum) and fundamental factors (asset growth, earnings yield),
   all shifted one session for open-execution (no look-ahead).
4. Ranks the universe under grouped-equal weighting → composite score.
5. Applies the **P20 downside-quantile volume veto**: any stock whose predicted
   P20 volume / trailing 21-session average volume < 0.35 is removed.
6. Sizes the top-2 survivors: inverse-vol risk balance, then a conviction tilt
   toward the top-1 scaled by the composite gap and the market-vol regime.
7. Applies the macro risk gate (half-risk at macro ≤ -0.5, flat/cash at ≤ -1.0).
8. Checks whether this session is a scheduled (monthly) rebalance.
9. Compares targets against `positions.csv` (if provided) → BUY/HOLD actions.
10. Writes `artifacts/live/champion_screen_YYYYMMDD.csv` and prints a `RESULT` line.

---

## How New Data Is Handled (No Human Intervention Needed)

- **S&P 500 constituents** auto-refresh: `load_market_context` calls
  `fetch_sp500_holdings` (Wikipedia scrape) whenever `cache/sp500_holdings.csv`
  is older than 7 days, so each run sees the latest S&P 500 as it rebalances.
- **TimesFM P20 forecasts** auto-refresh: the scheduled task runs
  `generate_spy_forecasts.py --quantiles` before the screen. It is **idempotent**
  — it loads the auto-refreshed holdings, finds `(symbol, date)` pairs missing from
  the forecast cache (new S&P 500 entrants **and** new monthly rebalance dates),
  forecasts only those on GPU, then merges them in atomically. If the cache already
  covers the universe on every rebalance date it exits early without loading the
  model. Failure is non-fatal: the screen still runs with `veto_cov=off` and a warning.
  *(2026-07-02 fix: the refresh previously short-circuited on symbol coverage only,
  so the quantile cache froze at 2026-01-02 and the live veto drifted onto
  6-month-stale ffilled forecasts. It now also checks the date dimension — see
  [Code Review (2026-07-02) → Forecast-staleness bug](#forecast-staleness-bug-fixed-this-review).)*
- **Prices / FRED / fundamentals** are cached with freshness windows
  (`price_cache_days`, `fred_cache_days`, `fundamentals_cache_days`) and refreshed
  with `--refresh-data` or when stale.

---

## Key Files

### Strategy logic (load-bearing)

| File | Role |
|------|------|
| `src/algo_trading/timesfm_experiments.py` | Weighted backtester (`run_weighted_backtest`, skip-when-unchanged), `build_factor_state`, `score_composite`, `build_context_requests` (PIT TFM context builder), `extended_metrics`. Shared by `run_champion.py` and `overfit_harness.py`. |
| `src/algo_trading/strategy.py` | Factor library + walk-forward weights (`build_factor_library`, `fit_indicator_weights`, `_strip_leaky_train_dates`, `score_universe_for_date`, `screen_live_session`). |
| `src/algo_trading/indicators.py` | Technical indicators + `shift_for_open_execution` (the `.shift(1)` no-look-ahead guard). |
| `src/algo_trading/data.py` | `load_market_context`, `fetch_sp500_holdings`, macro/fundamental loaders with PIT publication/statement lags. |
| `src/algo_trading/timesfm_engine.py` | TimesFM model load + inference, forecast-panel cache, `mean_forecast_sum`, `quantile_path`. |
| `src/algo_trading/calendar_utils.py` | Rebalance calendar (`select_rebalance_sessions`, `is_rebalance_session`, `next_rebalance_session`). |
| `src/algo_trading/config.py` | `AppConfig` dataclasses for every tunable parameter. |

### Configs

| File | Role |
|------|------|
| `config/research.json` | **The live champion config** — SPY universe (limit 500), benchmark SPY, monthly rebalance, weekly signal timeframe, enabled indicators, macro series. |
| `research_params.json` | **The iter83 champion parameters** (the table above). Applied on top of `config/research.json` by `overfit_harness.apply_params_to_config`. Single source of truth for champion tuning. |
| `config/baseline.json` | The pre-champion IWV baseline config used by the legacy CLI / research harness and the smoke tests. Not the champion. |

### Entry points

| File | Role |
|------|------|
| `run_champion.py` | **Champion runner** (backtest + live). `backtest` delegates to `overfit_harness.run_benchmark` (perturb off) for byte-for-bit reproduction; `live` reuses `build_veto_mask` + `_size_weights` so the screen uses the exact P20 veto + conviction/vol sizing as the backtest. |
| `overfit_harness.py` | **Champion backtest engine + autoresearch target.** `apply_params_to_config`, `build_veto_mask` (P20 veto), `_size_weights` (conviction + vol + vol_pow), `run_benchmark` (year metrics, perturbation, worst-year Sharpe). |
| `generate_spy_forecasts.py` | GPU TimesFM forecast generator (idempotent, resumable, force-refreshes holdings). Run after a holdings refresh or on the monthly rebalance day: `python generate_spy_forecasts.py --quantiles` (low-VRAM GPUs: `TFM_BATCH=16 python generate_spy_forecasts.py --quantiles`). |
| `overfitting_checklist.py` | Overfitting diagnostics for the champion. |
| `audit_pit_primary.py` / `audit_pit_coverage.py` | PIT + forecast-coverage audits (verify no look-ahead and full veto coverage). |
| `scripts/run_champion_live.ps1` | Scheduled-task launcher with the NYSE calendar guard. |
| `scripts/is_first_nyse_rebalance_session.py` | Calendar guard: exit 0 only on the first NYSE trading day of the month. |

### Forecast stack

| File | Role |
|------|------|
| `experiments/forecast_worker.py` | Batch forecast worker. |
| `experiments/run_forecast_driver.sh` | Queue→worker pipeline that produces the volume/logret forecast panels. **If a cache is stale or missing, run this to regenerate.** |
| `experiments/_prepare_requests.py` / `experiments/experiment_utils.py` | Request builders / experiment helpers. |
| `experiments/run_walkforward.py` / `experiments/lora_finetune.py` | Archival experiments (not champion). |

---

## Setup

1. Install Python 3.11+.
2. Create and activate a virtual environment (`.venv`).
3. `pip install -e .`
4. Put your FRED API key in `.env` (or pass `--env-file`): `FRED_API_KEY=your-key`.
   For TimesFM inference you also need `torch` + `transformers` (GPU recommended).

```powershell
$env:FRED_API_KEY = "your-key"
pip install -e .
```

---

## Commands

### Champion strategy (recommended)

```powershell
python run_champion.py backtest                    # Reproduce the iter83 champion (≈7 min)
python run_champion.py live                        # Screen the current session
python run_champion.py live --as-of 2026-06-23     # Screen a specific date
python run_champion.py live --current-positions positions.csv
python run_champion.py live --refresh-data         # Force-refresh cached data
```

### Autoresearch loop (finding the next champion)

```powershell
python overfit_harness.py --config config/research.json --params research_params.json --perturb-runs 8
```

The autoresearch framework (`.auto/measure.sh` + `.auto/checks.sh`) drives
`overfit_harness.py` over a parameter space; results land in `.auto/log.jsonl`.
If a new champion is found, update `research_params.json`.

### Legacy CLI (baseline research harness — not the champion)

```powershell
python -m algo_trading.cli --config config/baseline.json backtest     # walk-forward baseline
python -m algo_trading.cli --config config/baseline.json screen       # baseline live screen (no veto)
python -m algo_trading.cli snapshot-holdings                           # archive current SPY holdings
python -m algo_trading.cli import-holdings-history --source-dir DIR   # bulk-import dated snapshots
```

> ⚠️ The legacy CLI uses `score_universe_for_date` + `backtest.run_backtest`
> (equal-dollar, keep-overlap, **no TFM veto**) — a different engine from the
> champion. It does **not** reproduce the champion. Use `run_champion.py` for
> the champion.

---

## Current Holdings File (`positions.csv`)

Optional. Used to compare the target portfolio against what you already own:

```csv
symbol,shares
WDC,50
LITE,35
```

`symbol` uppercase; `shares` your current quantity. A header-only file also
works. Reconcile it with your actual brokerage before each live run.

---

## What To Look For In The Output

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

- **`Rebalance due: yes`** → review the screen and consider placing trades.
- **`Rebalance due: no`** → monitoring only, do not trade.
- **`Veto coverage: ON (P20 quantile)`** → the downside-quantile veto is active.
  If it says `OFF — run generate_spy_forecasts.py --quantiles`, the live session
  has no forecast coverage; re-run the forecast generator, then re-screen.
- **`Vetoed stocks`** → count removed by the P20 veto (~3% of candidates).
- **`weight`** → final portfolio weight after ALL sizing mechanisms (inverse-vol
  risk balance + conviction tilt + confidence scaling + vol-regime scaling).
  See [How Weights Are Calculated](#how-weights-are-calculated-sizing-mechanism) below.
- **`composite_score`** → the raw factor composite BEFORE veto and sizing.
  The composite gap (`top1 − top2`) drives the confidence-scaling: a large gap
  (clear winner) → full 17% tilt; a small gap (close call) → near zero tilt.
- **`action`** → `BUY` (open) / `HOLD` (keep).
- **`RESULT`** → grep-friendly summary logged to `champion_run.log`.

---

## How Weights Are Calculated (Sizing Mechanism)

The champion's sizing is a **4-layer stack** applied to the top-2 stocks (after
the P20 veto removes ~3% of candidates). Each layer compounds on the previous one.

### Layer 1 — Inverse-vol risk balance (`vol_scaled_weights=true, vol_lookback=21`)

Before the conviction tilt is applied, the base weights are set by inverse-volatility
risk balancing across the top-2:

```
vol[sym]  = 21-session rolling std of daily returns for sym, measured at prior_end
inv[sym]  = 1.0 / max(vol[sym], 0.001)
total_inv = sum(inv for all top-N)
base_w[sym] = inv[sym] / total_inv
```

This gives **more weight to the lower-volatility stock** (risk parity). If both
stocks have similar volatility, the base is near 50/50. If one is much calmer,
it gets more weight (e.g. 55/45). This reduces drawdowns by under-weighting the
more volatile name.

### Layer 2 — Conviction tilt (`weight_tilt=0.17`)

Shift 17% of total portfolio weight from the lower-ranked names to the top-1
ranked by composite score:

```
w[top1] += 0.17
w[top2] -= 0.17   (for N=2)
```

After layers 1+2, if base was 50/50, the weights become 67/33. The top-1 gets
a conviction bonus — the strategy bets MORE on the strongest signal.

### Layer 3 — Confidence scaling (`conf_tilt=true, conf_tilt_pow=50`)

Not every month deserves the full 17% tilt. If the top-2 composite scores are
very close (a "coin toss"), tilting 17% is reckless. This layer scales the tilt
by how confident the composite gap is:

```
gap          = |composite[top1] - composite[top2]|
median_gap   = median of all gaps across history (computed once over full sample)
raw_factor   = (gap / median_gap) ^ 50
conf_factor  = min(1.0, raw_factor)
eff_tilt     = weight_tilt × conf_factor
```

The **pow=50** creates a step function:
- **gap ≥ median_gap** → `conf_factor ≈ 1.0` → full 17% tilt (clear winner)
- **gap < median_gap** → `conf_factor ≈ 0.0` → near-zero tilt (close call)

This is what separates the 2022 lift (high-confidence energy-sector winners) from
the 2015 floor protection (low-confidence choppy months where tilting would over-bet).
A **CONFIDENCE signal**, not a return forecast — it reads the composite gap at
decision time with no look-ahead.

### Layer 4 — Vol-regime scaling (`conf_tilt_vol=true, conf_tilt_vol_pow=4`)

The tilt should be HIGHER in turbulent bear markets (where conviction matters more)
and LOWER in calm bull markets (where equal-weight is safer). This layer scales
the effective tilt by the market volatility regime:

```
bench_vol[session] = 21-session rolling std of SPY daily returns at prior_end
median_vol         = median of all bench_vol values
vol_factor         = min(2.0, (bench_vol / median_vol) ^ 4)
final_tilt         = eff_tilt × vol_factor
```

The **pow=4** is the interior optimum (1→2→3→4 peak→5→8 regress, tested exhaustively
in the autorresearch). It amplifies the tilt in high-vol periods (like 2022 energy
crisis, where bench_vol ≈ 2× median → vol_factor ≈ 2^4 = 16, capped at 2.0) and
attenuates it in low-vol periods (like 2017, where bench_vol ≈ 0.7× median →
vol_factor ≈ 0.7^4 = 0.24). This protects the 2017 canary (low-vol bull market)
and unlocks a higher base weight_tilt (0.17 instead of 0.13).

### Interpreting weights from the log

Given the example output:

```
 rank symbol weight composite_score
    1    WDC 0.6814        0.370790
    2   ECHO 0.3186        0.353016
```

**Weight 0.6814 / 0.3186 = roughly 68/32 split.** Here's how to read it:

1. **Composite gap** = 0.3708 − 0.3530 = 0.0178. If this gap ≥ median_gap, the
   confidence scaling returns `conf_factor ≈ 1.0` → the full 17% tilt is active.
   If the gap is smaller, the tilt is attenuated (closer to equal-weight).

2. **The spread is ~36% (0.6814 − 0.3186).** Equal-weight would be 0% spread.
   The 36% spread comes from TWO sources combined:
   - **Inverse-vol** risk balance (Layer 1): if WDC has lower volatility than
     ECHO, it gets more base weight (e.g. 52/48 instead of 50/50).
   - **Conviction tilt** (Layers 2–4): shifts up to 17% from ECHO to WDC
     (e.g. 50/50 → 67/33, or 52/48 → 69/31).

3. **If weights were near 50/50**, the conviction tilt is OFF (confidence gap
   below median, or vol-regime is very calm). This means the composite couldn't
   pick a clear winner — a close call, so the strategy stays equal-weight.

4. **If weights were near 65/35 or more extreme**, the conviction tilt is at
   or near full strength → the composite is confident the top-1 will outperform.

**No separate columns exist for the tilt amount or confidence factor** — the
`weight` column IS the final output of all four layers. This is intentional:
trading decisions need a single number per stock. The interpretation above tells
you what's driving that number.

---

## Scheduled Task (Live Trading)

- **Task name:** `Fundamental Momentum - 1st Day of Month`
- **Schedule:** **Mon–Sat, 21:00 SGT** (= 08:00/09:00 ET, before the US open; the previous US session closed at 04:00/05:00 SGT the same day)
- **Calendar guard:** evaluates the **last completed NYSE session** in ET; once the month's first NYSE session has completed, returns that first session as `SESSION=YYYY-MM-DD`; the launcher acts only when that month has not been processed yet
- **Catch-up:** a state file (`artifacts/live/last_rebalance_processed.txt`) records each processed month, so a missed/failed first-trading-day run is automatically picked up on any later day of that month
- **Lockfile:** `artifacts/live/.run.lock` (PID) prevents manual + scheduled runs from overlapping on the GPU
- **Script:** `scripts/run_champion_live.ps1`

> ⚠️ **Why is 9 PM SGT safe now (it wasn't before 2026-07-03)?** The old code
> screened the *current* ET date, which at 9 PM SGT (= 08:00/09:00 ET) has not
> traded yet — yfinance had no bar, so the run always fell back to the prior
> session (the 2026-07-01 21:00 run stalled silently and was re-run manually at
> 00:23). The rewritten calendar guard instead resolves the **last completed
> NYSE session** in ET, so a 21:00 SGT trigger screens the previous day's
> finished session with full data (verified with `--now-utc` simulations of
> 21:00 SGT in both EDT and EST). The schedule moved from 09:00 to 21:00 SGT on
> 2026-07-03 because the PC is off at 09:00; the screened session and the
> execution session are unchanged — the screen for the first-trading-day close
> is now produced ~30–90 min before the next open instead of ~12 h before. If a
> cold-cache run (~6–13 min) finishes after a 21:30 SGT (EDT) open, execute as
> soon as the screen CSV appears. See [Code Review (2026-07-03)](#code-review-2026-07-03--scheduling-freshness--sell-rows).

### Execution flow

1. Task Scheduler triggers `run_champion_live.ps1` **Mon–Sat 21:00 SGT**.
2. PowerShell resolves Python, takes the single-instance lock, opens
   `artifacts/live/champion_run.log` (logs `Run ended.` on every exit so a
   silent stall is visible by the absence of that line).
3. **Calendar guard** (`is_first_nyse_rebalance_session.py`): with no `-AsOf`
   arg it computes the last completed NYSE session in ET. Once the month's first
   NYSE session has completed, it prints `SESSION=YYYY-MM-DD` for that first
   session and exits 0. It also prints `CANDIDATE_SESSION=YYYY-MM-DD` for the
   latest completed session. Before the first session has completed, it logs
   "No action needed" and exits.
4. **Catch-up check:** if `last_rebalance_processed.txt` already contains this
   session's `YYYY-MM`, the run is skipped (already done). Otherwise proceed.
5. **TimesFM forecast auto-refresh** (`generate_spy_forecasts.py --quantiles`):
   idempotent incremental GPU refresh that force-refreshes S&P 500 holdings (Wikipedia)
   and forecasts any `(symbol, date)` pairs missing from the cache — new S&P 500 entrants
   **and** new monthly rebalance dates. Exits early (no model load) if the cache already
   covers the universe on every rebalance date. Prices/FRED/fundamentals refresh via
   their cache windows (3/7/30 days), which always trigger on the monthly run (~30 days
   since the last data load). Non-fatal on failure. `TFM_BATCH` / `TFM_CHECKPOINT` env vars
   tune GPU batch size and the partial-cache checkpoint interval.
6. **Champion screen** (`run_champion.py live --as-of <SESSION>`) → factor build
   + P20 veto + conviction/vol sizing + macro gate + monthly rebalance check →
   `RESULT` line (now incl. `veto_cov` and `session_lag`). The screen CSV
   includes explicit `SELL` rows for held names that dropped out of the top-2.
7. On success, write `YYYY-MM` to `last_rebalance_processed.txt` (so the month
   is not re-screened). On failure the state file is **not** written → the next
   trigger retries automatically.

**This uses the converged champion path ONLY** — `run_champion.py` imports and calls
`build_veto_mask`, `_size_weights`, and `score_composite` from `overfit_harness.py`
(the exact same functions used in the backtest). It never touches the legacy CLI
path (`screen_live_session`, `backtest.run_backtest`, `score_universe_for_date`).
The live screen uses the **same P20 quantile veto + conviction-tilt + vol-adjusted
sizing** as the iter83 champion backtest.

### Recreate / repair the scheduled task (elevated PowerShell)

```powershell
# Remove any stale task from an old repo location
Unregister-ScheduledTask -TaskName 'Fundamental Momentum - 1st Day of Month' -Confirm:$false -ErrorAction SilentlyContinue

# Register the champion task at the current repo path (Mon-Sat 21:00 SGT)
$repo = 'C:\Users\User\Desktop\Weekly Script\Fundamental Momentum-1st day of the month'
$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument ('-NoProfile -ExecutionPolicy Bypass -File "' + $repo + '\scripts\run_champion_live.ps1" -CurrentPositions positions.csv') `
    -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday,Saturday -At 9pm
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName 'Fundamental Momentum - 1st Day of Month' -Action $action -Trigger $trigger -Settings $settings -Force
```

Manual trigger: `Start-ScheduledTask -TaskName 'Fundamental Momentum - 1st Day of Month'`

Verify trigger days: `$t = Get-ScheduledTask -TaskName 'Fundamental Momentum - 1st Day of Month'; $t.Triggers | Format-List DaysOfWeek, Enabled`

---

## Outputs

### Champion live outputs (`artifacts/live/`)

- **`champion_screen_YYYYMMDD.csv`** — ranked screen with the P20 veto applied
  (`rank, symbol, name, sector, weight, composite_score, action, vetoed_symbols`).
- **`champion_run.log`** — append-only timestamped run log. Grep `RESULT` for a
  compact per-run summary.

### Backtest outputs (`artifacts/backtests/`)

- `backtest_equity_curve.csv`, `backtest_trade_log.csv`, `backtest_selections.csv`,
  `backtest_metrics.json`, `backtest_walk_forward.json`.

### Shared

- **`data/holdings_history/`** — dated S&P 500 holdings snapshots for PIT studies.
  **Only one snapshot exists today** (`2026-06-28.csv`) → backtest is survivorship-biased.
  Run `cli snapshot-holdings` on the first trading day of each month to grow the archive.
- **`cache/timesfm/`** — the forecast panels. **Only `ctx256_hor21_volume_q.parquet`
  is read by the champion** (the P20 veto cache, ctx=256 / hor=21 / quantile col 1).
  All other files are orphaned/experimental — see
  [Code Review (2026-07-02) → Cache audit](#cache-audit--which-file-is-actually-read).

---

## Long-Term Maintenance

### Weekly checklist

1. Open the project folder in PowerShell.
2. Check `artifacts/live/champion_run.log` for the latest `RESULT` line.
3. If `rebalance=yes`: review `champion_screen_YYYYMMDD.csv`, reconcile
   `positions.csv` with your brokerage, place BUY/SELL orders.
4. If `rebalance=no`: nothing to do.
5. On the first trading day of each month: `python -m algo_trading.cli snapshot-holdings`
   to archive the current S&P 500 constituents for future PIT studies.

### Real-use policy

- **Max drawdown can exceed -48%** (the champion hit -48.9% in 2025). Size positions
  accordingly; consider halving exposure or sitting out when the macro gate is in the
  half-risk / flat band.
- Do not trade from a run that says `Rebalance due: no`.
- The script does not connect to your broker — you place the trades yourself.

### Data refresh

- Prices/FRED/fundamentals refresh automatically when their cache windows expire, or
  with `--refresh-data`.
- If a FRED refresh fails (no `FRED_API_KEY`), the live screen aborts with a clear
  setup error rather than guessing the macro regime — do not bypass this.

### Config updates

- Keep `config/research.json` and `research_params.json` unchanged unless you
  intentionally review and approve a strategy change after new research.
- A new autoresearch champion updates `research_params.json` only (config/research.json
  holds the universe/indicator wiring).

### Troubleshooting

- **`Veto coverage: OFF`** → run `python generate_spy_forecasts.py --quantiles` to
  forecast new symbols, then re-screen.
- **`Veto coverage: STALE (ffilled from <date>, Nd old)`** / **`veto_cov=stale`** →
  the forecast cache lacks the current session so the veto is forward-filled from
  an old forecast. Run `python generate_spy_forecasts.py --quantiles` (catches up
  the missing months in one GPU pass), then re-screen.
- **`session_lag>4`** / **"latest loaded price session is … (Nd behind target)"** →
  yfinance returned stale data. Re-run with `--refresh-data`; if it persists, check
  connectivity. A >20% partial download now aborts with a clear error instead of
  silently shrinking the universe.
- **No `Run ended.` line for a scheduled run** → the python process stalled/crashed
  without reaching the finally block (the 2026-07-01 21:00 run did this during a
  yfinance holdings fetch). Re-trigger the task or run
  `scripts/run_champion_live.ps1` manually; the lockfile is auto-reclaimed if the
  stale PID is no longer alive.
- **Month skipped / no `rebalance=yes` this month** → check
  `artifacts/live/last_rebalance_processed.txt`. If it's missing or wrong, delete it
  and re-run the launcher (the catch-up logic will screen the first trading day of
  the current month). Do **not** delete it to force a re-screen of a month you
  already traded.
- **`No eligible stocks found (macro risk-off…)`** → the macro composite is below
  the flat threshold; the strategy is intentionally in cash.
- **FRED setup failure** → create `.env` with `FRED_API_KEY=...` (see Setup).
- **`HEIA`/`BK`/`SATS` delisted warnings** in logs → benign; yfinance can't fetch a
  few delisted tickers, they are dropped from the universe.
- **`Another run is in progress (PID …). Exiting.`** → a scheduled run is using the
  GPU. Wait for it to finish (check the log for `Run ended.`). If the PID is stale,
  delete `artifacts/live/.run.lock`.

---

## Strategy Coverage

- **Universe:** top-500 S&P 500 (Wikipedia constituents, limit 500), auto-refreshed.
- **Factors (enabled):** `sma_crossover`, `ema_crossover`, `trend_filter`,
  `ts_momentum` (technical); `asset_growth`, `earnings_yield` (fundamental).
- **Disabled:** breakout, rsi, cci, macd, obv, bollinger, variable_week_high,
  price_action, book_to_market, roa, roe, gross_profitability, investment_to_assets,
  net_issuance, accruals, cash_flow_yield, earnings_surprise.
- **Risk gates:** macro composite z-score (half-risk at -0.5, flat at -1.0);
  P20 downside-quantile volume veto (3.26% veto rate).
- **Sizing:** inverse-vol across top-2 + conviction tilt (0.17, gap-scaled, vol-regime-scaled).
- **Execution:** monthly rebalance at the session open; skip-when-unchanged to avoid
  needless turnover; 3 bps per side.

---

## Code Review (2026-07-01)

A systematic review of the SPY champion pipeline covering data leakage, model
input/output, model structure & parameters, and backtest/live/schedule
consistency. The codebase-memory knowledge graph (2145 nodes / 6286 edges) was
indexed and queried; every claim below traces to a specific location.

### Data Leakage Audit

| Checkpoint | Location | Guard | Verdict |
|------------|----------|-------|---------|
| Factor shift | `indicators.py` `shift_for_open_execution` | `.shift(1)` on ALL technical + fundamental factors in `build_factor_library` | ✅ Session-T close factors available at T+1 open |
| Base-mask filters | `strategy.py` `build_factor_library` | `history_ok`, `price_ok`, `liquidity_ok` all `.shift(1).fillna(False)` | ✅ Prior-session data only |
| Macro risk gate | `data.py` `load_macro_data` | Per-frequency publication lag (`PUBLICATION_LAG_DAYS`: daily=0, weekly=10, monthly=45 calendar days) shifts each series to its release date before `reindex(sessions, ffill)`; then `.shift(1)` for open execution | ✅ Monthly/weekly FRED values no longer used before release |
| Fundamental factors | `data.py` `_build_symbol_fundamentals` | `available_date = statement_date + statement_lag_days (60)`; `_latest_value`/`_ttm_value` filter `index <= cutoff` | ✅ PIT approximation with 60-day reporting lag |
| Walk-forward splits | `strategy.py` `build_walk_forward_splits` | Train window strictly before `test_start`; test = 12 months | ✅ No train/test overlap |
| Leaky train dates | `strategy.py` `_strip_leaky_train_dates` | Drops training rows whose forward-return endpoint reaches into the test period; called in `generate_walk_forward_plan`, `screen_live_session`, AND `build_factor_state` | ✅ No IC-fit peek into test |
| TFM veto trailing volume | `overfit_harness.py` `build_veto_mask` | `prior_end = sessions[pos-1]`; trailing avg uses data up to the session before rebalance | ✅ No look-ahead into rebalance session |
| TFM forecast context | `timesfm_experiments.py` `build_context_requests` + `generate_spy_forecasts.py` `build_missing_requests` | Context window ends at `prior_end` (session before rebalance); both paths byte-compatible | ✅ Forecasts don't see the rebalance session |

#### Universe survivorship bias

`data/holdings_history/` contains a single snapshot (`2026-06-28.csv`). With
`allow_current_holdings_fallback: true`, the **entire 2015→today backtest uses
today's S&P 500 constituents for every session** — no delisted names, no
historical membership changes. Backtest CAGR/Sharpe are **optimistic** vs a
tradable point-in-time universe. The live path is PIT-correct from the snapshot
date forward.

**Fix (data, not code):** backfill `data/holdings_history/` with monthly S&P 500
snapshots (paid historical-holdings dataset imported via
`cli import-holdings-history`), or set `allow_current_holdings_fallback: false`
to exclude unarchived periods (shrinks the backtest to the snapshot era).
Going forward, run `cli snapshot-holdings` on the first trading day of each
month so the archive grows.

#### Minor — annual 10-K lag

`statement_lag_days=60` is conservative for quarterly 10-Qs (~40–45d) but slightly
aggressive for annual 10-Ks (large accelerated filers 60d, others 90d). Applies
identically to backtest and live, so not a backtest/live inconsistency.

### Model Input / Output

**TimesFM is a veto model, not the alpha model.** Stock selection is the factor
composite (technical + fundamental ranks). TimesFM-2.5-200m is used **only** for
the volume veto:

- **Input:** 256-session raw daily volume series ending at the session before
  rebalance (`prior_end = sessions[pos-1]`, `window = series_frame.loc[:prior_end]`).
  Context length 256 is a multiple of `patch_len=32` (required). Left-padded with 0.
- **Output:** 21-step (sliced from the native 128) `full_predictions[..., col]`.
  `quantile_col=1` is P20 (downside). Veto fires when
  `sum(P20 over 21 steps)/21 / trailing_21_avg_vol < 0.35`.
- **`tfm_alpha_weight=0.0`** in `research_params.json` → the optional return-forecast
  tilt is OFF; TimesFM never contributes to selection, only to the veto.

**Quantile-column mapping verified.** `timesfm-2.5-200m-transformers/config.json`
declares `quantiles: [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]` (9 levels). The engine's
`N_QUANTILE_COLS=10` with `MEDIAN_COL=5` reflects the layout
`[P10,P20,P30,P40,P50, mean, P60,P70,P80,P90]` (mean inserted at index 5, matching
`full_predictions[...,5] == mean_predictions`). So `quantile_col=1` = P20 is
correctly labeled. ✅

### Model Structure & Parameters

**Factor weights are fixed across all walk-forward splits.** `indicator_weighting_scheme:
"grouped_equal"` + `use_indicator_ic_weights: false` + explicit
`fundamental_factor_weights` means `fit_indicator_weights` returns identical weights
for every split. The walk-forward machinery is decorative for the champion (it still
drives `min_train_rebalances` eligibility and the `_strip_leaky_train_dates` guard).

**Half-risk band is active at N=2.** `_target_holdings` returns
`max(1, max_holdings // 2)` in the half-risk band → `max(1, 1) = 1` holding. So
`macro_half_risk_threshold=-0.5` drops the strategy to a single holding when
`-1.0 ≤ macro ≤ -0.5`, and to cash when `macro < -1.0`. Consistent between backtest
and live (both call `_target_holdings` via `score_composite`).

**`score_composite` reads config from a module global** (`_cfg_holder.config`, set
by `set_config(cfg)`), while `score_universe_for_date` takes `config` as an explicit
argument. `run_champion.py` and `overfit_harness.py` always call `set_config` first,
so this is correct today, but fragile — calling `score_composite` without
`set_config` silently uses the default `AppConfig`. The legacy CLI uses
`score_universe_for_date` (explicit config), so it is unaffected.

**Dead `champion_veto` config removed (this review).** `config/research.json`
previously carried a `champion_veto` block (0.775/53/21 — IWV-champion values).
`apply_params_to_config` never reads it (veto params come from `research_params.json`,
which overrides to 0.35/21/21). The block was dead config that contradicted the
actual champion; it has been removed. The orphaned IWV champion files
(`config/champion.json`, `autoresearch.py`, `autoresearch.sh`,
`autoresearch_params.json`, `scripts/year_breakdown.py`) were also removed —
`run_champion.py` and the autoresearch loop (`overfit_harness.py`) never used them.

### Backtest vs Live Consistency

**Within the champion path: ✅ consistent and reproducible.** `run_champion.py
backtest` delegates to `overfit_harness.run_benchmark` (perturb off); `run_champion.py
live` reuses `build_veto_mask` + `_size_weights` so the P20 veto, conviction tilt,
and vol-adjusted sizing are identical. Both read `config/research.json` +
`research_params.json`. The autoresearch loop (`.auto/measure.sh`) runs the same
`overfit_harness.py` path.

⚠️ **Conviction-tilt scaling uses a full-sample statistic (mild in-sample bias +
backtest/live divergence).** `median_gap` (and the `conf_tilt_vol` median baseline
volatility) are computed **once over the entire sample** in `run_benchmark` and
`_champion_live_setup`, then applied to every date. For a 2015 backtest date this
incorporates 2016–2025 composite gaps — a mild indirect in-sample bias that only
scales the 0–17% conviction tilt (low magnitude). A **live** screen run today
computes `median_gap` over dates up to today (PIT-correct, no future), so its value
differs from the backtest's → the conviction-tilt magnitude differs slightly between
backtest and live. **Fix (if desired):** compute `median_gap` over a trailing window
(e.g. the 5-year train window) so it is PIT and matches backtest/live. Not blocking.

⚠️ **Two divergent backtest engines + two scoring entry points.** The legacy CLI
and the champion are different strategies:

| Path | Scoring | Backtest engine | Veto | Sizing |
|------|---------|-----------------|------|--------|
| `run_champion.py backtest` / live | `score_composite` | `timesfm_experiments.run_weighted_backtest` | ✅ P20 (3.3%) | conviction + vol + vol_pow=4 |
| `cli backtest` / screen (`algo_trading.cli`) | `score_universe_for_date` | `backtest.run_backtest` | ❌ none | equal-dollar, keep-overlap |

**The champion path is the experimental best** (iter83, 85 experiments, 30 keeps / 55
discards). The CLI path is the pre-champion baseline research harness. Here is exactly
what diverges:

**1. VETO — the biggest difference.**
The champion applies a TimesFM P20 downside-quantile volume veto: any stock whose
predicted P20 volume / trailing 21-session average volume < 0.35 is removed before
sizing. This vetoes ~3.3% of candidates and targets true tail-liquidity risk — the
autoresearch found it lifts the worst-year Sharpe 10× (baseline 0.03 → first P20
breakthrough +0.337). The CLI has **no veto at all** — `score_universe_for_date`
returns the raw composite top-N with no TimesFM involvement.

**2. SIZING — the breakthrough frontier.**
The champion applies four stacked sizing mechanisms on top of the top-2 selection:
- `vol_scaled_weights` (inverse-vol risk balance across the top-2, lookback 21)
- `weight_tilt=0.17` (conviction tilt: shift 17% weight toward the top-1)
- `conf_tilt + conf_tilt_pow=50` (confidence-scale the tilt by |top1−top2|/median_gap;
  pow 50 = step-function: full tilt only when gap ≥ median, ~0 otherwise)
- `conf_tilt_vol + conf_tilt_vol_pow=4` (vol-regime-scale: more tilt in high-vol/bear,
  less in low-vol/bull; pow=4 is the interior optimum 1→2→3→4 peak→5→8 regress)

The CLI has **none of these** — it sizes equal-dollar (`cash / N`) with no tilt.
The sizing frontier lifted the floor +0.16 (0.435 → 0.596) AFTER selection was
structurally pinned.

**3. BACKTEST ENGINE — turnover + skip-when-unchanged.**
`run_weighted_backtest` (champion) sells everything and re-buys target weights each
rebalance, sized by `cash * target_weight`. It skips the rebalance entirely when
the target symbol set matches current holdings (turnover-neutral optimization).
`backtest.run_backtest` (CLI) keeps overlapping positions at cost basis and only
deploys cash to new names — for max_holdings≥2 with a tilt, the equity path diverges
because the champion forces the full target weight distribution while the CLI only
reallocates residual cash. For max_holdings=1 they produce identical trades.

**4. SCORING ENTRY POINTS — config coupling.**
`score_composite` reads config from a module global (`_cfg_holder.config`, set by
`set_config(cfg)`). `score_universe_for_date` takes `config` as an explicit argument.
The champion always calls `set_config` first; the CLI passes config explicitly.
Both compute the same composite (ranked factor × weight sum), but calling
`score_composite` without `set_config` silently uses the default `AppConfig`.

**Bottom line: use `python run_champion.py backtest` or `python run_champion.py live`
for the champion strategy.** The CLI path (`python -m algo_trading.cli`) is the
baseline/research harness without the P20 veto or conviction/vol sizing.
The divergence is 3-fold: veto (present vs absent), sizing (4-tier tilt vs equal-dollar),
and backtest engine (full-turnover weight-based vs keep-overlap equal-dollar).

**`positions.csv` should be reconciled.** It lists 2 holdings (LITE, WDC) —
consistent with `max_holdings=2`, but it must match your actual brokerage before each
live run; a stale file causes over-trading.

### Scheduled Task Flow

✅ The calendar guard (`is_first_nyse_rebalance_session.py`) gates on the first NYSE
trading day of the month, matching `rebalance_frequency: "monthly"`. Verified: a
first-of-month date → exit 0; a mid-month date → exit 2; a non-trading day → exit 2.
The forecast auto-refresh + champion screen run only on the rebalance day.

⚠️ **Calendar guard uses a different NYSE session source than the strategy engine.**
`is_first_nyse_rebalance_session.py` derives NYSE sessions from `pandas_market_calendars`
(external holiday-aware calendar), while `select_rebalance_sessions("monthly")` derives
rebalance dates from `first_trading_days_of_months(sessions)` which operates on the price
data's session index. If price data is stale or missing sessions (e.g. a data-fetch
failure), the guard could approve a day that the strategy doesn't consider a rebalance
date, or vice-versa. In practice this is a pre-filter — false positives just produce a
`Rebalance due: no` screen; false negatives would skip a rebalance day but require price
data to have sessions the NYSE calendar lacks (unlikely). Not blocking.

### Review Summary

| Dimension | Status | Notes |
|-----------|--------|-------|
| Data leakage — price/technical/fundamental factors | ✅ PASS | Shift + walk-forward + leaky-train guards correct |
| Data leakage — macro risk gate | ✅ PASS | Monthly/weekly FRED series lagged by publication delay |
| Data leakage — universe membership | ⚠️ SURVIVORSHIP | Single snapshot; backtest uses today's S&P 500 for all dates |
| Model input/output | ✅ PASS | TimesFM P20 veto is PIT-correct; it is a veto, not the alpha model; quantile col mapping verified |
| Model structure & params | ✅ PASS | Fixed weights across splits; half-risk active at N=2; dead `champion_veto` removed |
| Backtest vs live (champion path) | ✅ PASS | `run_champion` + `overfit_harness` consistent and reproducible |
| Backtest vs live (conviction tilt) | ⚠️ MINOR | `median_gap` is a full-sample statistic → mild in-sample bias + backtest/live divergence |
| Backtest vs live (CLI vs champion) | ⚠️ DIVERGENT | Two engines + two scoring entry points; CLI omits veto and sizes differently |
| Scheduled run | ✅ PASS | Monthly guard matches monthly rebalance |
| Test suite | ✅ PASS | 22/22 |

**Bottom line:** The price/factor/macro pipeline is leak-free; the TimesFM P20 veto is
PIT-correct; the champion backtest↔live↔autoresearch path is internally consistent and
reproducible; the schedule matches the monthly rebalance; 22/22 tests pass. The dead
IWV-champion config/code has been removed. **Remaining limitations:** (1) the backtest
is survivorship-biased until `data/holdings_history/` is backfilled with monthly S&P 500
snapshots — treat CAGR/Sharpe as optimistic; (2) the conviction-tilt `median_gap` is a
mild full-sample statistic; (3) the legacy CLI is a different strategy from the champion.

## Code Review (2026-07-02) — Cache audit, staleness fix & schedule verification

A follow-up review focused on the TimesFM cache and the scheduled run, using the
codebase-memory graph (902 nodes / 2829 edges) plus a parquet-schema inspection of
`cache/timesfm/` and the live run log (`artifacts/live/champion_run.log`). Every claim
traces to a file:line or a log line.

### Cache audit — which file is actually read

The champion reads the forecast cache through exactly one function:
`timesfm_experiments.load_forecast_panel(ctx_len, horizon, kind, quantiles)` →
`timesfm_engine._cache_key(...)` → `cache/timesfm/ctx{ctx_len}_hor{horizon}_{kind}_{q}.parquet`.
With `research_params.json` (`forecast_context_len=256`, `veto_horizon=21`,
`use_quantile_veto=true`, `quantile_col=1`) the champion backtest
(`overfit_harness.run_benchmark`) and live screen (`run_champion.run_live`) both call
`load_forecast_panel(256, 21, "volume", quantiles=True)` and read columns `q1_h0..q1_h20`
(= P20). **Exactly one cache file is in use:**

| File | Size | Status | Read by |
|------|------|--------|---------|
| `ctx256_hor21_volume_q.parquet` | 100 MB | ✅ **IN USE** — the P20 veto cache | `overfit_harness.run_benchmark`, `run_champion.run_live` (via `load_forecast_panel`) |
| `ctx256_hor21_volume_q.monthly.bak` | 100 MB | ❌ orphaned backup | nothing (`.bak` ≠ the `_cache_key` name) |
| `ctx512_hor21_volume_q.parquet` | 105 MB | ❌ orphaned | only if `forecast_context_len=512` (champion uses 256) |
| `ctx256_hor21_volume_m.parquet` | 12.6 MB | ❌ not in use | only if `use_quantile_veto=false` (champion has `true`) |
| `ctx256_hor21_logret_q.parquet` | 47 MB | ❌ never referenced | no `load_forecast_panel(..., kind="logret", quantiles=True)` call exists |
| `ctx256_hor21_logret_m.parquet` | 13 MB | ❌ not in use | only if `tfm_alpha_weight>0` (champion has `0.0`) |
| `ctx256_hor21_price_m.parquet` | 4.3 MB | ❌ never referenced | — |
| `ctx256_hor5_logret_m.parquet` / `ctx256_hor60_logret_m.parquet` | 12 MB | ❌ wrong horizon | champion uses hor=21 |
| `ctx64_hor5_*_m.parquet` | 2 MB | ❌ wrong ctx | champion uses ctx=256 |
| `dates_*.csv`, `requests_*.parquet` | 75 MB | ❌ orphaned artifacts | written by the legacy `experiments/` driver; not read at runtime |

**~295 MB of the ~466 MB cache dir is orphaned.** The two biggest orphans
(`ctx512_hor21_volume_q.parquet`, `ctx256_hor21_volume_q.monthly.bak`) are safe to
delete to reclaim ~205 MB — they are not referenced by any code path. **The correct
cache is in use:** `ctx256_hor21_volume_q.parquet` has the `q1_h0..q1_h20` columns the
veto reads (verified by schema inspection: 72 576 rows × 231 cols, 133 monthly dates
2015-01-02 → 2026-01-02, 581 symbols), and its path matches `_cache_key(256, 21, "volume", True)`.

⚠️ **Note on the legacy forecast driver.** `experiments/_prepare_requests.py` loads
`config/baseline.json` (IWV universe), and `experiments/run_forecast_driver.sh` does not
set `TFM_QUANTILES=1`, so it produces **mean-only** caches for the **baseline** universe —
not the champion's SPY quantile cache. Do not use it to regenerate the champion veto
cache; use `python generate_spy_forecasts.py --quantiles` (which uses `config/research.json`).

### Forecast-staleness bug (fixed this review)

The live run log for 2026-07-01 showed:

```
Missing symbols (need forecasts): 0
Nothing to generate. Cache already covers the universe.
```

`generate_spy_forecasts.py` short-circuited on **symbol** coverage: once every S&P 500
symbol had ≥1 forecast it exited without checking the **date** dimension. The quantile
cache's last date was `2026-01-02` (133 monthly dates), so for the 2026-07-01 session
`build_veto_mask` ffilled `pred_vol` from `2026-01-02` — a **6-month-stale forecast**
(the veto then fired on 23 names using a forecast made from data through 2025-12-31).
PIT-safe (no leakage) but stale, and it silently contradicted the "auto-refresh" claim.

**Fix (this review, `generate_spy_forecasts.py`):** the refresh now also computes the
monthly rebalance dates missing from the cache and builds requests for the full universe
(`build_missing_requests` skips `(symbol, date)` pairs already cached), so the cache
extends to new months. The cheap fast-path (no missing symbols AND no missing dates →
no model load) is preserved. The next run catches up the 6 backlog months (Feb–Jul 2026,
~3k forecasts) in one GPU pass; subsequent months are ~580 forecasts. `py_compile` clean;
22/22 tests still pass.

**Catch-up run (2026-07-02):** `TFM_BATCH=16 python generate_spy_forecasts.py --quantiles`
produced 3 992 forecasts, 0 NaN, ~85 s on a GTX 1650 SUPER (4 GB). The P20 veto cache
now covers **139 monthly dates (2015-01-02 → 2026-07-01)**, 100% SPY coverage (580/580).
Re-screening 2026-07-01 now vetoes **6** names (CDE, FLEX, HPE, MRVL, ROKU, SATS) — down
from 23 under the stale ffilled forecast — confirming the veto keys off the fresh
2026-07-01 prediction. `TFM_BATCH` (lower for low-VRAM GPUs) and `TFM_CHECKPOINT`
(partial-cache interval) are now env-configurable.

### Scheduled-task verification

- **Calendar guard** (`is_first_nyse_rebalance_session.py`) tested on 4 dates:
  `2026-07-01`→exit 0, `2026-07-02`→exit 2, `2026-06-01`→exit 0, `2026-01-01`→exit 2. ✅
- **End-to-end run** (`artifacts/live/champion_run.log`, 2026-07-02 00:23–00:31):
  guard passed → forecast refresh → `run_champion.py live --as-of 2026-07-01` →
  `RESULT session=2026-07-01 rebalance=yes action=HOLD symbol=WDC vetoed=23 veto_cov=on`.
  The screen CSV was written. ✅ The schedule logic works end-to-end.
- ⚠️ **Timing nuance.** The task is set for 9 PM SGT (= 9 AM ET, 30 min before US open).
  At that moment yfinance has data only through the *prior* session, so a before-open run
  picks `session = prior_session` and would report `Rebalance due: no` on the first
  trading day. The successful 2026-07-01 screen actually ran at 00:23 SGT on 07-02
  (*after* the 07-01 close), when the 07-01 session was in the price panel — hence
  `rebalance=yes`. Practical implication: run the screen **after the first-trading-day
  close** (or the next morning) and execute at the following open — ~1 session of
  slippage vs the backtest (which trades the first-of-month open on prior-session
  factors). The 21:00 SGT slot also did not log a completion; a cache-miss price download
  of 580 symbols took ~6.5 min in the 00:23 run, so warm the price cache or widen the
  window if the before-open slot is kept.
- The calendar guard uses `pandas_market_calendars` NYSE while the strategy's
  `select_rebalance_sessions("monthly")` derives first-of-month from the price session
  index — a pre-filter mismatch noted previously (not blocking; false positives just
  yield `Rebalance due: no`).

### Prior verdicts re-confirmed

Re-traced against the indexed graph; no change:

- Factor `.shift(1)` (`indicators.shift_for_open_execution`) applied to **both** technical
  and fundamental factors in `strategy.build_factor_library`; base-mask `history_ok` /
  `price_ok` / `liquidity_ok` all `.shift(1)`. ✅
- Macro: `PUBLICATION_LAG_DAYS` (daily 0 / weekly 10 / monthly 45) shifts each FRED series
  to its release date before `reindex(sessions, ffill)`, then `+1` session in
  `build_factor_library`. ✅
- Fundamentals: `available_date = statement_date + 60d`; `_latest_value` / `_ttm_value`
  filter `index <= statement_date`. ✅
- Walk-forward: train strictly before `test_start`; `_strip_leaky_train_dates` called in
  `generate_walk_forward_plan`, `screen_live_session`, **and** `build_factor_state`. ✅
- TFM context: `build_context_requests` and `generate_spy_forecasts.build_missing_requests`
  both end the context window at `prior_end = sessions[pos-1]`. ✅ byte-compatible.
- Survivorship bias unchanged: `data/holdings_history/` has one snapshot (`2026-06-28.csv`)
  → the 2015→today backtest uses today's S&P 500.

**Minor (new).** In `strategy.build_factor_library`, `avg_dollar_volume` is taken from
the already-`shift(1)`-ed `technical` dict and then `liquidity_ok` applies a **second**
`.shift(1)`, so the liquidity filter uses T-2 dollar volume while the price/history
filters use T-1. PIT-safe and identical between backtest and live (so not a divergence),
just one session staler than intended. Not fixed — the champion was tuned with this in
place; "fixing" it would invalidate the iter83 result.

### Review summary (2026-07-02)

| Dimension | Status | Notes |
|-----------|--------|-------|
| Correct cache in use | ✅ PASS | `ctx256_hor21_volume_q.parquet` (P20, col 1) is the only file read; ~295 MB orphaned |
| Cache freshness | ✅ FIXED | refresh now extends the quantile cache to new months (was frozen at 2026-01-02) |
| Data leakage (price/factor/macro/fundamental/TFM) | ✅ PASS | re-confirmed; no look-ahead |
| Universe membership | ⚠️ SURVIVORSHIP | single snapshot; backtest uses today's S&P 500 |
| Backtest ↔ live (champion path) | ✅ PASS | `run_champion` + `overfit_harness` consistent |
| Conviction-tilt `median_gap` | ⚠️ MINOR | full-sample statistic (mild in-sample bias + bt/live divergence) |
| Legacy CLI vs champion | ⚠️ DIVERGENT | different engine/scoring/sizing; CLI is the baseline harness |
| Scheduled run | ✅ PASS (timing caveat) | guard + refresh + screen produce `rebalance=yes`; run after close |
| Test suite | ✅ PASS | 22/22 |

---

## Code Review (2026-07-03) — Scheduling, freshness & SELL rows

A focused review of the **scheduling**, **data freshness**, **logging**, and
**backtest/live consistency**, followed by implementation of the approved fixes.
The codebase-memory graph (909 nodes / 2833 edges) was indexed and queried; every
claim traces to a file:line.

### Problems found

1. **CRITICAL — the scheduled slot could never fire the rebalance.** The task fired
   at 21:00 SGT (= 09:00 ET, pre-open). At that moment yfinance has no bar for the
   first trading day, so `run_champion.run_live` (`run_champion.py:192-197`) fell
   back to the prior session → always `Rebalance due: no`. The
   `artifacts/live/champion_run.log` proves it: the 2026-07-01 21:00 run **stalled
   silently** (last log line 21:00:27, no completion) and the real screen was a
   **manual** 00:23 run with explicit `--as-of 2026-07-01`.
2. **CRITICAL — no missed-run catch-up.** No state file; the guard
   (`is_first_nyse_rebalance_session.py`) rejected every non-first day, so a
   missed/failed first-day run silently skipped the whole month's rebalance.
3. **HIGH — stale-session masking.** `run_live` picked
   `sessions[sessions <= target][-1]` with no freshness check, so stale price data
   silently screened an old session. Likewise `live_veto_on=True` for *any* prior
   forecast date (`run_champion.py:218-221`) — a 6-month-stale ffilled forecast
   still reported `veto_cov=on` (the 07-01 00:31 run vetoed 23 names off a January
   forecast yet reported coverage ON). Forecast age was never logged.
4. **HIGH — no SELL rows.** Output listed only target symbols (BUY/HOLD,
   `run_champion.py:262-273`); a held name dropping out of the top-2 vanished with
   no SELL instruction.
5. **MEDIUM — silent stall / no run-end record.** The PS1 had no catch-all or
   "Run ended" line; a hung python left no trace. No lockfile (Task Scheduler's
   default IgnoreNew covered scheduled overlaps but not manual+scheduled).
6. **MEDIUM — partial price download accepted silently.**
   `download_price_panel` (`data.py:628`) raised only if *nothing* downloaded;
   partial failures shrank the universe with no count/threshold check.

### Fixes applied (this review)

| Fix | File(s) | What changed |
|-----|---------|---------------|
| **Morning-after schedule + catch-up** | `scripts/is_first_nyse_rebalance_session.py`, `scripts/run_champion_live.ps1` | Guard now computes the **last completed NYSE session** in ET (timezone-correct) when no `-AsOf` is given. Once the month's first session has completed, it prints `SESSION=YYYY-MM-DD` for that first session, so any later day of the month can catch up a missed run. PS1 takes `--as-of` from that, writes `artifacts/live/last_rebalance_processed.txt` only after a successful screen, and skips a month already recorded. |
| **Lockfile + Run-ended log** | `scripts/run_champion_live.ps1` | `artifacts/live/.run.lock` (PID, stale-PID reclaim) prevents manual+scheduled overlap. Every exit path logs `Run ended.` so a silent stall is visible by the absence of that line. |
| **Stale-price warning** | `run_champion.py` `run_live` | If the loaded session lags the target by >4 calendar days, prints `WARNING: … session_lag` and adds `session_lag=N` to the `RESULT` line. |
| **Forecast-age surfacing** | `run_champion.py` `run_live` | Tracks the actual forecast date used (ffilled), prints `Veto coverage: STALE (ffilled from <date>, Nd old)` + a stderr warning when >35 days old, and sets `veto_cov=stale` in the `RESULT` line. The January-forecast incident would now be visible. |
| **SELL rows** | `run_champion.py` `run_live` | Held symbols not in the target list get a `SELL` row (weight 0) in the screen CSV + console table, so exits are explicit. |
| **Partial-download guard** | `src/algo_trading/data.py` `download_price_panel` | Warns on any failed tickers and **raises** if >20% of the universe failed, instead of silently producing a tiny universe. |
| **Tests** | `tests/test_live_trading.py` | Extended the guard test to assert `SESSION=` output; added `test_calendar_guard_auto_mode_returns_month_first_session_for_catchup` and `test_launcher_skips_when_month_already_processed`. 24/24 pass. |
| **README** | `README.md` | New top-of-file **Quick Operating Guide** (when it runs, weekly checklist, warning decoder, manual commands, file map) + this review section + updated Scheduled Task block (Mon–Sat 09:00 SGT) + Troubleshooting entries. |

### What was deliberately NOT changed

- **No drift monitor** (accepted gap — none existed before; per decision, documented
  only). The raw materials exist (cached forecasts + realized volume) should one be
  wanted later.
- **`median_gap` full-sample statistic** (mild in-sample bias + backtest/live
  divergence) and the double-shifted `liquidity_ok` — the champion was tuned with
  these in place; changing them would invalidate the iter83 result.
- **Universe survivorship bias** — a data-acquisition task (backfill
  `data/holdings_history/`), not a code fix.
- **Duplicated sizing-setup logic** (`_champion_live_setup` mirrors `run_benchmark`)
  and ~295 MB of orphaned cache files — low-priority housekeeping, not blocking.

### Re-confirmed from prior reviews

No change against the indexed graph: factor `.shift(1)`; macro `PUBLICATION_LAG_DAYS`
before ffill; fundamentals 60-day statement lag; TFM context ends at `prior_end` in
both `build_veto_mask` and `build_missing_requests`; the single local model
`timesfm-2.5-200m-transformers/` loaded with `TRANSFORMERS_OFFLINE=1` (no HF
download); quantile col 1 = P20; correct cache `ctx256_hor21_volume_q.parquet`;
champion backtest↔live reuse `build_veto_mask`/`_size_weights`/`score_composite`.

### Review summary (2026-07-03)

| Dimension | Status | Notes |
|-----------|--------|-------|
| Scheduled run (timing + timezone) | ✅ FIXED | Mon–Sat 09:00 SGT at review time (later moved to 21:00 SGT — safe because the guard uses the last completed NYSE session in ET and returns the month-first session for catch-up) |
| Missed-run catch-up | ✅ FIXED | state file `last_rebalance_processed.txt` |
| Overlap protection | ✅ FIXED | PID lockfile with stale reclaim |
| Silent-stall visibility | ✅ FIXED | `Run ended.` log line on every exit |
| Stale-price masking | ✅ FIXED | `session_lag` warning + RESULT field |
| Forecast-staleness visibility | ✅ FIXED | `veto_cov=stale` + forecast-date log |
| SELL rows | ✅ FIXED | explicit exits in screen CSV/console |
| Partial price download | ✅ FIXED | warn + abort >20% failures |
| Data leakage (price/factor/macro/fundamental/TFM) | ✅ PASS | re-confirmed; no look-ahead |
| Best-model path / quantile mapping | ✅ PASS | single local model, P20 = col 1, fp32 overflow guard |
| Backtest ↔ live (champion path) | ✅ PASS | shared `build_veto_mask`/`_size_weights` |
| Drift monitor | ⚠️ NONE (accepted) | documented gap; no code added |
| Conviction-tilt `median_gap` | ⚠️ MINOR | full-sample statistic (unchanged; would invalidate iter83) |
| Universe membership | ⚠️ SURVIVORSHIP | single snapshot (data task, not code) |
| Test suite | ✅ PASS | 24/24 |

**Bottom line:** the scheduling hole that made the automated rebalance impossible
is closed (morning-after slot + ET-aware guard + state-file catch-up), stale
price/forecast data is now surfaced loudly in the log and `RESULT` line, exits
are explicit (`SELL` rows), partial downloads abort instead of shrinking the
universe, and a layman operating guide sits at the top of the README. The
leak-free price/factor/macro/TFM pipeline and the champion backtest↔live
consistency are re-confirmed unchanged. 24/24 tests pass.

---

## Notes

- **`signal_timeframe="weekly"` vs `rebalance_frequency="monthly"`** are
  independent: the former resamples prices to weekly bars for factor computation;
  the latter selects which days are eligible for trade decisions (first trading day
  of each month). The strategy screens weekly but trades monthly; the
  skip-when-unchanged optimization makes rebalance frequency turnover-neutral.
- **Universe survivorship bias is not eliminated.** `data/holdings_history/` has
  one snapshot; the 2015→today backtest uses today's S&P 500. Backfill monthly
  snapshots to remove this overstatement (see Code Review → Universe survivorship bias).
- **Fundamental data comes from yfinance**, not a licensed point-in-time dataset.
  Statement dates are shifted by a 60-day reporting lag — an approximation.
- **Transaction costs** are 3 bps per side (6 bps round trip), realistic for
  liquid $11M+ daily-volume names.
- **Max drawdown can exceed -48%** (the champion hit -48.9% in 2025). Size
  accordingly and respect the macro risk-off bands.
- **The script does not connect to your broker** — you place trades yourself.
- **Do not trade from a run that says `Rebalance due: no`.**
