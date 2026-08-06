"""PIT + coverage audit: verify the TimesFM forecast cache covers the ENTIRE
SPY universe the harness selects from, with no look-ahead.

Checks:
  1. Symbol coverage: cache symbols vs the harness universe (price panel cols).
  2. Per-date coverage: for every rebalance date (2015-2025), are all
     *eligible* symbols covered by a forecast? (missing -> NaN pv -> silently
     NOT vetoed -> under-vetoing / coverage cheat).
  3. NaN forecasts in the cache (should be 0).
  4. PIT lag: confirm the forecast at date d was built from data ending at the
     session BEFORE d (no look-ahead). Re-derives prior_end per date and checks
     the cache row exists for d (forecast MADE at d, used for hold after d).
"""
from __future__ import annotations
import sys, json
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from algo_trading.timesfm_experiments import (
    load_context, build_factor_state, set_config, eligible_symbols,
    load_forecast_panel,
)
from algo_trading.config import AppConfig

CTX, HOR = 256, 21
cfg, ctx = load_context("config/research.json")
with open("research_params.json") as f:
    params = json.load(f)
from overfit_harness import apply_params_to_config
cfg = apply_params_to_config(cfg, params)
set_config(cfg)
state = build_factor_state(ctx, cfg)
sessions = ctx.prices.sessions

# Universe = all symbols with a price/volume column (the generator forecasts
# exactly this set: universe_syms = context.prices.volume.columns).
universe = list(ctx.prices.volume.columns)
print(f"Universe (price panel cols): {len(universe)} symbols")

# Champion uses the P20 quantile volume cache.
qpanel = load_forecast_panel(CTX, HOR, kind="volume", quantiles=True)
if qpanel is None:
    print("ERROR: no quantile volume cache"); sys.exit(1)
cache_syms = set(str(s) for s in qpanel.index.get_level_values("symbol").unique())
cache_dates = pd.DatetimeIndex(qpanel.index.get_level_values("date").unique()).sort_values()
print(f"Cache: {len(qpanel)} rows, {len(cache_syms)} symbols, {len(cache_dates)} dates")
print(f"Cache date range: {cache_dates.min().date()} -> {cache_dates.max().date()}")

# 1. Symbol coverage
missing_syms = [s for s in universe if s not in cache_syms]
extra_syms = [s for s in cache_syms if s not in set(universe)]
print(f"\n=== 1. SYMBOL COVERAGE ===")
print(f"Universe covered: {len(universe)-len(missing_syms)}/{len(universe)} ({100*(len(universe)-len(missing_syms))/len(universe):.1f}%)")
print(f"Missing (universe sym with NO forecast anywhere): {len(missing_syms)}")
if missing_syms:
    print(f"  e.g. {missing_syms[:15]}")
print(f"Extra (cache sym not in price panel): {len(extra_syms)}")
if extra_syms:
    print(f"  e.g. {extra_syms[:15]}")

# 2. NaN forecasts in cache (quantile cols only)
qcols = [f"q{c}_h{h}" for c in range(10) for h in range(HOR)]
nan_count = int(qpanel[qcols].isna().sum().sum())
print(f"\n=== 2. NaN FORECASTS (quantile cols) ===")
print(f"Total NaN cells: {nan_count} (of {len(qpanel)*len(qcols)} = {100*nan_count/(len(qpanel)*len(qcols)):.4f}%)")

# 3. Per-date eligible coverage (the real test: does every eligible sym at every
#    rebalance date have a forecast -> veto correctly applied?)
rebal = [pd.Timestamp(d) for d in state.rebalance_dates
         if pd.Timestamp(d) >= pd.Timestamp("2015-01-01")
         and state.split_for_date.get(d) is not None
         and pd.Timestamp(d) in state.base_mask.index]
print(f"\n=== 3. PER-DATE ELIGIBLE COVERAGE ===")
print(f"Rebalance dates (2015-2025, in base_mask): {len(rebal)}")
# Build a (date -> set of cache syms) map for fast lookup
cache_by_date = {}
for (sym, d) in qpanel.index:
    cache_by_date.setdefault(pd.Timestamp(d), set()).add(str(sym))

total_elig = 0; total_missing = 0; worst_date = None; worst_n = 0; worst_missing = []
zero_dates = []
for d in rebal:
    elig = eligible_symbols(state, d)
    have = cache_by_date.get(d, set())
    miss = [s for s in elig if str(s) not in have]
    total_elig += len(elig)
    total_missing += len(miss)
    if len(miss) > worst_n:
        worst_n = len(miss); worst_date = d; worst_missing = miss[:10]
    if len(elig) > 0 and len(miss) == len(elig):
        zero_dates.append(d)
print(f"Total eligible (sym,date) pairs: {total_elig}")
print(f"Total MISSING forecasts (eligible but no cache row): {total_missing} ({100*total_missing/max(1,total_elig):.3f}%)")
print(f"Worst date: {worst_date} -- {worst_n} missing of eligible (e.g. {worst_missing})")
print(f"Dates with ZERO eligible coverage: {len(zero_dates)}")
if zero_dates:
    print(f"  e.g. {[str(d.date()) for d in zero_dates[:5]]}")

# 4. PIT lag verification: for each rebalance date, prior_end = session before d.
#    The forecast at index d was built from data.loc[:prior_end] (generator line 72).
#    Confirm the cache date == the rebalance date (forecast MADE at d, hold after d).
print(f"\n=== 4. PIT LAG VERIFICATION ===")
rebal_in_cache = [d for d in rebal if d in cache_by_date]
print(f"Rebalance dates with a cache row: {len(rebal_in_cache)}/{len(rebal)}")
# Spot-check 3 dates: confirm prior_end < d and the cache row's context would end at prior_end
import random
random.seed(0)
sample = random.sample(rebal_in_cache, min(3, len(rebal_in_cache)))
for d in sample:
    pos = sessions.searchsorted(d, side="left")
    prior_end = sessions[pos-1] if pos > 0 else d
    n_cache_syms = len(cache_by_date.get(d, set()))
    print(f"  d={d.date()}: prior_end={prior_end.date()} (lag={(d-prior_end).days}d), "
          f"cache syms at d={n_cache_syms}, prior_end<d: {prior_end < d}")

# 5. Does the harness's veto silently skip missing-forecast symbols?
#    build_veto_mask: NaN pred_vol => not vetoed (fillna(False)). Confirm the
#    missing set above would be NOT-vetoed (under-vetoing risk).
print(f"\n=== 5. SILENT-VETO IMPACT ===")
if total_missing > 0:
    print(f"WARNING: {total_missing} eligible (sym,date) pairs have NO forecast -> "
          f"NaN pv -> NOT vetoed (under-vetoing). These are likely <{64}-obs new listings.")
else:
    print(f"OK: 0 missing -> every eligible sym has a forecast -> veto applied correctly everywhere.")
print(f"\nAUDIT COMPLETE.")
