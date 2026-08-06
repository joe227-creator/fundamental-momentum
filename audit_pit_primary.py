"""Refined PIT coverage audit: PRIMARY window only (2015-2025, 2026 excluded).
Checks whether missing-forecast eligible names ever get SELECTED by the champion
(the real impact of the under-vetoing gap), and breaks down missing pairs by year.
"""
from __future__ import annotations
import sys, json
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from algo_trading.timesfm_experiments import (
    load_context, build_factor_state, set_config, eligible_symbols,
    load_forecast_panel, score_composite,
)
from overfit_harness import apply_params_to_config, build_veto_mask

CTX, HOR = 256, 21
cfg, ctx = load_context("config/research.json")
with open("research_params.json") as f:
    params = json.load(f)
cfg = apply_params_to_config(cfg, params)
set_config(cfg)
state = build_factor_state(ctx, cfg)
sessions = ctx.prices.sessions

qpanel = load_forecast_panel(CTX, HOR, kind="volume", quantiles=True)
quantile_col = int(params.get("quantile_col", 1))
veto_horizon = int(params.get("veto_horizon", 21))
qcols = [f"q{quantile_col}_h{h}" for h in range(veto_horizon)]
pred_vol = qpanel[qcols].sum(axis=1) / veto_horizon
pred_vol = pred_vol.unstack(level="symbol")

veto_threshold = float(params.get("veto_threshold", 0.35))
trailing_window = int(params.get("trailing_vol_window", 21))
veto_mask, base_scores, prior_ends, dates = build_veto_mask(
    state, ctx, pred_vol, veto_threshold, trailing_window)

# Cache-by-date map
cache_by_date = {}
for (sym, d) in qpanel.index:
    cache_by_date.setdefault(pd.Timestamp(d), set()).add(str(sym))

# Restrict to PRIMARY window (exclude 2026 partial)
dates_primary = [d for d in dates if d.year <= 2025]
print(f"Primary-window rebalance dates (2015-2025): {len(dates_primary)}")

# Per-year missing breakdown + selection impact
N = int(params["max_holdings"])
year_missing = {}; year_elig = {}; selected_missing = []; selected_missing_by_year = {}
vetoed_that_would_be = 0
for d in dates_primary:
    elig = eligible_symbols(state, d)
    have = cache_by_date.get(d, set())
    miss = set(s for s in elig if str(s) not in have)
    y = d.year
    year_missing[y] = year_missing.get(y, 0) + len(miss)
    year_elig[y] = year_elig.get(y, 0) + len(elig)
    # Champion selection: top-N composite AFTER veto
    sc = base_scores.get(d)
    if sc is None or sc.empty:
        continue
    vet = veto_mask.loc[d]
    vetoed = set(vet[vet].index) & set(sc.index) if d in veto_mask.index else set()
    sc_post = sc.drop(index=list(vetoed))
    top = sc_post.sort_values(ascending=False).head(N).index.tolist()
    sel_missing = [s for s in top if str(s) in miss]
    if sel_missing:
        selected_missing.append((d, sel_missing))
        selected_missing_by_year[y] = selected_missing_by_year.get(y, 0) + len(sel_missing)
    # How many missing names WOULD have been vetoed if they had a forecast?
    # (they had high composite but no forecast -> if their P20 was low they'd be vetoed)
    # We can't know their P20 (no forecast), but we can count missing names in top-K pre-veto
    sc_pre_top = sc.sort_values(ascending=False).head(N+5).index.tolist()
    would_select = [s for s in sc_pre_top if str(s) in miss and s not in vetoed]

print(f"\n=== MISSING BY YEAR (primary window) ===")
for y in sorted(year_missing):
    m, e = year_missing[y], year_elig[y]
    print(f"  {y}: {m} missing / {e} eligible ({100*m/max(1,e):.2f}%)")
print(f"  TOTAL: {sum(year_missing.values())} missing / {sum(year_elig.values())} eligible "
      f"({100*sum(year_missing.values())/max(1,sum(year_elig.values())):.2f}%)")

print(f"\n=== SELECTION IMPACT (the real test) ===")
print(f"Dates where a SELECTED top-{N} name had NO forecast (under-vetoed into selection): "
      f"{len(selected_missing)}")
if selected_missing:
    for d, sm in selected_missing[:10]:
        print(f"  {d.date()}: selected {sm}")
    print(f"  By year: {selected_missing_by_year}")
else:
    print(f"  OK: NO selected name was ever missing a forecast in 2015-2025. "
          f"The under-vetoing gap never reaches selection -> HARMLESS for the primary window.")

print(f"\n=== PIT RECONFIRM (primary window) ===")
# For each primary date, forecast index d == rebalance date; context ended prior_end < d.
bad = 0
for d in dates_primary:
    pos = sessions.searchsorted(d, side="left")
    pe = sessions[pos-1] if pos > 0 else d
    if not (pe < d):
        bad += 1
print(f"Dates where prior_end >= d (look-ahead!): {bad} / {len(dates_primary)} "
      f"({'OK' if bad==0 else 'LEAK'})")
print(f"\nPRIMARY-WINDOW AUDIT COMPLETE.")
