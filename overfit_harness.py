"""Overfit-aware research harness for the IWV rotation strategy.

Reproduces the champion backtest path (load_context -> build_factor_state ->
score_composite + TimesFM volume veto -> run_weighted_backtest) but with a
**vectorized veto** (precomputed trailing-volume / predicted-volume ratio
matrix) that cuts the per-iteration cost from ~140s to ~30s without changing
any result. Adds:

  * per-calendar-year Sharpe/CAGR/MaxDD (the robustness lens the champion
    README reports by hand);
  * score-perturbation runs (the deterministic-strategy analog of multi-seed
    training) -> mean +/- std of the key metrics, the core overfitting signal;
  * a single primary metric `worst_year_sharpe` (maximize the floor across
    full years 2015-2025) plus secondary robustness/return monitors.

Primary metric: worst_year_sharpe (higher is better)
Secondary: overall sharpe/cagr/calmar/maxdd/sortino, mean/std annual sharpe,
           robust_score=mean-0.5*std, perturbation stds, per-year sharpes.

CLI:
  python overfit_harness.py --config config/research.json --params research_params.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")

from algo_trading.timesfm_experiments import (
    load_context, build_factor_state, set_config,
    run_weighted_backtest, extended_metrics,
    benchmark_close, score_composite, top_n_equal_weights,
    eligible_symbols, load_forecast_panel,
)
from algo_trading.timesfm_engine import mean_forecast_sum
from algo_trading.config import AppConfig

CTX, HOR = 256, 21
START_DATE = pd.Timestamp("2015-01-02")


# ---------------------------------------------------------------------------
# Parameter application (superset of autoresearch.apply_params_to_config so the
# champion params reproduce the champion backtest bit-for-bit).
# ---------------------------------------------------------------------------

def apply_params_to_config(cfg: AppConfig, p: dict) -> AppConfig:
    cfg.strategy.max_holdings = int(p["max_holdings"])
    cfg.strategy.require_positive_composite = bool(p["require_positive_composite"])
    cfg.strategy.macro_half_risk_threshold = float(p["macro_half_risk_threshold"])
    cfg.strategy.macro_flat_threshold = float(p["macro_flat_threshold"])
    cfg.strategy.base_filter_requires_trend = bool(p["base_filter_requires_trend"])
    if "allow_cash_when_risk_off" in p:
        cfg.strategy.allow_cash_when_risk_off = bool(p["allow_cash_when_risk_off"])
    if "rebalance_frequency" in p:
        cfg.strategy.rebalance_frequency = str(p["rebalance_frequency"])
    if "close_positions_before_new_positions" in p:
        cfg.strategy.close_positions_before_new_positions = bool(p["close_positions_before_new_positions"])
    if "full_turnover_rebalance" in p:
        cfg.strategy.full_turnover_rebalance = bool(p["full_turnover_rebalance"])

    if "min_avg_dollar_volume" in p:
        cfg.universe.min_avg_dollar_volume = float(p["min_avg_dollar_volume"])
    if "min_history_days" in p:
        cfg.universe.min_history_days = int(p["min_history_days"])
    if "min_price" in p:
        cfg.universe.min_price = float(p["min_price"])
    if "macro_zscore_window" in p:
        cfg.macro.zscore_window = int(p["macro_zscore_window"])
    if "statement_lag_days" in p:
        cfg.universe.statement_lag_days = int(p["statement_lag_days"])
    if "universe_limit" in p:
        cfg.universe.universe_limit = int(p["universe_limit"])

    cfg.indicators.sma_fast = int(p["sma_fast"])
    cfg.indicators.sma_slow = int(p["sma_slow"])
    cfg.indicators.sma_trend = int(p.get("sma_trend", 201))
    cfg.indicators.ema_fast = int(p.get("ema_fast", 21))
    cfg.indicators.ema_slow = int(p.get("ema_slow", 21))
    cfg.indicators.ts_mom_lookback_months = int(p["ts_mom_lookback_months"])
    cfg.indicators.ts_mom_skip_months = int(p["ts_mom_skip_months"])
    if "enable_indicators" in p:
        for name, enabled in p["enable_indicators"].items():
            if name in cfg.indicators.enabled_indicators:
                cfg.indicators.enabled_indicators[name] = bool(enabled)

    cfg.walk_forward.fundamental_group_weight = float(p["fundamental_group_weight"])
    fw = dict(cfg.walk_forward.fundamental_factor_weights)
    fw["asset_growth"] = float(p["asset_growth_weight"])
    fw["earnings_yield"] = float(p["earnings_yield_weight"])
    for key, field_name in [
        ("cash_flow_yield_weight", "cash_flow_yield"),
        ("book_to_market_weight", "book_to_market"),
        ("gross_profitability_weight", "gross_profitability"),
        ("roe_weight", "roe"),
        ("roa_weight", "roa"),
        ("investment_to_assets_weight", "investment_to_assets"),
        ("net_issuance_weight", "net_issuance"),
        ("accruals_weight", "accruals"),
    ]:
        if key in p:
            fw[field_name] = float(p[key])
    cfg.walk_forward.fundamental_factor_weights = fw
    cfg.walk_forward.use_indicator_ic_weights = bool(p["use_indicator_ic_weights"])
    cfg.walk_forward.indicator_weighting_scheme = str(p["indicator_weighting_scheme"])
    if "train_years" in p:
        cfg.walk_forward.train_years = int(p["train_years"])
    if "step_months" in p:
        cfg.walk_forward.step_months = int(p["step_months"])
    if "min_train_rebalances" in p:
        cfg.walk_forward.min_train_rebalances = int(p["min_train_rebalances"])
    return cfg


# ---------------------------------------------------------------------------
# Vectorized veto precomputation (exact equivalent of the champion per-date loop)
# ---------------------------------------------------------------------------

def build_veto_mask(
    state, ctx, pred_vol: pd.DataFrame, veto_threshold: float, trailing_window: int,
    use_relative_veto: bool = False, relative_veto_pct: float = 0.05,
) -> tuple[pd.DataFrame, dict[pd.Timestamp, pd.Series]]:
    """Return (veto_mask[dates x symbols], base_scores{d: pre-veto composite}).

    veto_mask.loc[d, sym] == True  <=>  champion would veto sym at d.
    Equivalent to:
        prior_end = sessions[pos-1]
        tv = volume[sym].loc[:prior_end].tail(window).mean()
        pv = pred_vol.loc[d, sym]
        ratio = pv / tv ; veto if ratio < veto_threshold  (ABSOLUTE, default)
        OR if use_relative_veto: veto if ratio is in the bottom `relative_veto_pct`
        fraction of the cross-section each date (RELATIVE, regime-adaptive -- the
        cutoff tracks the volume-distribution shift across bull/bear markets).
    NaN/<=0 trailing vol or NaN predicted vol => not vetoed (matches champion).
    """
    sessions = ctx.prices.sessions
    volume = ctx.prices.volume

    dates = [
        pd.Timestamp(d) for d in state.rebalance_dates
        if pd.Timestamp(d) >= pd.Timestamp("2015-01-01")
        and state.split_for_date.get(d) is not None
    ]
    dates = [d for d in dates if d in state.base_mask.index]

    # Trailing average volume over the last `trailing_window` sessions
    # (NaN-skipped, min_periods=1) -- vectorized over the whole panel.
    vol_trail = volume.rolling(trailing_window, min_periods=1).mean()

    prior_ends = []
    for d in dates:
        pos = sessions.searchsorted(d, side="left")
        prior_ends.append(sessions[pos - 1] if pos > 0 else d)

    # tv at prior_end for each date (rows align with `dates` order)
    tv_rows = vol_trail.reindex(prior_ends)
    tv_rows.index = pd.DatetimeIndex(dates)
    # predicted vol at each rebalance date. Forward-fill month-end forecasts to
    # intra-month rebalance dates (weekly/biweekly) -- PIT-safe: the month-end
    # forecast was known at month-end, so using it for subsequent intra-month dates
    # is leak-free. No-op for monthly rebalance (dates == cache dates -> exact
    # match, no ffill needed). Without this, weekly rebalance + monthly cache
    # yields NaN pred_vol for non-month-end weeks -> veto silently OFF (the
    # iter36 confound). pred_vol.index is monotonic (cache written in date order).
    pv_rows = pred_vol.reindex(dates, method="ffill")
    ratio = pv_rows.div(tv_rows)
    if use_relative_veto:
        # Relative (cross-sectional percentile) veto: each date, veto the bottom
        # `relative_veto_pct` fraction of symbols by P20 ratio. Regime-adaptive --
        # the cutoff tracks the volume distribution shift across bull/bear markets,
        # so it may veto a different (potentially better) set in 2022 than the
        # fixed absolute 0.35 threshold.
        cutoff = ratio.quantile(relative_veto_pct, axis=1)
        veto_mask = ratio.lt(cutoff, axis=0).fillna(False)
    else:
        veto_mask = (ratio < veto_threshold).fillna(False)

    # Precompute base composite scores (pre-veto) once, reused by perturbation.
    base_scores: dict[pd.Timestamp, pd.Series] = {}
    for d in dates:
        sc = score_composite(state, d)
        if not sc.empty:
            base_scores[d] = sc
    return veto_mask, base_scores, prior_ends, dates


def _size_weights(scores: pd.Series, N: int, d: pd.Timestamp,
                  vol_prior_frame: pd.DataFrame | None, vol_scaled: bool,
                  weight_tilt: float = 0.0, conf_tilt: bool = False,
                  median_gap=None, conf_tilt_pow: float = 1.0,
                  conf_tilt_vol: bool = False, vol_factor_map=None,
                  liquidity_ratio_frame: pd.DataFrame | None = None,
                  liquidity_power: float = 0.0) -> dict[str, float]:
    """Top-N sizing. Equal weight by default; inverse-vol weight across the
    top-N when vol_scaled (risk-balanced sizing, targets drawdown reduction).
    If weight_tilt>0: shift that fraction of weight from the lower-ranked names
    to the top-1 (conviction tilt toward the strongest composite score). No-op
    when 0."""
    s = scores.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty or N <= 0:
        return {}
    top = s.sort_values(ascending=False).head(N).index.tolist()
    if not top:
        return {}
    if vol_scaled and vol_prior_frame is not None and len(top) > 1 and d in vol_prior_frame.index:
        vols = vol_prior_frame.loc[d].reindex(top)
        inv = 1.0 / vols.clip(lower=0.001).replace([np.inf, -np.inf], np.nan).fillna(0.001)
        tot = float(inv.sum())
        w = {sym: float(inv[sym] / tot) for sym in top} if tot > 0 else {sym: 1.0 / len(top) for sym in top}
    else:
        w = {sym: 1.0 / len(top) for sym in top}
    if weight_tilt > 0 and len(top) >= 2:
        # shift weight_tilt from the lower-ranked names to the top-1 (conviction).
        # conf_tilt: scale the tilt by the composite-gap confidence (|top1-top2| /
        # median_gap, capped at 1) -- tilt fully when the top-1 is a clear winner
        # (large gap, e.g. 2022 energy), tilt less when it's a close call (small gap,
        # e.g. 2015 choppy). A CONFIDENCE signal (not a return forecast). PIT-safe.
        # No-op when conf_tilt=False or median_gap is None.
        eff = weight_tilt
        if conf_tilt and median_gap and median_gap > 0:
            gap = abs(float(s.loc[top[0]]) - float(s.loc[top[1]]))
            eff = min(weight_tilt, weight_tilt * (gap / median_gap) ** conf_tilt_pow)
            if conf_tilt_vol and vol_factor_map is not None:
                # vol-adjusted: scale the tilt by the market-vol regime (more tilt in
                # high-vol/bear periods like 2022, less in low-vol/bull like 2017).
                # Protects the 2017 canary (low vol) and may allow a higher weight_tilt.
                eff = min(1.0, eff * vol_factor_map.get(d, 1.0))
        if eff > 0:
            per = eff / (len(top) - 1)
            w[top[0]] = min(1.0, w[top[0]] + eff)
            for sym in top[1:]:
                w[sym] = max(0.0, w[sym] - per)
    if liquidity_power > 0 and liquidity_ratio_frame is not None and d in liquidity_ratio_frame.index:
        ratios = liquidity_ratio_frame.loc[d].dropna()
        median_ratio = float(ratios.median()) if not ratios.empty else np.nan
        selected_ratios = ratios.reindex(top)
        if np.isfinite(median_ratio) and median_ratio > 0:
            normalized = (selected_ratios / median_ratio).clip(0.25, 4.0)
            adjusted = {
                sym: weight * float(normalized.get(sym, 1.0) ** liquidity_power)
                for sym, weight in w.items()
            }
            total = float(sum(adjusted.values()))
            target_total = float(sum(w.values()))
            if total > 0:
                w = {sym: value * target_total / total for sym, value in adjusted.items()}
    return w


def _apply_corr_filter(sc, d, threshold, rets_panel, prior_map, sessions):
    """Correlation diversification (optional, PIT-safe). Pick the top-1 score, then
    restrict to candidates whose trailing 63-session return correlation to the top-1
    is <= threshold (negative correlation is KEPT -- it diversifies). Falls back to
    the original scores if no uncorrelated candidate exists. Uses returns up to
    prior_end only (no look-ahead). For N==2: returns top-1 + best uncorrelated."""
    if len(sc) < 2 or threshold <= 0 or prior_map is None or sessions is None:
        return sc
    pe = prior_map.get(d)
    if pe is None or pe not in rets_panel.index:
        return sc
    pos = sessions.searchsorted(d, side="left")
    win_start = sessions[max(0, pos - 63)]
    r = rets_panel.loc[win_start:pe]
    if len(r) < 20:
        return sc
    top1 = sc.idxmax()
    if top1 not in r.columns:
        return sc
    top1_ret = r[top1]
    if top1_ret.dropna().std() == 0:
        return sc
    cands = sc.drop(top1).index.intersection(r.columns)
    if len(cands) == 0:
        return sc
    corrs = r[cands].corrwith(top1_ret)
    uncorrelated = corrs[corrs <= threshold].index
    if len(uncorrelated) == 0:
        return sc  # all correlated -> keep original top-N (don't over-constrain)
    best = sc.loc[uncorrelated].idxmax()
    return sc.loc[[top1, best]]


def apply_veto_and_select(base_scores, veto_mask, N, rng=None, perturb_sigma=0.0,
                          vol_prior_frame=None, vol_scaled=False,
                          corr_threshold=0.0, rets_panel=None, corr_prior_map=None,
                          corr_sessions=None, weight_tilt: float = 0.0,
                          conf_tilt: bool = False, median_gap=None,
                          conf_tilt_pow: float = 1.0, conf_tilt_vol: bool = False,
                          vol_factor_map=None, liquidity_ratio_frame=None,
                          liquidity_power: float = 0.0):
    """Build weighted selection dict. If perturb_sigma>0, add Gaussian noise to
    each date's composite scores (deterministic-strategy perturbation analog).
    If corr_threshold>0: apply the correlation-diversification filter (top-1 +
    best uncorrelated name) before sizing -- PIT-safe, no-op when 0."""
    wsel = {}
    for d, sc in base_scores.items():
        if perturb_sigma > 0 and rng is not None:
            scale = float(sc.std()) if sc.std() and sc.std() > 0 else 1.0
            noise = rng.normal(0.0, perturb_sigma * scale, size=len(sc))
            sc = sc + pd.Series(noise, index=sc.index)
            sc = sc[sc > 0.0]  # honor require_positive after perturbation
        vet = veto_mask.loc[d]
        vetoed = set(vet[vet].index) & set(sc.index) if d in veto_mask.index else set()
        sc2 = sc.drop(index=[s for s in vetoed if s in sc.index])
        if corr_threshold > 0 and N >= 2:
            sc2 = _apply_corr_filter(sc2, d, corr_threshold, rets_panel, corr_prior_map, corr_sessions)
        wsel[d] = _size_weights(sc2, N, d, vol_prior_frame, vol_scaled, weight_tilt,
                                conf_tilt, median_gap, conf_tilt_pow, conf_tilt_vol,
                                vol_factor_map, liquidity_ratio_frame, liquidity_power)
    return wsel


# ---------------------------------------------------------------------------
# Year-by-year metrics
# ---------------------------------------------------------------------------

def year_metrics(equity_curve: pd.DataFrame, trade_log: pd.DataFrame, bench: pd.Series,
                 full_years_only: bool = True) -> dict[int, dict[str, float]]:
    eq = equity_curve["equity"].dropna()
    out: dict[int, dict[str, float]] = {}
    years = sorted(set(d.year for d in eq.index))
    today_year = pd.Timestamp.today().year
    for yr in years:
        if full_years_only and yr >= today_year:
            # partial current year -- measured separately, excluded from primary
            continue
        sub = eq[eq.index.year == yr]
        if len(sub) < 2:
            continue
        sub_eq = sub.to_frame("equity")
        tm = trade_log.copy()
        if not tm.empty:
            tm["yr"] = pd.to_datetime(tm["date"]).dt.year
            sub_trades = tm[tm["yr"] == yr].drop(columns=["yr"])
        else:
            sub_trades = trade_log
        bench_sub = bench.reindex(sub.index).ffill()
        ym = extended_metrics(sub_eq, sub_trades, bench_sub)
        if ym:
            out[yr] = ym
    return out


def summarize_year_sharpes(ym: dict[int, dict[str, float]]) -> dict[str, float]:
    sharpes = {yr: m.get("sharpe", np.nan) for yr, m in ym.items()}
    vals = np.array([v for v in sharpes.values() if np.isfinite(v)])
    if vals.size == 0:
        return {"worst_year_sharpe": np.nan, "mean_year_sharpe": np.nan,
                "std_year_sharpe": np.nan, "best_year_sharpe": np.nan, "n_years": 0}
    return {
        "worst_year_sharpe": float(np.min(vals)),
        "best_year_sharpe": float(np.max(vals)),
        "mean_year_sharpe": float(np.mean(vals)),
        "std_year_sharpe": float(np.std(vals, ddof=0)),
        "n_years": int(vals.size),
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(args) -> int:
    global CTX
    t0 = time.time()
    with open(args.params) as f:
        params = json.load(f)

    # Forecast context length (default 256 = champion; longer sees more history).
    # Read here so load_forecast_panel(CTX, ...) below picks up the configured ctx.
    CTX = int(params.get("forecast_context_len", 256))

    cfg, ctx = load_context(args.config)
    cfg = apply_params_to_config(cfg, params)
    set_config(cfg)

    state = build_factor_state(ctx, cfg)
    sessions = ctx.prices.sessions
    bench = benchmark_close(ctx, "SPY")

    N = int(params["max_holdings"])
    veto_threshold = float(params.get("veto_threshold", 0.775))
    trailing_window = int(params.get("trailing_vol_window", 53))
    veto_horizon = int(params.get("veto_horizon", 21))
    tc_rate = float(params.get("cost_per_side", 0.0003))

    use_quantile_veto = bool(params.get("use_quantile_veto", False))
    quantile_col = int(params.get("quantile_col", 0))  # 0=P10, 1=P20, 4=P50, 9=P90
    use_relative_veto = bool(params.get("use_relative_veto", False))
    relative_veto_pct = float(params.get("relative_veto_pct", 0.05))
    if use_quantile_veto:
        qpanel = load_forecast_panel(CTX, HOR, kind="volume", quantiles=True)
        if qpanel is None:
            print("ERROR: no quantile volume forecast cache (use_quantile_veto=true)", file=sys.stderr)
            return 1
        # Downside-quantile predicted volume summed over horizon / horizon for avg.
        # P20 (col 1) encodes tail-liquidity risk: veto only if the DOWNSIDE volume
        # forecast is low, not the mean -> fewer, smarter vetoes (New directions Phase 3).
        qcols = [f"q{quantile_col}_h{h}" for h in range(veto_horizon)]
        pred_vol = qpanel[qcols].sum(axis=1) / veto_horizon
        pcts = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        print(f"Using QUANTILE veto: P{pcts[quantile_col]} (col {quantile_col})", file=sys.stderr)
    else:
        panel = load_forecast_panel(CTX, HOR, kind="volume", quantiles=False)
        if panel is None:
            print("ERROR: no volume forecast cache", file=sys.stderr)
            return 1
        pred_vol = mean_forecast_sum(panel, veto_horizon) / veto_horizon
    pred_vol = pred_vol.unstack(level="symbol")

    veto_mask, base_scores, prior_ends, dates = build_veto_mask(state, ctx, pred_vol, veto_threshold, trailing_window, use_relative_veto=use_relative_veto, relative_veto_pct=relative_veto_pct)

    # --- TimesFM return-alpha tilt (optional, PIT-safe) ---
    # tfm_alpha_weight=0.0 (default) -> no-op, champion reproduces bit-for-bit.
    # When >0: load the logret forecast panel (forecast made at prior_end -- the
    # SAME PIT alignment as the volume veto, so NO look-ahead), take the cumulative
    # 21-step mean log-return forecast per (date, symbol), cross-sectionally z-score
    # per rebalance date, and add tfm_alpha_weight*z to base_scores[d]. This adds a
    # forward-looking-but-past-data-only return forecast as a selection tilt. The
    # perturbation loop below reuses the tilted base_scores (correct: perturbation
    # must test the full strategy including the alpha).
    tfm_alpha_weight = float(params.get("tfm_alpha_weight", 0.0))
    if tfm_alpha_weight > 0.0:
        logret_panel = load_forecast_panel(CTX, HOR, kind="logret", quantiles=False)
        if logret_panel is None:
            print("ERROR: no logret forecast cache (tfm_alpha_weight>0)", file=sys.stderr)
            return 1
        mcols = [f"m{h}" for h in range(HOR)]
        alpha_cum = logret_panel[mcols].sum(axis=1)  # cumulative log-ret over horizon
        alpha_wide = alpha_cum.unstack(level="symbol")  # rows=date, cols=symbol
        n_tilted = 0
        for d in list(base_scores.keys()):
            if d not in alpha_wide.index:
                continue
            a = alpha_wide.loc[d].dropna()
            if a.empty:
                continue
            mu, sd = float(a.mean()), float(a.std())
            z = (a - mu) / sd if sd > 0 else a * 0.0
            sc = base_scores[d]
            z = z.reindex(sc.index).fillna(0.0)
            base_scores[d] = sc + tfm_alpha_weight * z
            n_tilted += 1
        print(f"Using TFM_ALPHA tilt: weight={tfm_alpha_weight}, tilted {n_tilted}/{len(base_scores)} dates", file=sys.stderr)

    vol_scaled = bool(params.get("vol_scaled_weights", False))
    vol_lookback = int(params.get("vol_lookback", 63))
    vol_prior_frame = None
    if vol_scaled:
        ret_std = ctx.prices.close.pct_change().rolling(vol_lookback, min_periods=10).std()
        vol_prior_frame = ret_std.reindex(prior_ends)
        vol_prior_frame.index = pd.DatetimeIndex(dates)

    # --- Correlation diversification filter (optional, PIT-safe) ---
    # corr_threshold=0.0 (default) -> no-op, champion reproduces bit-for-bit.
    # When >0: pick top-1 by composite, then the highest-scoring name whose
    # trailing 63-session return corr to the top-1 is <= threshold (negative corr
    # kept -- it diversifies). Targets the 2-stock concentration risk: the 2022
    # top-2 are mean +0.45 correlated (3.5x the universe avg +0.13), clustered in
    # the same trending sectors (energy/oil). PIT-safe: returns up to prior_end.
    corr_threshold = float(params.get("corr_threshold", 0.0))
    rets_panel = ctx.prices.close.pct_change() if corr_threshold > 0 else None
    corr_prior_map = dict(zip(dates, prior_ends)) if corr_threshold > 0 else None
    corr_sessions = ctx.prices.sessions if corr_threshold > 0 else None
    if corr_threshold > 0:
        print(f"Using CORR diversification: threshold={corr_threshold} (top-1 + best "
              f"uncorrelated, 63-session trailing corr)", file=sys.stderr)

    # --- Conviction weight tilt (optional) ---
    # weight_tilt=0.0 (default) -> no-op, champion reproduces bit-for-bit.
    # When >0: shift that fraction of weight from the lower-ranked names to the
    # top-1 (strongest composite score). A sizing lever (does NOT change the
    # names -- so it cannot crash 2022 like the selection modifications). EV check:
    # the 2022 top-1 outperformed the top-2 by mean +0.0172/month (3/5 months).
    weight_tilt = float(params.get("weight_tilt", 0.0))
    if weight_tilt > 0:
        print(f"Using WEIGHT_TILT: {weight_tilt} (conviction tilt toward top-1)", file=sys.stderr)

    # --- Regime-cash overlay (optional, PIT-safe) ---
    # use_regime_cash=false (default) -> no-op, champion reproduces bit-for-bit.
    # When true: scale all position weights by regime_cash_fraction in months where
    # the BENCHMARK (SPY) close at prior_end < its regime_cash_sma-day SMA (bear
    # market). Reuses sma_trend (210) by default -- no new hyperparameter. The
    # residual weight stays cash (the backtester holds cash for weight sum < 1,
    # same as the macro_half_risk mechanism). PIT-safe: uses prior_end data only.
    use_regime_cash = bool(params.get("use_regime_cash", False))
    regime_sma = int(params.get("regime_cash_sma", int(params.get("sma_trend", 201))))
    regime_frac = float(params.get("regime_cash_fraction", 0.5))
    regime_scale = None
    if use_regime_cash:
        bench_sma = bench.rolling(regime_sma, min_periods=10).mean()
        prior_end_map = dict(zip(dates, prior_ends))
        regime_scale = {}
        n_bear = 0
        for d in dates:
            pe = prior_end_map.get(d)
            if pe is not None and pe in bench.index and pe in bench_sma.index:
                bc, bs = float(bench.loc[pe]), float(bench_sma.loc[pe])
                if np.isfinite(bc) and np.isfinite(bs) and bc < bs:
                    regime_scale[d] = regime_frac
                    n_bear += 1
                else:
                    regime_scale[d] = 1.0
            else:
                regime_scale[d] = 1.0
        print(f"Using REGIME_CASH overlay: sma={regime_sma}, bear_frac={regime_frac}, "
              f"bear months={n_bear}/{len(dates)}", file=sys.stderr)

    def _apply_regime_cash(wsel_dict):
        if regime_scale is None:
            return wsel_dict
        out = {}
        for d, w in wsel_dict.items():
            s = regime_scale.get(d, 1.0)
            out[d] = {sym: wt * s for sym, wt in w.items()} if s < 1.0 else w
        return out

    # --- Confidence-scaled conviction tilt (optional, PIT-safe) ---
    # conf_tilt=false (default) -> no-op. When true: scale the weight_tilt by the
    # composite-gap confidence (|top1-top2| / median_gap, capped at 1). Tilt fully
    # when the top-1 is a clear winner (large gap, e.g. 2022 energy), tilt less
    # when it's a close call (small gap, e.g. 2015 choppy). Aims to separate the
    # 2022 lift (high confidence) from the 2015 drop (low confidence) -- a
    # CONFIDENCE signal (not a return forecast), which may work where return-based
    # conditionals (cond_tilt, regime_tilt) failed. PIT-safe (composite scores only).
    conf_tilt = bool(params.get("conf_tilt", False))
    conf_tilt_pow = float(params.get("conf_tilt_pow", 1.0))
    conf_tilt_vol = bool(params.get("conf_tilt_vol", False))
    median_gap = None
    vol_factor_map = None
    if conf_tilt:
        _gaps = []
        for _d, _sc in base_scores.items():
            _s = _sc.replace([np.inf, -np.inf], np.nan).dropna()
            if len(_s) >= 2:
                _t = _s.sort_values(ascending=False).head(2)
                _gaps.append(abs(float(_t.iloc[0]) - float(_t.iloc[1])))
        median_gap = float(np.median(_gaps)) if _gaps else None
        if conf_tilt_vol:
            _vw = int(params.get("conf_tilt_vol_window", 21))
            _vcap = float(params.get("conf_tilt_vol_cap", 2.0))
            _vpow = float(params.get("conf_tilt_vol_pow", 1.0))
            _bv = bench.pct_change().rolling(_vw, min_periods=5).std()
            _pem = dict(zip(dates, prior_ends))
            _vols = [float(_bv.loc[pe]) for pe in [_pem.get(d) for d in dates]
                     if pe is not None and pe in _bv.index and np.isfinite(_bv.loc[pe]) and _bv.loc[pe] > 0]
            _mv = float(np.median(_vols)) if _vols else 1.0
            vol_factor_map = {}
            for d in dates:
                pe = _pem.get(d)
                vf = 1.0
                if pe is not None and pe in _bv.index and _mv > 0:
                    v = float(_bv.loc[pe])
                    if np.isfinite(v) and v > 0:
                        vf = min(_vcap, (v / _mv) ** _vpow) if _vpow != 1.0 else min(_vcap, v / _mv)
                vol_factor_map[d] = vf
            print(f"Using CONF_TILT_VOL: window={_vw} median_vol={_mv:.4f} cap={_vcap} pow={_vpow}", file=sys.stderr)
        print(f"Using CONF_TILT: median_gap={median_gap:.4f} pow={conf_tilt_pow} "
              f"vol={conf_tilt_vol} (confidence-scaled tilt)", file=sys.stderr)
    # Base selection + backtest
    wsel = apply_veto_and_select(base_scores, veto_mask, N, vol_prior_frame=vol_prior_frame, vol_scaled=vol_scaled, corr_threshold=corr_threshold, rets_panel=rets_panel, corr_prior_map=corr_prior_map, corr_sessions=corr_sessions, weight_tilt=weight_tilt, conf_tilt=conf_tilt, median_gap=median_gap, conf_tilt_pow=conf_tilt_pow, conf_tilt_vol=conf_tilt_vol, vol_factor_map=vol_factor_map)
    wsel = _apply_regime_cash(wsel)
    res = run_weighted_backtest(ctx, wsel, cfg, tc_rate=tc_rate, start_date=START_DATE)
    m = extended_metrics(res["equity_curve"], res["trade_log"], bench)
    ym = year_metrics(res["equity_curve"], res["trade_log"], bench, full_years_only=True)
    ys = summarize_year_sharpes(ym)

    # veto rate (matches champion accounting: vetoed-in-scores / total candidates)
    veto_count = 0
    total_cand = 0
    for d, sc in base_scores.items():
        vet = veto_mask.loc[d]
        vetoed = set(vet[vet].index) & set(sc.index) if d in veto_mask.index else set()
        total_cand += len(sc)
        veto_count += len(vetoed)
    veto_pct = veto_count / max(1, total_cand)

    # Perturbation runs (deterministic multi-seed analog): reuse base_scores +
    # veto_mask, only re-score (with noise) + re-backtest. Cheap.
    perturb_sharpes = []
    perturb_worst = []
    perturb_robust = []
    n_perturb = int(args.perturb_runs)
    sigma = float(args.perturb_sigma)
    base_seed = int(args.seed)
    for k in range(n_perturb):
        rng = np.random.default_rng(base_seed + 1000 + k)
        wsel_k = apply_veto_and_select(base_scores, veto_mask, N, rng=rng, perturb_sigma=sigma,
                                        vol_prior_frame=vol_prior_frame, vol_scaled=vol_scaled,
                                        corr_threshold=corr_threshold, rets_panel=rets_panel,
                                        corr_prior_map=corr_prior_map, corr_sessions=corr_sessions,
                                        weight_tilt=weight_tilt, conf_tilt=conf_tilt,
                                        median_gap=median_gap, conf_tilt_pow=conf_tilt_pow,
                                        conf_tilt_vol=conf_tilt_vol, vol_factor_map=vol_factor_map)
        wsel_k = _apply_regime_cash(wsel_k)
        res_k = run_weighted_backtest(ctx, wsel_k, cfg, tc_rate=tc_rate, start_date=START_DATE)
        mk = extended_metrics(res_k["equity_curve"], res_k["trade_log"], bench)
        ymk = year_metrics(res_k["equity_curve"], res_k["trade_log"], bench, full_years_only=True)
        ysk = summarize_year_sharpes(ymk)
        if np.isfinite(mk.get("sharpe", np.nan)):
            perturb_sharpes.append(mk["sharpe"])
        if np.isfinite(ysk["worst_year_sharpe"]):
            perturb_worst.append(ysk["worst_year_sharpe"])
        if np.isfinite(ysk["mean_year_sharpe"]) and np.isfinite(ysk["std_year_sharpe"]):
            perturb_robust.append(ysk["mean_year_sharpe"] - 0.5 * ysk["std_year_sharpe"])

    def _std(vals):
        a = np.array([v for v in vals if np.isfinite(v)])
        return float(np.std(a, ddof=0)) if a.size else np.nan

    perturb_sharpe_std = _std(perturb_sharpes)
    perturb_worst_std = _std(perturb_worst)
    perturb_mean_worst = float(np.mean(perturb_worst)) if perturb_worst else np.nan
    perturb_mean_sharpe = float(np.mean(perturb_sharpes)) if perturb_sharpes else np.nan
    perturb_robust_std = _std(perturb_robust)

    robust_score = (ys["mean_year_sharpe"] - 0.5 * ys["std_year_sharpe"]) if np.isfinite(ys["mean_year_sharpe"]) else np.nan

    # ---- Emit METRIC lines ----
    print(f"METRIC worst_year_sharpe={ys['worst_year_sharpe']:.6f}")
    print(f"METRIC robust_score={robust_score:.6f}")
    print(f"METRIC sharpe={m['sharpe']:.6f}")
    print(f"METRIC cagr={m['cagr']:.6f}")
    print(f"METRIC max_drawdown={m['max_drawdown']:.6f}")
    print(f"METRIC calmar={m['calmar']:.6f}")
    print(f"METRIC sortino={m['sortino']:.6f}")
    print(f"METRIC information_ratio={m['information_ratio']:.6f}")
    print(f"METRIC mean_year_sharpe={ys['mean_year_sharpe']:.6f}")
    print(f"METRIC std_year_sharpe={ys['std_year_sharpe']:.6f}")
    print(f"METRIC best_year_sharpe={ys['best_year_sharpe']:.6f}")
    print(f"METRIC n_years={ys['n_years']}")
    print(f"METRIC perturb_sharpe_std={perturb_sharpe_std:.6f}")
    print(f"METRIC perturb_worst_year_std={perturb_worst_std:.6f}")
    print(f"METRIC perturb_mean_worst_year={perturb_mean_worst:.6f}")
    print(f"METRIC perturb_mean_sharpe={perturb_mean_sharpe:.6f}")
    print(f"METRIC perturb_robust_std={perturb_robust_std:.6f}")
    print(f"METRIC veto_pct={veto_pct:.6f}")
    for yr in sorted(ym.keys()):
        print(f"METRIC year_{yr}_sharpe={ym[yr].get('sharpe', np.nan):.6f}")
        print(f"METRIC year_{yr}_cagr={ym[yr].get('cagr', np.nan):.6f}")
        print(f"METRIC year_{yr}_maxdd={ym[yr].get('max_drawdown', np.nan):.6f}")
    print(f"METRIC runtime={time.time()-t0:.1f}")

    print(
        f"# worst_year_sharpe={ys['worst_year_sharpe']:.4f} robust={robust_score:.4f} "
        f"sharpe={m['sharpe']:.4f} cagr={m['cagr']:.4f} maxdd={m['max_drawdown']:.4f} "
        f"meanY={ys['mean_year_sharpe']:.4f} stdY={ys['std_year_sharpe']:.4f} "
        f"perturb_worst_std={perturb_worst_std:.4f} vetoed={veto_pct:.1%}",
        flush=True,
    )
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/research.json")
    ap.add_argument("--params", default="research_params.json")
    ap.add_argument("--perturb-runs", type=int, default=8)
    ap.add_argument("--perturb-sigma", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    sys.exit(run_benchmark(args))


if __name__ == "__main__":
    main()
