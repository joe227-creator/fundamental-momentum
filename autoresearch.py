"""Autoresearch benchmark: Experiment 13 (TimesFM volume-implied exhaustion veto).

Reads parameters from autoresearch_params.json, builds the factor state with
those parameters, runs the volume-veto backtest, and prints METRIC lines.

Primary metric: CAGR (maximize)
Secondary metrics: Sharpe, MaxDD, Calmar, Sortino, subperiod Sharpes
"""
import json, sys, time, warnings, os
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
    run_weighted_backtest, extended_metrics, subperiod_metrics,
    benchmark_close, score_composite, top_n_equal_weights,
    eligible_symbols, load_forecast_panel,
)
from algo_trading.timesfm_engine import mean_forecast_sum
from algo_trading.indicators import cross_sectional_rank
from algo_trading.config import AppConfig

PARAMS_FILE = "autoresearch_params.json"
CTX, HOR = 256, 21


def load_params():
    with open(PARAMS_FILE, "r") as f:
        return json.load(f)


def apply_params_to_config(cfg: AppConfig, p: dict) -> AppConfig:
    """Apply tunable parameters to a copy of the config."""
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
    if "cash_flow_yield_weight" in p:
        fw["cash_flow_yield"] = float(p["cash_flow_yield_weight"])
    if "book_to_market_weight" in p:
        fw["book_to_market"] = float(p["book_to_market_weight"])
    if "gross_profitability_weight" in p:
        fw["gross_profitability"] = float(p["gross_profitability_weight"])
    if "roe_weight" in p:
        fw["roe"] = float(p["roe_weight"])
    if "investment_to_assets_weight" in p:
        fw["investment_to_assets"] = float(p["investment_to_assets_weight"])
    if "net_issuance_weight" in p:
        fw["net_issuance"] = float(p["net_issuance_weight"])
    if "accruals_weight" in p:
        fw["accruals"] = float(p["accruals_weight"])
    if "cash_flow_yield_weight" in p:
        fw["cash_flow_yield"] = float(p["cash_flow_yield_weight"])
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


def trailing_avg(volume_series, end_session, window):
    s = volume_series.loc[:end_session]
    return s.tail(window).mean()


def run_benchmark():
    t0 = time.time()
    params = load_params()

    # Load market context (cached pickles, ~14s)
    cfg, ctx = load_context("config/baseline.json")
    cfg = apply_params_to_config(cfg, params)
    set_config(cfg)

    # Build factor state with tuned parameters (~19s)
    state = build_factor_state(ctx, cfg)

    sessions = ctx.prices.sessions
    volume = ctx.prices.volume
    bench = benchmark_close(ctx, "SPY")

    # Load volume forecast panel
    panel = load_forecast_panel(CTX, HOR, kind="volume", quantiles=False)
    if panel is None:
        print("ERROR: no volume forecast cache", file=sys.stderr)
        sys.exit(1)

    veto_horizon = int(params.get("veto_horizon", HOR))
    pred_vol = mean_forecast_sum(panel, veto_horizon) / veto_horizon
    pred_vol = pred_vol.unstack(level="symbol")

    # Optionally load quantile panel for downside-conservative veto
    quantile_panel = None
    if params.get("use_quantile_veto", False):
        quantile_panel = load_forecast_panel(CTX, HOR, kind="volume", quantiles=True)
        if quantile_panel is not None:
            qcol = int(params.get("quantile_col", 0))
            from algo_trading.timesfm_engine import quantile_path
            qpath = quantile_path(quantile_panel, qcol, HOR)
            pred_vol_q = (qpath.sum(axis=1) / HOR).unstack(level="symbol")

    # Optionally load logret forecast panel for alpha blend
    tfm_alpha_weight = float(params.get("tfm_alpha_weight", 0.0))
    tfm_alpha_panel = None
    if tfm_alpha_weight > 0 or params.get("veto_mode") == "tfm_return_veto" or float(params.get("sup_return_threshold", 0.0)) != 0.0:
        tfm_alpha_panel = load_forecast_panel(CTX, HOR, kind="logret", quantiles=False)
        if tfm_alpha_panel is not None:
            tfm_alpha_forecast = mean_forecast_sum(tfm_alpha_panel, HOR).unstack(level="symbol")

    # Optionally load 60-day logret forecast for longer-horizon veto
    sup60_threshold = float(params.get("sup60_return_threshold", 0.0))
    tfm60_panel = None
    if sup60_threshold != 0.0 or float(params.get("tfm60_alpha_weight", 0.0)) > 0:
        tfm60_panel = load_forecast_panel(CTX, 60, kind="logret", quantiles=False)
        if tfm60_panel is not None:
            tfm60_forecast = mean_forecast_sum(tfm60_panel, 60).unstack(level="symbol")

    # Optionally load 5-day logret forecast for short-term crash veto
    sup5_threshold = float(params.get("sup5_return_threshold", 0.0))
    tfm5_panel = None
    if sup5_threshold != 0.0:
        tfm5_panel = load_forecast_panel(CTX, 5, kind="logret", quantiles=False)
        if tfm5_panel is not None:
            tfm5_forecast = mean_forecast_sum(tfm5_panel, 5).unstack(level="symbol")

    # Optionally load ctx64/hor5 volume forecast for short-term volume veto
    sup_vol_threshold = float(params.get("sup_vol_threshold", 0.0))
    sup_vol_panel = None
    if sup_vol_threshold > 0:
        sup_vol_panel = load_forecast_panel(64, 5, kind="volume", quantiles=False)
        if sup_vol_panel is not None:
            sup_vol_pred = (mean_forecast_sum(sup_vol_panel, 5) / 5).unstack(level="symbol")

    # Optionally load price forecast for price-ratio veto
    price_veto_threshold = float(params.get("price_veto_threshold", 0.0))
    price_panel = None
    if price_veto_threshold > 0:
        price_panel = load_forecast_panel(CTX, HOR, kind="price", quantiles=False)
        if price_panel is not None:
            # Predicted avg price over horizon / last observed price
            price_pred = mean_forecast_sum(price_panel, HOR) / HOR
            price_pred = price_pred.unstack(level="symbol")

    # Build dates list
    dates = [pd.Timestamp(d) for d in state.rebalance_dates
             if pd.Timestamp(d) >= pd.Timestamp("2015-01-01")
             and state.split_for_date.get(d) is not None]

    # Tunable parameters
    veto_threshold = float(params["veto_threshold"])
    soft_threshold = float(params.get("veto_soft_threshold", 0.0))
    vol_confidence_weight = float(params.get("vol_confidence_weight", 0.0))
    trailing_window = int(params["trailing_vol_window"])
    veto_mode = params.get("veto_mode", "tfm")
    N = int(params["max_holdings"])
    tc_rate = float(params.get("cost_per_side", 0.001))

    # Run backtest with volume veto
    wsel = {}
    veto_count = 0
    total_cand = 0
    for d in dates:
        if d not in state.base_mask.index:
            continue
        # Build extra factors for TFM alpha blend
        extra_factors = None
        if tfm_alpha_weight > 0 and tfm_alpha_panel is not None:
            if d in tfm_alpha_forecast.index:
                tfm_scores = tfm_alpha_forecast.loc[d].dropna()
                if not tfm_scores.empty:
                    tfm_ranks = tfm_scores.rank(pct=True)
                    tfm_ranks = tfm_ranks.reindex(state.base_mask.columns).fillna(0.0)
                    extra_factors = {"tfm_alpha": (tfm_ranks, tfm_alpha_weight)}
        tfm60_alpha_weight = float(params.get("tfm60_alpha_weight", 0.0))
        if tfm60_alpha_weight > 0 and tfm60_panel is not None:
            if d in tfm60_forecast.index:
                tfm60_scores = tfm60_forecast.loc[d].dropna()
                if not tfm60_scores.empty:
                    tfm60_ranks = tfm60_scores.rank(pct=True)
                    tfm60_ranks = tfm60_ranks.reindex(state.base_mask.columns).fillna(0.0)
                    if extra_factors is None:
                        extra_factors = {}
                    extra_factors["tfm60_alpha"] = (tfm60_ranks, tfm60_alpha_weight)
        scores = score_composite(state, d, extra_factors=extra_factors)
        if scores.empty:
            continue

        if veto_mode in ("tfm", "tfm_quantile", "naive"):
            v = set()
            elig = eligible_symbols(state, d)
            pos = sessions.searchsorted(d, side="left")
            prior_end = sessions[pos - 1] if pos > 0 else d
            for sym in elig:
                tv = trailing_avg(volume[sym], prior_end, trailing_window) if sym in volume.columns else np.nan
                if not np.isfinite(tv) or tv <= 0:
                    continue
                if veto_mode == "naive":
                    tv_short = trailing_avg(volume[sym], prior_end, 21)
                    if not np.isfinite(tv_short):
                        continue
                    ratio = tv_short / tv
                else:
                    pv_source = pred_vol_q if (veto_mode == "tfm_quantile" and quantile_panel is not None) else pred_vol
                    pv = pv_source.loc[d, sym] if (d in pv_source.index and sym in pv_source.columns) else np.nan
                    if not np.isfinite(pv):
                        continue
                    ratio = pv / tv
                if ratio < veto_threshold:
                    v.add(sym)
                elif soft_threshold > 0 and ratio < soft_threshold and sym in scores.index:
                    # Soft veto: penalize score proportionally
                    penalty = ratio / soft_threshold
                    scores[sym] = scores[sym] * penalty
                elif vol_confidence_weight > 0 and sym in scores.index and ratio > 1.0:
                    # Volume confidence boost: stocks with predicted volume increase get higher score
                    scores[sym] = scores[sym] * (1.0 + vol_confidence_weight * (ratio - 1.0))
            total_cand += len(scores)
            veto_count += len(set(scores.index) & v)
            scores = scores.drop(index=[s for s in v if s in scores.index])

        # Supplementary TFM return veto (applies on top of volume veto)
        sup_return_threshold = float(params.get("sup_return_threshold", 0.0))
        if sup_return_threshold != 0.0 and tfm_alpha_panel is not None:
            v2 = set()
            for sym in scores.index:
                if d in tfm_alpha_forecast.index and sym in tfm_alpha_forecast.columns:
                    pred_ret = tfm_alpha_forecast.loc[d, sym]
                    if np.isfinite(pred_ret) and pred_ret < sup_return_threshold:
                        v2.add(sym)
            veto_count += len(v2)
            scores = scores.drop(index=[s for s in v2 if s in scores.index])

        # Supplementary short-term volume veto (ctx64/hor5)
        if sup_vol_threshold > 0 and sup_vol_panel is not None:
            v_vol = set()
            elig = eligible_symbols(state, d)
            pos = sessions.searchsorted(d, side="left")
            prior_end = sessions[pos - 1] if pos > 0 else d
            for sym in scores.index:
                tv = trailing_avg(volume[sym], prior_end, trailing_window) if sym in volume.columns else np.nan
                if not np.isfinite(tv) or tv <= 0:
                    continue
                pv = sup_vol_pred.loc[d, sym] if (d in sup_vol_pred.index and sym in sup_vol_pred.columns) else np.nan
                if not np.isfinite(pv):
                    continue
                ratio = pv / tv
                if ratio < sup_vol_threshold:
                    v_vol.add(sym)
            veto_count += len(v_vol)
            scores = scores.drop(index=[s for s in v_vol if s in scores.index])

        # Supplementary price forecast veto
        if price_veto_threshold > 0 and price_panel is not None:
            v_price = set()
            pos = sessions.searchsorted(d, side="left")
            prior_end = sessions[pos - 1] if pos > 0 else d
            for sym in scores.index:
                if sym not in ctx.prices.close.columns:
                    continue
                last_close = ctx.prices.close[sym].loc[prior_end] if prior_end in ctx.prices.close.index else np.nan
                if not np.isfinite(last_close) or last_close <= 0:
                    continue
                pp = price_pred.loc[d, sym] if (d in price_pred.index and sym in price_pred.columns) else np.nan
                if not np.isfinite(pp):
                    continue
                price_ratio = pp / last_close
                if price_ratio < price_veto_threshold:
                    v_price.add(sym)
            veto_count += len(v_price)
            scores = scores.drop(index=[s for s in v_price if s in scores.index])

        # Supplementary 60-day TFM return veto
        if sup60_threshold != 0.0 and tfm60_panel is not None:
            v3 = set()
            for sym in scores.index:
                if d in tfm60_forecast.index and sym in tfm60_forecast.columns:
                    pred_ret = tfm60_forecast.loc[d, sym]
                    if np.isfinite(pred_ret) and pred_ret < sup60_threshold:
                        v3.add(sym)
            veto_count += len(v3)
            scores = scores.drop(index=[s for s in v3 if s in scores.index])

        # Supplementary 5-day TFM return veto (short-term crash detection)
        if sup5_threshold != 0.0 and tfm5_panel is not None:
            v4 = set()
            for sym in scores.index:
                if d in tfm5_forecast.index and sym in tfm5_forecast.columns:
                    pred_ret = tfm5_forecast.loc[d, sym]
                    if np.isfinite(pred_ret) and pred_ret < sup5_threshold:
                        v4.add(sym)
            veto_count += len(v4)
            scores = scores.drop(index=[s for s in v4 if s in scores.index])

        if veto_mode == "tfm_return_veto" and tfm_alpha_panel is not None:
            v = set()
            return_threshold = float(params.get("tfm_return_threshold", 0.0))
            for sym in scores.index:
                if d in tfm_alpha_forecast.index and sym in tfm_alpha_forecast.columns:
                    pred_ret = tfm_alpha_forecast.loc[d, sym]
                    if np.isfinite(pred_ret) and pred_ret < return_threshold:
                        v.add(sym)
            total_cand += len(scores)
            veto_count += len(set(scores.index) & v)
            scores = scores.drop(index=[s for s in v if s in scores.index])

        if params.get("vol_scaled_weights", False):
            top_syms = scores.head(N).index.tolist()
            if len(top_syms) > 0:
                pos = sessions.searchsorted(d, side="left")
                prior_end = sessions[pos - 1] if pos > 0 else d
                vols = {}
                for sym in top_syms:
                    if sym in ctx.prices.close.columns:
                        rets = ctx.prices.close[sym].loc[:prior_end].pct_change().tail(63)
                        vols[sym] = rets.std() if len(rets) > 10 else 0.02
                    else:
                        vols[sym] = 0.02
                inv_vol = {s: 1.0 / max(v, 0.001) for s, v in vols.items()}
                total_iv = sum(inv_vol.values())
                wsel[d] = {s: inv_vol[s] / total_iv for s in top_syms}
            else:
                wsel[d] = {}
        else:
            wsel[d] = top_n_equal_weights(scores, N)

    # Run backtest
    res = run_weighted_backtest(ctx, wsel, cfg, tc_rate=tc_rate,
                                start_date=pd.Timestamp("2015-01-02"))
    m = extended_metrics(res["equity_curve"], res["trade_log"], bench)
    sp = subperiod_metrics(res["equity_curve"], res["trade_log"], bench)

    veto_pct = veto_count / max(1, total_cand)

    # Output METRIC lines
    print(f"METRIC cagr={m['cagr']:.6f}")
    print(f"METRIC sharpe={m['sharpe']:.6f}")
    print(f"METRIC max_drawdown={m['max_drawdown']:.6f}")
    print(f"METRIC calmar={m['calmar']:.6f}")
    print(f"METRIC sortino={m['sortino']:.6f}")
    print(f"METRIC information_ratio={m['information_ratio']:.6f}")

    # Subperiod metrics for robustness
    for period_name, pm in sp.items():
        print(f"METRIC {period_name}_sharpe={pm['sharpe']:.6f}")
        print(f"METRIC {period_name}_cagr={pm['cagr']:.6f}")
        print(f"METRIC {period_name}_max_drawdown={pm['max_drawdown']:.6f}")

    print(f"METRIC veto_pct={veto_pct:.6f}")
    print(f"METRIC runtime={time.time()-t0:.1f}")

    # Summary line for logging
    print(f"# CAGR={m['cagr']:.4f} Sharpe={m['sharpe']:.4f} MaxDD={m['max_drawdown']:.4f} "
          f"Calmar={m['calmar']:.4f} Sortino={m['sortino']:.4f} vetoed={veto_pct:.1%}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(run_benchmark())
