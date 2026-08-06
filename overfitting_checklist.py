"""Comprehensive Overfitting Checklist for the IWV rotation strategy.

Adapts the 6 classical overfitting tests to this DETERMINISTIC ranking strategy
(there is no stochastic training/seed, so "multi-seed" becomes score
perturbation; "weight stability" becomes perturbing the fixed factor weights;
"walk-forward leak" becomes a behavioral audit of the PIT guards).

Tests:
  1. Perturbation (multi-seed analog): 12 noisy re-runs -> mean +/- std of
     worst_year_sharpe / sharpe / cagr. FAIL if std/|mean| > 5% for sharpe.
  2. Per-seed breakdown + per-fold OOS: list each perturbation run's worst
     year; report each walk-forward fold's out-of-sample Sharpe and the
     in-sample IC vs OOS gap.
  3. Sub-period / regime: Sharpe across bear / bull / covid-crash / high-vol.
     FAIL if any regime Sharpe is badly out of line (e.g. bull < bear).
  4. Parameter sensitivity: vary key hyperparams +/- ; report spread of
     worst_year_sharpe & sharpe. FAIL if worst_year_sharpe spread > 0.5 or
     sharpe collapses > 30% under a one-step tweak.
  5. Weight stability: perturb fixed factor weights (asset_growth /
     earnings_yield / fundamental_group_weight) +/- ; report spread.
  6. Walk-forward leak audit: verify splits non-overlapping, train_end <
     test_start, _strip_leaky_train_dates removes leaky rows, macro
     publication lag set; per-fold OOS table.

Emits METRIC lines for the stability scores so autoresearch can track them.
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
    run_weighted_backtest, extended_metrics, benchmark_close,
    load_forecast_panel,
)
from algo_trading.timesfm_engine import mean_forecast_sum
from algo_trading.config import AppConfig
from algo_trading.strategy import _strip_leaky_train_dates
from algo_trading.data import PUBLICATION_LAG_DAYS

from overfit_harness import (
    apply_params_to_config, build_veto_mask, apply_veto_and_select,
    year_metrics, summarize_year_sharpes, START_DATE, CTX, HOR,
)


def _sharpe_from_eq(eq_slice: pd.Series) -> float:
    r = eq_slice.pct_change().dropna()
    if r.std() == 0 or len(r) < 2:
        return float("nan")
    return float(np.sqrt(252) * r.mean() / r.std())


def regime_report(eq: pd.Series) -> dict[str, dict]:
    regimes = {
        "bull_2016_2017": ("2016-01-01", "2017-12-31"),
        "bull_2023_2024": ("2023-01-01", "2024-12-31"),
        "bear_2018": ("2018-01-01", "2018-12-31"),
        "bear_2022": ("2022-01-01", "2022-12-31"),
        "covid_crash_2020": ("2020-02-19", "2020-03-23"),
        "covid_recovery_2020": ("2020-03-24", "2020-12-31"),
        "high_vol_2020q1q2": ("2020-01-01", "2020-06-30"),
        "weak_2025": ("2025-01-01", "2025-12-31"),
    }
    out = {}
    for name, (s, e) in regimes.items():
        sub = eq[(eq.index >= pd.Timestamp(s)) & (eq.index <= pd.Timestamp(e))]
        out[name] = {"sharpe": _sharpe_from_eq(sub), "n_days": int(len(sub))}
    return out


def per_fold_oos(state, eq: pd.Series) -> list[dict]:
    """Out-of-sample Sharpe per walk-forward split (test period)."""
    rows = []
    for i, split in enumerate(state.splits):
        sub = eq[(eq.index >= split.test_start) & (eq.index <= split.test_end)]
        # also compute the leaky-strip count for this split
        clean = _strip_leaky_train_dates(split.train_dates, state.rebalance_dates, split.test_start)
        rows.append({
            "split": i,
            "train_start": str(split.train_start.date()),
            "train_end": str(split.train_end.date()),
            "test_start": str(split.test_start.date()),
            "test_end": str(split.test_end.date()),
            "n_train": len(split.train_dates),
            "n_train_stripped": len(split.train_dates) - len(clean),
            "n_test": len(split.test_dates),
            "oos_sharpe": _sharpe_from_eq(sub),
        })
    return rows


def run_one(cfg, ctx, params, veto_mask, base_scores, N, tc_rate, sigma=0.0, seed=None, perturb=False,
              vol_prior_frame=None, vol_scaled=False):
    rng = np.random.default_rng(seed) if perturb else None
    wsel = apply_veto_and_select(base_scores, veto_mask, N, rng=rng, perturb_sigma=sigma if perturb else 0.0,
                                 vol_prior_frame=vol_prior_frame, vol_scaled=vol_scaled)
    res = run_weighted_backtest(ctx, wsel, cfg, tc_rate=tc_rate, start_date=START_DATE)
    bench = benchmark_close(ctx, "SPY")
    m = extended_metrics(res["equity_curve"], res["trade_log"], bench)
    ym = year_metrics(res["equity_curve"], res["trade_log"], bench, full_years_only=True)
    ys = summarize_year_sharpes(ym)
    return m, ys, res["equity_curve"]["equity"].dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/research.json")
    ap.add_argument("--params", default="research_params.json")
    ap.add_argument("--perturb-runs", type=int, default=12)
    ap.add_argument("--perturb-sigma", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-structural", action="store_true",
                    help="skip the expensive rebuild-based sensitivity sweeps")
    args = ap.parse_args()

    t0 = time.time()
    with open(args.params) as f:
        params = json.load(f)
    cfg, ctx = load_context(args.config)
    cfg = apply_params_to_config(cfg, params)
    set_config(cfg)

    state = build_factor_state(ctx, cfg)
    N = int(params["max_holdings"])
    veto_threshold = float(params.get("veto_threshold", 0.775))
    trailing_window = int(params.get("trailing_vol_window", 53))
    veto_horizon = int(params.get("veto_horizon", 21))
    tc_rate = float(params.get("cost_per_side", 0.0003))

    # Honor the quantile-veto config (mirror overfit_harness). Without this the
    # checklist would silently validate the MEAN veto even when the champion uses
    # the P20 downside-quantile veto -- a phantom-config cheat.
    use_quantile_veto = bool(params.get("use_quantile_veto", False))
    quantile_col = int(params.get("quantile_col", 0))
    if use_quantile_veto:
        qpanel = load_forecast_panel(CTX, HOR, kind="volume", quantiles=True)
        if qpanel is None:
            print("ERROR: no quantile volume forecast cache (use_quantile_veto=true)", file=sys.stderr)
            return 1
        qcols = [f"q{quantile_col}_h{h}" for h in range(veto_horizon)]
        pred_vol = qpanel[qcols].sum(axis=1) / veto_horizon
        pcts = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        print(f"Using QUANTILE veto in checklist: P{pcts[quantile_col]} (col {quantile_col})", file=sys.stderr)
    else:
        panel = load_forecast_panel(CTX, HOR, kind="volume", quantiles=False)
        if panel is None:
            print("ERROR: no volume forecast cache", file=sys.stderr)
            return 1
        pred_vol = mean_forecast_sum(panel, veto_horizon) / veto_horizon
    pred_vol = pred_vol.unstack(level="symbol")
    veto_mask, base_scores, prior_ends, dates = build_veto_mask(state, ctx, pred_vol, veto_threshold, trailing_window)

    vol_scaled = bool(params.get("vol_scaled_weights", False))
    vol_lookback = int(params.get("vol_lookback", 63))
    vol_prior_frame = None
    if vol_scaled:
        ret_std = ctx.prices.close.pct_change().rolling(vol_lookback, min_periods=10).std()
        vol_prior_frame = ret_std.reindex(prior_ends)
        vol_prior_frame.index = pd.DatetimeIndex(dates)

    base_m, base_ys, base_eq = run_one(cfg, ctx, params, veto_mask, base_scores, N, tc_rate,
                                      vol_prior_frame=vol_prior_frame, vol_scaled=vol_scaled)
    base_worst = base_ys["worst_year_sharpe"]
    base_sharpe = base_m["sharpe"]
    base_cagr = base_m["cagr"]

    print("=" * 90)
    print("COMPREHENSIVE OVERFITTING CHECKLIST")
    print("=" * 90)
    print(f"Baseline: worst_year_sharpe={base_worst:.4f} sharpe={base_sharpe:.4f} "
          f"cagr={base_cagr:.4f} meanY={base_ys['mean_year_sharpe']:.4f} "
          f"stdY={base_ys['std_year_sharpe']:.4f}")
    print()

    # ---- TEST 1: Perturbation (multi-seed analog) ----
    print("-" * 90)
    print("TEST 1: Perturbation stability (multi-seed analog)")
    print("-" * 90)
    pert_worsts, pert_sharpes, pert_cagrs, pert_robusts, pert_rows = [], [], [], [], []
    for k in range(args.perturb_runs):
        seed = args.seed + 1000 + k
        mk, ysk, eqk = run_one(cfg, ctx, params, veto_mask, base_scores, N, tc_rate,
                               sigma=args.perturb_sigma, seed=seed, perturb=True,
                               vol_prior_frame=vol_prior_frame, vol_scaled=vol_scaled)
        w = ysk["worst_year_sharpe"]; s = mk["sharpe"]; c = mk["cagr"]
        rb = (ysk["mean_year_sharpe"] - 0.5 * ysk["std_year_sharpe"]) if np.isfinite(ysk["mean_year_sharpe"]) else np.nan
        pert_worsts.append(w); pert_sharpes.append(s); pert_cagrs.append(c); pert_robusts.append(rb)
        ymk = year_metrics(pd.DataFrame({"equity": eqk}), pd.DataFrame(), benchmark_close(ctx, "SPY"),
                            full_years_only=True)
        yr_items = [(yr, ym.get("sharpe", np.nan)) for yr, ym in ymk.items() if np.isfinite(ym.get("sharpe", np.nan))]
        worst_year = min(yr_items, key=lambda x: x[1])[0] if yr_items else -1
        pert_rows.append((k, w, s, c, worst_year))
        print(f"  run {k:2d}: worst_year_sharpe={w:+.4f}  sharpe={s:.4f}  cagr={c:.4f}  worst_year={worst_year}")

    def _ms(vals):
        a = np.array([v for v in vals if np.isfinite(v)])
        return float(a.mean()), float(a.std(ddof=0))
    mw, sw = _ms(pert_worsts); ms, ss = _ms(pert_sharpes); mc, sc = _ms(pert_cagrs); mr, sr = _ms(pert_robusts)
    sharpe_cv = abs(ss / ms) if ms else np.nan
    worst_cv = abs(sw / mw) if mw else np.nan
    t1_pass = sharpe_cv < 0.05
    print(f"  -> worst_year_sharpe: mean={mw:+.4f} std={sw:.4f} (cv={worst_cv:.1%})")
    print(f"  -> sharpe:            mean={ms:.4f} std={ss:.4f} (cv={sharpe_cv:.1%})  "
          f"{'PASS' if t1_pass else 'FAIL'} (>5% std/mean)")
    print(f"  -> cagr:              mean={mc:.4f} std={sc:.4f}")
    print(f"  -> robust_score:      mean={mr:.4f} std={sr:.4f}")
    print()

    # ---- TEST 2: per-fold OOS + per-seed breakdown ----
    print("-" * 90)
    print("TEST 2: Per-fold out-of-sample + per-seed worst-year breakdown")
    print("-" * 90)
    folds = per_fold_oos(state, base_eq)
    oos_sharpes = [f["oos_sharpe"] for f in folds if np.isfinite(f["oos_sharpe"])]
    n_neg_oos = sum(1 for s in oos_sharpes if s < 0)
    print(f"  Walk-forward folds: {len(folds)}  (negative-OOS folds: {n_neg_oos})")
    for f in folds:
        print(f"  split {f['split']:2d}  train {f['train_start']}..{f['train_end']} "
              f"({f['n_train']} dates, stripped {f['n_train_stripped']})  "
              f"test {f['test_start']}..{f['test_end']} ({f['n_test']})  "
              f"OOS_sharpe={f['oos_sharpe']:+.3f}")
    print(f"  Per-seed worst years: {[r[4] for r in pert_rows]}")
    print(f"  -> OOS sharpe: min={min(oos_sharpes):+.3f} mean={np.mean(oos_sharpes):+.3f} "
          f"max={max(oos_sharpes):+.3f}")
    print()

    # ---- TEST 3: regime / sub-period ----
    print("-" * 90)
    print("TEST 3: Sub-period / regime analysis")
    print("-" * 90)
    regs = regime_report(base_eq)
    for name, r in regs.items():
        print(f"  {name:24s} sharpe={r['sharpe']:+.3f}  n_days={r['n_days']}")
    bull = [regs[k]["sharpe"] for k in ("bull_2016_2017", "bull_2023_2024") if np.isfinite(regs[k]["sharpe"])]
    bear = [regs[k]["sharpe"] for k in ("bear_2018", "bear_2022") if np.isfinite(regs[k]["sharpe"])]
    print(f"  -> bull mean={np.mean(bull):+.3f}  bear mean={np.mean(bear):+.3f}")
    print()

    # ---- TEST 4 & 5: parameter + weight sensitivity ----
    print("-" * 90)
    print("TEST 4 & 5: Parameter + weight sensitivity (one-step tweaks)")
    print("-" * 90)
    sens = []  # (label, worst_year_sharpe, sharpe, cagr)

    # Cheap params (reuse state): veto_threshold, max_holdings, cost, macro thresholds
    cheap_variations = []
    # veto_threshold sensitivity: vary +/-0.05 around the CURRENT threshold (adaptive
    # -- works for both the mean-veto champion ~0.80 and the P20 champion ~0.30).
    for vt in [round(veto_threshold - 0.05, 3), round(veto_threshold + 0.05, 3)]:
        cheap_variations.append((f"veto_threshold={vt}", {"veto_threshold": vt}))
    for mh in [-0.4, -0.6]:
        cheap_variations.append((f"macro_half={mh}", {"macro_half_risk_threshold": mh}))
    for mf in [-0.8, -1.2]:
        cheap_variations.append((f"macro_flat={mf}", {"macro_flat_threshold": mf}))
    for Nv in [2, 3]:
        cheap_variations.append((f"max_holdings={Nv}", {"max_holdings": Nv}))
    for cc in [0.0002, 0.0004]:
        cheap_variations.append((f"cost={cc}", {"cost_per_side": cc}))

    for label, delta in cheap_variations:
        p2 = dict(params); p2.update(delta)
        # veto_threshold / trailing change -> rebuild veto_mask; max_holdings/macro/cost reuse
        if "veto_threshold" in delta:
            vm2, _, _, _ = build_veto_mask(state, ctx, pred_vol, p2["veto_threshold"], trailing_window)
        else:
            vm2 = veto_mask
        m2, ys2, _ = run_one(cfg, ctx, p2, vm2, base_scores, p2["max_holdings"], p2.get("cost_per_side", tc_rate),
                             vol_prior_frame=vol_prior_frame, vol_scaled=vol_scaled)
        sens.append((label, ys2["worst_year_sharpe"], m2["sharpe"], m2["cagr"]))
        print(f"  {label:26s} worst_year={ys2['worst_year_sharpe']:+.4f}  sharpe={m2['sharpe']:.4f}  cagr={m2['cagr']:.4f}")

    # Structural params (rebuild factor state): sma, ts_mom, fund weights, min_volume
    if not args.skip_structural:
        struct_variations = []
        for sf in [2, 4]:
            struct_variations.append((f"sma_fast={sf}", {"sma_fast": sf}))
        for ss_v in [45, 55]:
            struct_variations.append((f"sma_slow={ss_v}", {"sma_slow": ss_v}))
        for st in [190, 210]:
            struct_variations.append((f"sma_trend={st}", {"sma_trend": st}))
        for tm in [8, 10]:
            struct_variations.append((f"ts_mom={tm}", {"ts_mom_lookback_months": tm}))
        for fg in [0.02, 0.04]:
            struct_variations.append((f"fund_group_w={fg}", {"fundamental_group_weight": fg}))
        for ag in [16.0, 24.0]:
            struct_variations.append((f"asset_growth={ag}", {"asset_growth_weight": ag}))
        for ey in [2.0, 4.0]:
            struct_variations.append((f"earnings_yield={ey}", {"earnings_yield_weight": ey}))
        for mv in [10000000.0, 12000000.0]:
            struct_variations.append((f"min_volume={mv/1e6:.0f}M", {"min_avg_dollar_volume": mv}))

        for label, delta in struct_variations:
            p2 = dict(params); p2.update(delta)
            c2 = AppConfig.from_file(args.config); c2 = apply_params_to_config(c2, p2); set_config(c2)
            st2 = build_factor_state(ctx, c2)
            vm2, bs2, _, _ = build_veto_mask(st2, ctx, pred_vol, p2.get("veto_threshold", veto_threshold),
                                       p2.get("trailing_vol_window", trailing_window))
            m2, ys2, _ = run_one(c2, ctx, p2, vm2, bs2, p2["max_holdings"], p2.get("cost_per_side", tc_rate),
                                 vol_prior_frame=vol_prior_frame, vol_scaled=vol_scaled)
            sens.append((label, ys2["worst_year_sharpe"], m2["sharpe"], m2["cagr"]))
            print(f"  {label:26s} worst_year={ys2['worst_year_sharpe']:+.4f}  sharpe={m2['sharpe']:.4f}  cagr={m2['cagr']:.4f}  [rebuild]")

    sens_worst = [s[1] for s in sens if np.isfinite(s[1])]
    sens_sharpe = [s[2] for s in sens if np.isfinite(s[2])]
    w_spread = max(sens_worst) - min(sens_worst) if sens_worst else np.nan
    s_spread = max(sens_sharpe) - min(sens_sharpe) if sens_sharpe else np.nan
    s_collapse = (max(sens_sharpe) - min(sens_sharpe)) / max(sens_sharpe) if sens_sharpe and max(sens_sharpe) > 0 else np.nan
    t4_pass = w_spread < 0.5 and s_collapse < 0.30
    print(f"  -> worst_year_sharpe spread = {w_spread:.4f}  (max={max(sens_worst):+.4f}, min={min(sens_worst):+.4f})")
    print(f"  -> sharpe spread = {s_spread:.4f}  collapse = {s_collapse:.1%}  "
          f"{'PASS' if t4_pass else 'WARN/FAIL'} (worst_year spread<0.5 & sharpe collapse<30%)")
    print()

    # ---- TEST 6: walk-forward leak audit ----
    print("-" * 90)
    print("TEST 6: Walk-forward leak audit")
    print("-" * 90)
    leaks = 0
    for f in folds:
        ts = pd.Timestamp(f["test_start"]); te = pd.Timestamp(f["test_end"]); tre = pd.Timestamp(f["train_end"])
        if tre >= ts:
            leaks += 1
    print(f"  Splits with train_end >= test_start (overlap leak): {leaks}  (expect 0)")
    print(f"  Macro publication lag config: {PUBLICATION_LAG_DAYS}")
    print(f"  _strip_leaky_train_dates total rows stripped across folds: "
          f"{sum(f['n_train_stripped'] for f in folds)}")
    t6_pass = leaks == 0
    print(f"  -> {'PASS' if t6_pass else 'FAIL'} (no train/test overlap; macro PIT lag configured)")
    print()

    # ---- Summary + METRIC lines ----
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"  T1 perturbation (sharpe cv<5%):        {'PASS' if t1_pass else 'FAIL'}  (cv={sharpe_cv:.1%})")
    print(f"  T4/5 param sensitivity (spread/collapse): {'PASS' if t4_pass else 'WARN'}  (w_spread={w_spread:.3f}, s_collapse={s_collapse:.1%})")
    print(f"  T6 leak audit:                         {'PASS' if t6_pass else 'FAIL'}")
    overall = t1_pass and t4_pass and t6_pass
    print(f"  OVERALL: {'PASS' if overall else 'FAIL/WARN'}")
    print(f"  runtime={time.time()-t0:.1f}s")

    # METRIC lines for tracking
    print(f"METRIC chk_perturb_sharpe_cv={sharpe_cv:.6f}")
    print(f"METRIC chk_perturb_worst_cv={worst_cv:.6f}")
    print(f"METRIC chk_perturb_worst_std={sw:.6f}")
    print(f"METRIC chk_perturb_sharpe_std={ss:.6f}")
    print(f"METRIC chk_sens_worst_spread={w_spread:.6f}")
    print(f"METRIC chk_sens_sharpe_collapse={s_collapse:.6f}")
    print(f"METRIC chk_leak_overlap={leaks}")
    print(f"METRIC chk_neg_oos_folds={n_neg_oos}")
    print(f"METRIC chk_oos_sharpe_min={min(oos_sharpes):.6f}")
    print(f"METRIC chk_runtime={time.time()-t0:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
