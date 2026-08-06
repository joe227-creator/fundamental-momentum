"""Walk-forward validation for the alpha-blend (Exp 1) strategy.

For each test year Y from 2018 → 2026:
  1. Optimize alpha on the PRIOR window [2015, Y-1] (in-sample)
  2. Apply that alpha to year Y (out-of-sample)
  3. Record OOS metrics

This tests whether the alpha=0.5/0.75 result is stable under proper
walk-forward selection (not just a single full-period optimization).

Also produces a sub-period robustness heatmap:
  rows = alpha ∈ {0, 0.25, 0.5, 0.75, 1.0}
  cols = period ∈ {pre-COVID(2015-2019), COVID(2020), post-COVID(2021-2026), full}
  cell = Sharpe ratio
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from experiment_utils import setup, backtest, save_summary, fmt
from algo_trading.timesfm_experiments import (
    score_composite, top_n_equal_weights, load_forecast_panel, eligible_symbols,
)
from algo_trading.indicators import cross_sectional_rank

CTX, HOR, N = 256, 21, 2
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
TEST_YEARS = list(range(2018, 2027))


def blend_scores(state, d, tfm_rank, alpha):
    """Blend ts_momentum rank with TFM rank at given alpha (Exp 1 factor-level blend)."""
    ts_rank = state.ranked_factors["ts_momentum"].loc[d] if d in state.ranked_factors["ts_momentum"].index else pd.Series(dtype=float)
    if tfm_rank is not None and d in tfm_rank.index:
        tf_ranks = tfm_rank.loc[d]
        blended = (1.0 - alpha) * ts_rank + alpha * tf_ranks
        blended = blended.dropna()
        if blended.empty and alpha > 0:
            blended = ts_rank.dropna()
    else:
        blended = ts_rank.dropna()
    if blended.empty:
        return None
    scores = score_composite(state, d, override={"ts_momentum": blended})
    return scores


def run_backtest_subset(cfg, ctx, state, dates_subset, tfm_rank, alpha, bench):
    wsel = {}
    for d in dates_subset:
        if d not in state.base_mask.index:
            continue
        blended = blend_scores(state, d, tfm_rank, alpha)
        if blended is not None and not blended.empty:
            wsel[d] = top_n_equal_weights(blended, N)
    if not wsel:
        return None
    m, sp, res = backtest(f"wf_a{alpha}", wsel, cfg, ctx, bench)
    return m


def main():
    cfg, ctx, state, dates, bench = setup("2015-01-01")
    panel = load_forecast_panel(CTX, HOR, "logret", False)
    if panel is None:
        print("[wf] Run ctx256/hor21 mean forecast driver first."); return

    # Build TFM rank by date
    from algo_trading.timesfm_engine import mean_forecast_sum
    tfm_scores = mean_forecast_sum(panel, HOR).unstack(level="symbol")
    tfm_rank = cross_sectional_rank(tfm_scores.T).T if tfm_scores is not None else None

    dates_ts = pd.DatetimeIndex(dates)

    # ---- Part 1: Walk-forward alpha selection ----
    wf_results = []
    print("=" * 70, flush=True)
    print("WALK-FORWARD VALIDATION (optimize alpha on prior, test on next year)", flush=True)
    print("=" * 70, flush=True)
    for test_year in TEST_YEARS:
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year}-12-31")
        if test_start > dates_ts.max():
            break

        # Optimize alpha on prior window
        prior_dates = [d for d in dates if d < test_start]
        test_dates = [d for d in dates if test_start <= d <= test_end]
        if not prior_dates or not test_dates:
            continue

        best_alpha, best_sharpe = 0.0, -999
        for alpha in ALPHAS:
            m = run_backtest_subset(cfg, ctx, state, prior_dates, tfm_rank, alpha, bench)
            if m and m["sharpe"] > best_sharpe:
                best_sharpe = m["sharpe"]
                best_alpha = alpha

        # Apply best_alpha to test year (OOS)
        m_oos = run_backtest_subset(cfg, ctx, state, test_dates, tfm_rank, best_alpha, bench)
        if m_oos:
            wf_results.append({
                "test_year": test_year,
                "selected_alpha": best_alpha,
                "oos_sharpe": m_oos["sharpe"],
                "oos_cagr": m_oos["cagr"],
                "oos_maxdd": m_oos["max_drawdown"],
                "oos_calmar": m_oos["calmar"],
            })
            print(f"  {test_year}: alpha={best_alpha} → OOS Sharpe={m_oos['sharpe']:.3f} "
                  f"CAGR={m_oos['cagr']:.3f} MaxDD={m_oos['max_drawdown']:.3f}", flush=True)

    # Aggregate OOS
    if wf_results:
        avg_sharpe = np.mean([r["oos_sharpe"] for r in wf_results])
        alpha_counts = pd.Series([r["selected_alpha"] for r in wf_results]).value_counts().to_dict()
        print(f"\n[wf] Avg OOS Sharpe: {avg_sharpe:.3f}", flush=True)
        print(f"[wf] Alpha selection distribution: {alpha_counts}", flush=True)

    # ---- Part 2: Sub-period robustness heatmap ----
    print("\n" + "=" * 70, flush=True)
    print("SUB-PERIOD ROBUSTNESS HEATMAP (Sharpe by alpha × period)", flush=True)
    print("=" * 70, flush=True)
    periods = {
        "pre-COVID (2015-2019)": ("2015-01-01", "2019-12-31"),
        "COVID (2020)": ("2020-01-01", "2020-12-31"),
        "post-COVID (2021-2026)": ("2021-01-01", "2026-12-31"),
        "full (2015-2026)": ("2015-01-01", "2026-12-31"),
    }
    heatmap = {}
    header = f"{'alpha':>8s}"
    for pname in periods:
        header += f" | {pname[:18]:>18s}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for alpha in ALPHAS:
        row = f"{alpha:>8.2f}"
        heatmap[alpha] = {}
        for pname, (ps, pe) in periods.items():
            subset = [d for d in dates if pd.Timestamp(ps) <= d <= pd.Timestamp(pe)]
            m = run_backtest_subset(cfg, ctx, state, subset, tfm_rank, alpha, bench)
            sharpe = m["sharpe"] if m else float("nan")
            heatmap[alpha][pname] = sharpe
            row += f" | {sharpe:>18.3f}"
        print(row, flush=True)

    # Save
    if wf_results:
        pd.DataFrame(wf_results).to_csv("artifacts/timesfm/walk_forward_results.csv", index=False)
    # Save heatmap separately
    heatmap_df = pd.DataFrame(heatmap).T  # alpha as index, period as columns
    heatmap_df.to_csv("artifacts/timesfm/subperiod_heatmap.csv")
    save_summary("walk_forward", wf_results if wf_results else [{"status": "no_results"}])
    print("\n[wf] DONE — saved walk_forward_results.csv and subperiod_heatmap.csv", flush=True)


if __name__ == "__main__":
    main()
