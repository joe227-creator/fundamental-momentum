# Champion (iter21) on SPY/S&P 500 vs IWV — Generalization Test

**Question:** Does the IWV-tuned champion transfer to the S&P 500? Are they the same?

**Setup:** Champion iter21 params applied to the S&P 500 universe (503 constituents,
Wikipedia list, top-500 truncated) with SPY as benchmark. Same survivorship caveat
as IWV (current constituents = survivors; returns OPTIMISTIC vs tradable PIT).

## Headline: They are NOT the same — SPY is BETTER, and the champion generalizes (not overfit)

| Metric | IWV champion (iter21) | SPY/S&P 500 (champion) | Verdict |
|---|---|---|---|
| **worst_year_sharpe (PRIMARY)** | 0.312 (2022) | **0.350** (2022) | SPY higher |
| robust_score | 0.78 | **0.83** | SPY higher |
| sharpe | 0.97 | **1.24** | SPY higher |
| cagr | 0.44 | **0.57** | SPY higher |
| max_drawdown | -0.42 | -0.44 | ~same |
| calmar | 1.06 | **1.31** | SPY higher |
| sortino | 1.40 | **1.80** | SPY higher |
| mean_year_sharpe | 0.93 | **1.10** | SPY higher |
| std_year_sharpe | 0.30 | 0.54 | SPY MORE dispersed |
| perturb_worst_year_std | 0.28 | **0.09** | SPY 3x more stable |
| perturb_mean_worst_year | +0.04 | **+0.08** | SPY higher (perturbed floor) |
| veto_pct | 8.5% | 8.3% | ~same |
| years positive | 11/11 | 11/11 | same |

## Per-year Sharpe (2022 is the floor for BOTH)

| Year | IWV | SPY |
|---|---|---|
| 2015 | 0.81 | 0.42 |
| 2016 | 1.41 | 1.06 |
| 2017 | 1.12 | 1.38 |
| 2018 | 0.94 | 1.60 |
| 2019 | 0.72 | 0.68 |
| 2020 | 1.40 | 1.03 |
| 2021 | 0.70 | 1.64 |
| **2022 (floor)** | **0.31** | **0.35** |
| 2023 | 0.92 | 1.18 |
| 2024 | 1.10 | 2.18 |
| 2025 | 0.83 | 0.60 |

## Key findings
1. **NOT the same.** SPY/S&P 500 beats IWV on the primary (worst_year 0.35 vs 0.31)
   and almost every secondary (sharpe, cagr, robust, calmar, sortino).
2. **The champion generalizes — strong evidence it is NOT overfit to IWV.** IWV-tuned
   params IMPROVE on SPY rather than degrade. An overfit champion would collapse on a
   different universe; this one strengthens.
3. **SPY is more peak-dependent** (stdY 0.54 vs 0.30): huge 2024 (2.18), 2018 (1.60),
   2021 (1.64). Large-cap momentum has bigger runs but more dispersion.
4. **SPY perturbation stability is 3x better** (perturb_worst_std 0.09 vs 0.28) —
   large-caps are far more stable under selection noise.
5. **2022 is the worst year for BOTH** and the champion protects it in both universes
   (floor positive). The weak-year robustness objective holds across universes.

## Caveats (honest)
1. **Survivorship bias** (SAME as IWV): SPY universe = current S&P 500 survivors, no
   delisted names. Returns OPTIMISTIC vs tradable PIT. NOT eliminated.
2. **Veto coverage gap**: only 195/503 SPY symbols have cached TimesFM volume
   forecasts; 308 (61%) are never vetoed. The champion's volume veto (#2 lever) is
   largely inactive for SPY — yet SPY still performs BETTER, suggesting large-caps
   need the liquidity veto less (inherently liquid). A faithful full-veto SPY test
   needs regenerating 308 symbols' forecasts (TimesFM model; torch NOT installed in
   this env, so not feasible here without setup).
3. Benchmark = SPY price; universe = S&P 500 constituents (Wikipedia list = same
   names as IVV/SPY hold).
4. Champion params were TUNED on IWV; the SPY result is an out-of-sample UNIVERSE
   test (not tuned on SPY) — which is why the clean generalization is meaningful.

## Artifacts
- SPY run output: `.auto/spy_champion_run.txt`
- SPY holdings list: `cache/spy500_holdings.csv`
- IWV holdings backup: `cache/iwv_holdings.csv.iwv_backup` (restored to cache/iwv_holdings.csv)
- IWV config restored: config/research.json (etf_ticker/benchmark_ticker = IWV)
- SPY price panel cached in cache/prices/ (reusable if switching back to SPY)
