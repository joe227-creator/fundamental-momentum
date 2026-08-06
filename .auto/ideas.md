# Ideas backlog — SPY/S&P 500 full-veto optimization + TSFM research directions

## STATUS: SPY OPTIMIZATION CONVERGED at iter83 champion (worst_year 0.596, 19.9x baseline).
85 experiments (30 keeps / 55 discards). AUTORESEARCH CONCLUDED -- no high-impact untested
direction remains. iter83 = iter20 base (ts_mom10 + sma3/50/210 + P20 veto 0.35 + vol_scaled
+ max_holdings=2) + conviction tilt (weight_tilt=0.17, conf_tilt, pow=50) + vol-adjusted
(conf_tilt_vol) + vol_pow (conf_tilt_vol_pow=4, interior optimum). PIT+coverage audit CLEAN
(100% universe coverage 2015-2025, 0 NaN, 0 look-ahead). Floor 0.435->0.596 (+37%) via
SIZING (not selection). Two breakthroughs this final session: (1) the vol_pow sweep
(iter78-80, +0.034, interior optimum at 4: 1->2->3->4 PEAK->5->8 regress); (2) the
vol_pow=4 unlocked weight_tilt=0.17 (iter83, robust 0.9499 at 0.95 stop; 0.18 breaks it).
weight_tilt CONVERGED at 0.17. ALL directions exhausted (selection/sizing/veto/rebalance/
TSFM-upgrades). Validated not-overfit (OOS 0 neg folds min +0.730, perturb guard +0.020<0.10).
See .auto/prompt.md CONCLUSION section.

## VALIDATED (New directions.txt phases)
- **Phase 3 — Quantile P20 veto: VALIDATED.** The breakthrough. P20 (col 1,
  threshold 0.30, 1.4% veto) replaced the mean veto (20.7%) -> 10x floor lift,
  all 6 tests PASS. Quantile bracketed: P10 (col 0) uncalibratable (0% veto at
  reasonable thresholds), P20 BEST (0.337), P30 (col 2) under-vetoes (0.333).

## REJECTED (New directions.txt phases — tested, wrong direction)
- **Phase 4 — Volatility gate: REJECTED.** A realized-vol z-score veto HURT the
  2022 floor (0.34 -> 0.11) because in bear markets the good momentum names ARE
  high-vol (vol expands for momentum names), so a vol-gate systematically vetoes
  the 2022 WINNERS. Volatility correlates with the momentum signal itself; the P20
  volume veto works because LIQUIDITY != VOLATILITY. Do NOT retry vol-gates on this
  momentum strategy.

## NEW TSFM RESEARCH DIRECTIONS (from New directions.txt — high-value roadmap)
The TimesFM volume veto is now FULL coverage (100%) on SPY. New directions.txt
surveys 5 recent TSFM-in-finance papers with a 5-phase upgrade roadmap. Ranked
by effort/impact/risk for THIS harness:

### Phase 3 — Probabilistic/Quantile veto (LOW effort, LOW risk, MODERATE impact) ★TRY FIRST
TimesFM natively outputs quantile forecasts (P10..P90). Replace the current
MEAN-ratio veto (pv_mean/tv < threshold) with a DOWNSIDE-quantile veto
(P20 forecast volume / trailing volume < threshold). This encodes tail-liquidity
risk explicitly and should reduce losses in liquidity crises (the 2022/2018
floors). The harness already loads quantile panels (`load_forecast_panel(...,
quantiles=True)`); the veto would need a new code path in a NEW harness file
(the core `build_veto_mask` uses mean only — off-limits to modify). Generate
the quantile cache with `generate_spy_forecasts.py` adapted to `quantiles=True`.
This is the most principled, lowest-overfitting-risk TSFM upgrade and directly
targets the weak-year floor.

### Phase 4 — TSFM volatility gate as a SECOND risk gate (MEDIUM effort, MEDIUM risk, HIGH impact)
Add a parallel veto on forecasted VOLATILITY (not just volume): veto/half-size
if forecasted vol z-score (vs 52-week rolling) exceeds a threshold. Targets the
recurring drawdowns. Needs a volatility forecast cache (TimesFM on log-returns,
kind="logret") — the engine supports it. New harness code path.

### Phase 1 — Kronos backbone replacement (LOW effort, LOW risk, HIGH impact)
Kronos (arXiv:2508.02739, NeurIPS 2025) is a finance-native TSFM pre-trained on
12B K-line (OHLCV) records. It natively encodes volume alongside price and
improves price-forecast RankIC by 93% and vol-forecast MAE by 9% vs generic
TSFMs. Open-source on HuggingFace. Drop-in replacement for TimesFM in the
forecast generator: same veto logic, new model. Expect better veto precision.
Risk: need to install kronos + adapt the generator's forward call.

### Phase 2 — TimesFM-ICF in-context forecasting (VERY LOW effort, VERY LOW risk, MODERATE impact)
TimesFM-ICF (ICML 2025) accepts multiple related time-series as in-context
examples. Feed 5-10 same-GICS-sector stocks' recent volume as context alongside
the target -> ~6.8% lower volume forecast error, no retraining. Requires the
ICF checkpoint and adapting `build_context_requests` to emit multi-series
contexts (in a new harness file).

### Phase 5 — Prune-then-finetune adapter (HIGH effort, MEDIUM-HIGH risk, VERY HIGH impact)
Structured pruning (30-50% sparsity) of TimesFM/Kronos then adapter-only fine-
tuning on SPY OHLCV (per "Trading with the Devil": adapters preserve systematic
backbone risk; full fine-tuning adds unrewarded idiosyncratic noise). Highest
effort; defer until Phases 1-4 explored.

### Critical research caveats (from the papers)
- Chronos study: even Sharpe>3 gross collapses to negative NET after 3bps
  slippage -> use TSFM signals only as COARSE gates (as now), not high-turnover
  alpha. The 3bps cost model + skip-when-unchanged already mitigate this.
- "Trading with the Devil": foundation-model alpha DECAYS as strategies crowd ->
  build a rolling-window veto-effectiveness monitor.
- "Re(Visiting) TSFMs in Finance": off-the-shelf TSFMs disappoint zero-shot;
  finance-native pre-training (Kronos) is the most urgent upgrade path.

## EXHAUSTED on SPY (do NOT retry)
- veto_threshold: 0.35 is the P20 Pareto sweet spot (FULLY BRACKETED: 0.25 under-vetoes
  -> 0.333, 0.30 floor-robust -> 0.4347, 0.35 Pareto -> 0.4347 + best secondaries,
  0.40 crashes floor -> 0.166). Mean-veto 0.80 was a sharp fragile peak (over-vetoed 20.7%).
  NOTE: 0.35 was return-chasing on the 201 base (iter7) but a clean Pareto win on the
  210 base (iter20) -- the 210 base absorbs the extra vetoing without destabilizing.
- quantile_col: 1 (P20) is the sweet spot. P10 uncalibratable (0% veto), P30 under-vetoes.
- trailing_vol_window: 21 (34 crashes 2022+2023).
- veto_horizon: 21 (10 is a BIT-IDENTICAL no-op at P20 threshold 0.30 -- vetoed set
  is horizon-invariant; the P20 veto is robust to the forecast-averaging window).
- max_holdings: 2 (3 crashes 2022+2018 negative).
- ts_mom_lookback: 10. GRID FULLY MAPPED (iter31 robustness check): 8(0.190 crash),
  9(0.313 return-chasing), 10(0.435 BEST), 11(0.267 overfit). SHARP stable peak at 10.
  9 is return-chasing (lifts 2017/19/24, drops 2022 floor + dispersion).
- ts_mom_skip: 0 (10-1 crashes 2022 to 0.12 + perturb_worst_std explodes 0.10->0.22;
  same IWV iter25 pattern -- skip=1 chases avg return at the floor's expense).
- sma_trend: 210 is the sweet spot. GRID FULLY MAPPED (iter30 robustness check):
  190(0.337), 200(0.337), 210(0.435 BEST), 220(0.356). SHARP stable peak at 210
  (perturb_worst_std 0.089 at 210 vs 0.132 at 200). Boundary between 200 and 210 is
  sharp (critical ~205-210-day trend timescale). 205 untested = boundary-fishing (skip).
- sma_slow: 50. GRID FULLY MAPPED (iter37 medium-term regime test): 45(0.356),
  50(0.435 BEST), 55(0.316), 100(0.147 -- medium-term too lagging, crashed 2022).
  Sharp peak at 50. The 3/50 (short-term) is optimal -- responsive enough to catch
  2022 trends without whipsawing; medium-term (100) misses the choppy bear turns.
- sma_fast: 3 (2 drops to 0.316 on 210 base; 4 no-op). Bracketed.
- vol_lookback: 21 (34 crashes 2022 to 0.10 + perturb explodes; same pattern as ts_mom_skip).
- vol-gate (Phase 4): REJECTED (hurts 2022 -- vol correlates with momentum signal;
  vetoing high-vol names vetoes the 2022 winners).
- ema_crossover: REJECTED (iter23). Activating the degenerate 21/21 -> 21/63 is
  return-chasing: boosts strong years (sharpe 1.50->1.58) but hurts the 2022 floor
  (0.43->0.29). Trend signals whipsaw in 2022's choppy bear market, adding noise to
  the composite. Same lesson as vol-gate. The degenerate 21/21 (EMA OFF) is correct.
- fundamentals (fund_group/asset_growth/earnings_yield): INERT at 2% weight (iter22:
  bit-identical no-op) and HARMFUL at >=4% (iter27: fund_group 0.10 drops 2022 floor
  0.43->0.39; checklist 0.04 drops sharpe). 98% technical dominates. macro thresholds:
  dead (IWV lesson; no-op on SPY too).
- min_volume: INERT for the floor (iter40). Champion's 11M filters 0 SPY names
  (IWV-inherited no-op). 50M filters 79 names (16.2%) but the 2022 top-2 are ALWAYS
  high-liquidity (>50M adv) -- 79 filtered names never selected in 2022. Primary
  bit-identical (0.4347). P20 veto (3%) is the ONLY liquidity mechanism touching
  the selection. min_volume CLOSED (11M no-op is fine).
- ema_crossover: REJECTED (iter23, return-chasing, hurts 2022 floor).
- tfm_alpha (return forecast): REJECTED (iters25-26, anti-predictive at any weight).
- relative veto (cross-sectional percentile): REJECTED (iter28, holds 2022 but trades
  off 2015; can't LIFT the structural 0.4347 floor).
- regime-cash overlay (trend-following, 50% cash when SPY<210-SMA): REJECTED (iter29,
  HURTS 2022 -- lagging 210-SMA cuts after drop, misses V-rebounds; dilutes winners).
- require_positive_composite: non-binding no-op (iter24, top-2 always positive).
- indicator-disable (rsi): INERT no-op (iter32, bit-identical). The composite is
  ts_mom+sma DOMINATED; ~8 secondary indicators (rsi/cci/macd/obv/bollinger/breakout/
  variable_week_high/price_action/ema_crossover) don't reshape the top-2. Not worth
  sweeping disables (all inert -- top-2 is ts_mom+sma determined regardless).
- use_indicator_ic_weights/grouped_ic: DISASTER on IWV (iter23, worst_year -1.41,
  overfit per-fold). grouped_equal is optimal.
- vol_scaled_weights: TRUE is optimal on SPY (iter33, confirmed). vol_scaled is a REAL
  lever (unlike inert indicators) -- inverse-vol lifts the 2015 weak year (0.425->0.489).
  Equal-weight (false) is MORE perturbation-stable (0.085 vs 0.095) but lower floor
  (0.4253 vs 0.4347, 2015 becomes floor). Champion correctly chose the higher floor.

## DEFERRED (bigger efforts, not tried — from New directions.txt)
- **Phase 1 — Kronos backbone replacement: ASSESSED, MISMATCHED with the P20 champion.**
  Kronos (NeoQuasar/Kronos-base, OHLCV-native) has NO native quantile output —
  `predict(sample_count=1)` returns a single mean/median forecast. My champion's
  breakthrough is the P20 DOWNSIDE-quantile veto, which needs native quantiles.
  Getting P20 from Kronos requires sample_count=20 + computing the 20th percentile
  = 20x compute (~7h GPU for 80K forecasts). And Kronos-mean would reproduce the
  MEAN veto that over-vetoes (the root cause of the fragile 0.034 floor). So Kronos
  is NOT a clean drop-in for the P20 veto. TimesFM's native P10-P90 (one forward
  pass) is BETTER suited to the downside-quantile veto. Kronos would only make sense
  if reverting to a mean veto (a step backward). DEFERRED unless the veto mechanism
  changes. Repo: github.com/shiyu-coder/Kronos (repo-based install, old HF API,
  dep-conflict risk with transformers 5.12).
- **tfm_alpha_weight (TimesFM return-alpha signal): TESTED + REJECTED (iters25-26).**
  Did the big effort: adapted generate_spy_forecasts.py for --kind logret, regenerated
  the logret cache for 581 SPY symbols on GPU (49K forecasts, 15min, 100% coverage),
  added a PIT-safe tfm_alpha code path to overfit_harness.py (z-scored cumulative
  21-step log-ret forecast added to base_scores; weight=0.0 verified bit-identical
  no-op). RESULT: both weight=1.0 AND weight=0.1 CRASH 2022 to -0.44 (3 negative
  years at w=1.0, 2 at w=0.1). The return forecast is ANTI-PREDICTIVE at ANY positive
  weight -- it chases recent returns, picking mean-reverting stocks in bear markets
  (pro-cyclical harm to the weak-year floor). NO sweet spot. CONFIRMS New directions.txt
  caveat: 'off-the-shelf TSFMs disappoint zero-shot for RETURN forecasting.' The
  volume veto works because LIQUIDITY != RETURN DIRECTION. Infrastructure retained
  (logret cache + harness path, commit 9f3f97a); tfm_alpha_weight=0.0 (OFF) in champion.
  Do NOT retry return-alpha signals on this momentum strategy.
- **Phase 2 — TimesFM-ICF in-context: INFEASIBLE (no public checkpoint).** Web
  search (2026-06) confirms TimesFM-ICF (ICML 2025) is a DISTINCT advance requiring
  continued pre-training; the base `timesfm-2.5-200m-transformers` does NOT support
  in-context multi-series prompting, and NO public ICF checkpoint exists on HuggingFace.
  Would improve the P20 forecast ACCURACY (not the mechanism) -> could tighten the
  veto. DEAD until Google releases an ICF checkpoint. Check HF for `timesfm-*icf*`.
- **Phase 5 — Prune-then-finetune adapter**: prune TimesFM 30-50% then adapter-only
  fine-tune on SPY OHLCV (per "Trading with the Devil": adapters preserve
  systematic backbone risk). Highest effort; could improve P20 forecast accuracy.
- **Lifting the 2022 floor above 0.43**: structural for 2-stock monthly momentum.
  Would need a fundamentally different signal (e.g., a defensive/bear-market factor
  tilt) or a 3rd conditional holding -- not currently supported by AppConfig.

## DATA INTEGRITY (not optimization)
- Backfill data/holdings_history/ with paid historical S&P 500 constituents to
  reduce survivorship bias. Until then, current-survivors caveat holds.
- generate_spy_forecasts.py is resumable (checkpoints every 8000 forecasts);
  re-run after any universe change to refresh coverage.

## Do NOT try (overfitting traps, confirmed on IWV)
- Don't tune SMA windows to year extremes.
- Don't zero/raise transaction costs.
- Don't optimize to 2026 partial year.
- Don't chase worst_year point-estimate gains that drop robust_score.
- Don't use partial-veto coverage as a "result" (it's coverage cheating).

## CONVICTON TILT + VOL-ADJUSTED + POW-SWEEP BREAKTHROUGH (iters42-76) -- at 0.554
The 2022 floor was "structural at 0.4347" (iter20-42, sizing-invariant). The CONVICTION
TILT broke through: floor 0.4347 -> 0.554 (+0.119, +27%). 22 keeps in one session
(iter43, 46-52, 54-56, 61-62, 67, 69-76). The key insight: SIZING (not selection) lifts the floor.

### Champion (iter76, commit bb09f05): weight_tilt=0.16 + conf_tilt + pow=50 + vol-adjusted
worst_year=0.554 (2015=0.554 floor, 2022=0.727, 2025=0.978 canary). sharpe=1.47,
cagr=0.80, robust=0.953, perturb_worst_std=0.167. ALL checks PASS.
Strategy = iter20 base (ts_mom10 + sma3/50/210 + P20 veto 0.35 + vol_scaled +
max_holdings=2) + conviction tilt (weight_tilt=0.16, conf_tilt=true, pow=50,
cap=weight_tilt) + vol-adjusted (conf_tilt_vol=true, window=21, cap=2.0).

### The mechanism (composite-gap CONFIDENCE signal -- NOT a return forecast)
The conviction tilt shifts weight toward the top-1 (strongest composite). The
CONFIDENCE-SCALED variant (conf_tilt) scales the tilt by the composite-gap confidence
(|top1-top2| / median_gap, capped at weight_tilt, raised to conf_tilt_pow). Tilt FULLY
in high-confidence months (large gap, clear winner -- e.g. 2022 energy) and LESS in
low-confidence months (small gap, close call -- e.g. 2015 choppy). This SEPARATES the
2022 lift (high confidence, top-1 outperforms) from the 2015 drop (low confidence, less
tilt -> 2015 protected). A CONFIDENCE signal works where return-based conditionals
(cond_tilt trailing-return iter44, regime_tilt 210-SMA iter45) FAILED -- both missed
the early-2022 outperformance (SPY was >210-SMA in early 2022).

### The pow sweep (steeper confidence = more 2015 protection)
conf_tilt_pow: 1.0(linear, 0.471) -> 2.0(quadratic, 0.492) -> 3.0(cubic, 0.497) ->
4.0(quartic, 0.499) -> 5.0(0.4996, CONVERGED). Each step STRICTLY DOMINATED (higher
floor + MORE stable + robust improved). The steeper pow drops the tilt faster in
low-confidence months -> 2015 protected more. The 2015 crossed 0.50 at pow=5.0.
Converged because the 2022 (high-conf months at full tilt cap) became the bottleneck.

### EXHAUSTED on the conviction tilt + vol-adjusted + pow-sweep (do NOT retry)
- weight_tilt magnitude (WITH conf_tilt_vol + pow=50): 0.14(0.536) -> 0.15(0.549) ->
  **0.16(0.554, robust 0.953, PEAK/iter76)** -> 0.17(0.559, robust 0.948, STOP/iter77).
  Robustness drops ~0.005/0.01 weight_tilt. The pow sweep's robustness cushion shifted
  the stop from 0.13 (pow=5/iter57) to 0.16 (pow=50). The 2025 canary binds (drops
  1.016->0.978 at 0.16). (At pow=5, the stop was at 0.13/iter57; vol-adjusted shifted
  it to 0.15/iter63.)
- conf_tilt_pow sweep (iter69-74): CONVERGED at 50 (the step-function asymptote).
  pow 5->6->7->8->10->20->50, each STRICTLY DOMINATED (floor up + robust up + perturb
  down). Floor converged at 0.5437 (pow=20==50); higher pow only improves robustness.
  The binary gate (full tilt when gap>=median, ~0 else) is the pow->inf limit. The pow
  sweep RE-OPENED at weight_tilt=0.14+vol-adjusted (was converged at 5.0 on iter48 base).
  Re-tuning pow at 0.16 is unnecessary (50 is the asymptote regardless of weight_tilt).
- conf_tilt_cap=false (NO-CAP, iter53): DISASTER. Over-tilts high-conf -> crashes 2017/2023/2025.
- conf_tilt_vol_cap (iter68): INERT. vol_factor never reaches 2.0; 2.5 is bit-identical.
- conf_tilt_vol_window (iter64-65): 10 noisy, 21 sweet spot, 63 too smooth. CONVERGED at 21.
- cond_tilt (iter44): DISCARD. Skips some 2022 months.
- regime_tilt (iter45): DISCARD. Misses early-2022 outperformance.
- dir-adjusted (iter66): REJECTED. Helps underperformers but HURTS the 2015 floor.
- 2-tier bonus (iter58-60): REJECTED. 2015/2017 both very-high-conf -- no percentile separates.
- The 2025 canary (moderate-vol vf=1.06, POSITIVE-return underperformer) is the binding
  constraint for weight_tilt. The 2015 floor (0.554) is the bottleneck. No PIT signal
  separates 2015 (outperformer) from 2025 (underperformer) -- both high-conf + positive-return.

### DEFERRED (could lift the floor above 0.554)
- A 2nd regime signal that protects the 2025 canary (moderate-vol, positive-return
  underperformer) without hurting the 2015 (moderate-vol, ~0-return outperformer).
  vol-adjusted fails (both moderate-vol vf~1); dir-adjusted hurts the 2015. Need a
  non-vol, non-direction signal (e.g. sector/bear-market factor) -- none found.
- A fundamentally different selection mechanism (all selection mods crashed 2022:
  tfm_alpha/EMA/fundamentals/corr). Or a new strategy architecture.
- Phase 5 (prune-then-finetune TimesFM) won't help -- the floor is selection-determined
  (veto-invariant), not veto-determined.
