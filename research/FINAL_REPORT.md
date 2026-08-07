# Fundamental Momentum Research Report

## Objective

Fixed Research score:

`mean_rolling_6m_return + 0.20*return_on_risk + 0.10*win_rate - 0.35*max(0,-0.50-maximum_drawdown) - 0.15*max(0,0.80-Sharpe) - 0.10*max(0,turnover-baseline_turnover)`

Evaluation uses 23 complete non-overlapping 126-session windows from 2015-01-02, unchanged fees, open execution, frozen baseline turnover, and cached point-in-time inputs.

## Baseline

- Experiment `316058b8-7191-4747-bcd7-c9cff510688d`, commit `93abc97ed06328645bccea2af4fdd97bde2be7f0`.
- Score `0.6149878552`; mean window return `0.3776043679`; return-on-risk `0.7738739580`.
- Win rate `0.8260869565`; max drawdown `-0.4879403991`; Sharpe `1.4116455037`.
- Turnover `98.5930757562`; five-seed perturb mean `0.5740708711`, std `0.0430848490`, min `0.5008842863`.

## Promoted Candidate

Nonlinear rank aggregation plus static persistence keeps validated base selection while emphasizing strong cross-sectional factor ranks.

- `rebalance_persistence=0.2872700594236812`.
- `rank_power=1.9966462104925915`.
- Point score `0.7130601543`; mean window return `0.4458845771`; return-on-risk `0.9009879030`.
- Win rate `0.8695652174`; max drawdown `-0.4948840887`; Sharpe `1.4515913614`.
- Turnover `65.6311684011`; drawdown penalty included by fixed objective.
- Ten-seed score mean `0.6762969477`; median `0.6742146697`; std `0.0198114240`; min `0.6493130878`; max `0.7193872420`.
- Every validation seed beats baseline score. Robust floor proxy `mean-1sd=0.6564855237`.
- Regime means: pre-COVID `0.269314`, COVID `0.616127`, post-COVID `0.605475`.
- Final promoted-branch reproduction: score `0.7130773067`; perturb mean `0.6693636357`; std `0.0168591783`; min `0.6493349203`; max `0.6978934682`; turnover `65.6700033189`.

## Research Tree

Optuna tested context length, forecast uncertainty sizing, continuous liquidity sizing, persistence strength, deadband, persistence plus uncertainty, adaptive persistence, score smoothing, asymmetric persistence, walk-forward training windows, minimum holding period, nonlinear rank aggregation, median consensus, macro-conditioned persistence, joint rank/persistence, factor-specific powers, factor agreement, sector concentration cap, soft score weighting, and confidence-conditioned breadth. Nonlinear rank aggregation produced robust leader. Adaptive persistence had high point score but catastrophic perturbation instability. Consensus and asymmetric candidates failed robust comparison. Macro, factor-specific, agreement, sector, soft-weight, and confidence-breadth paths were inert.

Full IDs, branches, parameters, metrics, and decisions: `research/experiment_ledger.csv`.

## Integrity

- Primary PIT audit 2015-2025: 0 missing of 62,248 eligible forecast pairs.
- Selected-name missing forecast count: 0.
- Look-ahead violations: 0 of 132 rebalance dates.
- Current-data limitation: historical holdings snapshots remain incomplete outside primary audit window; survivorship bias remains.
- Candidate max drawdown exceeds `-0.50`; objective penalizes this, but deployment risk remains material.
- Verification: evaluator modules compile; 22 tests passed and 2 skipped. Two inherited live-trading tests fail outside research path: blank position symbols serialize as `NAN`, and trade-log dates mix date-only and datetime strings. These were not changed during research promotion.

## Reproduction

```bash
bash .auto/measure.sh
```

Run from commit on branch `orx/promoted-nonlinear-rank-candidate`. Shared cache must be available at `/mnt/c/Users/User/Desktop/Test/Fundamental Momentum/cache` in WSL.
