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

Static persistence keeps validated base selection and gradually moves holdings toward each monthly target.

- `rebalance_persistence=0.2872700594236812`.
- Point score `0.6806187648`; mean window return `0.4400518225`; return-on-risk `0.8257742913`.
- Win rate `0.8695652174`; max drawdown `-0.5328960070`; Sharpe `1.4396242880`.
- Turnover `78.7676118414`; drawdown penalty included by fixed objective.
- Ten-seed score mean `0.6486837050`; median `0.6526871073`; std `0.0322228874`; min `0.5819193522`; max `0.7108035343`.
- Robust floor proxy `mean-1sd=0.6164608176`, narrowly above baseline point score.
- Regime means: pre-COVID `0.276098`, COVID `0.609620`, post-COVID `0.584259`.
- Final promoted-branch reproduction: score `0.6806700401`; perturb mean `0.6568753335`; perturb min `0.6099800264`; turnover `78.7655903000`.

## Research Tree

Optuna tested context length, forecast uncertainty sizing, continuous liquidity sizing, persistence strength, deadband, persistence plus uncertainty, adaptive persistence, score smoothing, asymmetric persistence, walk-forward training windows, and minimum holding period. Only static persistence produced a robustness-adjusted gain. Adaptive persistence had the highest point score but catastrophic perturbation instability. Asymmetric persistence improved point score but failed the `mean-1σ` gate. Minimum holding period was inert at monthly cadence.

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

Run from commit on branch `orx/promoted-static-persistence-candidate`. Shared cache must be available at `/mnt/c/Users/User/Desktop/Test/Fundamental Momentum/cache` in WSL.
