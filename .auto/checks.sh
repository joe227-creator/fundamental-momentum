#!/bin/bash
set -euo pipefail
cd "C:/Users/User/Desktop/Weekly Script/Fundamental Momentum-1st day of the month"
export PYTHONPATH=src TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1

# 1) Validate params JSON has required keys and sane ranges.
.venv/Scripts/python.exe -c "
import json
p=json.load(open('research_params.json'))
req=['max_holdings','sma_fast','sma_slow','sma_trend','ts_mom_lookback_months','ts_mom_skip_months',
     'fundamental_group_weight','asset_growth_weight','earnings_yield_weight','macro_half_risk_threshold',
     'macro_flat_threshold','veto_threshold','trailing_vol_window','veto_horizon','cost_per_side',
     'min_avg_dollar_volume','min_price','universe_limit','require_positive_composite',
     'base_filter_requires_trend','use_indicator_ic_weights','indicator_weighting_scheme']
missing=[k for k in req if k not in p]
assert not missing, f'missing keys: {missing}'
assert p['max_holdings']>=1, 'max_holdings<1'
assert p['universe_limit']>=50, 'universe_limit<50'
assert p['cost_per_side']>=0.0002, 'cost_per_side below 2bps (unrealistic)'
assert 1<=p['sma_fast']<p['sma_slow']<p['sma_trend'], 'sma order violated'
print('params OK')
"

# 2) Compile-check harness + checklist (catch syntax errors fast).
.venv/Scripts/python.exe -m py_compile overfit_harness.py overfitting_checklist.py

# 3) Guard against overall collapse / NaN from the last measure.sh run.
.venv/Scripts/python.exe -c "
import re, math
try:
    txt=open('.auto/last_metrics.txt').read()
except FileNotFoundError:
    raise SystemExit('no last_metrics.txt')
def g(k):
    m=re.search(rf'^METRIC {k}=([-+\w\.]+)', txt, re.M)
    return float(m.group(1)) if m else None
w=g('worst_year_sharpe'); s=g('sharpe'); c=g('cagr'); n=g('n_years'); ps=g('perturb_worst_year_std')
assert w is not None and not math.isnan(w), 'worst_year_sharpe missing/NaN'
assert n==11, f'n_years={n} (expect 11 full years)'
assert s is not None and s>=0.5, f'sharpe {s} below 0.5 floor (overall collapse)'
assert c is not None and c>=0.15, f'cagr {c} below 0.15 floor (overall collapse)'
print(f'checks OK: worst_year={w:.4f} sharpe={s:.4f} cagr={c:.4f} perturb_worst_std={ps}')
"
