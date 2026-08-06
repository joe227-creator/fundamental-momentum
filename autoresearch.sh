#!/bin/bash
# Autoresearch benchmark entrypoint.
# Runs the Experiment 13 (TimesFM volume veto) backtest with current parameters.
# Outputs METRIC lines: primary=cagr, secondary=sharpe, max_drawdown, etc.
set -e
cd "$(dirname "$0")"
export PYTHONPATH="src;experiments"
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python autoresearch.py
