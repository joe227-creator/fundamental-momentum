#!/bin/bash
set -euo pipefail
cd "C:/Users/User/Desktop/Fundamental Momentum Test"
export PYTHONPATH=src
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
.venv/Scripts/python.exe overfit_harness.py \
  --config config/research.json \
  --params research_params.json \
  --perturb-runs 8 \
  --perturb-sigma 0.03 \
  --seed 42 2>&1 | tee .auto/last_metrics.txt
