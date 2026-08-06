#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH=src
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Local CPU runs share immutable cached market data outside Git clones.
SHARED_CACHE="/mnt/c/Users/User/Desktop/Weekly Script/Fundamental Momentum-1st day of the month/cache"
if [[ ! -e cache && -d "$SHARED_CACHE" ]]; then
  ln -s "$SHARED_CACHE" cache
fi

mkdir -p .openresearch/artifacts
/home/user/venv/bin/python research_score.py \
  --config config/research.json \
  --params research_params.json \
  --reference research/baseline_reference.json \
  --artifact-dir .openresearch/artifacts \
  --perturb-runs 5 \
  --perturb-sigma 0.03 \
  --seed 42 2>&1 | tee .auto/last_metrics.txt
