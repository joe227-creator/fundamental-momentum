#!/usr/bin/env bash
# Crash-resilient driver: forecasts every date in the dates CSV by invoking the
# per-date worker. Skips dates already in the cache. Continues past segfaults
# (exit 139) / OOM; a final re-run mops up any missed dates.
# Usage: ./run_forecast_driver.sh [dates_csv] [ctx] [hor] [kind] [field]
set -u
DATES_CSV="${1:-cache/timesfm/dates_ctx256_logret_close.csv}"
CTX="${2:-256}"; HOR="${3:-21}"; KIND="${4:-logret}"; FIELD="${5:-close}"
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src
export TFM_BATCH=1
tail -n +2 "$DATES_CSV" | while IFS=, read -r d; do
  [ -z "$d" ] && continue
  echo "=== $d ==="
  python -u experiments/forecast_worker.py "$d" "$CTX" "$HOR" "$KIND" "$FIELD" 2>&1 | grep -v "Loading weights"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then echo "!!! worker exit $rc for $d (will retry on next run)"; fi
done
echo "DRIVER DONE"
