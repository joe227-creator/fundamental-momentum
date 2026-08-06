"""TimesFM inference engine for the momentum experiments.

Wraps the *transformers* port of google/timesfm-2.5-200m-transformers.

Reality notes (see EXPERIMENT_NOTES.md):
  * The transformers port has NO `.forecast(inputs=, freq=)` method (that is the
    official google `timesfm` paxtorch API). We call
    ``model(past_values=[...], forecast_context_len=L)`` and slice the first H
    steps of the 128-step output.
  * Context length must be a multiple of patch_len=32. We left-pad / truncate
    to the requested context length.
  * Output: ``mean_predictions`` (B,128) and ``full_predictions`` (B,128,10).
    Column 5 of full_predictions == mean_predictions. Columns ascend from the
    lower tail (col0, downside proxy) to the upper tail (col9, upside proxy).
  * CPU inference is segfault-prone on Windows with multi-threading / large
    batches. We pin to a single torch thread and a small batch size with retry.
  * All forecasts are disk-cached keyed by (context_len, horizon, series_kind,
    quantile) so walk-forward / sub-period reruns never recompute.
"""
from __future__ import annotations

import os
import hashlib
import threading
import time
from pathlib import Path
from typing import Sequence

import gc
import numpy as np
import pandas as pd

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_MODEL = None
_MODEL_LOCK = threading.Lock()

PATCH_LEN = 32
NATIVE_HORIZON = 128
N_QUANTILE_COLS = 10
MEDIAN_COL = 5  # full_predictions[...,5] == mean_predictions
DOWNSIDE_COL = 0  # lower-tail proxy for the spec's 5th percentile
UPSIDE_COL = 9  # upper-tail proxy for the spec's 95th percentile

DEFAULT_BATCH = 8
MAX_RETRIES = 4

CACHE_ROOT = Path("cache/timesfm")


def get_model():
    """Lazy singleton load of the TimesFM model."""
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                import torch

                torch.set_num_threads(1)
                from transformers import TimesFm2_5ModelForPrediction

                _MODEL = (
                    TimesFm2_5ModelForPrediction.from_pretrained(
                        "timesfm-2.5-200m-transformers"
                    )
                    .to(torch.float32)
                    .eval()
                )
    return _MODEL


def _patch_align(length: int) -> int:
    """Round a context length up to the nearest multiple of PATCH_LEN."""
    if length <= PATCH_LEN:
        return PATCH_LEN
    return int(PATCH_LEN * np.ceil(length / PATCH_LEN))


def _prepare_context(series: Sequence[float], context_len: int, kind: str) -> np.ndarray:
    """Return a 1-D float32 array of length context_len.

    For log-return / volume series we left-pad with 0.0 (neutral log-return) so
    the most recent observations sit at the right edge of the context window.
    For price series we left-pad with the earliest available value.
    """
    arr = np.asarray(series, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        arr = np.zeros(1, dtype=np.float32)
    if arr.size >= context_len:
        return arr[-context_len:].astype(np.float32)
    pad_value = 0.0 if kind in ("logret", "volume", "logvol") else float(arr[0])
    pad = np.full(context_len - arr.size, pad_value, dtype=np.float32)
    return np.concatenate([pad, arr]).astype(np.float32)


def _raw_forward(batch: list[np.ndarray], context_len: int, horizon: int, quantiles: bool):
    """Run one model forward pass. Returns (mean (B,horizon), full (B,horizon,10) or None)."""
    import torch

    model = get_model()
    past = [torch.tensor(a, dtype=torch.float32) for a in batch]
    with torch.no_grad():
        out = model(past_values=past, forecast_context_len=context_len, horizon_length=horizon)
    mean = out.mean_predictions[:, :horizon].cpu().numpy()
    full = None
    if quantiles:
        full = out.full_predictions[:, :horizon, :].cpu().numpy()
    del out, past
    return mean, full


def _batched_forward(
    arrays: list[np.ndarray], context_len: int, horizon: int, quantiles: bool, batch: int = DEFAULT_BATCH
):
    n = len(arrays)
    means = np.empty((n, horizon), dtype=np.float32)
    fulls = np.empty((n, horizon, N_QUANTILE_COLS), dtype=np.float32) if quantiles else None
    for start in range(0, n, batch):
        end = min(start + batch, n)
        chunk = arrays[start:end]
        b = len(chunk)
        m, f = None, None
        for attempt in range(MAX_RETRIES):
            try:
                m, f = _raw_forward(chunk, context_len, horizon, quantiles)
                means[start:end] = m
                if quantiles:
                    fulls[start:end] = f
                break
            except RuntimeError:
                # CUDA/OOM-style; halve batch and retry
                if b <= 1:
                    raise
                # fall through to single-item loop
                for j, a in enumerate(chunk):
                    m1, f1 = _raw_forward([a], context_len, horizon, quantiles)
                    means[start + j] = m1[0]
                    if quantiles:
                        fulls[start + j] = f1[0]
                break
        else:
            raise RuntimeError("TimesFM forward failed after retries")
        # free intermediate tensors to avoid memory accumulation segfaults
        if m is not None:
            del m
        if f is not None:
            del f
        gc.collect()
    return means, fulls


def _cache_key(context_len: int, horizon: int, kind: str, quantiles: bool) -> Path:
    q = "q" if quantiles else "m"
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return CACHE_ROOT / f"ctx{context_len}_hor{horizon}_{kind}_{q}.parquet"


def forecast_panel(
    requests: pd.DataFrame,
    context_len: int,
    horizon: int,
    kind: str = "logret",
    quantiles: bool = False,
    batch: int = DEFAULT_BATCH,
) -> pd.DataFrame:
    """Forecast a panel of (symbol, date) -> context series.

    Parameters
    ----------
    requests : DataFrame with columns ['symbol','date','series'] where
        'series' is an array-like of the raw input series (pre-padding). Index
        values must be unique per (symbol, date).
    context_len : desired context length; rounded up to a multiple of 32.
    horizon : forecast horizon (<=128); output is sliced to this.
    kind : 'logret' | 'volume' | 'logvol' | 'price' — controls padding.
    quantiles : if True, also return full quantile columns.

    Returns a DataFrame indexed by (symbol, date) with columns
    m0..m{horizon-1} and, if quantiles, q{c}_h{h} for c in 0..9, h in 0..horizon-1.
    """
    ctx = _patch_align(context_len)
    cache_path = _cache_key(ctx, horizon, kind, quantiles)

    index_cols = ["symbol", "date"]
    requests = requests.copy()
    requests["date"] = pd.to_datetime(requests["date"]).dt.normalize()
    requests = requests.set_index(index_cols).sort_index()
    requested_keys = requests.index

    results: dict[tuple, np.ndarray] = {}
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if not isinstance(cached.index, pd.MultiIndex):
            cached = cached.set_index(index_cols)
        cached = cached.sort_index()
        cached_keys = cached.index.intersection(requested_keys)
        mean_cols = [c for c in cached.columns if c.startswith("m") and c[1:].isdigit()]
        H_cached = len(mean_cols)
        if H_cached >= horizon:
            use_mean = [f"m{h}" for h in range(horizon)]
            for key in cached_keys:
                results[key] = (cached.loc[key, use_mean].to_numpy(dtype=np.float32), None)
            if quantiles:
                qcols = [c for c in cached.columns if c.startswith("q")]
                if qcols:
                    use_q = [f"q{c}_h{h}" for c in range(N_QUANTILE_COLS) for h in range(horizon)]
                    use_q = [c for c in use_q if c in cached.columns]
                    # rebuild full arrays
                    for key in list(results.keys()):
                        row = cached.loc[key]
                        full = np.empty((horizon, N_QUANTILE_COLS), dtype=np.float32)
                        for c in range(N_QUANTILE_COLS):
                            for h in range(horizon):
                                col = f"q{c}_h{h}"
                                full[h, c] = row[col] if col in cached.columns else np.nan
                        results[key] = (results[key][0], full)

    missing = requested_keys.difference(results.keys())
    if len(missing):
        sub = requests.loc[missing]
        arrays = [_prepare_context(s, ctx, kind) for s in sub["series"]]
        means, fulls = _batched_forward(arrays, ctx, horizon, quantiles)
        for i, key in enumerate(missing):
            results[key] = (means[i], fulls[i] if fulls is not None else None)

    # assemble output
    rows = []
    for key in requested_keys:
        m, f = results[key]
        row = {"symbol": key[0], "date": key[1]}
        for h in range(horizon):
            row[f"m{h}"] = float(m[h])
        if quantiles and f is not None:
            for c in range(N_QUANTILE_COLS):
                for h in range(horizon):
                    row[f"q{c}_h{h}"] = float(f[h, c])
        rows.append(row)
    out = pd.DataFrame(rows).set_index(index_cols).sort_index()

    # persist (merge with existing cache) -- atomic write to avoid corruption on crash
    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
        except Exception:
            # Cache file corrupt — start fresh
            cached = None
        if cached is not None:
            if not isinstance(cached.index, pd.MultiIndex):
                cached = cached.set_index(index_cols)
            out = pd.concat([cached, out[~out.index.isin(cached.index)]])
            out = out[~out.index.duplicated(keep="last")]
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    out.to_parquet(tmp)
    os.replace(tmp, cache_path)
    return out


def mean_forecast_sum(panel: pd.DataFrame, horizon: int) -> pd.Series:
    """Sum of the first `horizon` mean-forecast steps, indexed by (symbol,date)."""
    cols = [f"m{h}" for h in range(horizon)]
    return panel[cols].sum(axis=1)


def quantile_path(panel: pd.DataFrame, col: int, horizon: int) -> pd.DataFrame:
    """Return a (symbol,date) x horizon DataFrame for quantile column `col`."""
    cols = [f"q{col}_h{h}" for h in range(horizon)]
    return panel[cols].rename(columns={c: int(c[3:].split("_h")[1]) for c in cols})
