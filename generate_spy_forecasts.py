"""Generate TimesFM volume forecasts for the S&P 500 (SPY) universe on GPU.

This is a NEW harness file (the off-limits core ``timesfm_engine.py`` runs on CPU
and cannot be modified). It reuses the core's PURE functions
(``_prepare_context``, ``_cache_key``) and faithfully replicates
``build_context_requests`` (eligible_only=False) + ``forecast_panel`` so the
output parquet is byte-compatible with the cache the harness reads.

It only forecasts (symbol, date) pairs MISSING from the existing cache, then
merges them in (atomic write). Idempotent: re-running is cheap.
"""
from __future__ import annotations
import os, sys, time, gc, argparse
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
from algo_trading.config import AppConfig
from algo_trading.data import load_market_context
from algo_trading.calendar_utils import select_rebalance_sessions
from algo_trading.timesfm_engine import _prepare_context, _cache_key, N_QUANTILE_COLS

CTX = 256
HOR = 21
KIND = "volume"
MIN_OBS = 64
BATCH = 128
CHECKPOINT_EVERY = 8000  # write partial cache every N forecasts (resumable)

# Mode set via CLI (--quantiles). Default: mean-only cache (_m.parquet).
QUANTILES = False
CACHE_PATH = _cache_key(CTX, HOR, KIND, quantiles=False)


def load_existing_cache() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(CACHE_PATH)
    if not isinstance(df.index, pd.MultiIndex):
        df = df.set_index(["symbol", "date"])
    return df.sort_index()


def build_missing_requests(context, config, missing_syms, existing_index):
    """Replicate build_context_requests(eligible_only=False) for missing syms.

    Series frame depends on KIND: 'volume' -> raw volume; 'logret' ->
    log_returns(close) = np.log(close).diff() (byte-compatible with the core's
    build_context_requests kind='logret', so the merged cache is uniform).
    """
    sessions = context.prices.sessions
    rebal = select_rebalance_sessions(sessions, config.strategy.rebalance_frequency)
    dates = [pd.Timestamp(d) for d in rebal
             if pd.Timestamp(d) >= pd.Timestamp("2015-01-01") and pd.Timestamp(d) in sessions]
    if KIND == "logret":
        series_frame = np.log(context.prices.close).diff()
    else:
        series_frame = context.prices.volume
    missing_set = set(missing_syms)
    # only columns we actually have prices for
    missing_set = {s for s in missing_set if s in series_frame.columns}
    rows = []
    for d in dates:
        pos = sessions.searchsorted(d, side="left")
        prior_end = sessions[pos - 1] if pos > 0 else d
        window = series_frame.loc[:prior_end]
        if window.empty:
            continue
        avail = window.tail(CTX + 4)  # 260
        for sym in missing_set:
            if (sym, d) in existing_index:
                continue
            s = avail[sym].dropna().to_numpy() if sym in avail.columns else np.array([])
            if s.size < MIN_OBS:
                continue
            rows.append({"symbol": sym, "date": d, "series": s[-(CTX + 4):]})
    return pd.DataFrame(rows)


def _checkpoint(new_rows_df: pd.DataFrame, existing: pd.DataFrame) -> None:
    """Merge partial results into the cache (atomic) so a killed run is resumable."""
    if new_rows_df.empty:
        return
    merged = pd.concat([existing, new_rows_df]) if not existing.empty else new_rows_df
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    tmp = CACHE_PATH.with_suffix(CACHE_PATH.suffix + ".tmp")
    merged.to_parquet(tmp)
    os.replace(tmp, CACHE_PATH)


def _row_cols():
    """Output columns for the current mode."""
    cols = [f"m{h}" for h in range(HOR)]
    if QUANTILES:
        cols += [f"q{c}_h{h}" for c in range(N_QUANTILE_COLS) for h in range(HOR)]
    return cols


def gpu_forward(rows: pd.DataFrame, batch: int = BATCH, existing: pd.DataFrame | None = None):
    """Run GPU batched forward. Returns DataFrame indexed by (symbol,date).
    Mean mode: m0..m{HOR-1}. Quantile mode: + q{c}_h{h} (c=0..9, h=0..HOR-1).
    Checkpoints partial results to the cache every CHECKPOINT_EVERY forecasts."""
    from transformers import TimesFm2_5ModelForPrediction
    print(f"Loading TimesFM model on GPU (fp16, quantiles={QUANTILES})...", flush=True)
    model = (TimesFm2_5ModelForPrediction
             .from_pretrained("timesfm-2.5-200m-transformers", torch_dtype=torch.float16)
             .to("cuda").eval())
    torch.set_grad_enabled(False)

    n = len(rows)
    means = np.empty((n, HOR), dtype=np.float32)
    fulls = np.empty((n, HOR, N_QUANTILE_COLS), dtype=np.float32) if QUANTILES else None
    series_arr = rows["series"].tolist()
    keys = list(zip(rows["symbol"], rows["date"]))

    def run_batch(chunk, bsz):
        prepared = [_prepare_context(s, CTX, KIND) for s in chunk]
        # float32 inputs (NOT float16): raw volume ~1e6-1e9 overflows float16 (max 65504) -> inf -> NaN.
        past = [torch.tensor(a, dtype=torch.float32, device="cuda") for a in prepared]
        out = model(past_values=past, forecast_context_len=CTX, horizon_length=HOR)
        m = out.mean_predictions[:, :HOR].cpu().numpy().astype(np.float32)
        f = None
        if QUANTILES:
            f = out.full_predictions[:, :HOR, :].cpu().numpy().astype(np.float32)
        del out, past
        return m, f

    i = 0
    t0 = time.time()
    last_ckpt = 0
    while i < n:
        bsz = batch
        while bsz >= 1:
            try:
                chunk = series_arr[i:i + bsz]
                m, f = run_batch(chunk, bsz)
                means[i:i + len(chunk)] = m
                if QUANTILES:
                    fulls[i:i + len(chunk)] = f
                i += len(chunk)
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "OOM" in str(e):
                    torch.cuda.empty_cache(); gc.collect()
                    if bsz <= 1:
                        # single-item also OOM -> skip this one with NaN
                        means[i] = np.nan
                        i += 1
                        break
                    bsz = max(1, bsz // 2)
                    print(f"  OOM -> retry batch {bsz}", flush=True)
                else:
                    raise
        if i % (batch * 5) == 0 or i >= n:
            done = i
            rate = done / max(1e-9, time.time() - t0)
            nan_so_far = int(np.isnan(means[:i]).all(axis=1).sum())
            print(f"  {done}/{n} forecasts ({rate:.1f}/s, batch {bsz}) | NaN-so-far: {nan_so_far}", flush=True)
        # incremental checkpoint (resumable on timeout/kill)
        if i - last_ckpt >= CHECKPOINT_EVERY:
            done_so_far = _assemble(means[:i], fulls[:i] if QUANTILES else None, keys[:i])
            _checkpoint(done_so_far, existing if existing is not None else pd.DataFrame())
            last_ckpt = i
            print(f"  checkpoint: wrote {i} partial forecasts to cache", flush=True)

    return _assemble(means, fulls, keys)


def _assemble(means: np.ndarray, fulls, keys) -> pd.DataFrame:
    """Build the output DataFrame from mean (+optional quantile) arrays."""
    data = {f"m{h}": means[:, h] for h in range(HOR)}
    if QUANTILES and fulls is not None:
        for c in range(N_QUANTILE_COLS):
            for h in range(HOR):
                data[f"q{c}_h{h}"] = fulls[:, h, c]
    out = pd.DataFrame(data)
    out["symbol"] = [k[0] for k in keys]
    out["date"] = [k[1] for k in keys]
    return out.set_index(["symbol", "date"]).sort_index()


def main():
    global QUANTILES, CACHE_PATH, KIND, CTX
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantiles", action="store_true", help="generate the quantile cache (_q.parquet)")
    ap.add_argument("--kind", default="volume", choices=["volume", "logret"],
                    help="series kind: 'volume' (raw volume) or 'logret' (log_returns close)")
    ap.add_argument("--ctx", type=int, default=256,
                    help="forecast context length (default 256; model supports up to 16384). "
                         "A longer context sees more history -> potentially more accurate forecasts. "
                         "Produces a SEPARATE cache (ctx{N}_hor21_*).")
    ap.add_argument("--rebalance-freq", default=None, choices=["weekly", "biweekly", "monthly"],
                    help="override config rebalance_frequency for request dates (default: use config). "
                         "WARNING: weekly/biweekly OVERWRITE the shared cache key (ctx{N}_hor21_*_q.parquet) "
                         "with weekly-dated forecasts -- back up the monthly cache first.")
    args = ap.parse_args()
    CTX = args.ctx
    QUANTILES = args.quantiles
    KIND = args.kind
    CACHE_PATH = _cache_key(CTX, HOR, KIND, quantiles=QUANTILES)
    print(f"Cache path: {CACHE_PATH} (kind={KIND}, quantiles={QUANTILES})", flush=True)
    print("Loading SPY market context (cache hit expected for prices)...", flush=True)
    config = AppConfig.from_file("config/research.json")
    if args.rebalance_freq:
        config.strategy.rebalance_frequency = args.rebalance_freq
        print(f"Override rebalance_frequency = {args.rebalance_freq} (for request dates)", flush=True)
    context = load_market_context(config=config, cache_dir="cache", refresh=False)
    universe_syms = list(context.prices.volume.columns)
    print(f"SPY universe symbols with prices: {len(universe_syms)}", flush=True)

    existing = load_existing_cache()
    existing_syms = set(str(s) for s in existing.index.get_level_values("symbol").unique()) if not existing.empty else set()
    existing_index = set(existing.index) if not existing.empty else set()
    print(f"Existing cache: {len(existing)} rows, {len(existing_syms)} symbols", flush=True)

    missing_syms = [s for s in universe_syms if s not in existing_syms]
    print(f"Missing symbols (need forecasts): {len(missing_syms)}", flush=True)
    if not missing_syms:
        print("Nothing to generate. Cache already covers the universe.")
        return

    rows = build_missing_requests(context, config, missing_syms, existing_index)
    print(f"Built {len(rows)} (symbol,date) request rows (after min_obs={MIN_OBS} filter)", flush=True)
    if rows.empty:
        print("No forecastable rows. Done.")
        return

    new_df = gpu_forward(rows, batch=BATCH, existing=existing)
    new_df = new_df.dropna(how="all")
    print(f"Generated {len(new_df)} new forecasts", flush=True)

    # merge with existing (atomic write)
    merged = pd.concat([existing, new_df]) if not existing.empty else new_df
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    tmp = CACHE_PATH.with_suffix(CACHE_PATH.suffix + ".tmp")
    merged.to_parquet(tmp)
    os.replace(tmp, CACHE_PATH)
    print(f"Wrote merged cache: {len(merged)} rows, "
          f"{len(set(str(s) for s in merged.index.get_level_values('symbol').unique()))} symbols", flush=True)

    # coverage report vs universe
    have = set(str(s) for s in merged.index.get_level_values("symbol").unique())
    cov = len(have & set(universe_syms))
    print(f"Forecast coverage of SPY universe: {cov}/{len(universe_syms)} ({100*cov/len(universe_syms):.1f}%)", flush=True)


if __name__ == "__main__":
    main()
