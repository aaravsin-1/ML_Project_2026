"""Loader for the Indian NSE 5-level Limit Order Book Parquet tick store (Kite + NSE).

Reads bucketed Parquet files directly from D:\\kite_data\\ticks (no PostgreSQL required)
and resamples each full regular trading day (09:15:00-15:29:59 IST) onto a clean
1-second clock while preserving day boundaries so features, lags, labels, and
sequence windows NEVER cross overnight gaps.

Layout of canonical 5-level book `lob_20`:
    [ap1, aq1, bp1, bq1, ap2, aq2, bp2, bq2, ..., ap5, aq5, bp5, bq5]
matching the existing 10-level `lob_40` column convention (ask_p, ask_q, bid_p, bid_q).
"""

from __future__ import annotations

import glob
import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from src.config import DATA_DIR

DEFAULT_TICK_DIR = Path(os.environ.get("MARKET_TICKS", r"D:\kite_data\ticks"))
NSE_CACHE_DIR = DATA_DIR / "processed" / "nse"
NSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Full regular session: 09:15:00 to 15:29:59 IST = 6h 15m = 22,500 seconds
SESSION_OPEN_SEC = 9 * 3600 + 15 * 60       # 33,300
SESSION_SECONDS = 6 * 3600 + 15 * 60        # 22,500
VALID_START_DATE = "2026-08-07"             # first day with full 5-level depth
EXCLUDE_DATES = {"2026-11-08"}              # Sunday Diwali Muhurat 1-hour session

# Realistic Indian market execution parameter presets (all-in statutory + exchange, ₹0 brokerage)
# Intraday equity: STT 2.5 bp on sell (~1.25 bp/side) + NSE txn ~0.30 + SEBI/stamp/GST ~= 2.0 bp/side
NSE_EQUITY_EXEC = {
    "initial_cash": 10_000_000.0,   # ₹1 Cr
    "lot_notional": 100_000.0,      # ~₹1 Lakh per trade -> lot_size computed from median price
    "maker_fee_bps": 0.0,
    "taker_fee_bps": 2.0,
    "base_latency_ms": 1.0,
    "latencies_ms": [0.0, 1000.0, 2000.0, 5000.0, 10000.0, 30000.0],
}
# Index futures (post-Union Budget 2026, effective Apr 1, 2026):
# STT 5.0 bp on sell (~2.50 bp/side) + NSE txn ~0.173 + SEBI ~0.01 + stamp ~0.10 + GST ~0.033 ~= 2.8 bp/side
# NIFTY 50 lot size = 65 units (verified by GCD of all resting L1 quantities in the book)
NSE_FUT_EXEC = {
    "initial_cash": 10_000_000.0,
    "lot_size": 65.0,               # 1 NIFTY futures lot (65 units)
    "max_inventory": 325.0,         # ±5 lots
    "maker_fee_bps": 0.0,
    "taker_fee_bps": 2.8,
    "base_latency_ms": 1.0,
    "latencies_ms": [0.0, 1000.0, 2000.0, 5000.0, 10000.0, 30000.0],
}

BOOK_COLS_5 = (
    [f"ap{i}" for i in range(1, 6)]
    + [f"aq{i}" for i in range(1, 6)]
    + [f"bp{i}" for i in range(1, 6)]
    + [f"bq{i}" for i in range(1, 6)]
    + [f"ao{i}" for i in range(1, 6)]
    + [f"bo{i}" for i in range(1, 6)]
)
TICK_COLS = [
    "ts", "seq", "symbol", "last_price", "last_qty", "volume",
    "total_buy_qty", "total_sell_qty",
] + BOOK_COLS_5


@dataclass
class NSESymbolData:
    """Full multi-day 1-second resampled 5-level order book for one NSE instrument."""
    symbol: str
    asset: str                        # "EQUITY" or "DERIVATIVE"
    days: List[str]
    lob_20: np.ndarray                # (N, 20) float64 [ap1, aq1, bp1, bq1, ..., ap5, aq5, bp5, bq5]
    order_counts: np.ndarray          # (N, 10) float32 [ao1, bo1, ..., ao5, bo5]
    total_qty: np.ndarray             # (N, 2)  float32 [total_buy_qty, total_sell_qty]
    day_id: np.ndarray                # (N,)    int32   0..len(days)-1
    sec_in_day: np.ndarray            # (N,)    int32   0..22499
    timestamps_ms: np.ndarray         # (N,)    float64 epoch UTC ms
    trades_df: pd.DataFrame           # (M, 5)  timestamp_ms, day_id, price, size, direction
    tick_size: float
    meta: Dict[str, Any] = field(default_factory=dict)


def list_valid_days(tick_dir: Path = DEFAULT_TICK_DIR, asset: str = "EQUITY") -> List[str]:
    base = tick_dir / asset
    if not base.is_dir():
        raise FileNotFoundError(f"Tick directory not found: {base}")
    return sorted(
        d for d in os.listdir(base)
        if d >= VALID_START_DATE and d not in EXCLUDE_DATES and (base / d).is_dir()
    )


def _infer_tick_size(lob_20: np.ndarray) -> float:
    ap = lob_20[:, 0::4]
    bp = lob_20[:, 2::4]
    diffs = np.concatenate([
        np.abs(np.diff(ap, axis=1)).ravel(),
        np.abs(np.diff(bp, axis=1)).ravel(),
        (lob_20[:, 0] - lob_20[:, 2]).ravel(),
    ])
    diffs = diffs[np.isfinite(diffs) & (diffs > 1e-6)]
    if len(diffs) == 0:
        return 0.05
    # Use 1st percentile / rounded min to ignore rare float artifacts
    raw = float(np.min(diffs))
    for std_tick in (0.01, 0.05, 0.10, 0.20, 0.25, 0.50, 1.0):
        if abs(raw - std_tick) < 1e-4:
            return std_tick
    return round(raw, 4)


def _process_single_day(
    day: str,
    symbol: str,
    asset: str,
    tick_dir: Path,
    grid_sec: int = 1,
) -> Optional[Dict[str, Any]]:
    """Loads one day for one symbol (or front-month FUT if asset == 'DERIVATIVE') onto a 1 s clock."""
    t0 = pd.Timestamp(f"{day} 09:15:00", tz="Asia/Kolkata").tz_convert("UTC")
    t1 = pd.Timestamp(f"{day} 15:30:00", tz="Asia/Kolkata").tz_convert("UTC")

    if asset == "EQUITY":
        b = zlib.crc32(symbol.encode()) % 16
        files = glob.glob(str(tick_dir / "EQUITY" / day / f"b{b:02d}__*.parquet"))
        if not files:
            return None
        filt = (ds.field("symbol") == symbol) & (ds.field("ts") >= t0) & (ds.field("ts") < t1)
        df = ds.dataset(files).to_table(columns=TICK_COLS, filter=filt).to_pandas()
    else:
        # Front-month futures for `symbol` (underlying), selected by highest day volume
        files = glob.glob(str(tick_dir / "DERIVATIVE" / day / "*.parquet"))
        if not files:
            return None
        filt = (
            (ds.field("underlying") == symbol)
            & (ds.field("instrument_type") == "FUT")
            & (ds.field("ts") >= t0)
            & (ds.field("ts") < t1)
        )
        df = ds.dataset(files).to_table(columns=TICK_COLS, filter=filt).to_pandas()
        if len(df) == 0:
            return None
        front_sym = df.groupby("symbol")["volume"].max().idxmax()
        df = df[df["symbol"] == front_sym].copy()

    if len(df) < 1000:
        return None

    df = df.sort_values(["ts", "seq"], kind="mergesort").reset_index(drop=True)

    # Filter out crossed/missing top-of-book ticks (e.g. auction artifacts)
    valid_book = (
        df["bp1"].notna() & df["ap1"].notna()
        & (df["bp1"] > 0) & (df["ap1"] > df["bp1"])
        & (df["bq1"] > 0) & (df["aq1"] > 0)
    )
    df = df.loc[valid_book].reset_index(drop=True)
    if len(df) < 1000:
        return None

    # Seconds since 09:15:00 IST (0 .. 22499), unit-agnostic (handles us or ns timestamps)
    sec_float = (df["ts"] - t0).dt.total_seconds().to_numpy(dtype=np.float64)
    sec_offset = np.floor(sec_float).astype(np.int64)
    in_sess = (sec_offset >= 0) & (sec_offset < SESSION_SECONDS)
    df = df.loc[in_sess].reset_index(drop=True)
    sec_offset = sec_offset[in_sess]
    sec_float = sec_float[in_sess]
    if len(df) < 1000:
        return None

    # Build reconstructed trade tape from positive cumulative volume diffs
    vol = df["volume"].to_numpy(dtype=np.float64)
    dvol = np.diff(vol, prepend=vol[0])
    dvol[0] = 0.0
    dvol = np.where(dvol > 0, dvol, 0.0)
    lp = df["last_price"].to_numpy(dtype=np.float64)
    mid_tick = (df["bp1"].to_numpy(dtype=np.float64) + df["ap1"].to_numpy(dtype=np.float64)) / 2.0
    prev_mid = np.roll(mid_tick, 1)
    prev_mid[0] = mid_tick[0]
    # Lee-Ready quote rule against previous snapshot mid; fallback to tick rule on ties
    direction = np.where(lp > prev_mid, 1, np.where(lp < prev_mid, -1, 0)).astype(np.int8)
    dlp = np.diff(lp, prepend=lp[0])
    tie = direction == 0
    direction[tie] = np.where(dlp[tie] > 0, 1, np.where(dlp[tie] < 0, -1, 0)).astype(np.int8)
    # Forward-fill remaining 0s with last non-zero direction (default +1)
    last_d = 1
    for i in range(len(direction)):
        if direction[i] != 0:
            last_d = int(direction[i])
        else:
            direction[i] = last_d

    tr_mask = dvol > 0
    t0_ms = float(t0.timestamp() * 1000.0)
    tick_epoch_ms = t0_ms + sec_float * 1000.0
    trades_ts = tick_epoch_ms[tr_mask]
    trades_price = lp[tr_mask]
    trades_size = dvol[tr_mask]
    trades_dir = direction[tr_mask]

    # Resample onto 1-second grid: row j = last tick with sec_offset <= j
    grid_secs = np.arange(0, SESSION_SECONDS, grid_sec, dtype=np.int64)
    idx = np.searchsorted(sec_offset, grid_secs, side="right") - 1
    valid_grid = idx >= 0
    first_valid = int(np.argmax(valid_grid))
    grid_secs = grid_secs[first_valid:]
    idx = idx[first_valid:]

    # Build canonical 5-level array: [ap1, aq1, bp1, bq1, ..., ap5, aq5, bp5, bq5]
    n_grid = len(idx)
    lob_20 = np.zeros((n_grid, 20), dtype=np.float64)
    order_counts = np.zeros((n_grid, 10), dtype=np.float32)

    for lvl in range(1, 6):
        ap = df[f"ap{lvl}"].to_numpy(dtype=np.float64)[idx]
        aq = df[f"aq{lvl}"].to_numpy(dtype=np.float64)[idx]
        bp = df[f"bp{lvl}"].to_numpy(dtype=np.float64)[idx]
        bq = df[f"bq{lvl}"].to_numpy(dtype=np.float64)[idx]
        ao = df[f"ao{lvl}"].to_numpy(dtype=np.float32)[idx]
        bo = df[f"bo{lvl}"].to_numpy(dtype=np.float32)[idx]
        # Replace missing/zero deeper levels with previous level price and 0 qty
        if lvl > 1:
            bad_a = (~np.isfinite(ap)) | (ap <= 0)
            ap = np.where(bad_a, lob_20[:, (lvl - 2) * 4], ap)
            aq = np.where(bad_a | (~np.isfinite(aq)), 0.0, aq)
            bad_b = (~np.isfinite(bp)) | (bp <= 0)
            bp = np.where(bad_b, lob_20[:, (lvl - 2) * 4 + 2], bp)
            bq = np.where(bad_b | (~np.isfinite(bq)), 0.0, bq)
        lob_20[:, (lvl - 1) * 4] = ap
        lob_20[:, (lvl - 1) * 4 + 1] = np.nan_to_num(aq, nan=0.0)
        lob_20[:, (lvl - 1) * 4 + 2] = bp
        lob_20[:, (lvl - 1) * 4 + 3] = np.nan_to_num(bq, nan=0.0)
        order_counts[:, (lvl - 1) * 2] = np.nan_to_num(ao, nan=0.0)
        order_counts[:, (lvl - 1) * 2 + 1] = np.nan_to_num(bo, nan=0.0)

    total_qty = np.column_stack([
        np.nan_to_num(df["total_buy_qty"].to_numpy(dtype=np.float32)[idx], nan=0.0),
        np.nan_to_num(df["total_sell_qty"].to_numpy(dtype=np.float32)[idx], nan=0.0),
    ])

    grid_epoch_ms = t0_ms + grid_secs.astype(np.float64) * 1000.0

    return {
        "day": day,
        "lob_20": lob_20,
        "order_counts": order_counts,
        "total_qty": total_qty,
        "sec_in_day": grid_secs.astype(np.int32),
        "timestamps_ms": grid_epoch_ms,
        "trades_ts": trades_ts,
        "trades_price": trades_price,
        "trades_size": trades_size,
        "trades_dir": trades_dir,
        "raw_ticks": len(df),
    }


def load_nse_symbol(
    symbol: str,
    asset: str = "EQUITY",
    days: Optional[Sequence[str]] = None,
    tick_dir: Path = DEFAULT_TICK_DIR,
    cache: bool = True,
    workers: int = 8,
) -> NSESymbolData:
    """Loads (and caches) 1-second 5-level LOB snapshots across all valid trading days."""
    tag = f"{symbol}_FUT" if asset == "DERIVATIVE" else symbol
    cache_path = NSE_CACHE_DIR / f"{tag}_1s_5lvl.npz"

    if cache and cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        trades = pd.DataFrame({
            "timestamp_ms": z["trade_ts"],
            "day_id": z["trade_day_id"],
            "price": z["trade_price"],
            "size": z["trade_size"],
            "direction": z["trade_dir"],
        })
        return NSESymbolData(
            symbol=tag,
            asset=asset,
            days=list(z["days"]),
            lob_20=z["lob_20"],
            order_counts=z["order_counts"],
            total_qty=z["total_qty"],
            day_id=z["day_id"],
            sec_in_day=z["sec_in_day"],
            timestamps_ms=z["timestamps_ms"],
            trades_df=trades,
            tick_size=float(z["tick_size"]),
            meta={"raw_ticks_total": int(z["raw_ticks_total"])},
        )

    day_list = list(days) if days is not None else list_valid_days(tick_dir, asset)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda d: _process_single_day(d, symbol, asset, tick_dir), day_list))

    kept = [r for r in results if r is not None]
    if not kept:
        raise RuntimeError(f"No valid days found for {symbol} ({asset})")

    kept_days = [r["day"] for r in kept]
    lob_20 = np.vstack([r["lob_20"] for r in kept])
    order_counts = np.vstack([r["order_counts"] for r in kept])
    total_qty = np.vstack([r["total_qty"] for r in kept])
    sec_in_day = np.concatenate([r["sec_in_day"] for r in kept])
    timestamps_ms = np.concatenate([r["timestamps_ms"] for r in kept])
    day_id = np.concatenate([np.full(len(r["lob_20"]), i, dtype=np.int32) for i, r in enumerate(kept)])

    trade_ts = np.concatenate([r["trades_ts"] for r in kept])
    trade_day_id = np.concatenate([np.full(len(r["trades_ts"]), i, dtype=np.int32) for i, r in enumerate(kept)])
    trade_price = np.concatenate([r["trades_price"] for r in kept])
    trade_size = np.concatenate([r["trades_size"] for r in kept])
    trade_dir = np.concatenate([r["trades_dir"] for r in kept])
    raw_ticks_total = sum(r["raw_ticks"] for r in kept)

    tick_size = _infer_tick_size(lob_20)

    if cache:
        np.savez_compressed(
            cache_path,
            days=np.array(kept_days),
            lob_20=lob_20,
            order_counts=order_counts,
            total_qty=total_qty,
            day_id=day_id,
            sec_in_day=sec_in_day,
            timestamps_ms=timestamps_ms,
            trade_ts=trade_ts,
            trade_day_id=trade_day_id,
            trade_price=trade_price,
            trade_size=trade_size,
            trade_dir=trade_dir,
            tick_size=np.array(tick_size),
            raw_ticks_total=np.array(raw_ticks_total),
        )

    trades = pd.DataFrame({
        "timestamp_ms": trade_ts,
        "day_id": trade_day_id,
        "price": trade_price,
        "size": trade_size,
        "direction": trade_dir,
    })
    return NSESymbolData(
        symbol=tag,
        asset=asset,
        days=kept_days,
        lob_20=lob_20,
        order_counts=order_counts,
        total_qty=total_qty,
        day_id=day_id,
        sec_in_day=sec_in_day,
        timestamps_ms=timestamps_ms,
        trades_df=trades,
        tick_size=tick_size,
        meta={"raw_ticks_total": raw_ticks_total},
    )
