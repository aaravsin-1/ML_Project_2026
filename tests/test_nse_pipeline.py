"""Automated tests for the NSE 5-level LOB pipeline, day-boundary safety, and DeepLOB5."""

import numpy as np
import pandas as pd
import torch

from src.data.nse_loader import NSESymbolData
from src.microstructure.nse_pipeline import build_nse_segments, prepare_nse_segment
from src.models.deeplob5 import DeepLOB5


def _synthetic_multiday_nse(n_days: int = 8, rows_per_day: int = 400, seed: int = 0) -> NSESymbolData:
    rng = np.random.default_rng(seed)
    n = n_days * rows_per_day
    day_id = np.repeat(np.arange(n_days, dtype=np.int32), rows_per_day)
    sec_in_day = np.tile(np.arange(rows_per_day, dtype=np.int32), n_days)
    ts_ms = (day_id.astype(np.float64) * 86_400_000.0) + sec_in_day.astype(np.float64) * 1000.0

    # Random walk mid with sharp overnight jumps between days
    mid = np.zeros(n, dtype=np.float64)
    for d in range(n_days):
        a, b = d * rows_per_day, (d + 1) * rows_per_day
        base = 1000.0 + d * 50.0
        mid[a:b] = base + np.cumsum(rng.normal(0, 0.1, size=rows_per_day))

    lob_20 = np.zeros((n, 20), dtype=np.float64)
    for lvl in range(5):
        lob_20[:, lvl * 4] = np.round((mid + (lvl + 0.5) * 0.05) / 0.05) * 0.05
        lob_20[:, lvl * 4 + 1] = rng.integers(50, 500, size=n).astype(np.float64)
        lob_20[:, lvl * 4 + 2] = np.round((mid - (lvl + 0.5) * 0.05) / 0.05) * 0.05
        lob_20[:, lvl * 4 + 3] = rng.integers(50, 500, size=n).astype(np.float64)

    order_counts = rng.integers(1, 20, size=(n, 10)).astype(np.float32)
    total_qty = rng.integers(5000, 50000, size=(n, 2)).astype(np.float32)
    trades = pd.DataFrame({
        "timestamp_ms": ts_ms[::2],
        "day_id": day_id[::2],
        "price": mid[::2],
        "size": rng.integers(10, 200, size=len(ts_ms[::2])).astype(np.float64),
        "direction": rng.choice([-1, 1], size=len(ts_ms[::2])).astype(np.int8),
    })
    return NSESymbolData(
        symbol="TEST_NSE",
        asset="EQUITY",
        days=[f"2026-08-{10 + i:02d}" for i in range(n_days)],
        lob_20=lob_20,
        order_counts=order_counts,
        total_qty=total_qty,
        day_id=day_id,
        sec_in_day=sec_in_day,
        timestamps_ms=ts_ms,
        trades_df=trades,
        tick_size=0.05,
    )


def test_nse_day_boundary_and_split_isolation():
    sym = _synthetic_multiday_nse(n_days=16, rows_per_day=300)
    segs = build_nse_segments(sym, n_segments=1)
    prep = prepare_nse_segment(segs[0], horizons=[5, 20])
    sp = prep.splits(20)
    seg = segs[0]

    # Train/val/test are strictly separated by whole trading days
    assert seg.day_id[sp["train"]].max() < seg.day_id[sp["val"]].min()
    assert seg.day_id[sp["val"]].max() < seg.day_id[sp["test"]].min()

    # No test or train label crosses an overnight boundary
    for split_name in ("train", "val", "test"):
        rows = sp[split_name]
        assert np.all(seg.day_id[rows] == seg.day_id[rows + 20])
        assert np.all(seg.pos_in_day[rows] >= 99)

    # First row of every day has zero OFI and zero lag features (no overnight diff)
    day_starts = np.nonzero(seg.pos_in_day == 0)[0]
    ofi_col = prep.feat_names.index("ofi_l1")
    lag1_col = prep.feat_names.index("ofi_lag_1")
    assert np.allclose(prep.tab_feat[day_starts, ofi_col], 0.0)
    assert np.allclose(prep.tab_feat[day_starts, lag1_col], 0.0)


def test_nse_features_strictly_causal():
    sym = _synthetic_multiday_nse(n_days=8, rows_per_day=250, seed=7)
    seg_orig = build_nse_segments(sym, n_segments=1)[0]
    prep_orig = prepare_nse_segment(seg_orig, horizons=[10])

    # Mutate the last 2 test days drastically — earlier days must be bit-identical
    sym_mut = _synthetic_multiday_nse(n_days=8, rows_per_day=250, seed=7)
    cutoff = 6 * 250
    sym_mut.lob_20[cutoff:, :] *= 3.0
    sym_mut.order_counts[cutoff:, :] *= 5.0
    sym_mut.total_qty[cutoff:, :] *= 4.0
    seg_mut = build_nse_segments(sym_mut, n_segments=1)[0]
    prep_mut = prepare_nse_segment(seg_mut, horizons=[10])

    assert np.allclose(prep_orig.tab_feat[:cutoff], prep_mut.tab_feat[:cutoff], atol=1e-6)
    assert np.allclose(prep_orig.deep_feat[:cutoff], prep_mut.deep_feat[:cutoff], atol=1e-6)


def test_deeplob5_forward_shape():
    m = DeepLOB5(num_classes=3)
    x = torch.randn(4, 100, 20)
    y = m(x)
    assert y.shape == (4, 3)
