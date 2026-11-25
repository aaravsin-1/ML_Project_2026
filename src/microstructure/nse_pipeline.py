"""Day-boundary-safe 5-level Microstructure Feature Pipeline & Segment Preparation for NSE.

Guarantees zero leakage across both split boundaries and overnight gaps:
- OFI, rolling means, lags, forward returns, and 100-step deep windows NEVER cross a day boundary.
- Segments are split by whole trading days (e.g., 11 train / 2 val / 3 test inside each 16-day segment).
- Normalizer (P0, V0), MLOFI PCA, VPIN bucket volume, and label thresholds are fitted on training days only.
- Eliminates the ofi_l1 == mlofi_lvl_1 duplicate column (every column is distinct).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.normalizers import DecimalPrecisionNormalizer
from src.data.nse_loader import NSE_EQUITY_EXEC, NSE_FUT_EXEC, NSESymbolData
from src.microstructure.ofi import OrderFlowImbalance
from src.microstructure.vpin import VPINEngine

SEQ_LEN = 100
NSE_HORIZONS = [5, 10, 20, 50, 100]  # seconds


@dataclass
class NSESegment:
    """One contiguous multi-day segment (e.g. 16 trading days) with whole-day train/val/test split."""
    symbol: str
    asset: str
    segment_id: int
    days: List[str]
    lob_20: np.ndarray              # (N, 20) float64
    order_counts: np.ndarray        # (N, 10) float32
    total_qty: np.ndarray           # (N, 2)  float32
    day_id: np.ndarray              # (N,)    int32 (0..n_days-1 local to segment)
    pos_in_day: np.ndarray          # (N,)    int32 (0..len_day-1)
    timestamps_ms: np.ndarray       # (N,)    float64 epoch UTC ms
    trades_df: pd.DataFrame
    tick_size: float
    train_days: int                 # days [0, train_days)
    val_days: int                   # days [train_days, train_days + val_days)
    test_days: int                  # days [train_days + val_days, total_days)
    exec_params: Dict[str, Any]


@dataclass
class NSEPreparedSegment:
    segment: NSESegment
    tab_feat: np.ndarray            # (N, 30) float32 (25 core + 5 NSE-native)
    deep_feat: np.ndarray           # (N, 20) float32
    feat_names: List[str]
    core_indices: List[int]         # indices of the 25 standard 5-level LOB features
    nse_extra_indices: List[int]    # indices of the 5 Kite order-count / full-book features
    pca_explained_var: np.ndarray   # (3,) explained variance ratio of MLOFI PCA on train
    labels_dict: Dict[int, np.ndarray]
    alphas_dict: Dict[int, float]
    train_end: int                  # first row index of val split
    split_cache: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)

    def labels(self, k: int) -> np.ndarray:
        return self.labels_dict[k]

    def splits(self, k: int) -> Dict[str, np.ndarray]:
        """Exact valid train/val/test row indices for horizon k (same rows for tabular & sequence)."""
        if k not in self.split_cache:
            seg = self.segment
            n = len(seg.lob_20)
            same_day_fwd = np.zeros(n, dtype=bool)
            if k < n:
                same_day_fwd[:-k] = seg.day_id[:-k] == seg.day_id[k:]
            min_pos = max(SEQ_LEN - 1, k)
            valid = (seg.pos_in_day >= min_pos) & same_day_fwd
            tr = np.nonzero(valid & (seg.day_id < seg.train_days))[0]
            va = np.nonzero(valid & (seg.day_id >= seg.train_days) & (seg.day_id < seg.train_days + seg.val_days))[0]
            te = np.nonzero(valid & (seg.day_id >= seg.train_days + seg.val_days))[0]
            self.split_cache[k] = {"train": tr, "val": va, "test": te}
        return self.split_cache[k]


def _day_smoothed_forward_returns(mid: np.ndarray, day_id: np.ndarray, k: int, smoothing: int = 5) -> np.ndarray:
    """Trailing-mean forward return strictly within each trading day (NaN when crossing day end)."""
    n = len(mid)
    s = np.empty(n, dtype=np.float64)
    # Compute trailing mean resetting at every day boundary
    changes = np.concatenate([[0], np.nonzero(day_id[1:] != day_id[:-1])[0] + 1, [n]])
    for a, b in zip(changes[:-1], changes[1:]):
        seg_mid = mid[a:b]
        if smoothing <= 1:
            s[a:b] = seg_mid
        else:
            cs = np.cumsum(np.insert(seg_mid, 0, 0.0))
            idx = np.arange(1, len(seg_mid) + 1)
            win = np.minimum(idx, smoothing)
            s[a:b] = (cs[idx] - cs[idx - win]) / win

    ret = np.full(n, np.nan, dtype=np.float64)
    if k < n:
        same = day_id[:-k] == day_id[k:]
        valid_idx = np.nonzero(same)[0]
        ret[valid_idx] = (s[valid_idx + k] - s[valid_idx]) / s[valid_idx]
    return ret


def build_nse_segments(
    sym_data: NSESymbolData,
    n_segments: int = 4,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> List[NSESegment]:
    """Splits a multi-day NSESymbolData into `n_segments` contiguous whole-day segments."""
    total_days = len(sym_data.days)
    day_bounds = np.linspace(0, total_days, n_segments + 1).astype(int)
    segments: List[NSESegment] = []

    for seg_id in range(n_segments):
        d_lo, d_hi = int(day_bounds[seg_id]), int(day_bounds[seg_id + 1])
        n_seg_days = d_hi - d_lo
        n_tr = max(1, int(round(n_seg_days * train_frac)))
        n_va = max(1, int(round(n_seg_days * val_frac)))
        n_te = n_seg_days - n_tr - n_va
        if n_te < 1:
            n_tr -= 1
            n_te = 1

        mask = (sym_data.day_id >= d_lo) & (sym_data.day_id < d_hi)
        lob = sym_data.lob_20[mask]
        oc = sym_data.order_counts[mask]
        tq = sym_data.total_qty[mask]
        local_day = (sym_data.day_id[mask] - d_lo).astype(np.int32)
        ts = sym_data.timestamps_ms[mask]

        # Position within each day (0, 1, 2, ...)
        pos_in_day = np.zeros(len(local_day), dtype=np.int32)
        changes = np.concatenate([[0], np.nonzero(local_day[1:] != local_day[:-1])[0] + 1, [len(local_day)]])
        for a, b in zip(changes[:-1], changes[1:]):
            pos_in_day[a:b] = np.arange(b - a, dtype=np.int32)

        tr_mask = (sym_data.trades_df["day_id"] >= d_lo) & (sym_data.trades_df["day_id"] < d_hi)
        trades = sym_data.trades_df.loc[tr_mask].copy().reset_index(drop=True)
        trades["day_id"] = (trades["day_id"] - d_lo).astype(np.int32)

        # Execution parameters calibrated to the instrument's median price
        if sym_data.asset == "DERIVATIVE":
            exec_params = dict(NSE_FUT_EXEC)
        else:
            med_mid = float(np.median((lob[:, 0] + lob[:, 2]) / 2.0))
            lot_shares = max(1.0, round(NSE_EQUITY_EXEC["lot_notional"] / med_mid))
            exec_params = dict(
                NSE_EQUITY_EXEC,
                lot_size=lot_shares,
                max_inventory=lot_shares * 5.0,
            )

        segments.append(NSESegment(
            symbol=sym_data.symbol,
            asset=sym_data.asset,
            segment_id=seg_id,
            days=sym_data.days[d_lo:d_hi],
            lob_20=lob,
            order_counts=oc,
            total_qty=tq,
            day_id=local_day,
            pos_in_day=pos_in_day,
            timestamps_ms=ts,
            trades_df=trades,
            tick_size=sym_data.tick_size,
            train_days=n_tr,
            val_days=n_va,
            test_days=n_te,
            exec_params=exec_params,
        ))
    return segments


def _compute_day_safe_vpin(
    timestamps_ms: np.ndarray,
    day_id: np.ndarray,
    trades_df: pd.DataFrame,
    bucket_vol: float,
    window_buckets: int = 50,
) -> np.ndarray:
    """Computes volume-clock VPIN causally per day (never carrying partial buckets across nights)."""
    n = len(timestamps_ms)
    out = np.full(n, np.nan, dtype=np.float32)
    if trades_df is None or len(trades_df) == 0 or bucket_vol <= 0:
        return out
    engine = VPINEngine(bucket_volume=bucket_vol, window_buckets=window_buckets)
    for d in np.unique(day_id):
        row_idx = np.nonzero(day_id == d)[0]
        td = trades_df[trades_df["day_id"] == d]
        if len(td) == 0:
            continue
        bt, vv, _ = engine.compute_from_trades(td)
        if len(bt) > 0:
            out[row_idx] = engine.map_vpin_to_lob(timestamps_ms[row_idx], bt, vv)
    return out


def prepare_nse_segment(
    seg: NSESegment,
    horizons: Sequence[int] = NSE_HORIZONS,
    smoothing: int = 5,
    flat_share: float = 1.0 / 3.0,
) -> NSEPreparedSegment:
    """Fits all stateful transforms on training days only and extracts 30 causal features + (N, 20) deep tensor."""
    lob = seg.lob_20
    n = len(lob)
    train_mask = seg.day_id < seg.train_days
    train_end = int(np.nonzero(~train_mask)[0][0])  # first row after training days

    # 1. Decimal-precision normalizer for deep models (fitted on first 100 training rows)
    norm = DecimalPrecisionNormalizer()
    norm.fit(lob[:train_end, 0], lob[:train_end, 1])
    deep_feat = norm.transform_lob(lob).astype(np.float32)
    deep_feat = np.nan_to_num(deep_feat, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Book geometry (5 levels)
    ask_p1, ask_v1 = lob[:, 0], lob[:, 1]
    bid_p1, bid_v1 = lob[:, 2], lob[:, 3]
    mid = (ask_p1 + bid_p1) / 2.0
    spread = ask_p1 - bid_p1

    denom1 = bid_v1 + ask_v1 + 1e-8
    imbalance_l1 = ((bid_v1 - ask_v1) / denom1).astype(np.float32)
    micro_price = (bid_v1 * ask_p1 + ask_v1 * bid_p1) / denom1
    micro_dev = (micro_price - mid).astype(np.float32)
    spread_ticks = np.rint(spread / seg.tick_size).astype(np.float32)

    ask_v_all = lob[:, 1::4]  # (N, 5)
    bid_v_all = lob[:, 3::4]  # (N, 5)
    ask_p_all = lob[:, 0::4]  # (N, 5)
    bid_p_all = lob[:, 2::4]  # (N, 5)

    cum_b3, cum_a3 = bid_v_all[:, :3].sum(axis=1), ask_v_all[:, :3].sum(axis=1)
    imbalance_l3 = ((cum_b3 - cum_a3) / (cum_b3 + cum_a3 + 1e-8)).astype(np.float32)

    cum_b5, cum_a5 = bid_v_all.sum(axis=1), ask_v_all.sum(axis=1)
    imbalance_l5 = ((cum_b5 - cum_a5) / (cum_b5 + cum_a5 + 1e-8)).astype(np.float32)

    slope_bid = ((bid_p1 - bid_p_all[:, -1]) / (cum_b5 + 1e-8)).astype(np.float32)
    slope_ask = ((ask_p_all[:, -1] - ask_p1) / (cum_a5 + 1e-8)).astype(np.float32)

    # 3. Multi-level OFI (5 levels), zeroed at every day start so no overnight diff leaks
    mlofi = OrderFlowImbalance.compute_multi_level_ofi(lob, num_levels=5)
    day_start = seg.pos_in_day == 0
    mlofi[day_start] = 0.0

    # Fit 3-component PCA on training days' 5-level OFI only
    _, pca_model = OrderFlowImbalance.reduce_mlofi_pca(mlofi[:train_end], n_components=3, pca_model=None)
    mlofi_pca, _ = OrderFlowImbalance.reduce_mlofi_pca(mlofi, n_components=3, pca_model=pca_model)
    pca_var = np.array(pca_model.explained_variance_ratio_, dtype=np.float64)

    # 4. Volume-clock VPIN (bucket = average training volume per minute, ~1/375 of daily train volume)
    tr_trades = seg.trades_df[seg.trades_df["day_id"] < seg.train_days]
    train_minutes = max(seg.train_days * 375.0, 1.0)
    bucket_vol = float(tr_trades["size"].sum() / train_minutes) if len(tr_trades) else 1000.0
    vpin = _compute_day_safe_vpin(seg.timestamps_ms, seg.day_id, seg.trades_df, bucket_vol)

    # 5. Assemble 25 core distinct features (no duplicate ofi_l1 == mlofi_lvl_1)
    feat_dict: Dict[str, np.ndarray] = {
        "imbalance_l1": imbalance_l1,
        "imbalance_l3": imbalance_l3,
        "imbalance_l5": imbalance_l5,
        "spread_ticks": spread_ticks,
        "micro_deviation": micro_dev,
        "slope_bid": slope_bid,
        "slope_ask": slope_ask,
        "vpin": vpin,
        "ofi_l1": mlofi[:, 0],
        "ofi_pca1": mlofi_pca[:, 0],
        "ofi_pca2": mlofi_pca[:, 1],
        "ofi_pca3": mlofi_pca[:, 2],
        "mlofi_lvl_2": mlofi[:, 1],
        "mlofi_lvl_3": mlofi[:, 2],
        "mlofi_lvl_4": mlofi[:, 3],
        "mlofi_lvl_5": mlofi[:, 4],
    }

    # Day-safe lags of L1 signals (1, 2, 5 s)
    for lag in (1, 2, 5):
        cross = seg.pos_in_day < lag
        for name, arr in (("ofi", mlofi[:, 0]), ("imbalance", imbalance_l1), ("micro_dev", micro_dev)):
            lagged = np.roll(arr, lag)
            lagged[cross] = 0.0
            feat_dict[f"{name}_lag_{lag}"] = lagged.astype(np.float32)

    n_core = len(feat_dict)  # 25

    # 6. 5 NSE-native features from Kite (full-book total buy/sell qty + order counts)
    tbq = seg.total_qty[:, 0].astype(np.float64)
    tsq = seg.total_qty[:, 1].astype(np.float64)
    ao1, bo1 = seg.order_counts[:, 0].astype(np.float64), seg.order_counts[:, 1].astype(np.float64)
    ao_all = seg.order_counts[:, 0::2].sum(axis=1).astype(np.float64)
    bo_all = seg.order_counts[:, 1::2].sum(axis=1).astype(np.float64)

    feat_dict["total_book_imbalance"] = ((tbq - tsq) / (tbq + tsq + 1e-8)).astype(np.float32)
    feat_dict["order_imbalance_l1"] = ((bo1 - ao1) / (bo1 + ao1 + 1e-8)).astype(np.float32)
    feat_dict["order_imbalance_l5"] = ((bo_all - ao_all) / (bo_all + ao_all + 1e-8)).astype(np.float32)
    feat_dict["avg_order_size_ratio_l1"] = (
        np.log1p(bid_v1 / np.maximum(bo1, 1.0)) - np.log1p(ask_v1 / np.maximum(ao1, 1.0))
    ).astype(np.float32)
    feat_dict["depth_ratio_l5_to_total"] = ((cum_b5 + cum_a5) / (tbq + tsq + 1e-8)).astype(np.float32)

    feat_names = list(feat_dict.keys())
    tab_feat = np.column_stack([feat_dict[k] for k in feat_names]).astype(np.float32)
    tab_feat = np.nan_to_num(tab_feat, nan=0.0, posinf=0.0, neginf=0.0)

    # 7. Calibrate per-horizon directional labels strictly on training days
    labels_dict: Dict[int, np.ndarray] = {}
    alphas_dict: Dict[int, float] = {}
    for k in horizons:
        ret = _day_smoothed_forward_returns(mid, seg.day_id, k, smoothing=smoothing)
        tr_ret = ret[:train_end]
        tr_ret = tr_ret[np.isfinite(tr_ret)]
        alpha = float(np.quantile(np.abs(tr_ret), flat_share)) if len(tr_ret) else 0.0
        y = np.zeros(n, dtype=np.int64)
        y[ret > alpha] = 1
        y[ret < -alpha] = 2
        labels_dict[k] = y
        alphas_dict[k] = alpha

    return NSEPreparedSegment(
        segment=seg,
        tab_feat=tab_feat,
        deep_feat=deep_feat,
        feat_names=feat_names,
        core_indices=list(range(n_core)),
        nse_extra_indices=list(range(n_core, len(feat_names))),
        pca_explained_var=pca_var,
        labels_dict=labels_dict,
        alphas_dict=alphas_dict,
        train_end=train_end,
    )
