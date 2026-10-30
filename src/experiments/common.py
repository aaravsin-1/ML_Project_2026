"""Shared dataset preparation for all experiments.

One place decides how a session is obtained, where the splits are, and what the
feature pipeline is allowed to see, so every experiment is evaluated identically:
- the pipeline (PCA, normalizer) is fit on training rows only;
- splits are chronological with a purge gap of k rows between them;
- tabular and sequence models are scored on exactly the same test rows.

A session is either one synthetic run (one generator seed) or one contiguous segment
of real Coinbase BTC-USD data; experiments do not need to know which.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import DATA_DIR
from src.data.synthetic_generator import SyntheticLOBGenerator
from src.data.crypto_loader import CryptoLOBLoader, COINBASE_BTC_TICK
from src.data.labels import direction_labels
from src.data.lob_dataset import purged_split_indices
from src.microstructure.pipeline import MicrostructureFeaturePipeline

SEQ_LEN = 100
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
HORIZONS = [10, 20, 50, 100]
BTC_PATH = DATA_DIR / "raw" / "crypto" / "BTC_1sec.csv"

SYNTHETIC_EXEC = {
    "initial_cash": 1_000_000.0, "lot_size": 100.0, "max_inventory": 500.0,
    "maker_fee_bps": -0.5, "taker_fee_bps": 1.5,
    "base_latency_ms": 1.0,
    "latencies_ms": [0.0, 0.05, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0],
}
# Coinbase Pro fee schedule (April 2021), top volume tier: 0 bp maker, 4 bp taker.
# Snapshots arrive once per second, so the latency sweep is in whole seconds.
BTC_EXEC = {
    "initial_cash": 1_000_000.0, "lot_size": 0.1, "max_inventory": 0.5,
    "maker_fee_bps": 0.0, "taker_fee_bps": 4.0,
    "base_latency_ms": 1.0,
    "latencies_ms": [0.0, 1000.0, 2000.0, 5000.0, 10000.0, 30000.0],
}


@dataclass
class MarketSession:
    lob_40: np.ndarray
    labels_dict: Dict[int, np.ndarray]
    trades_df: pd.DataFrame
    timestamps_ms: np.ndarray
    seed: int
    name: str = "synthetic"
    tick_size: Optional[float] = None
    vpin_bucket_volume: float = 1000.0
    exec_params: Dict[str, Any] = field(default_factory=lambda: dict(SYNTHETIC_EXEC))


@dataclass
class PreparedData:
    session: MarketSession
    tab_feat: np.ndarray
    deep_feat: np.ndarray
    feat_names: List[str]
    pipeline: MicrostructureFeaturePipeline
    train_end: int
    split_cache: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)

    def labels(self, horizon_k: int) -> np.ndarray:
        return self.session.labels_dict[horizon_k]

    def splits(self, horizon_k: int) -> Dict[str, np.ndarray]:
        """Purged row indices for horizon k. Train starts after SEQ_LEN - 1 rows of warm-up
        so sequence models and tabular models train on the same rows too."""
        if horizon_k not in self.split_cache:
            self.split_cache[horizon_k] = purged_split_indices(
                len(self.tab_feat), horizon_k, TRAIN_RATIO, VAL_RATIO, min_start=SEQ_LEN - 1
            )
        return self.split_cache[horizon_k]


def generate_session(seed: int, n_events: int, regime: str = "large_tick") -> MarketSession:
    gen = SyntheticLOBGenerator(base_price=100.0, tick_size=0.01, regime=regime, seed=seed)
    lob_40, labels_dict, trades_df, timestamps_ms = gen.generate_session(n_events=n_events)
    return MarketSession(lob_40, labels_dict, trades_df, timestamps_ms, seed, name=f"synthetic_seed{seed}")


def btc_sessions(n_segments: int = 4, path: Path = BTC_PATH, max_rows: Optional[int] = None) -> List[MarketSession]:
    """Splits the Coinbase BTC-USD 1-second data into contiguous, non-overlapping segments.

    Each segment is an independent replication with its own 70/15/15 split, label
    thresholds calibrated on its own training rows, and VPIN bucket size set to its
    average training volume per minute.
    """
    lob_all, ts_all, trades_all = CryptoLOBLoader.load_coinbase_snapshots(path)
    if max_rows is not None:
        lob_all, ts_all = lob_all[:max_rows], ts_all[:max_rows]
    bounds = np.linspace(0, len(lob_all), n_segments + 1).astype(int)
    sessions = []
    for i in range(n_segments):
        lo, hi = bounds[i], bounds[i + 1]
        t0, t1 = ts_all[lo], ts_all[hi - 1]
        lob = lob_all[lo:hi]
        ts = ts_all[lo:hi] - t0
        in_seg = (trades_all["timestamp_ms"] >= t0) & (trades_all["timestamp_ms"] <= t1)
        trades = trades_all.loc[in_seg].copy()
        trades["timestamp_ms"] -= t0

        train_end = int(len(lob) * TRAIN_RATIO)
        mid = (lob[:, 0] + lob[:, 2]) / 2.0
        labels = direction_labels(mid, HORIZONS, fit_end=train_end)
        train_trades = trades[trades["timestamp_ms"] < ts[train_end]]
        bucket = float(train_trades["size"].sum() / max(ts[train_end] / 60000.0, 1.0))

        sessions.append(MarketSession(
            lob, labels, trades.reset_index(drop=True), ts, seed=i, name=f"btc_segment_{i}",
            tick_size=COINBASE_BTC_TICK, vpin_bucket_volume=bucket, exec_params=dict(BTC_EXEC),
        ))
    return sessions


def prepare_data(session: MarketSession) -> PreparedData:
    """Fits the feature pipeline on the training period and transforms the full session."""
    n = len(session.lob_40)
    train_end = int(n * TRAIN_RATIO)
    pipeline = MicrostructureFeaturePipeline(
        num_levels=10, pca_components=3, tick_size=session.tick_size,
        vpin_bucket_vol=session.vpin_bucket_volume,
    )
    tab_feat, deep_feat, feat_names = pipeline.fit_transform(
        session.lob_40, session.trades_df, session.timestamps_ms, fit_end=train_end
    )
    return PreparedData(session, tab_feat, deep_feat, feat_names, pipeline, train_end)


def load_prepared(seed: int = 42, n_events: int = 30000) -> PreparedData:
    return prepare_data(generate_session(seed, n_events))
