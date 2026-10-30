"""Unified Microstructural Feature Pipeline.

Orchestrates OFI, MLOFI with PCA, VPIN, Micro-Price, and Book Geometry
into dense feature matrices for Tabular and Deep Learning models.

Every feature is causal (row t only uses snapshots <= t). The only fitted
state (PCA loadings and the price/volume normalizer) is estimated on the
training rows alone, so it carries no information from the evaluation period.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.microstructure.ofi import OrderFlowImbalance
from src.microstructure.vpin import VPINEngine
from src.microstructure.microprice import MicroPriceEstimator
from src.data.normalizers import DecimalPrecisionNormalizer


class MicrostructureFeaturePipeline:
    """End-to-end feature extraction pipeline for LOB alpha modeling."""

    def __init__(
        self,
        num_levels: int = 10,
        pca_components: int = 3,
        vpin_bucket_vol: float = 1000.0,
        vpin_window: int = 50,
        tick_size: Optional[float] = None
    ):
        self.num_levels = num_levels
        self.pca_components = pca_components
        self.vpin_engine = VPINEngine(bucket_volume=vpin_bucket_vol, window_buckets=vpin_window)
        self.pca_model: Optional[PCA] = None
        self.normalizer = DecimalPrecisionNormalizer()
        self.tick_size = tick_size
        self.feature_names: List[str] = []

    @staticmethod
    def infer_tick_size(lob_40: np.ndarray) -> float:
        """Smallest price increment between adjacent ask levels (robust to float noise)."""
        diffs = np.abs(np.diff(lob_40[:, 0::4], axis=1)).ravel()
        diffs = diffs[diffs > 1e-9]
        return float(np.round(np.min(diffs), 10)) if len(diffs) else 0.01

    def fit(self, lob_40_train: np.ndarray) -> "MicrostructureFeaturePipeline":
        """Fits the normalizer and MLOFI PCA on training snapshots only."""
        if self.tick_size is None:
            self.tick_size = self.infer_tick_size(lob_40_train)
        self.normalizer.fit(lob_40_train[:, 0], lob_40_train[:, 1])
        mlofi_train = OrderFlowImbalance.compute_multi_level_ofi(lob_40_train, num_levels=self.num_levels)
        _, self.pca_model = OrderFlowImbalance.reduce_mlofi_pca(
            mlofi_train, n_components=self.pca_components, pca_model=None
        )
        return self

    def fit_transform(
        self,
        lob_40: np.ndarray,
        trades_df: Optional[pd.DataFrame] = None,
        timestamps_ms: Optional[np.ndarray] = None,
        fit_end: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Fits on rows [0, fit_end) and transforms the full series.

        `fit_end` should be the end of the training split. It defaults to the
        whole array only for convenience in unit tests; experiments must pass it.

        Returns:
            tabular_features: (N, n_features) matrix for tree / linear models
            deep_features: (N, 40) normalized array for DeepLOB / TLOB
            feature_names: List of column names
        """
        end = len(lob_40) if fit_end is None else fit_end
        self.fit(lob_40[:end])
        tabular_features, deep_features = self.transform(lob_40, trades_df, timestamps_ms)
        return tabular_features, deep_features, self.feature_names

    def compute_vpin(
        self,
        n_rows: int,
        trades_df: Optional[pd.DataFrame],
        timestamps_ms: Optional[np.ndarray]
    ) -> np.ndarray:
        """Volume-clock VPIN mapped causally onto LOB snapshot times."""
        if trades_df is None or len(trades_df) == 0:
            # No trade tape: VPIN is undefined, so emit a constant that carries no signal.
            return np.full(n_rows, np.nan, dtype=np.float32)
        if timestamps_ms is None:
            raise ValueError("VPIN needs LOB snapshot timestamps to map volume buckets onto the book")
        bucket_t, vpin_vals, _ = self.vpin_engine.compute_from_trades(trades_df)
        return self.vpin_engine.map_vpin_to_lob(timestamps_ms, bucket_t, vpin_vals)

    def transform(
        self,
        lob_40: np.ndarray,
        trades_df: Optional[pd.DataFrame] = None,
        timestamps_ms: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms data using fitted parameters (no leakage)."""
        assert self.pca_model is not None, "Call fit() before transform()"
        deep_features = self.normalizer.transform_lob(lob_40)
        geom = MicroPriceEstimator.compute_features(lob_40)
        mlofi = OrderFlowImbalance.compute_multi_level_ofi(lob_40, num_levels=self.num_levels)
        mlofi_pca, _ = OrderFlowImbalance.reduce_mlofi_pca(
            mlofi, n_components=self.pca_components, pca_model=self.pca_model
        )
        vpin_lob = self.compute_vpin(len(lob_40), trades_df, timestamps_ms)

        # Spread measured in ticks. Relative spread (spread / mid) is avoided on purpose:
        # with a fixed tick it is a monotone function of the price level, which lets
        # models memorize where the price is rather than learn book dynamics.
        spread_ticks = np.rint(
            (lob_40[:, 0].astype(np.float64) - lob_40[:, 2]) / self.tick_size
        ).astype(np.float32)

        feature_dict: Dict[str, np.ndarray] = {
            "imbalance_l1": geom["imbalance_l1"],
            "imbalance_l3": geom["imbalance_l3"],
            "imbalance_l5": geom["imbalance_l5"],
            "imbalance_l10": geom["imbalance_l10"],
            "spread_ticks": spread_ticks,
            "micro_deviation": geom["micro_deviation"],
            "slope_bid": geom["slope_bid"],
            "slope_ask": geom["slope_ask"],
            "vpin": vpin_lob,
            "ofi_l1": mlofi[:, 0],
            "ofi_pca1": mlofi_pca[:, 0],
            "ofi_pca2": mlofi_pca[:, 1],
            "ofi_pca3": mlofi_pca[:, 2] if self.pca_components >= 3 else np.zeros_like(spread_ticks),
        }

        for lvl in range(min(5, self.num_levels)):
            feature_dict[f"mlofi_lvl_{lvl+1}"] = mlofi[:, lvl]

        for lag in [1, 2, 5]:
            feature_dict[f"ofi_lag_{lag}"] = np.roll(mlofi[:, 0], lag)
            feature_dict[f"imbalance_lag_{lag}"] = np.roll(geom["imbalance_l1"], lag)
            feature_dict[f"micro_dev_lag_{lag}"] = np.roll(geom["micro_deviation"], lag)
            feature_dict[f"ofi_lag_{lag}"][:lag] = 0.0
            feature_dict[f"imbalance_lag_{lag}"][:lag] = 0.0
            feature_dict[f"micro_dev_lag_{lag}"][:lag] = 0.0

        self.feature_names = list(feature_dict.keys())
        tabular_features = np.column_stack([feature_dict[k] for k in self.feature_names]).astype(np.float32)
        tabular_features = np.nan_to_num(tabular_features, nan=0.0, posinf=0.0, neginf=0.0)
        deep_features = np.nan_to_num(deep_features, nan=0.0, posinf=0.0, neginf=0.0)

        return tabular_features, deep_features
