"""Unified Microstructural Feature Pipeline.

Orchestrates OFI, MLOFI with PCA, VPIN, Micro-Price, and Book Geometry
into dense feature matrices for Tabular and Deep Learning models.
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
        vpin_window: int = 50
    ):
        self.num_levels = num_levels
        self.pca_components = pca_components
        self.vpin_engine = VPINEngine(bucket_volume=vpin_bucket_vol, window_buckets=vpin_window)
        self.pca_model: Optional[PCA] = None
        self.normalizer = DecimalPrecisionNormalizer()
        self.feature_names: List[str] = []

    def fit_transform(
        self,
        lob_40: np.ndarray,
        trades_df: Optional[pd.DataFrame] = None,
        timestamps_ms: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Fits normalizer and PCA on initial data, then transforms features.
        
        Returns:
            tabular_features: (N, n_features) matrix for tree / linear models
            deep_features: (N, 40) normalized array for DeepLOB / TLOB
            feature_names: List of column names
        """
        # Fit normalizer
        ask_p1 = lob_40[:, 0]
        ask_v1 = lob_40[:, 1]
        self.normalizer.fit(ask_p1, ask_v1)
        
        # Transform deep features
        deep_features = self.normalizer.transform_lob(lob_40)

        # Micro-price & book geometry
        geom = MicroPriceEstimator.compute_features(lob_40)

        # OFI & MLOFI
        mlofi = OrderFlowImbalance.compute_multi_level_ofi(lob_40, num_levels=self.num_levels)
        mlofi_pca, self.pca_model = OrderFlowImbalance.reduce_mlofi_pca(
            mlofi, n_components=self.pca_components, pca_model=None
        )

        # VPIN
        if trades_df is not None and len(trades_df) > 0 and timestamps_ms is not None:
            bucket_t, vpin_vals, _ = self.vpin_engine.compute_from_trades(trades_df)
            vpin_lob = self.vpin_engine.map_vpin_to_lob(timestamps_ms, bucket_t, vpin_vals)
        else:
            # Synthetic proxy from order flow imbalance volatility
            vpin_lob = np.clip(np.abs(geom["imbalance_l1"]) * 0.5 + 0.3, 0.0, 1.0).astype(np.float32)

        # Build Tabular Feature Matrix
        feature_dict = {
            "imbalance_l1": geom["imbalance_l1"],
            "imbalance_l3": geom["imbalance_l3"],
            "imbalance_l5": geom["imbalance_l5"],
            "imbalance_l10": geom["imbalance_l10"],
            "rel_spread": geom["rel_spread"],
            "micro_deviation": geom["micro_deviation"],
            "slope_bid": geom["slope_bid"],
            "slope_ask": geom["slope_ask"],
            "vpin": vpin_lob,
            "ofi_l1": mlofi[:, 0],
            "ofi_pca1": mlofi_pca[:, 0],
            "ofi_pca2": mlofi_pca[:, 1],
            "ofi_pca3": mlofi_pca[:, 2] if self.pca_components >= 3 else np.zeros_like(vpin_lob),
        }

        # Add multi-level OFI features
        for lvl in range(min(5, self.num_levels)):
            feature_dict[f"mlofi_lvl_{lvl+1}"] = mlofi[:, lvl]

        # Add lag features for top signals
        for lag in [1, 2, 5]:
            feature_dict[f"ofi_lag_{lag}"] = np.roll(mlofi[:, 0], lag)
            feature_dict[f"imbalance_lag_{lag}"] = np.roll(geom["imbalance_l1"], lag)
            feature_dict[f"micro_dev_lag_{lag}"] = np.roll(geom["micro_deviation"], lag)
            # Zero out boundary
            feature_dict[f"ofi_lag_{lag}"][:lag] = 0.0
            feature_dict[f"imbalance_lag_{lag}"][:lag] = 0.0
            feature_dict[f"micro_dev_lag_{lag}"][:lag] = 0.0

        # Construct tabular array
        self.feature_names = list(feature_dict.keys())
        tabular_features = np.column_stack([feature_dict[k] for k in self.feature_names]).astype(np.float32)

        # Replace NaNs / Infs
        tabular_features = np.nan_to_num(tabular_features, nan=0.0, posinf=0.0, neginf=0.0)
        deep_features = np.nan_to_num(deep_features, nan=0.0, posinf=0.0, neginf=0.0)

        return tabular_features, deep_features, self.feature_names

    def transform(
        self,
        lob_40: np.ndarray,
        trades_df: Optional[pd.DataFrame] = None,
        timestamps_ms: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms out-of-sample data using fitted parameters (no leakage)."""
        deep_features = self.normalizer.transform_lob(lob_40)
        geom = MicroPriceEstimator.compute_features(lob_40)
        mlofi = OrderFlowImbalance.compute_multi_level_ofi(lob_40, num_levels=self.num_levels)
        mlofi_pca, _ = OrderFlowImbalance.reduce_mlofi_pca(
            mlofi, n_components=self.pca_components, pca_model=self.pca_model
        )

        if trades_df is not None and len(trades_df) > 0 and timestamps_ms is not None:
            bucket_t, vpin_vals, _ = self.vpin_engine.compute_from_trades(trades_df)
            vpin_lob = self.vpin_engine.map_vpin_to_lob(timestamps_ms, bucket_t, vpin_vals)
        else:
            vpin_lob = np.clip(np.abs(geom["imbalance_l1"]) * 0.5 + 0.3, 0.0, 1.0).astype(np.float32)

        feature_dict = {
            "imbalance_l1": geom["imbalance_l1"],
            "imbalance_l3": geom["imbalance_l3"],
            "imbalance_l5": geom["imbalance_l5"],
            "imbalance_l10": geom["imbalance_l10"],
            "rel_spread": geom["rel_spread"],
            "micro_deviation": geom["micro_deviation"],
            "slope_bid": geom["slope_bid"],
            "slope_ask": geom["slope_ask"],
            "vpin": vpin_lob,
            "ofi_l1": mlofi[:, 0],
            "ofi_pca1": mlofi_pca[:, 0],
            "ofi_pca2": mlofi_pca[:, 1],
            "ofi_pca3": mlofi_pca[:, 2] if self.pca_components >= 3 else np.zeros_like(vpin_lob),
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

        tabular_features = np.column_stack([feature_dict[k] for k in self.feature_names]).astype(np.float32)
        tabular_features = np.nan_to_num(tabular_features, nan=0.0, posinf=0.0, neginf=0.0)
        deep_features = np.nan_to_num(deep_features, nan=0.0, posinf=0.0, neginf=0.0)

        return tabular_features, deep_features
