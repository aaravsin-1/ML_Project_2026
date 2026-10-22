"""Micro-Price Estimator and Book Geometry Features.

Based on Stoikov (2018): 'The Micro-Price: A High-Frequency Estimator of Future Prices'.
Calculates theoretical micro-price, top-of-book and multi-level depth imbalances,
spread dynamics, and book slope.
"""

from typing import Dict, Tuple
import numpy as np


class MicroPriceEstimator:
    """Calculates micro-price, multi-level depth imbalances, and book slope."""

    @staticmethod
    def compute_features(lob_40: np.ndarray) -> Dict[str, np.ndarray]:
        """Extracts microstructural geometry features from 40-feature LOB array.
        
        lob_40 layout: [ask_p1, ask_v1, bid_p1, bid_v1, ..., ask_p10, ask_v10, bid_p10, bid_v10]
        """
        ask_p1 = lob_40[:, 0]
        ask_v1 = lob_40[:, 1]
        bid_p1 = lob_40[:, 2]
        bid_v1 = lob_40[:, 3]

        mid_price = (ask_p1 + bid_p1) / 2.0
        spread = ask_p1 - bid_p1
        rel_spread = spread / (mid_price + 1e-8)

        # Level 1 Imbalance
        denom_1 = ask_v1 + bid_v1 + 1e-8
        imbalance_l1 = (bid_v1 - ask_v1) / denom_1

        # Stoikov Micro-price
        micro_price = (bid_v1 * ask_p1 + ask_v1 * bid_p1) / denom_1
        micro_deviation = micro_price - mid_price

        # Multi-level cumulative depth imbalances: k = 3, 5, 10
        # Ask volumes are cols 1, 5, 9, 13, 17, 21, 25, 29, 33, 37
        # Bid volumes are cols 3, 7, 11, 15, 19, 23, 27, 31, 35, 39
        ask_v_all = lob_40[:, 1::4]  # (N, 10)
        bid_v_all = lob_40[:, 3::4]  # (N, 10)
        ask_p_all = lob_40[:, 0::4]  # (N, 10)
        bid_p_all = lob_40[:, 2::4]  # (N, 10)

        cum_ask_v3 = np.sum(ask_v_all[:, :3], axis=1)
        cum_bid_v3 = np.sum(bid_v_all[:, :3], axis=1)
        imbalance_l3 = (cum_bid_v3 - cum_ask_v3) / (cum_bid_v3 + cum_ask_v3 + 1e-8)

        cum_ask_v5 = np.sum(ask_v_all[:, :5], axis=1)
        cum_bid_v5 = np.sum(bid_v_all[:, :5], axis=1)
        imbalance_l5 = (cum_bid_v5 - cum_ask_v5) / (cum_bid_v5 + cum_ask_v5 + 1e-8)

        cum_ask_v10 = np.sum(ask_v_all[:, :10], axis=1)
        cum_bid_v10 = np.sum(bid_v_all[:, :10], axis=1)
        imbalance_l10 = (cum_bid_v10 - cum_ask_v10) / (cum_bid_v10 + cum_ask_v10 + 1e-8)

        # Book slopes
        depth_price_bid_diff = bid_p1 - bid_p_all[:, -1]
        depth_price_ask_diff = ask_p_all[:, -1] - ask_p1
        slope_bid = depth_price_bid_diff / (cum_bid_v10 + 1e-8)
        slope_ask = depth_price_ask_diff / (cum_ask_v10 + 1e-8)

        return {
            "mid_price": mid_price.astype(np.float32),
            "spread": spread.astype(np.float32),
            "rel_spread": rel_spread.astype(np.float32),
            "micro_price": micro_price.astype(np.float32),
            "micro_deviation": micro_deviation.astype(np.float32),
            "imbalance_l1": imbalance_l1.astype(np.float32),
            "imbalance_l3": imbalance_l3.astype(np.float32),
            "imbalance_l5": imbalance_l5.astype(np.float32),
            "imbalance_l10": imbalance_l10.astype(np.float32),
            "slope_bid": slope_bid.astype(np.float32),
            "slope_ask": slope_ask.astype(np.float32),
            "total_depth_bid": cum_bid_v10.astype(np.float32),
            "total_depth_ask": cum_ask_v10.astype(np.float32),
        }
