"""Volume-Synchronized Probability of Informed Trading (VPIN).

Formalized by Easley, Lopez de Prado, and O'Hara (2012):
'Flow Toxicity and Liquidity in a High-Frequency World'.
Operates on a volume clock rather than a time clock to measure order flow toxicity.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd


class VPINEngine:
    """Calculates Volume-Synchronized Probability of Informed Trading (VPIN)."""

    def __init__(self, bucket_volume: float = 1000.0, window_buckets: int = 50):
        self.bucket_volume = bucket_volume
        self.window_buckets = window_buckets

    def compute_from_trades(
        self,
        trades_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes VPIN buckets from a trade DataFrame.
        
        trades_df must contain:
        - timestamp_ms
        - price
        - size
        - direction (+1: buyer initiated, -1: seller initiated, or inferred via tick rule)
        
        Returns:
            bucket_times: (N_buckets,) timestamp of bucket completion
            vpin_values: (N_buckets,) rolling VPIN toxicity metric in [0, 1]
            bucket_imbalances: (N_buckets,) raw signed volume imbalance (V_B - V_S) / V
        """
        assert len(trades_df) > 0, "Trade DataFrame cannot be empty"

        sizes = trades_df["size"].values.astype(np.float64)
        prices = trades_df["price"].values.astype(np.float64)
        timestamps = trades_df["timestamp_ms"].values.astype(np.float64)

        # Check or infer direction
        if "direction" in trades_df.columns:
            directions = trades_df["direction"].values.astype(np.int8)
        else:
            # Tick rule
            directions = self._tick_rule(prices)

        # Build volume buckets
        bucket_times = []
        bucket_imbalances = []  # |V_b - V_s|

        current_v_buy = 0.0
        current_v_sell = 0.0
        accumulated_vol = 0.0

        for t in range(len(sizes)):
            vol_remaining = sizes[t]
            d = directions[t]

            while vol_remaining > 0:
                needed = self.bucket_volume - accumulated_vol
                take = min(vol_remaining, needed)

                if d > 0:
                    current_v_buy += take
                else:
                    current_v_sell += take

                accumulated_vol += take
                vol_remaining -= take

                if accumulated_vol >= self.bucket_volume - 1e-8:
                    # Bucket is filled!
                    bucket_times.append(timestamps[t])
                    abs_imbalance = abs(current_v_buy - current_v_sell)
                    bucket_imbalances.append(abs_imbalance)

                    # Reset bucket
                    current_v_buy = 0.0
                    current_v_sell = 0.0
                    accumulated_vol = 0.0

        bucket_times = np.array(bucket_times, dtype=np.float64)
        bucket_imbalances = np.array(bucket_imbalances, dtype=np.float64)

        if len(bucket_imbalances) == 0:
            return np.array([]), np.array([]), np.array([])

        # Rolling VPIN over window_buckets
        # VPIN = sum(|V_B - V_S|) / (N * V)
        vpin_series = pd.Series(bucket_imbalances).rolling(
            self.window_buckets, min_periods=1
        ).sum() / (self.window_buckets * self.bucket_volume)

        vpin_values = np.clip(vpin_series.values, 0.0, 1.0).astype(np.float32)
        return bucket_times, vpin_values, bucket_imbalances

    def map_vpin_to_lob(
        self,
        lob_timestamps: np.ndarray,
        bucket_times: np.ndarray,
        vpin_values: np.ndarray
    ) -> np.ndarray:
        """Causally maps rolling VPIN values onto LOB snapshot timestamps via forward fill (asof).
        
        Guarantees strictly no lookahead: at timestamp t, only buckets completed at or before t are used.
        """
        n_lob = len(lob_timestamps)
        lob_vpin = np.zeros(n_lob, dtype=np.float32)

        if len(bucket_times) == 0 or len(vpin_values) == 0:
            # Baseline default neutral VPIN
            lob_vpin.fill(0.35)
            return lob_vpin

        # Use searchsorted to find last bucket completed <= lob_timestamp
        # searchsorted with side='right' gives index of first element > timestamp, so idx-1 is last <= timestamp
        indices = np.searchsorted(bucket_times, lob_timestamps, side="right") - 1
        valid_mask = indices >= 0
        lob_vpin[valid_mask] = vpin_values[indices[valid_mask]]
        
        # Warmup fill with the first available VPIN
        if not np.all(valid_mask):
            first_val = vpin_values[0] if len(vpin_values) > 0 else 0.35
            lob_vpin[~valid_mask] = first_val

        return lob_vpin

    @staticmethod
    def _tick_rule(prices: np.ndarray) -> np.ndarray:
        """Standard tick rule for direction classification."""
        n = len(prices)
        directions = np.zeros(n, dtype=np.int8)
        prev_dir = 1
        for i in range(1, n):
            delta = prices[i] - prices[i - 1]
            if delta > 0:
                directions[i] = 1
                prev_dir = 1
            elif delta < 0:
                directions[i] = -1
                prev_dir = -1
            else:
                directions[i] = prev_dir
        return directions
