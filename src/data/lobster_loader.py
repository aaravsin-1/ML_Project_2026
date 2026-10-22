"""Loader and Parser for NASDAQ LOBSTER Event and Orderbook Reconstructions."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


class LOBSTERLoader:
    """Parser for LOBSTER limit order book and message reconstruction datasets.
    
    LOBSTER Orderbook layout:
    - Ask Price 1, Ask Size 1, Bid Price 1, Bid Size 1, Ask Price 2, Ask Size 2, ...
    - Prices are scaled by 10,000 (dollar * 10000)
    
    LOBSTER Message layout:
    - Time (seconds from midnight, up to nanosecond precision)
    - Type:
      1: Submission of a new limit order
      2: Cancellation (partial deletion)
      3: Deletion (total deletion)
      4: Execution of a visible limit order
      5: Execution of a hidden limit order
      7: Trading halt indicator
    - Order ID
    - Size (number of shares)
    - Price (dollar * 10000)
    - Direction (1: Buy, -1: Sell)
    """

    @staticmethod
    def load_orderbook(
        orderbook_file: Union[str, Path],
        num_levels: int = 10,
        price_scaling: float = 1e-4
    ) -> np.ndarray:
        """Loads and scales LOBSTER orderbook file.
        
        Returns:
            np.ndarray of shape (N, num_levels * 4) with prices converted to dollars.
        """
        path = Path(orderbook_file)
        if not path.exists():
            raise FileNotFoundError(f"LOBSTER orderbook file not found: {path}")

        df = pd.read_csv(path, header=None)
        expected_cols = num_levels * 4
        assert df.shape[1] >= expected_cols, f"Requested {num_levels} levels ({expected_cols} cols) but file has {df.shape[1]}"
        
        arr = df.iloc[:, :expected_cols].values.astype(np.float32)
        # Scale price columns (columns 0, 2, 4, 6...)
        for c in range(0, expected_cols, 2):
            arr[:, c] *= price_scaling
            
        return arr

    @staticmethod
    def load_messages(message_file: Union[str, Path]) -> pd.DataFrame:
        """Loads and formats LOBSTER message file."""
        path = Path(message_file)
        if not path.exists():
            raise FileNotFoundError(f"LOBSTER message file not found: {path}")

        columns = ["timestamp", "event_type", "order_id", "size", "price", "direction"]
        df = pd.read_csv(path, header=None, names=columns)
        df["price"] = df["price"] * 1e-4
        return df

    @staticmethod
    def compute_labels(
        mid_prices: np.ndarray,
        horizons: list[int] = [10, 20, 50, 100],
        alpha: float = 0.0002,
        smoothing_window: int = 5
    ) -> Dict[int, np.ndarray]:
        """Calculates discrete UP (1), FLAT (0), DOWN (2) labels with smoothing."""
        n_samples = len(mid_prices)
        labels = {}

        # Smoothed mid-price series
        smoothed = pd.Series(mid_prices).rolling(smoothing_window, min_periods=1, center=False).mean().values

        for k in horizons:
            lbl = np.zeros(n_samples, dtype=np.int64)
            # Future smoothed mid-price at t + k
            future_smoothed = np.roll(smoothed, -k)
            # Future return
            returns = (future_smoothed - smoothed) / (smoothed + 1e-8)
            
            lbl[returns > alpha] = 1   # UP
            lbl[returns < -alpha] = 2  # DOWN
            # Last k samples invalid
            lbl[-k:] = 0
            labels[k] = lbl

        return labels
