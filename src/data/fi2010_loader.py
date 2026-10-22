"""Loader and Preprocessor for the Benchmark FI-2010 Limit Order Book Dataset."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


class FI2010Loader:
    """Parser for FI-2010 Benchmark Dataset.
    
    FI-2010 layout:
    - Shape: (149, N_events) or (N_events, 149)
    - Rows 0:40: 10 price levels [ask_p1, ask_v1, bid_p1, bid_v1, ..., ask_p10, ask_v10, bid_p10, bid_v10]
    - Rows 40:144: Additional technical / event features
    - Rows 144:149: Ground truth labels for horizons k in [10, 20, 30, 50, 100]
    - Label mapping in raw files: 1 = UP, 2 = FLAT/HOLD, 3 = DOWN
      Standardized to: 0 = FLAT, 1 = UP, 2 = DOWN
    """
    HORIZONS = [10, 20, 30, 50, 100]

    @staticmethod
    def parse_file(file_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """Parses a raw or normalized FI-2010 text or CSV file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"FI-2010 file not found at: {file_path}")

        # Load file (typically space-separated or comma-separated)
        data = np.loadtxt(file_path)
        # FI-2010 raw files are typically formatted with 149 rows and N columns
        if data.shape[0] == 149:
            data = data.T  # Transpose to (N_events, 149)
        elif data.shape[1] != 149 and data.shape[0] != 149:
            raise ValueError(f"Unexpected shape for FI-2010 data: {data.shape}, expected 149 features/labels")

        # 40 LOB features (first 10 levels)
        lob_features = data[:, :40].astype(np.float32)

        # Labels for horizons k = [10, 20, 30, 50, 100] located at columns 144:149
        labels_by_horizon = {}
        for idx, k in enumerate(FI2010Loader.HORIZONS):
            raw_labels = data[:, 144 + idx].astype(np.int64)
            # Map raw labels (1 -> UP: 1, 2 -> FLAT: 0, 3 -> DOWN: 2)
            mapped_labels = np.zeros_like(raw_labels, dtype=np.int64)
            mapped_labels[raw_labels == 1] = 1  # UP
            mapped_labels[raw_labels == 2] = 0  # FLAT
            mapped_labels[raw_labels == 3] = 2  # DOWN
            labels_by_horizon[k] = mapped_labels

        return lob_features, labels_by_horizon

    @staticmethod
    def extract_level_arrays(lob_40: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Separates 40 features into ask_prices, ask_volumes, bid_prices, bid_volumes.
        
        Returns:
            ask_prices: (N, 10)
            ask_volumes: (N, 10)
            bid_prices: (N, 10)
            bid_volumes: (N, 10)
        """
        ask_prices = lob_40[:, 0::4]
        ask_volumes = lob_40[:, 1::4]
        bid_prices = lob_40[:, 2::4]
        bid_volumes = lob_40[:, 3::4]
        return ask_prices, ask_volumes, bid_prices, bid_volumes
