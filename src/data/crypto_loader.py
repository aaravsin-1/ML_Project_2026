"""Loader and Parser for High-Frequency Cryptocurrency LOB Snapshots."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class CryptoLOBLoader:
    """Parser for 24/7 Cryptocurrency Limit Order Book Snapshots (e.g., BTC/USDT, ETH/USDT).
    
    Standard format handles top 10 to 15 bid/ask levels.
    """

    @staticmethod
    def parse_crypto_csv(
        file_path: Union[str, Path],
        num_levels: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Loads crypto LOB snapshots and extracts canonical 40-feature array and mid-prices.
        
        Expected columns:
        - bids: bid_price_0, bid_qty_0, ...
        - asks: ask_price_0, ask_qty_0, ...
        or interleaved.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Crypto LOB file not found: {path}")

        df = pd.read_csv(path)
        
        # Interleave features into canonical format: [ask_p0, ask_q0, bid_p0, bid_q0, ask_p1, ...]
        features = np.zeros((len(df), num_levels * 4), dtype=np.float32)
        
        for lvl in range(num_levels):
            ask_p_col = f"ask_price_{lvl}" if f"ask_price_{lvl}" in df.columns else f"asks[{lvl}].price"
            ask_q_col = f"ask_qty_{lvl}" if f"ask_qty_{lvl}" in df.columns else f"asks[{lvl}].amount"
            bid_p_col = f"bid_price_{lvl}" if f"bid_price_{lvl}" in df.columns else f"bids[{lvl}].price"
            bid_q_col = f"bid_qty_{lvl}" if f"bid_qty_{lvl}" in df.columns else f"bids[{lvl}].amount"
            
            base_idx = lvl * 4
            features[:, base_idx] = df[ask_p_col].values
            features[:, base_idx + 1] = df[ask_q_col].values
            features[:, base_idx + 2] = df[bid_p_col].values
            features[:, base_idx + 3] = df[bid_q_col].values
            
        mid_prices = (features[:, 0] + features[:, 2]) / 2.0
        return features, mid_prices
