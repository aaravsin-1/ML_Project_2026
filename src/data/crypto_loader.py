"""Loader for the Coinbase 1-second LOB snapshots on Kaggle.

Dataset: "High Frequency Crypto Limit Order Book Data" (martinsn, CC0), built from the
Coinbase websocket feed with sadighian/crypto-rl. Each row is a snapshot with:
- midpoint, spread
- buys / sells: notional of market buys / sells since the previous snapshot
- {bids,asks}_distance_x: level price as a fractional distance from the midpoint
- {bids,asks}_notional_x: resting notional (USD) at level x
(plus limit / market / cancel notional flows per level, unused here)
"""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
import pandas as pd

COINBASE_BTC_TICK = 0.01


class CryptoLOBLoader:
    """Converts the Coinbase snapshot format into the canonical 40-column LOB layout."""

    @staticmethod
    def load_absolute_times_ms(file_path: Union[str, Path]) -> np.ndarray:
        """Snapshot times as UTC epoch milliseconds (to align different coins in time)."""
        path = Path(file_path)
        cache_path = path.with_suffix(".times.npy")
        if cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
            return np.load(cache_path)
        ts = pd.to_datetime(pd.read_csv(path, usecols=["system_time"])["system_time"], utc=True)
        epoch_ms = (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy() * 1000.0
        np.save(cache_path, epoch_ms)
        return epoch_ms

    @staticmethod
    def load_coinbase_snapshots(
        file_path: Union[str, Path],
        num_levels: int = 10,
        tick_size: float = COINBASE_BTC_TICK,
        cache: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Loads snapshots and the per-second trade tape.

        Returns:
            lob_40: (N, num_levels * 4) [ask_p, ask_qty, bid_p, bid_qty, ...] with prices
                    snapped to the tick grid and quantities in coins (notional / price)
            timestamps_ms: (N,) milliseconds since the first snapshot
            trades_df: signed trade tape for VPIN (timestamp_ms, price, size, direction),
                       one buy row and one sell row per snapshot interval with volume
        """
        path = Path(file_path)
        cache_path = path.with_suffix(f".L{num_levels}.npz")
        if cache and cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
            z = np.load(cache_path)
            trades = pd.DataFrame({c: z[f"trade_{c}"] for c in ("timestamp_ms", "price", "size", "direction")})
            return z["lob_40"], z["timestamps_ms"], trades

        cols = ["system_time", "midpoint", "buys", "sells"] + [
            f"{side}_{field}_{lvl}"
            for side in ("bids", "asks") for field in ("distance", "notional") for lvl in range(num_levels)
        ]
        df = pd.read_csv(path, usecols=cols)
        ts = pd.to_datetime(df["system_time"])
        timestamps_ms = ((ts - ts.iloc[0]).dt.total_seconds() * 1000.0).to_numpy()
        mid = df["midpoint"].to_numpy(dtype=np.float64)

        lob_40 = np.zeros((len(df), num_levels * 4), dtype=np.float64)
        for lvl in range(num_levels):
            ask_p = np.round(mid * (1.0 + df[f"asks_distance_{lvl}"].to_numpy()) / tick_size) * tick_size
            bid_p = np.round(mid * (1.0 + df[f"bids_distance_{lvl}"].to_numpy()) / tick_size) * tick_size
            lob_40[:, lvl * 4] = ask_p
            lob_40[:, lvl * 4 + 1] = df[f"asks_notional_{lvl}"].to_numpy() / ask_p
            lob_40[:, lvl * 4 + 2] = bid_p
            lob_40[:, lvl * 4 + 3] = df[f"bids_notional_{lvl}"].to_numpy() / bid_p

        # Trade tape: the dataset gives buy/sell notional per interval, not individual
        # prints, so each interval contributes up to one aggregated buy and one sell.
        buys = df["buys"].to_numpy() / mid
        sells = df["sells"].to_numpy() / mid
        rows = []
        for size, direction in ((buys, 1), (sells, -1)):
            idx = np.nonzero(size > 0)[0]
            rows.append(pd.DataFrame({
                "timestamp_ms": timestamps_ms[idx], "price": mid[idx],
                "size": size[idx], "direction": np.full(len(idx), direction, dtype=np.int8),
            }))
        trades = pd.concat(rows).sort_values("timestamp_ms", kind="stable").reset_index(drop=True)

        if cache:
            np.savez(cache_path, lob_40=lob_40, timestamps_ms=timestamps_ms,
                     **{f"trade_{c}": trades[c].to_numpy() for c in trades.columns})
        return lob_40, timestamps_ms, trades
