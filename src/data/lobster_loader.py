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


LOBSTER_DUMMY_PRICE = 9_999_999_999  # empty levels: ask = +this, bid = -this (in $ x 10000)


def load_lobster_day_on_grid(
    data_dir: Union[str, Path],
    ticker: str,
    date: str = "2012-06-21",
    num_levels: int = 10,
    grid_ms: int = 100,
    flow_levels: int = 5,
) -> Dict[str, object]:
    """Resamples one LOBSTER day onto a fixed clock (default 100 ms).

    Row j is the book at the END of grid interval j (the last event at or before
    open + (j + 1) * grid_ms). Returns:
        lob_40:        (N, num_levels * 4) [ask_p, ask_q, bid_p, bid_q, ...] in dollars / shares.
                       Empty levels (LOBSTER dummy prices) are filled with the last real level.
        timestamps_ms: (N,) grid times in ms since the open
        trades_df:     executions for VPIN (timestamp_ms, price, size, direction), where the
                       direction is the aggressor's: executing a sell limit order is a buy trade
        flows:         {"{bids,asks}_{limit,market,cancel}": (N, flow_levels)} USD notional during
                       interval j at the price of level l of the book at the START of the interval
                       (row j - 1). "market" counts visible executions (type 4) only.
        tick_size:     0.01
    """
    data_dir = Path(data_dir)
    stem = f"{ticker}_{date}_34200000_57600000"
    ob = pd.read_csv(data_dir / f"{stem}_orderbook_{num_levels}.csv", header=None).to_numpy(np.float64, copy=True)
    msg = pd.read_csv(data_dir / f"{stem}_message_{num_levels}.csv", header=None,
                      names=["time", "type", "order_id", "size", "price", "direction"])

    # Replace dummy prices on empty levels by the last real level (volume is already 0)
    for side_col in (0, 2):
        prices = ob[:, side_col::4].copy()
        dummy = np.abs(prices) >= LOBSTER_DUMMY_PRICE
        for lvl in range(1, num_levels):
            prices[:, lvl] = np.where(dummy[:, lvl], prices[:, lvl - 1], prices[:, lvl])
        ob[:, side_col::4] = prices
    ob[:, 0::2] /= 1e4  # all price columns (0, 2, 4, ...) to dollars

    t_ms = (msg["time"].to_numpy() - 34200.0) * 1000.0
    n_grid = int(np.ceil((57600.0 - 34200.0) * 1000.0 / grid_ms))
    grid_end = (np.arange(n_grid) + 1) * grid_ms
    last_event = np.searchsorted(t_ms, grid_end, side="right") - 1
    valid = last_event >= 0
    lob = ob[np.maximum(last_event, 0)]
    first = int(np.argmax(valid))
    lob, grid_end = lob[first:], grid_end[first:]

    interval = np.minimum((t_ms // grid_ms).astype(np.int64) - first, len(lob) - 1)
    keep = interval >= 1  # events in the first kept interval have no start-of-interval book

    ex = msg[msg["type"].isin([4, 5])]
    trades = pd.DataFrame({
        "timestamp_ms": t_ms[ex.index], "price": ex["price"].to_numpy() / 1e4,
        "size": ex["size"].to_numpy().astype(np.float64), "direction": -ex["direction"].to_numpy(),
    })

    flows = {}
    kinds = {"limit": [1], "market": [4], "cancel": [2, 3]}
    start_book = np.vstack([lob[:1], lob[:-1]])  # book at the start of each interval
    for side, sign, p_col in (("bids", 1, 2), ("asks", -1, 0)):
        level_cents = np.rint(start_book[:, p_col::4][:, :flow_levels] * 100).astype(np.int64)
        for kind, types in kinds.items():
            sel = msg["type"].isin(types).to_numpy() & (msg["direction"].to_numpy() == sign) & keep
            j = interval[sel]
            cents = np.rint(msg["price"].to_numpy()[sel] / 100).astype(np.int64)
            usd = msg["size"].to_numpy()[sel] * msg["price"].to_numpy()[sel] / 1e4
            out = np.zeros((len(lob), flow_levels))
            for lvl in range(flow_levels):
                match = level_cents[j, lvl] == cents
                np.add.at(out[:, lvl], j[match], usd[match])
            flows[f"{side}_{kind}"] = out

    return {"lob_40": lob, "timestamps_ms": grid_end.astype(np.float64), "trades_df": trades,
            "flows": flows, "tick_size": 0.01}
