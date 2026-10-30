"""Microstructure-Grounded Synthetic Limit Order Book Generator.

Generates realistic high-frequency continuous double-auction LOB dynamics:
- Level 1 to 10 depth queues
- Large-tick vs Small-tick microstructure regimes
- Poisson & Hawkes self-exciting order arrival
- Informed trader adverse selection bursts
- Standardized 40-feature representation and multi-horizon labels
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class SyntheticLOBGenerator:
    """High-fidelity synthetic LOB simulator capturing microstructural dynamics."""

    def __init__(
        self,
        base_price: float = 100.0,
        tick_size: float = 0.01,
        num_levels: int = 10,
        regime: str = "large_tick",  # "large_tick" or "small_tick"
        seed: Optional[int] = 42
    ):
        self.base_price = base_price
        self.tick_size = tick_size
        self.num_levels = num_levels
        self.regime = regime
        self.rng = np.random.default_rng(seed)

        # Microstructural regime parameters
        if regime == "large_tick":
            # Deep queues, narrow spread (1 tick), high signal-to-noise ratio
            self.mean_queue_size = 800.0
            self.queue_std = 250.0
            self.spread_ticks = 1
            self.informed_flow_prob = 0.08
            self.noise_volatility = 0.00005
        else:
            # Small-tick: shallow queues, wider/variable spread (2-4 ticks), higher noise
            self.mean_queue_size = 120.0
            self.queue_std = 60.0
            self.spread_ticks = 3
            self.informed_flow_prob = 0.20
            self.noise_volatility = 0.0003

    def generate_session(
        self,
        n_events: int = 50000,
        dt_mean_ms: float = 10.0
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray], pd.DataFrame, np.ndarray]:
        """Generates a complete synthetic trading session.
        
        Returns:
            lob_features: (n_events, 40) float64 array in FI-2010 format, prices snapped to the tick grid:
                          [ask_p1, ask_v1, bid_p1, bid_v1, ..., ask_p10, ask_v10, bid_p10, bid_v10]
            labels: dict mapping horizon k -> (n_events,) int labels (0: FLAT, 1: UP, 2: DOWN)
            trades_df: DataFrame of executed trades for VPIN calculation (timestamp, price, size, direction)
            timestamps_ms: (n_events,) snapshot timestamps, needed to map volume-clock VPIN onto the LOB
        """
        # Initialize prices
        mid_price = self.base_price
        best_bid = round((mid_price - (self.spread_ticks * self.tick_size / 2.0)) / self.tick_size) * self.tick_size
        best_ask = best_bid + (self.spread_ticks * self.tick_size)

        # Pre-allocate output arrays
        lob_features = np.zeros((n_events, self.num_levels * 4), dtype=np.float64)
        timestamps = np.zeros(n_events, dtype=np.float64)
        current_time_ms = 0.0

        # Trade tracking
        trade_records = []

        # Current book state: (10 levels of bid_p, bid_v, ask_p, ask_v)
        bids = np.zeros((self.num_levels, 2), dtype=np.float64)  # [price, size]
        asks = np.zeros((self.num_levels, 2), dtype=np.float64)

        for lvl in range(self.num_levels):
            bids[lvl, 0] = best_bid - (lvl * self.tick_size)
            bids[lvl, 1] = max(10.0, self.rng.normal(self.mean_queue_size * (1 + 0.15 * lvl), self.queue_std))
            asks[lvl, 0] = best_ask + (lvl * self.tick_size)
            asks[lvl, 1] = max(10.0, self.rng.normal(self.mean_queue_size * (1 + 0.15 * lvl), self.queue_std))

        # Latent trend / informed trading regime state
        regime_state = 0  # 0: neutral, 1: toxic buy burst, -1: toxic sell burst
        regime_countdown = 0

        for t in range(n_events):
            # Advance clock
            current_time_ms += self.rng.exponential(dt_mean_ms)
            timestamps[t] = current_time_ms

            # Regime transitions (Hawkes / informed flow clustering)
            if regime_countdown <= 0:
                if self.rng.random() < self.informed_flow_prob:
                    regime_state = 1 if self.rng.random() > 0.5 else -1
                    regime_countdown = self.rng.integers(15, 60)
                else:
                    regime_state = 0
            else:
                regime_countdown -= 1

            # Order event types:
            # 1: Market order (execution)
            # 2: Limit order insertion
            # 3: Cancellation
            event_type = self.rng.choice([1, 2, 3], p=[0.40, 0.40, 0.20])

            if event_type == 1:
                # Market order: aggressive sweep
                if regime_state == 1:
                    is_buy = self.rng.random() < 0.85  # Toxic buy wave
                elif regime_state == -1:
                    is_buy = self.rng.random() < 0.15  # Toxic sell wave
                else:
                    is_buy = self.rng.random() < 0.50  # Balanced

                trade_size = max(5.0, self.rng.exponential(self.mean_queue_size * 0.3))

                if is_buy:
                    # Executes against resting asks
                    fill_price = asks[0, 0]
                    trade_records.append((current_time_ms, fill_price, trade_size, 1))
                    asks[0, 1] -= trade_size
                    if asks[0, 1] <= 0:
                        # Top level depleted! Shift book up
                        best_ask += self.tick_size
                        best_bid = best_ask - (self.spread_ticks * self.tick_size)
                        self._shift_book_up(bids, asks)
                else:
                    # Executes against resting bids
                    fill_price = bids[0, 0]
                    trade_records.append((current_time_ms, fill_price, trade_size, -1))
                    bids[0, 1] -= trade_size
                    if bids[0, 1] <= 0:
                        # Top level depleted! Shift book down
                        best_bid -= self.tick_size
                        best_ask = best_bid + (self.spread_ticks * self.tick_size)
                        self._shift_book_down(bids, asks)

            elif event_type == 2:
                # Limit order insertion
                lvl = self.rng.geometric(p=0.45) - 1
                lvl = min(lvl, self.num_levels - 1)
                size_add = max(10.0, self.rng.normal(self.mean_queue_size * 0.25, 50.0))
                
                # Asymmetric insertions driven by informed pressure
                p_bid = 0.70 if regime_state == 1 else (0.30 if regime_state == -1 else 0.50)
                if self.rng.random() < p_bid:
                    bids[lvl, 1] += size_add
                else:
                    asks[lvl, 1] += size_add

            elif event_type == 3:
                # Cancellation
                lvl = self.rng.integers(0, self.num_levels)
                # Cancellations occur more frequently on the opposite side of informed flow
                if regime_state == 1:
                    # Sellers pulling liquidity
                    asks[lvl, 1] = max(10.0, asks[lvl, 1] * self.rng.uniform(0.3, 0.8))
                elif regime_state == -1:
                    # Buyers pulling liquidity
                    bids[lvl, 1] = max(10.0, bids[lvl, 1] * self.rng.uniform(0.3, 0.8))
                else:
                    if self.rng.random() < 0.5:
                        bids[lvl, 1] = max(10.0, bids[lvl, 1] * self.rng.uniform(0.5, 0.9))
                    else:
                        asks[lvl, 1] = max(10.0, asks[lvl, 1] * self.rng.uniform(0.5, 0.9))

            # Record snapshot in canonical 40-feature representation
            # [ask_p1, ask_v1, bid_p1, bid_v1, ask_p2, ask_v2, bid_p2, bid_v2, ...]
            # Prices are snapped to the tick grid so repeated += tick_size does not leave
            # floating-point residue that models could pick up as a fake signal.
            for lvl in range(self.num_levels):
                idx = lvl * 4
                lob_features[t, idx] = round(asks[lvl, 0] / self.tick_size) * self.tick_size
                lob_features[t, idx + 1] = asks[lvl, 1]
                lob_features[t, idx + 2] = round(bids[lvl, 0] / self.tick_size) * self.tick_size
                lob_features[t, idx + 3] = bids[lvl, 1]

        # Calculate mid-prices
        mid_prices = (lob_features[:, 0] + lob_features[:, 2]) / 2.0

        # Multi-horizon labels: k in [10, 20, 50, 100]
        labels = self._generate_labels(mid_prices, horizons=[10, 20, 50, 100])

        trades_df = pd.DataFrame(
            trade_records,
            columns=["timestamp_ms", "price", "size", "direction"]
        )

        return lob_features, labels, trades_df, timestamps

    def _shift_book_up(self, bids: np.ndarray, asks: np.ndarray):
        """Shift depth levels up when ask level 1 is swept."""
        for lvl in range(self.num_levels - 1):
            asks[lvl, 0] = asks[lvl + 1, 0]
            asks[lvl, 1] = asks[lvl + 1, 1]
        asks[-1, 0] = asks[-2, 0] + self.tick_size
        asks[-1, 1] = max(10.0, self.rng.normal(self.mean_queue_size, self.queue_std))

        # Re-anchor bids
        best_bid = asks[0, 0] - (self.spread_ticks * self.tick_size)
        for lvl in range(self.num_levels):
            bids[lvl, 0] = best_bid - (lvl * self.tick_size)

    def _shift_book_down(self, bids: np.ndarray, asks: np.ndarray):
        """Shift depth levels down when bid level 1 is swept."""
        for lvl in range(self.num_levels - 1):
            bids[lvl, 0] = bids[lvl + 1, 0]
            bids[lvl, 1] = bids[lvl + 1, 1]
        bids[-1, 0] = bids[-2, 0] - self.tick_size
        bids[-1, 1] = max(10.0, self.rng.normal(self.mean_queue_size, self.queue_std))

        # Re-anchor asks
        best_ask = bids[0, 0] + (self.spread_ticks * self.tick_size)
        for lvl in range(self.num_levels):
            asks[lvl, 0] = best_ask + (lvl * self.tick_size)

    def _generate_labels(
        self,
        mid_prices: np.ndarray,
        horizons: List[int] = [10, 20, 50, 100],
        alpha: float = 0.00015
    ) -> Dict[int, np.ndarray]:
        """Calculates smoothed discrete labels {0: FLAT, 1: UP, 2: DOWN}."""
        n = len(mid_prices)
        labels = {}
        # Apply smoothing window of 5
        smoothed = pd.Series(mid_prices).rolling(5, min_periods=1).mean().values

        for k in horizons:
            lbl = np.zeros(n, dtype=np.int64)
            # Future smoothed mid-price at t+k
            fut = np.roll(smoothed, -k)
            ret = (fut - smoothed) / (smoothed + 1e-8)
            lbl[ret > alpha] = 1   # UP
            lbl[ret < -alpha] = 2  # DOWN
            lbl[-k:] = 0          # Invalidate boundary
            labels[k] = lbl
        return labels
