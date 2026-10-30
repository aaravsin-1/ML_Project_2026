"""Algorithmic Trading Strategies Evaluated Under Real-World Microstructure Friction."""

from typing import Dict, List, Optional, Tuple
import numpy as np

from src.execution.simulator import ExecutionSimulator, TradeExecution


class MicrostructureStrategyRunner:
    """Runs contrasting execution modes over an out-of-sample LOB sequence."""

    def __init__(self, simulator: ExecutionSimulator):
        self.simulator = simulator

    def run_academic_midprice(
        self,
        signals: np.ndarray,      # 0: FLAT, 1: UP, 2: DOWN
        mid_prices: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Academic Fantasy Mode: Fills at exact mid-price with 0 spread and 0 fees."""
        n = len(signals)
        cash = self.simulator.initial_cash
        pos = 0.0
        lot = self.simulator.lot_size
        max_inv = self.simulator.max_inventory
        equity = np.zeros(n, dtype=np.float64)
        num_trades = 0

        for t in range(n):
            sig = signals[t]
            p = mid_prices[t]

            # Signal 1 (UP) -> Buy, Signal 2 (DOWN) -> Sell
            if sig == 1 and (pos + lot) <= max_inv:
                cash -= lot * p
                pos += lot
                num_trades += 1
            elif sig == 2 and (pos - lot) >= -max_inv:
                cash += lot * p
                pos -= lot
                num_trades += 1

            equity[t] = cash + (pos * p)

        return {"equity": equity, "final_pnl": equity[-1] - self.simulator.initial_cash, "num_trades": num_trades}

    def run_realistic_aggressive(
        self,
        signals: np.ndarray,
        lob_40: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Realistic Aggressive Taker: Crosses spread, walks book, incurs taker fees & latency."""
        self.simulator.reset()
        n = len(signals)
        mid_prices = (lob_40[:, 0] + lob_40[:, 2]) / 2.0

        for t in range(n):
            sig = signals[t]
            if sig == 1:
                self.simulator.execute_aggressive_order(t, side=1, lob_40=lob_40)
            elif sig == 2:
                self.simulator.execute_aggressive_order(t, side=-1, lob_40=lob_40)

            self.simulator.update_mark_to_market(mid_prices[t])

        return {
            "equity": np.array(self.simulator.equity_curve),
            "final_pnl": self.simulator.equity_curve[-1] - self.simulator.initial_cash,
            "total_fees": self.simulator.total_fees,
            "total_slippage": self.simulator.total_slippage,
            "num_trades": len(self.simulator.trades)
        }

    def run_dynamic_passive_vpin(
        self,
        signals: np.ndarray,
        lob_40: np.ndarray,
        vpin_series: np.ndarray,
        toxic_vpin_thresh: float,
        max_wait_events: int = 50
    ) -> Dict[str, np.ndarray]:
        """Passive maker: joins the touch on the predicted side, stands down when VPIN is toxic.

        At most one working order exists at a time. It is posted at the touch when it
        reaches the exchange (after latency), fills only under the trade-through rule in
        `ExecutionSimulator.passive_fill_index`, and is cancelled when the signal flips,
        VPIN turns toxic, or it has waited `max_wait_events`.
        """
        self.simulator.reset()
        n = len(signals)
        mid_prices = (lob_40[:, 0] + lob_40[:, 2]) / 2.0
        toxic_avoidance_count = 0
        orders_posted = 0
        working = None  # (side, price, fill_idx or None, expiry_idx, mid_at_decision)

        for t in range(n):
            sig = signals[t]
            toxic = vpin_series[t] > toxic_vpin_thresh
            desired = 1 if sig == 1 else (-1 if sig == 2 else 0)

            # 1. Resolve the working order: fill, expire, or cancel.
            if working is not None:
                side, price, fill_idx, expiry, mid_dec = working
                if fill_idx is not None and fill_idx <= t:
                    self.simulator.fill_passive_order(fill_idx, side, price, mid_dec)
                    working = None
                elif t >= expiry or toxic or desired != side:
                    working = None

            if toxic:
                toxic_avoidance_count += 1
            elif working is None and desired != 0:
                # 2. Post a new order at the touch seen when it arrives at the exchange.
                post_idx = self.simulator.arrival_index(t, n)
                price = lob_40[post_idx, 2] if desired > 0 else lob_40[post_idx, 0]
                fill_idx = self.simulator.passive_fill_index(desired, price, post_idx, lob_40, max_wait_events)
                working = (desired, price, fill_idx, post_idx + max_wait_events, mid_prices[t])
                orders_posted += 1

            self.simulator.update_mark_to_market(mid_prices[t])

        return {
            "equity": np.array(self.simulator.equity_curve),
            "final_pnl": self.simulator.equity_curve[-1] - self.simulator.initial_cash,
            "total_fees": self.simulator.total_fees,
            "num_trades": len(self.simulator.trades),
            "orders_posted": orders_posted,
            "toxic_avoidance_count": toxic_avoidance_count
        }
