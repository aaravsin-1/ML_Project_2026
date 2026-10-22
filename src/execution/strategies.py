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

        for t in range(n):
            sig = signals[t]
            p = mid_prices[t]

            # Signal 1 (UP) -> Buy, Signal 2 (DOWN) -> Sell
            if sig == 1 and (pos + lot) <= max_inv:
                cash -= lot * p
                pos += lot
            elif sig == 2 and (pos - lot) >= -max_inv:
                cash += lot * p
                pos -= lot

            equity[t] = cash + (pos * p)

        return {"equity": equity, "final_pnl": equity[-1] - self.simulator.initial_cash}

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
        toxic_vpin_thresh: float = 0.65
    ) -> Dict[str, np.ndarray]:
        """Dynamic Passive Market Maker: Captures spread/rebates with active VPIN adverse defense."""
        self.simulator.reset()
        n = len(signals)
        mid_prices = (lob_40[:, 0] + lob_40[:, 2]) / 2.0
        toxic_avoidance_count = 0

        for t in range(n):
            sig = signals[t]
            vpin_t = vpin_series[t]

            # VPIN Toxicity Defense: Cancel resting quotes during toxic flow
            if vpin_t > toxic_vpin_thresh:
                toxic_avoidance_count += 1
                # Do not provide passive liquidity into informed toxic flow
            else:
                # Safe flow regime: provide liquidity on predicted trend side
                if sig == 1:
                    # Expect price rise: rest passive bid to buy at bid price
                    self.simulator.execute_passive_order(
                        decision_idx=t,
                        side=1,
                        lob_40=lob_40,
                        vpin_metric=vpin_t,
                        vpin_toxic_threshold=toxic_vpin_thresh
                    )
                elif sig == 2:
                    # Expect price fall: rest passive ask to sell at ask price
                    self.simulator.execute_passive_order(
                        decision_idx=t,
                        side=-1,
                        lob_40=lob_40,
                        vpin_metric=vpin_t,
                        vpin_toxic_threshold=toxic_vpin_thresh
                    )

            self.simulator.update_mark_to_market(mid_prices[t])

        return {
            "equity": np.array(self.simulator.equity_curve),
            "final_pnl": self.simulator.equity_curve[-1] - self.simulator.initial_cash,
            "total_fees": self.simulator.total_fees,
            "num_trades": len(self.simulator.trades),
            "toxic_avoidance_count": toxic_avoidance_count
        }
