"""High-Fidelity Limit Order Book Execution Simulator.

Accurately models:
- Spread-crossing penalties (buying at Ask, selling at Bid)
- Walking the book / multi-level market impact slippage
- Exchange fee schedules (Maker rebate vs Taker fee)
- Execution latency measured in wall-clock time against snapshot timestamps
- Passive fills under a conservative trade-through rule (adverse selection is endogenous)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TradeExecution:
    timestamp_idx: int
    side: int         # +1: Buy, -1: Sell
    order_type: str   # "aggressive" or "passive"
    fill_price: float
    mid_price_at_decision: float
    size: float
    fee_paid: float
    slippage: float
    gross_pnl: float = 0.0
    net_pnl: float = 0.0


class ExecutionSimulator:
    """Continuous double auction execution simulator with microstructural realism."""

    def __init__(
        self,
        initial_cash: float = 1000000.0,
        maker_fee_bps: float = -0.5,    # -0.5 bps maker rebate
        taker_fee_bps: float = 1.5,     # 1.5 bps taker fee
        latency_ticks: int = 1,         # events of delay, used when no timestamps are given
        lot_size: float = 100.0,
        max_inventory: float = 500.0,
        latency_ms: Optional[float] = None,
        timestamps_ms: Optional[np.ndarray] = None
    ):
        self.initial_cash = initial_cash
        self.maker_fee = maker_fee_bps / 10000.0
        self.taker_fee = taker_fee_bps / 10000.0
        self.latency_ticks = latency_ticks
        self.lot_size = lot_size
        self.max_inventory = max_inventory
        # With timestamps, an order sent at decision time t reaches the exchange at the
        # first snapshot with timestamp >= t + latency_ms. Otherwise fall back to a fixed
        # number of events.
        self.latency_ms = latency_ms
        self.timestamps_ms = timestamps_ms

        # Account state
        self.cash = initial_cash
        self.position = 0.0
        self.equity_curve: List[float] = []
        self.trades: List[TradeExecution] = []
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.total_slippage = 0.0

    def reset(self):
        self.cash = self.initial_cash
        self.position = 0.0
        self.equity_curve = []
        self.trades = []
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.total_slippage = 0.0

    def arrival_index(self, decision_idx: int, n_rows: int) -> int:
        """Snapshot index at which an order sent at `decision_idx` reaches the book."""
        if self.latency_ms is not None and self.timestamps_ms is not None:
            target = self.timestamps_ms[decision_idx] + self.latency_ms
            idx = int(np.searchsorted(self.timestamps_ms, target, side="left"))
        else:
            idx = decision_idx + self.latency_ticks
        return min(max(idx, decision_idx), n_rows - 1)

    def execute_aggressive_order(
        self,
        decision_idx: int,
        side: int,  # +1: Buy, -1: Sell
        lob_40: np.ndarray,
        size: Optional[float] = None
    ) -> Optional[TradeExecution]:
        """Executes an aggressive market order crossing the spread and walking depth."""
        order_size = size if size is not None else self.lot_size
        
        # Position limit check
        if side > 0 and (self.position + order_size) > self.max_inventory:
            return None
        if side < 0 and (self.position - order_size) < -self.max_inventory:
            return None

        # Simulate execution latency delay: arrives at exchange at arrival_idx
        arrival_idx = self.arrival_index(decision_idx, len(lob_40))
        snapshot = lob_40[arrival_idx]
        decision_snapshot = lob_40[decision_idx]

        mid_at_decision = (decision_snapshot[0] + decision_snapshot[2]) / 2.0

        # Unpack depth at arrival
        # ask_prices: 0, 4, 8... ask_volumes: 1, 5, 9...
        # bid_prices: 2, 6, 10... bid_volumes: 3, 7, 11...
        ask_prices = snapshot[0::4]
        ask_volumes = snapshot[1::4]
        bid_prices = snapshot[2::4]
        bid_volumes = snapshot[3::4]

        # Fill by walking the book
        remaining = order_size
        cost = 0.0

        if side > 0:
            # Buying: consumes asks from level 1 upward
            for lvl in range(len(ask_prices)):
                available = ask_volumes[lvl]
                take = min(remaining, available)
                cost += take * ask_prices[lvl]
                remaining -= take
                if remaining <= 0:
                    break
            # If not filled in top 10 levels, fill remainder at last ask
            if remaining > 0:
                cost += remaining * (ask_prices[-1] + 0.05)
            avg_price = cost / order_size
        else:
            # Selling: consumes bids from level 1 downward
            for lvl in range(len(bid_prices)):
                available = bid_volumes[lvl]
                take = min(remaining, available)
                cost += take * bid_prices[lvl]
                remaining -= take
                if remaining <= 0:
                    break
            if remaining > 0:
                cost += remaining * max(0.01, bid_prices[-1] - 0.05)
            avg_price = cost / order_size

        # Fee & slippage calculation
        fee = avg_price * order_size * self.taker_fee
        slippage = abs(avg_price - mid_at_decision) * order_size

        # Update portfolio
        if side > 0:
            self.cash -= (cost + fee)
            self.position += order_size
        else:
            self.cash += (cost - fee)
            self.position -= order_size

        self.total_fees += fee
        self.total_slippage += slippage

        trade = TradeExecution(
            timestamp_idx=arrival_idx,
            side=side,
            order_type="aggressive",
            fill_price=avg_price,
            mid_price_at_decision=mid_at_decision,
            size=order_size,
            fee_paid=fee,
            slippage=slippage
        )
        self.trades.append(trade)
        return trade

    def passive_fill_index(
        self,
        side: int,
        price: float,
        post_idx: int,
        lob_40: np.ndarray,
        max_wait: int
    ) -> Optional[int]:
        """First snapshot after `post_idx` where a resting order at `price` would have filled.

        Conservative trade-through rule: we assume we sit at the back of the queue, so a
        resting bid only fills once the best bid trades *below* our price (the whole queue
        at our level was consumed), and symmetrically for asks. Fills therefore happen
        exactly when the market moves against the order, which is adverse selection.
        """
        end = min(post_idx + max_wait, len(lob_40) - 1)
        if end <= post_idx:
            return None
        window = lob_40[post_idx + 1 : end + 1]
        eps = 1e-9
        if side > 0:
            hit = np.nonzero(window[:, 2] < price - eps)[0]
        else:
            hit = np.nonzero(window[:, 0] > price + eps)[0]
        return int(post_idx + 1 + hit[0]) if len(hit) else None

    def fill_passive_order(
        self,
        fill_idx: int,
        side: int,
        price: float,
        mid_at_decision: float,
        size: Optional[float] = None
    ) -> Optional[TradeExecution]:
        """Books a passive fill at `price`, earning the maker rebate."""
        order_size = size if size is not None else self.lot_size
        if side > 0 and (self.position + order_size) > self.max_inventory:
            return None
        if side < 0 and (self.position - order_size) < -self.max_inventory:
            return None

        fee = price * order_size * self.maker_fee
        if side > 0:
            self.cash -= (price * order_size + fee)
            self.position += order_size
        else:
            self.cash += (price * order_size - fee)
            self.position -= order_size
        self.total_fees += fee

        trade = TradeExecution(
            timestamp_idx=fill_idx,
            side=side,
            order_type="passive",
            fill_price=price,
            mid_price_at_decision=mid_at_decision,
            size=order_size,
            fee_paid=fee,
            slippage=0.0
        )
        self.trades.append(trade)
        return trade

    def update_mark_to_market(self, mid_price: float) -> float:
        """Computes current total equity = cash + position * mid_price."""
        equity = self.cash + (self.position * mid_price)
        self.equity_curve.append(equity)
        return equity
