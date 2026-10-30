"""High-Fidelity Limit Order Book Execution Simulator.

Accurately models:
- Spread-crossing penalties (buying at Ask, selling at Bid)
- Walking the book / multi-level market impact slippage, with an explicit policy for
  orders larger than the visible depth (extrapolate the book, partial fill, or reject)
- Exchange fee schedules (Maker rebate vs Taker fee)
- Execution latency measured in wall-clock time against snapshot timestamps
- Passive fills under a conservative trade-through rule (adverse selection is endogenous)

Not modelled: aggressive fills do not deplete the book in later snapshots, so two orders
that arrive on the same snapshot both see its full depth.
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
    requested_size: Optional[float] = None  # differs from `size` only for partial fills
    levels_consumed: int = 0                # visible levels the order walked through
    beyond_book_size: float = 0.0           # quantity filled past the visible depth


OVERFLOW_POLICIES = ("extrapolate", "partial", "reject")


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
        timestamps_ms: Optional[np.ndarray] = None,
        overflow_policy: str = "extrapolate"
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
        # What to do when an aggressive order is larger than the visible depth:
        #   "extrapolate": continue the book past its last visible level (see `_walk_book`)
        #   "partial":     fill only the visible depth and drop the rest
        #   "reject":      do not trade at all
        if overflow_policy not in OVERFLOW_POLICIES:
            raise ValueError(f"overflow_policy must be one of {OVERFLOW_POLICIES}, got {overflow_policy!r}")
        self.overflow_policy = overflow_policy

        # Account state
        self.cash = initial_cash
        self.position = 0.0
        self.equity_curve: List[float] = []
        self.trades: List[TradeExecution] = []
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self._reset_depth_counters()

    def _reset_depth_counters(self):
        self.n_orders_beyond_book = 0      # aggressive orders that ran past the visible depth
        self.qty_beyond_book = 0.0         # quantity filled past it ("extrapolate")
        self.qty_unfilled = 0.0            # quantity dropped ("partial") or refused ("reject")

    def reset(self):
        self.cash = self.initial_cash
        self.position = 0.0
        self.equity_curve = []
        self.trades = []
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self._reset_depth_counters()

    def arrival_index(self, decision_idx: int, n_rows: int) -> int:
        """Snapshot index at which an order sent at `decision_idx` reaches the book."""
        if self.latency_ms is not None and self.timestamps_ms is not None:
            target = self.timestamps_ms[decision_idx] + self.latency_ms
            idx = int(np.searchsorted(self.timestamps_ms, target, side="left"))
        else:
            idx = decision_idx + self.latency_ticks
        return min(max(idx, decision_idx), n_rows - 1)

    @staticmethod
    def _visible_levels(snapshot: np.ndarray, side: int) -> Tuple[np.ndarray, np.ndarray]:
        """Usable (prices, volumes) on the side an order of `side` trades against.

        Levels with a missing, non-positive or non-finite price or volume are dropped, so
        books padded with zeros or NaNs beyond their real depth are handled.
        """
        # Layout: ask_p 0, ask_q 1, bid_p 2, bid_q 3, then repeating per level
        prices = snapshot[0::4] if side > 0 else snapshot[2::4]
        volumes = snapshot[1::4] if side > 0 else snapshot[3::4]
        ok = np.isfinite(prices) & np.isfinite(volumes) & (prices > 0) & (volumes > 0)
        return prices[ok].astype(np.float64), volumes[ok].astype(np.float64)

    @staticmethod
    def _extrapolation_step(prices: np.ndarray, snapshot: np.ndarray) -> float:
        """Price gap between levels assumed past the visible book.

        The mean gap between visible levels. With a single visible level, the quoted
        spread is used instead.
        """
        if len(prices) >= 2:
            return float(np.abs(np.diff(prices)).mean())
        spread = float(snapshot[0] - snapshot[2])
        return spread if np.isfinite(spread) and spread > 0 else 0.0

    def _walk_book(
        self, snapshot: np.ndarray, side: int, order_size: float
    ) -> Optional[Tuple[float, float, int, float]]:
        """Fills `order_size` against `snapshot` under the overflow policy.

        Returns (filled_qty, cost, levels_consumed, beyond_book_qty), or None if no fill.

        "extrapolate" assumes the book keeps going past its last visible level with the
        same shape: levels spaced by the mean visible gap and each holding the mean visible
        level volume. The cost of the excess therefore grows with its size and scales with
        the instrument's own prices, unlike a fixed price add-on.
        """
        prices, volumes = self._visible_levels(snapshot, side)
        if len(prices) == 0:
            return None  # nothing to trade against

        cum = np.cumsum(volumes)
        n_full = int(np.searchsorted(cum, order_size, side="left"))
        if n_full < len(prices):
            # Fully filled inside the visible book
            take = volumes[: n_full + 1].copy()
            take[-1] = order_size - (cum[n_full - 1] if n_full > 0 else 0.0)
            return order_size, float(take @ prices[: n_full + 1]), n_full + 1, 0.0

        visible_qty = float(cum[-1])
        visible_cost = float(volumes @ prices)
        excess = order_size - visible_qty
        if self.overflow_policy == "reject":
            return None
        if self.overflow_policy == "partial":
            return visible_qty, visible_cost, len(prices), 0.0

        step = self._extrapolation_step(prices, snapshot)
        level_qty = float(volumes.mean())
        # Excess fills across extra levels j = 1, 2, ... priced at last +/- j * step:
        # m full levels of `level_qty`, then the remainder r at level m + 1
        m = int(excess // level_qty)
        r = excess - m * level_qty
        last = float(prices[-1])
        offsets = level_qty * step * m * (m + 1) / 2.0 + r * step * (m + 1)
        extra_cost = excess * last + side * offsets
        if side < 0:
            # A sell can never fill at a negative price
            extra_cost = max(extra_cost, 0.0)
        return order_size, visible_cost + extra_cost, len(prices), excess

    def execute_aggressive_order(
        self,
        decision_idx: int,
        side: int,  # +1: Buy, -1: Sell
        lob_40: np.ndarray,
        size: Optional[float] = None
    ) -> Optional[TradeExecution]:
        """Executes an aggressive market order crossing the spread and walking depth.

        Returns None if the position limit blocks the order, the book side is empty, or
        the order exceeds the visible depth under the "reject" policy.
        """
        order_size = size if size is not None else self.lot_size
        if order_size <= 0:
            return None

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

        walked = self._walk_book(snapshot, side, order_size)
        if walked is None:
            if self.overflow_policy == "reject" and len(self._visible_levels(snapshot, side)[0]):
                # Refused for exceeding the visible depth, not for an empty book
                self.n_orders_beyond_book += 1
                self.qty_unfilled += order_size
            return None
        filled, cost, levels, beyond = walked
        unfilled = order_size - filled
        if beyond > 0 or unfilled > 0:
            self.n_orders_beyond_book += 1
            self.qty_beyond_book += beyond
            self.qty_unfilled += unfilled
        if filled <= 0:
            return None
        avg_price = cost / filled

        # Fees, and slippage as signed cost against the decision mid: positive when the
        # fill is worse than the mid, negative if the price moved our way during latency
        fee = cost * self.taker_fee
        slippage = side * (avg_price - mid_at_decision) * filled

        # Update portfolio
        if side > 0:
            self.cash -= (cost + fee)
            self.position += filled
        else:
            self.cash += (cost - fee)
            self.position -= filled

        self.total_fees += fee
        self.total_slippage += slippage

        trade = TradeExecution(
            timestamp_idx=arrival_idx,
            side=side,
            order_type="aggressive",
            fill_price=avg_price,
            mid_price_at_decision=mid_at_decision,
            size=filled,
            fee_paid=fee,
            slippage=slippage,
            requested_size=order_size,
            levels_consumed=levels,
            beyond_book_size=beyond,
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
            slippage=0.0,
            requested_size=order_size
        )
        self.trades.append(trade)
        return trade

    def update_mark_to_market(self, mid_price: float) -> float:
        """Computes current total equity = cash + position * mid_price."""
        equity = self.cash + (self.position * mid_price)
        self.equity_curve.append(equity)
        return equity
