"""Unit Tests for High-Fidelity Limit Order Book Execution Simulator."""

import pytest
import numpy as np

from src.execution.simulator import ExecutionSimulator
from src.execution.metrics import PerformanceMetrics


def test_spread_crossing_aggressive():
    """Validates that aggressive orders cross the spread and pay the ask/bid correctly."""
    sim = ExecutionSimulator(
        initial_cash=100000.0,
        taker_fee_bps=2.0,  # 2 bps fee
        latency_ticks=0,
        lot_size=100.0
    )

    # LOB snapshot: Ask1 = 100.05 (size 500), Bid1 = 100.00 (size 500)
    lob = np.zeros((5, 40), dtype=np.float32)
    lob[:, 0] = 100.05
    lob[:, 1] = 500.0
    lob[:, 2] = 100.00
    lob[:, 3] = 500.0

    # 1. Buy order (crosses spread to ask)
    trade_buy = sim.execute_aggressive_order(decision_idx=0, side=1, lob_40=lob, size=100.0)
    assert trade_buy is not None
    assert np.isclose(trade_buy.fill_price, 100.05)
    # Fee = 100.05 * 100 * 0.0002 = 2.001
    assert np.isclose(trade_buy.fee_paid, 2.001)
    assert np.isclose(sim.position, 100.0)

    # 2. Sell order (crosses spread to bid)
    trade_sell = sim.execute_aggressive_order(decision_idx=1, side=-1, lob_40=lob, size=100.0)
    assert trade_sell is not None
    assert np.isclose(trade_sell.fill_price, 100.00)
    assert np.isclose(sim.position, 0.0)


def test_walking_the_book_slippage():
    """Validates that large orders consume depth across multiple levels and incur slippage."""
    sim = ExecutionSimulator(initial_cash=100000.0, latency_ticks=0)
    
    # LOB snapshot with shallow top levels:
    # Level 1: Ask = 100.05, Size = 50
    # Level 2: Ask = 100.10, Size = 50
    lob = np.zeros((2, 40), dtype=np.float32)
    lob[:, 0] = 100.05  # Ask 1
    lob[:, 1] = 50.0    # Ask 1 Size
    lob[:, 2] = 100.00  # Bid 1
    lob[:, 3] = 500.0
    lob[:, 4] = 100.10  # Ask 2
    lob[:, 5] = 100.0   # Ask 2 Size

    # Order of 100 shares must take 50 @ 100.05 and 50 @ 100.10
    # Average fill = (50 * 100.05 + 50 * 100.10) / 100 = 100.075
    trade = sim.execute_aggressive_order(decision_idx=0, side=1, lob_40=lob, size=100.0)
    assert np.isclose(trade.fill_price, 100.075)
    assert trade.fill_price > 100.05  # Proves slippage from walking the book!


def _two_level_book(rows: int = 2) -> np.ndarray:
    """Asks 100.05 x 50 and 100.10 x 50, bid 100.00 x 500, levels 3-10 zero-padded."""
    lob = np.zeros((rows, 40), dtype=np.float64)
    lob[:, 0], lob[:, 1] = 100.05, 50.0
    lob[:, 2], lob[:, 3] = 100.00, 500.0
    lob[:, 4], lob[:, 5] = 100.10, 50.0
    return lob


def test_overflow_extrapolates_the_book():
    """Past the visible depth the book continues with the mean level gap and volume."""
    sim = ExecutionSimulator(initial_cash=1e6, latency_ticks=0, taker_fee_bps=0.0)
    # 100 visible at 100.05/100.10; the extra 75 fill as 50 @ 100.15 and 25 @ 100.20.
    # The zero-padded levels 3-10 must not be treated as asks at price 0.
    trade = sim.execute_aggressive_order(0, side=1, lob_40=_two_level_book(), size=175.0)
    expected = (50 * 100.05 + 50 * 100.10 + 50 * 100.15 + 25 * 100.20) / 175
    assert np.isclose(trade.fill_price, expected)
    assert trade.size == 175.0 and trade.levels_consumed == 2
    assert np.isclose(trade.beyond_book_size, 75.0)
    assert sim.n_orders_beyond_book == 1 and np.isclose(sim.qty_beyond_book, 75.0)
    assert np.isclose(sim.cash, 1e6 - expected * 175)


def test_overflow_cost_grows_with_size():
    """Bigger overflows pay more per unit; the old fixed add-on did not."""
    lob = _two_level_book()
    prices = []
    for size in (100.0, 150.0, 300.0, 1000.0):
        sim = ExecutionSimulator(initial_cash=1e7, latency_ticks=0, max_inventory=1e6)
        prices.append(sim.execute_aggressive_order(0, 1, lob, size=size).fill_price)
    assert all(a < b for a, b in zip(prices, prices[1:]))


def test_overflow_sell_side_mirrors_buy_side():
    lob = np.zeros((1, 40))
    lob[0, 0], lob[0, 1] = 100.05, 500.0
    lob[0, 2], lob[0, 3] = 100.00, 50.0
    lob[0, 6], lob[0, 7] = 99.95, 50.0
    sim = ExecutionSimulator(initial_cash=1e6, latency_ticks=0, taker_fee_bps=0.0)
    trade = sim.execute_aggressive_order(0, side=-1, lob_40=lob, size=175.0)
    expected = (50 * 100.00 + 50 * 99.95 + 50 * 99.90 + 25 * 99.85) / 175
    assert np.isclose(trade.fill_price, expected)
    assert np.isclose(sim.position, -175.0)


def test_overflow_partial_and_reject_policies():
    lob = _two_level_book()
    partial = ExecutionSimulator(initial_cash=1e6, latency_ticks=0, overflow_policy="partial")
    trade = partial.execute_aggressive_order(0, 1, lob, size=175.0)
    assert trade.size == 100.0 and trade.requested_size == 175.0
    assert np.isclose(trade.fill_price, 100.075)
    assert np.isclose(partial.position, 100.0) and np.isclose(partial.qty_unfilled, 75.0)

    reject = ExecutionSimulator(initial_cash=1e6, latency_ticks=0, overflow_policy="reject")
    assert reject.execute_aggressive_order(0, 1, lob, size=175.0) is None
    assert reject.position == 0.0 and reject.n_orders_beyond_book == 1
    # Orders that fit are unaffected by the policy
    assert np.isclose(reject.execute_aggressive_order(0, 1, lob, size=80.0).fill_price,
                      (50 * 100.05 + 30 * 100.10) / 80)

    with pytest.raises(ValueError):
        ExecutionSimulator(overflow_policy="bogus")


def test_single_level_and_empty_books():
    """One visible level extrapolates with the spread; an empty side gives no fill."""
    lob = np.zeros((1, 40))
    lob[0, 0], lob[0, 1] = 100.05, 10.0
    lob[0, 2], lob[0, 3] = 100.00, 10.0
    sim = ExecutionSimulator(initial_cash=1e6, latency_ticks=0, taker_fee_bps=0.0)
    trade = sim.execute_aggressive_order(0, 1, lob, size=20.0)
    assert np.isclose(trade.fill_price, (10 * 100.05 + 10 * 100.10) / 20)

    lob[0, 1] = np.nan  # ask side has no usable level
    assert sim.execute_aggressive_order(0, 1, lob, size=1.0) is None


def test_slippage_is_signed_cost():
    """Slippage is positive when filled worse than the decision mid, negative if better."""
    lob = np.zeros((2, 40))
    lob[:, 1] = lob[:, 3] = 500.0
    lob[0, 0], lob[0, 2] = 100.05, 100.00   # decision mid 100.025
    lob[1, 0], lob[1, 2] = 99.95, 99.90     # price falls during latency
    sim = ExecutionSimulator(initial_cash=1e6, latency_ticks=1)
    buy = sim.execute_aggressive_order(0, 1, lob, size=10.0)
    assert np.isclose(buy.slippage, (99.95 - 100.025) * 10)   # bought below the mid: negative
    sell = sim.execute_aggressive_order(0, -1, lob, size=10.0)
    assert np.isclose(sell.slippage, (100.025 - 99.90) * 10)  # sold below the mid: positive


def test_passive_fill_requires_trade_through():
    """A resting bid fills only after the best bid trades below it, and earns the rebate."""
    sim = ExecutionSimulator(initial_cash=100000.0, maker_fee_bps=-0.5, latency_ticks=0)
    lob = np.zeros((6, 40), dtype=np.float64)
    lob[:, 0] = [100.01, 100.01, 100.01, 100.01, 100.00, 100.00]  # ask
    lob[:, 2] = [100.00, 100.00, 100.00, 100.00, 99.99, 99.99]    # bid drops at row 4
    lob[:, 1] = lob[:, 3] = 500.0

    # Bid at 100.00 posted at row 0: no fill while the bid holds, fill at row 4
    assert sim.passive_fill_index(side=1, price=100.00, post_idx=0, lob_40=lob, max_wait=10) == 4
    # A short wait window expires before the trade-through
    assert sim.passive_fill_index(side=1, price=100.00, post_idx=0, lob_40=lob, max_wait=3) is None
    # An ask at 100.01 never fills because the ask never trades above it
    assert sim.passive_fill_index(side=-1, price=100.01, post_idx=0, lob_40=lob, max_wait=10) is None

    trade = sim.fill_passive_order(fill_idx=4, side=1, price=100.00, mid_at_decision=100.005)
    assert np.isclose(trade.fill_price, 100.00)
    assert trade.fee_paid < 0  # maker rebate
    assert np.isclose(sim.position, 100.0)


def test_latency_uses_timestamps():
    """With timestamps, orders arrive at the first snapshot at or after t + latency."""
    ts = np.array([0.0, 4.0, 9.0, 15.0, 30.0])
    lob = np.zeros((5, 40))
    sim = ExecutionSimulator(latency_ms=10.0, timestamps_ms=ts)
    assert sim.arrival_index(0, len(lob)) == 3
    assert sim.arrival_index(3, len(lob)) == 4
    assert sim.arrival_index(4, len(lob)) == 4  # clipped at the end of the data
    assert ExecutionSimulator(latency_ms=0.0, timestamps_ms=ts).arrival_index(2, len(lob)) == 2


def test_sharpe_uses_time_bars():
    """Sharpe is computed on wall-clock bars and annualized by the bar count per year."""
    rng = np.random.default_rng(0)
    ts = np.arange(2000) * 10.0  # 10 ms events -> 20 one-second bars
    equity = 1e6 + np.cumsum(rng.normal(1.0, 5.0, size=2000))
    m = PerformanceMetrics.calculate_metrics(equity, 1e6, timestamps_ms=ts, bar_ms=1000.0)
    assert m["n_bars"] == 19
    bars = equity[99::100]
    r = np.diff(bars) / bars[:-1]
    expected = r.mean() / r.std() * np.sqrt(252 * 6.5 * 3600)
    assert np.isclose(m["sharpe_ratio"], expected)
    # Without timestamps the Sharpe is per step and not annualized
    m_raw = PerformanceMetrics.calculate_metrics(equity, 1e6)
    r_raw = np.diff(equity) / equity[:-1]
    assert np.isclose(m_raw["sharpe_ratio"], r_raw.mean() / r_raw.std())
