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
