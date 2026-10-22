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


def test_passive_adverse_selection():
    """Validates passive fills and adverse selection penalty when VPIN is high."""
    sim = ExecutionSimulator(
        initial_cash=100000.0,
        maker_fee_bps=-0.5,
        adverse_selection_cost_bps=2.0
    )
    lob = np.zeros((2, 40), dtype=np.float32)
    lob[:, 0] = 100.05
    lob[:, 1] = 500.0
    lob[:, 2] = 100.00
    lob[:, 3] = 500.0

    # Case A: Low VPIN (safe)
    trade_safe = sim.execute_passive_order(
        decision_idx=0, side=1, lob_40=lob, vpin_metric=0.30, vpin_toxic_threshold=0.65
    )
    assert not trade_safe.adverse_selected
    assert np.isclose(trade_safe.fill_price, 100.00)

    # Case B: High VPIN (toxic flow -> adverse fill penalty)
    trade_toxic = sim.execute_passive_order(
        decision_idx=0, side=1, lob_40=lob, vpin_metric=0.85, vpin_toxic_threshold=0.65
    )
    assert trade_toxic.adverse_selected
    assert trade_toxic.fill_price > 100.00  # Penalized by adverse selection
