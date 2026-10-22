"""Unit Tests for Microstructural Signals and Feature Engineering."""

import pytest
import numpy as np
import pandas as pd
import torch

from src.microstructure.ofi import OrderFlowImbalance
from src.microstructure.vpin import VPINEngine
from src.microstructure.microprice import MicroPriceEstimator
from src.data.normalizers import DecimalPrecisionNormalizer, CausalRollingZScore, BilinearNormalization


def test_ofi_exact_formula():
    """Validates Cont et al. Level-1 OFI formula under price changes and queue updates."""
    # Step 0: Base state
    # Bid = 100.0, Q_b = 500 | Ask = 100.01, Q_a = 500
    # Step 1: Bid price rises (Demand pressure) -> Delta_W = Q_b(t) = 600
    # Ask price stays same -> Delta_V = Q_a(t) - Q_a(t-1) = 0
    # OFI = 600 - 0 = 600
    bp = np.array([100.0, 100.01, 100.01], dtype=np.float32)
    bv = np.array([500.0, 600.0, 400.0], dtype=np.float32)
    ap = np.array([100.02, 100.02, 100.03], dtype=np.float32)
    av = np.array([500.0, 500.0, 300.0], dtype=np.float32)

    ofi = OrderFlowImbalance.compute_top_level_ofi(bp, bv, ap, av)
    assert len(ofi) == 3
    assert ofi[0] == 0.0  # Warmup
    assert ofi[1] == 600.0  # Bid price stepped up: Delta_W = 600, Delta_V = 0 -> OFI = 600


def test_microprice_and_imbalance():
    """Validates Stoikov Micro-price and Imbalance bounds."""
    # Synthetic 40-feature single snapshot
    lob = np.zeros((10, 40), dtype=np.float32)
    # Ask P1 = 100.02, Ask V1 = 200, Bid P1 = 100.00, Bid V1 = 800
    lob[:, 0] = 100.02
    lob[:, 1] = 200.0
    lob[:, 2] = 100.00
    lob[:, 3] = 800.0

    geom = MicroPriceEstimator.compute_features(lob)
    
    # Imbalance: (800 - 200) / (800 + 200) = 0.60
    assert np.isclose(geom["imbalance_l1"][0], 0.60, atol=1e-4)

    # Micro-price: (800 * 100.02 + 200 * 100.00) / 1000 = (80016 + 20000)/1000 = 100.016
    assert np.isclose(geom["micro_price"][0], 100.016, atol=1e-4)

    # Mid price = 100.01 -> Micro deviation = +0.006 (skewed toward ask)
    assert np.isclose(geom["micro_deviation"][0], 0.006, atol=1e-4)


def test_vpin_calculation():
    """Validates VPIN volume clock and toxicity bounds."""
    # Create trades: 5000 shares total, all buy-initiated -> VPIN should be close to 1.0
    trades = pd.DataFrame({
        "timestamp_ms": np.arange(10, dtype=float) * 100.0,
        "price": np.full(10, 100.0),
        "size": np.full(10, 500.0),
        "direction": np.ones(10, dtype=int)  # 100% buy orders
    })

    vpin_engine = VPINEngine(bucket_volume=1000.0, window_buckets=3)
    bucket_t, vpin_vals, imbs = vpin_engine.compute_from_trades(trades)

    assert len(bucket_t) == 5  # 5000 / 1000 = 5 buckets
    assert np.all(vpin_vals >= 0.0)
    assert np.all(vpin_vals <= 1.0)
    # Since all flow is one-sided buys, VPIN should approach 1.0
    assert vpin_vals[-1] > 0.90


def test_bilinear_normalization():
    """Validates Bilinear Normalization (BiN) PyTorch module."""
    bin_layer = BilinearNormalization(num_features=40, seq_len=100)
    x = torch.randn(8, 1, 100, 40)
    out = bin_layer(x)
    assert out.shape == x.shape
    # Check gradient backward pass
    loss = out.sum()
    loss.backward()
    assert bin_layer.gamma_feat.grad is not None
    assert bin_layer.gamma_time.grad is not None
