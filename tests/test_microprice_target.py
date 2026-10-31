"""Tests for Experiment K (micro-price as a prediction target)."""

import numpy as np

from src.experiments.exp_k_microprice import (
    block_bootstrap_ic_diff, evaluate_session, fit_beta, forward_change_bp, gap_bp, mid_and_micro, r2_vs_zero, score,
)


def _book(ask, ask_q, bid, bid_q):
    lob = np.zeros((len(ask), 40))
    lob[:, 0], lob[:, 1], lob[:, 2], lob[:, 3] = ask, ask_q, bid, bid_q
    return lob


def test_micro_price_is_imbalance_weighted_mid():
    lob = _book([100.02, 100.02, 100.02], [100, 300, 0], [100.00, 100.00, 100.00], [300, 100, 0])
    mid, micro = mid_and_micro(lob)
    assert np.allclose(mid, 100.01)
    # more size on the bid pushes the micro-price toward the ask: micro - mid = I * spread / 2
    imbalance = np.array([0.5, -0.5])
    assert np.allclose(micro[:2] - mid[:2], imbalance * 0.02 / 2)
    assert micro[2] == mid[2]  # empty level 1 falls back to the mid
    assert np.allclose(gap_bp(mid, micro)[:2], imbalance * 0.01 / 100.01 * 1e4)


def test_forward_change_uses_only_the_future_and_never_wraps():
    p = np.array([100.0, 101.0, 99.0, 100.0])
    f = forward_change_bp(p, 2)
    assert np.allclose(f[:2], [(99 - 100) / 100 * 1e4, (100 - 101) / 101 * 1e4])
    assert np.isnan(f[2:]).all()


def test_beta_and_r2():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20000)
    y = 0.7 * x + rng.normal(scale=0.5, size=x.size)
    assert abs(fit_beta(x, y) - 0.7) < 0.02
    assert r2_vs_zero(y, np.zeros_like(y)) == 0.0
    assert r2_vs_zero(y, y) == 1.0
    s = score(y, 0.7 * x)
    assert s["ic_pearson"] > 0.8 and s["hit_rate"] > 0.75 and s["r2"] > 0.6
    # calibration rescales a mis-scaled but correlated forecast before R^2
    assert score(y, 10 * x, calib=0.07)["r2"] > 0.6 > score(y, 10 * x)["r2"]


def test_bootstrap_ci_brackets_the_true_difference():
    rng = np.random.default_rng(1)
    signal = rng.normal(size=30000)
    y = signal + rng.normal(size=signal.size)
    good, bad = signal, signal + 3 * rng.normal(size=signal.size)
    lo, hi = block_bootstrap_ic_diff(y, good, bad, block=50, draws=300)
    true_diff = np.corrcoef(y, good)[0, 1] - np.corrcoef(y, bad)[0, 1]
    assert lo < true_diff < hi and lo > 0
    lo, hi = block_bootstrap_ic_diff(y, good, good, block=50, draws=100)
    assert lo == hi == 0.0


def test_evaluate_session_finds_a_planted_signal_and_the_control_finds_nothing():
    """End to end on a toy book: the next mid move follows the queue imbalance. LightGBM must find it,
    the micro-price rule must have beta > 0, and the shuffled-target control must score ~0."""
    rng = np.random.default_rng(2)
    n = 12000
    bid_q, ask_q = rng.integers(1, 100, n).astype(float), rng.integers(1, 100, n).astype(float)
    imb = (bid_q - ask_q) / (bid_q + ask_q)
    steps = np.sign(np.r_[0.0, imb[:-1]] + rng.normal(scale=0.6, size=n)) * 0.01  # move follows last imbalance
    mid = 100.0 + np.cumsum(steps)
    lob = _book(mid + 0.005, ask_q, mid - 0.005, bid_q)
    X = rng.normal(size=(n, 3)).astype(np.float32)  # irrelevant features; the gap is appended inside
    tr, va, te = np.arange(0, 8000), np.arange(8050, 10000), np.arange(10050, n - 5)

    rows, betas = evaluate_session("toy", "toy", lob, X, lambda k: {"train": tr, "val": va, "test": te},
                                   {"1": 1}, seed=0, block=100)
    by = {(r["model"], r["train_target"]): r for r in rows}
    assert betas[0]["beta_train"] > 0
    assert by[("LightGBM", "mid")]["ic_pearson"] > 0.2
    assert by[("Micro-price rule (beta x gap)", "mid")]["ic_pearson"] > 0.2
    # the control (shuffled train and validation targets, averaged over 3 shuffles) is far below the real model
    control = by[("LightGBM, shuffled-target control", "mid")]["ic_pearson"]
    assert abs(control) < 0.1
    assert by[("LightGBM", "mid")]["ic_pearson"] > 3 * abs(control)
