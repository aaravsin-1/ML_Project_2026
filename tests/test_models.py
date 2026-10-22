"""Unit Tests for Machine Learning Models (Tier 0 to Tier 5)."""

import pytest
import numpy as np
import torch

from src.models.baselines import NaiveMomentumBaseline, NaiveFlatBaseline, LogisticRegressionBaseline
from src.models.tabular_trees import RandomForestLOB, LightGBMLOB
from src.models.recurrent import LSTMLOBNet, GRULOBNet
from src.models.deeplob import DeepLOB
from src.models.tlob import TLOB


def test_baselines_shapes():
    X = np.random.randn(50, 10).astype(np.float32)
    y = np.random.choice([0, 1, 2], size=50)

    # Momentum
    mom = NaiveMomentumBaseline()
    preds_mom = mom.predict(y)
    assert len(preds_mom) == 50

    # Flat
    flat = NaiveFlatBaseline()
    preds_flat = flat.predict(X)
    assert np.all(preds_flat == 0)

    # Logistic
    lr = LogisticRegressionBaseline()
    lr.fit(X, y)
    preds_lr = lr.predict(X)
    probs_lr = lr.predict_proba(X)
    assert len(preds_lr) == 50
    assert probs_lr.shape == (50, 3)


def test_tabular_tree_models():
    X = np.random.randn(100, 15).astype(np.float32)
    y = np.random.choice([0, 1, 2], size=100)

    # Random Forest
    rf = RandomForestLOB(n_estimators=10, max_depth=4)
    rf.fit(X, y)
    preds_rf = rf.predict(X)
    assert len(preds_rf) == 100
    assert len(rf.feature_importances_) == 15

    # LightGBM
    lgb = LightGBMLOB(n_estimators=15, num_leaves=15, max_depth=4)
    lgb.fit(X, y)
    preds_lgb = lgb.predict(X)
    probs_lgb = lgb.predict_proba(X)
    assert len(preds_lgb) == 100
    assert probs_lgb.shape == (100, 3)


def test_deeplob_forward():
    """Validates DeepLOB architecture forward pass and tensor dimensions."""
    model = DeepLOB(num_classes=3, lstm_hidden=64)
    model.eval()
    # Batch=4, Channel=1, SeqLen=100, Features=40
    x = torch.randn(4, 1, 100, 40)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 3)


def test_tlob_forward():
    """Validates TLOB Transformer dual attention and Bilinear Normalization."""
    model = TLOB(num_features=40, seq_len=100, d_model=32, num_heads=2, num_layers=1)
    model.eval()
    x = torch.randn(4, 1, 100, 40)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 3)


def test_recurrent_models():
    lstm = LSTMLOBNet(num_features=40, hidden_dim=32, num_layers=1)
    gru = GRULOBNet(num_features=40, hidden_dim=32, num_layers=1)
    x = torch.randn(4, 1, 100, 40)

    out_lstm = lstm(x)
    out_gru = gru(x)
    assert out_lstm.shape == (4, 3)
    assert out_gru.shape == (4, 3)
