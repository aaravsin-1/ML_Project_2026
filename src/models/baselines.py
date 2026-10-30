"""Tier 0 (Naïve Baselines) and Tier 1 (Linear Baselines) for LOB Price Movement."""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier


class NaiveMomentumBaseline:
    """Tier 0: Predict that the most recent *realized* k-event move repeats.

    The label at row t compares the smoothed mid-price at t+k with t, so it is only
    known at time t+k. The latest label observable at decision time t is therefore
    y[t-k], not y[t-1] (y[t-1] already contains k-1 events of future prices).
    """
    def __init__(self, horizon_k: int = 20):
        self.horizon_k = horizon_k

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, y_full: np.ndarray, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Predictions for rows `indices` of the full label series (default: every row)."""
        if indices is None:
            indices = np.arange(len(y_full))
        src = np.asarray(indices) - self.horizon_k
        preds = np.zeros(len(src), dtype=np.int64)
        valid = src >= 0
        preds[valid] = y_full[src[valid]]
        return preds

    def predict_proba(self, y_full: np.ndarray, indices: Optional[np.ndarray] = None) -> np.ndarray:
        preds = self.predict(y_full, indices)
        proba = np.full((len(preds), 3), 0.1, dtype=np.float32)
        proba[np.arange(len(preds)), preds] = 0.8
        return proba


class NaiveFlatBaseline:
    """Tier 0: Predicts FLAT (0) unconditionally.
    
    Tests whether models can beat the majority class in highly imbalanced LOB data.
    """
    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        proba = np.zeros((n, 3), dtype=np.float32)
        proba[:, 0] = 1.0  # 100% prob for FLAT
        return proba


class LogisticRegressionBaseline:
    """Tier 1: Linear Statistical Baseline on Microstructure Features.
    
    Validates Cont et al.'s finding on the linear price impact of Order Flow Imbalance.
    """
    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
            class_weight="balanced"
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Features span very different scales (OFI in shares, imbalances in [-1, 1]);
        # standardize with training statistics so the L2 penalty treats them evenly.
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        self.model.fit((X - self.mean_) / self.std_, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict((X - self.mean_) / self.std_)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba((X - self.mean_) / self.std_)
