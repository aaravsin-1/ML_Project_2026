"""Tier 0 (Naïve Baselines) and Tier 1 (Linear Baselines) for LOB Price Movement."""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier


class NaiveMomentumBaseline:
    """Tier 0: Next price movement equals the previous movement.
    
    Tests whether simple momentum carries directional predictive information.
    """
    def __init__(self):
        self.last_label = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, y_lagged: np.ndarray) -> np.ndarray:
        """Shifts lagged labels by 1 to simulate momentum forecasting."""
        preds = np.roll(y_lagged, 1)
        preds[0] = 0
        return preds

    def predict_proba(self, y_lagged: np.ndarray) -> np.ndarray:
        n = len(y_lagged)
        proba = np.full((n, 3), 1.0 / 3.0, dtype=np.float32)
        preds = self.predict(y_lagged)
        for i in range(n):
            proba[i, :] = 0.1
            proba[i, preds[i]] = 0.8
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
            solver="lbfgs"
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
