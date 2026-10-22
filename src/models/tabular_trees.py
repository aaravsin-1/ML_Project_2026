"""Tier 2 (Random Forest) and Tier 3 (LightGBM) Tabular Microstructure Models."""

import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


class RandomForestLOB:
    """Tier 2: Random Forest Non-Linear Baseline.
    
    Captures non-linear thresholds and interactions between depth depletion and spread widening.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 12,
        min_samples_leaf: int = 10,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight="balanced_subsample"
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_


class LightGBMLOB:
    """Tier 3: State-of-the-art Tabular GBDT Baseline.
    
    Histogram-based splits and leaf-wise tree growth optimized for microsecond-scale execution.
    """
    def __init__(
        self,
        num_leaves: int = 63,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        n_estimators: int = 250,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight="balanced",
            verbosity=-1
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 15
    ):
        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)] if eval_set else None
        
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_
