"""Comprehensive Overfitting Diagnostics, Purged CV, and Leakage Verification Suite.

Designed to detect common high-frequency quantitative failure modes:
1. Generalization Gap (Train vs Val vs Test over-fitting)
2. Matthews Correlation Coefficient (MCC) & Cohen's Kappa for severe class imbalance
3. Target Permutation Check (model edge must collapse to 0 on shuffled targets)
4. Purged & embargoed time-series cross validation
5. Regime Generalization (Large-tick trained model tested on Small-tick regime)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, cohen_kappa_score, log_loss

from src.models.tabular_trees import LightGBMLOB
from src.data.purged_cv import PurgedTimeSeriesSplit


class OverfittingDiagnostics:
    """Rigorous statistical validation tools for high-frequency quantitative models."""

    @staticmethod
    def evaluate_generalization_gap(
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Calculates performance across Train, Validation, and Test splits.
        
        Monitors overfitting via delta gaps: (Train - Test).
        """
        preds_tr = model.predict(X_train)
        preds_va = model.predict(X_val)
        preds_te = model.predict(X_test)

        acc_tr = accuracy_score(y_train, preds_tr)
        acc_va = accuracy_score(y_val, preds_va)
        acc_te = accuracy_score(y_test, preds_te)

        f1_tr = f1_score(y_train, preds_tr, average="macro", zero_division=0)
        f1_va = f1_score(y_val, preds_va, average="macro", zero_division=0)
        f1_te = f1_score(y_test, preds_te, average="macro", zero_division=0)

        mcc_te = matthews_corrcoef(y_test, preds_te)
        kappa_te = cohen_kappa_score(y_test, preds_te)

        return {
            "train_accuracy": float(acc_tr),
            "val_accuracy": float(acc_va),
            "test_accuracy": float(acc_te),
            "train_macro_f1": float(f1_tr),
            "val_macro_f1": float(f1_va),
            "test_macro_f1": float(f1_te),
            "accuracy_gap": float(acc_tr - acc_te),
            "f1_gap": float(f1_tr - f1_te),
            "test_mcc": float(mcc_te),
            "test_kappa": float(kappa_te),
            "is_overfitted": bool((f1_tr - f1_te) > 0.25)
        }

    @staticmethod
    def run_permutation_reality_check(
        model_class,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_permutations: int = 5
    ) -> Dict[str, float]:
        """Target Permutation Check.

        Trains on shuffled training targets and scores on the real test targets. Any
        edge that survives points to a bug that ties features to labels by row position
        or to a test set that can be predicted without learning. It does NOT detect
        features that look at future prices: shuffling breaks that link as well, so a
        pass here is not evidence of zero lookahead leakage.
        """
        rng = np.random.default_rng(999)
        perm_f1s = []
        perm_mccs = []

        for _ in range(n_permutations):
            y_train_shuffled = rng.permutation(y_train)
            model = model_class(n_estimators=100, max_depth=6, num_leaves=31)
            model.fit(X_train, y_train_shuffled)
            preds = model.predict(X_test)

            perm_f1s.append(f1_score(y_test, preds, average="macro", zero_division=0))
            perm_mccs.append(matthews_corrcoef(y_test, preds))

        mean_f1 = float(np.mean(perm_f1s))
        mean_mcc = float(np.mean(perm_mccs))

        return {
            "permuted_macro_f1": mean_f1,
            "permuted_mcc": mean_mcc,
            "leakage_detected": bool(abs(mean_mcc) > 0.10 or mean_f1 > 0.45)
        }

    @staticmethod
    def run_purged_cv_evaluation(
        model_class,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        horizon_k: int = 20,
        embargo_pct: float = 0.01
    ) -> pd.DataFrame:
        """Executes Purged Time-Series Cross Validation with Embargo."""
        cv = PurgedTimeSeriesSplit(n_splits=n_splits, horizon_k=horizon_k, embargo_pct=embargo_pct)
        fold_records = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            model = model_class(n_estimators=120, max_depth=6, num_leaves=31)
            model.fit(X_tr, y_tr)

            preds = model.predict(X_te)
            acc = accuracy_score(y_te, preds)
            f1 = f1_score(y_te, preds, average="macro", zero_division=0)
            mcc = matthews_corrcoef(y_te, preds)

            fold_records.append({
                "fold": fold_idx + 1,
                "train_samples": len(train_idx),
                "test_samples": len(test_idx),
                "test_accuracy": acc,
                "test_macro_f1": f1,
                "test_mcc": mcc
            })

        return pd.DataFrame(fold_records)
