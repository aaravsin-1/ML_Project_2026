"""Unit tests for Purged Time-Series Cross Validation and Verification Diagnostics."""

import pytest
import numpy as np

from src.data.purged_cv import PurgedTimeSeriesSplit
from src.utils.validation_checks import OverfittingDiagnostics
from src.models.tabular_trees import LightGBMLOB


def test_purged_time_series_split():
    """Validates that PurgedTimeSeriesSplit respects embargo and horizon purging."""
    X = np.arange(1000).reshape(-1, 1)
    horizon_k = 20
    embargo_pct = 0.02  # 20 samples

    cv = PurgedTimeSeriesSplit(n_splits=5, horizon_k=horizon_k, embargo_pct=embargo_pct)
    splits = list(cv.split(X))

    assert len(splits) == 4  # 5 splits yield 4 test iterations

    for train_idx, test_idx in splits:
        test_min = test_idx[0]
        test_max = test_idx[-1]

        # Samples prior to test must end at test_min - horizon_k
        prior_train = train_idx[train_idx < test_min]
        if len(prior_train) > 0:
            assert prior_train[-1] <= test_min - horizon_k

        # Samples post test must start after test_max + embargo
        post_train = train_idx[train_idx > test_max]
        if len(post_train) > 0:
            assert post_train[0] >= test_max + int(1000 * embargo_pct)


def test_permutation_reality_check():
    """Validates that permutation test collapses MCC to ~0.0 when targets are randomized."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((600, 10)).astype(np.float32)
    y_train = rng.choice([0, 1, 2], size=600)
    X_test = rng.standard_normal((300, 10)).astype(np.float32)
    y_test = rng.choice([0, 1, 2], size=300)

    diag = OverfittingDiagnostics.run_permutation_reality_check(
        LightGBMLOB, X_train, y_train, X_test, y_test, n_permutations=2
    )

    assert not diag["leakage_detected"]
    assert abs(diag["permuted_mcc"]) < 0.10
    assert diag["permuted_macro_f1"] < 0.45
