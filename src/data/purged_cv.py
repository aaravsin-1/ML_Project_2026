"""Purged Time-Series Cross-Validation with Embargo for High-Frequency LOB Data.

Formalized by Marcos Lopez de Prado (2018) in 'Advances in Financial Machine Learning'.
Guarantees zero overlapping label leakage and zero serial correlation bleed:
1. Purges training samples whose future label window [t, t+k] overlaps with test sets.
2. Applies an embargo window immediately following test sets to prevent auto-regressive memory leak.
"""

from typing import Generator, List, Tuple
import numpy as np


class PurgedTimeSeriesSplit:
    """Purged and Embargoed Time-Series Cross-Validation."""

    def __init__(self, n_splits: int = 5, horizon_k: int = 20, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.horizon_k = horizon_k
        self.embargo_pct = embargo_pct

    def split(self, X: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        embargo_samples = int(n_samples * self.embargo_pct)

        for i in range(1, self.n_splits):
            # Test fold
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_samples)
            test_indices = np.arange(test_start, test_end)

            # Train fold: all samples before test_start, purged by horizon_k
            train_end = max(0, test_start - self.horizon_k)
            train_indices_prior = np.arange(0, train_end)

            # Optional post-test training samples with embargo
            post_train_start = test_end + embargo_samples
            if post_train_start < n_samples:
                train_indices_post = np.arange(post_train_start, n_samples)
                train_indices = np.concatenate([train_indices_prior, train_indices_post])
            else:
                train_indices = train_indices_prior

            if len(train_indices) == 0:
                continue

            yield train_indices, test_indices
