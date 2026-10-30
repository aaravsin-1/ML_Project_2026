"""Directional labels for real LOB data with a train-only threshold.

Label at row t compares the trailing-mean mid-price at t+k with the one at t:
UP if the return exceeds +alpha_k, DOWN if below -alpha_k, FLAT otherwise. On real data
the size of k-event moves differs by orders of magnitude across horizons, so alpha_k is
set per horizon as a quantile of |return| over training rows only (rows whose label
window ends inside the training period).
"""

from typing import Dict, List
import numpy as np
import pandas as pd


def smoothed_forward_returns(mid: np.ndarray, k: int, smoothing: int = 5) -> np.ndarray:
    """(s[t+k] - s[t]) / s[t] with s the trailing `smoothing`-row mean; NaN for the last k rows."""
    s = pd.Series(mid).rolling(smoothing, min_periods=1).mean().to_numpy()
    ret = np.full(len(mid), np.nan)
    ret[:-k] = (s[k:] - s[:-k]) / s[:-k]
    return ret


def calibrate_alpha(mid: np.ndarray, k: int, fit_end: int, flat_share: float = 1 / 3, smoothing: int = 5) -> float:
    """Threshold putting `flat_share` of training labels in FLAT."""
    ret = smoothed_forward_returns(mid[:fit_end], k, smoothing)
    return float(np.nanquantile(np.abs(ret), flat_share))


def direction_labels(
    mid: np.ndarray,
    horizons: List[int],
    fit_end: int,
    flat_share: float = 1 / 3,
    smoothing: int = 5
) -> Dict[int, np.ndarray]:
    """{k: labels} with 0 = FLAT, 1 = UP, 2 = DOWN; the last k rows (undefined) are FLAT."""
    labels = {}
    for k in horizons:
        alpha = calibrate_alpha(mid, k, fit_end, flat_share, smoothing)
        ret = smoothed_forward_returns(mid, k, smoothing)
        y = np.zeros(len(mid), dtype=np.int64)
        y[ret > alpha] = 1
        y[ret < -alpha] = 2
        labels[k] = y
    return labels
