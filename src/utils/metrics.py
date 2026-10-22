"""Classification and Regression Evaluation Utilities."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, precision_recall_fscore_support


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Computes full suite of classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    prec, rec, f1s, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "f1_flat": float(f1s[0]) if len(f1s) > 0 else 0.0,
        "f1_up": float(f1s[1]) if len(f1s) > 1 else 0.0,
        "f1_down": float(f1s[2]) if len(f1s) > 2 else 0.0,
    }

    if y_prob is not None:
        try:
            # Handle potential probability normalization
            probs = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
            probs = probs / probs.sum(axis=-1, keepdims=True)
            ce_loss = log_loss(y_true, probs, labels=[0, 1, 2])
            metrics["cross_entropy"] = float(ce_loss)
        except Exception:
            metrics["cross_entropy"] = 0.0

    return metrics
