"""Experiment B: The Horizon of Signal Survival (Alpha Decay Mapping).

Evaluates prediction quality across varying prediction horizons:
horizons in [10, 20, 50, 100] events (or time domains 100ms to 30s).
Constructs the empirical decay curve showing the half-life of microstructure alpha.
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR
from src.data.synthetic_generator import SyntheticLOBGenerator
from src.data.lob_dataset import temporal_train_val_test_split
from src.microstructure.pipeline import MicrostructureFeaturePipeline
from src.models.tabular_trees import LightGBMLOB
from src.utils.metrics import compute_classification_metrics
from src.utils.visualization import plot_signal_decay


def run_experiment_b(
    lob_40: Optional[np.ndarray] = None,
    labels_dict: Optional[Dict[int, np.ndarray]] = None,
    trades_df: Optional[pd.DataFrame] = None,
    horizons: List[int] = [10, 20, 50, 100],
    n_synthetic_events: int = 30000,
    save_artifacts: bool = True
) -> pd.DataFrame:
    """Executes Experiment B to generate the alpha decay curve."""
    print("\n=======================================================")
    print("   EXPERIMENT B: THE HORIZON OF SIGNAL SURVIVAL        ")
    print("=======================================================")

    if lob_40 is None or labels_dict is None:
        print(f"[+] Generating {n_synthetic_events} events for multi-horizon decay study...")
        gen = SyntheticLOBGenerator(base_price=100.0, tick_size=0.01, regime="large_tick", seed=42)
        lob_40, labels_dict, trades_df = gen.generate_session(n_events=n_synthetic_events)

    print(f"[+] Extracting features across {len(lob_40)} events...")
    pipeline = MicrostructureFeaturePipeline(num_levels=10, pca_components=3)
    tab_feat, _, _ = pipeline.fit_transform(lob_40, trades_df=trades_df)

    decay_records = []
    f1_list = []

    for k in horizons:
        print(f"\n[+] Evaluating Horizon k = {k} events...")
        target_y = labels_dict[k]
        
        splits_x, splits_y = temporal_train_val_test_split(tab_feat, target_y, 0.70, 0.15, 0.15)
        X_train, y_train = splits_x["train"], splits_y["train"]
        X_val, y_val = splits_x["val"], splits_y["val"]
        X_test, y_test = splits_x["test"], splits_y["test"]

        # Train optimized LightGBM
        model = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=150)
        model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=10)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        metrics = compute_classification_metrics(y_test, preds, probs)

        print(f"    Horizon k={k}: Macro-F1 = {metrics['macro_f1']:.4f} | Accuracy = {metrics['accuracy']:.4f}")
        decay_records.append({"horizon_k": k, **metrics})
        f1_list.append(metrics["macro_f1"])

    df_decay = pd.DataFrame(decay_records)
    print("\n================ EXPERIMENT B FINAL RESULTS ================")
    print(df_decay[["horizon_k", "macro_f1", "accuracy", "cross_entropy"]].to_string(index=False))
    print("============================================================\n")

    if save_artifacts:
        csv_path = ARTIFACTS_DIR / "experiment_b_signal_decay.csv"
        df_decay.to_csv(csv_path, index=False)
        plot_path = ARTIFACTS_DIR / "experiment_b_signal_decay.png"
        plot_signal_decay(horizons, f1_list, plot_path)
        print(f"[+] Saved decay results to {csv_path} and plot to {plot_path}")

    return df_decay


if __name__ == "__main__":
    run_experiment_b()
