"""Experiment B: The Horizon of Signal Survival (Alpha Decay Mapping).

Evaluates prediction quality across prediction horizons k in [10, 20, 50, 100] events.
Each horizon gets its own purged split (the purge gap equals k), and LightGBM is
reported next to the honest momentum baseline so the lift over a trivial rule is visible.
"""

from typing import List
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR
from src.experiments.common import PreparedData, load_prepared
from src.models.baselines import NaiveMomentumBaseline
from src.models.tabular_trees import LightGBMLOB
from src.utils.metrics import compute_classification_metrics
from src.utils.visualization import plot_signal_decay


def run_experiment_b(
    data: PreparedData,
    horizons: List[int] = [10, 20, 50, 100],
    save_artifacts: bool = True
) -> pd.DataFrame:
    """Executes Experiment B to generate the alpha decay curve."""
    print("\n=======================================================")
    print("   EXPERIMENT B: THE HORIZON OF SIGNAL SURVIVAL        ")
    print("=======================================================")
    seed = data.session.seed
    X = data.tab_feat
    records = []

    for k in horizons:
        y_all = data.labels(k)
        idx = data.splits(k)
        y_test = y_all[idx["test"]]

        model = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=150, random_state=seed)
        model.fit(X[idx["train"]], y_all[idx["train"]], X[idx["val"]], y_all[idx["val"]], early_stopping_rounds=10)
        lgb_metrics = compute_classification_metrics(y_test, model.predict(X[idx["test"]]), model.predict_proba(X[idx["test"]]))

        mom = NaiveMomentumBaseline(horizon_k=k)
        mom_metrics = compute_classification_metrics(y_test, mom.predict(y_all, idx["test"]))

        print(f"    k={k:<4} LightGBM Macro-F1 {lgb_metrics['macro_f1']:.4f} | Momentum Macro-F1 {mom_metrics['macro_f1']:.4f}")
        records.append({
            "seed": seed,
            "horizon_k": k,
            **lgb_metrics,
            "momentum_macro_f1": mom_metrics["macro_f1"],
            "test_frac_flat": float(np.mean(y_test == 0)),
        })

    df_decay = pd.DataFrame(records)

    if save_artifacts:
        df_decay.to_csv(ARTIFACTS_DIR / "experiment_b_signal_decay.csv", index=False)
        plot_signal_decay(horizons, df_decay["macro_f1"].tolist(), ARTIFACTS_DIR / "experiment_b_signal_decay.png")

    return df_decay


if __name__ == "__main__":
    run_experiment_b(load_prepared())
