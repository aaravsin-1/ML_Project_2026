"""Experiment C: Ablation and Feature Attribution (Information Isolation).

Nested feature sets, each adding one kind of information, all trained with LightGBM:
- Model A: Spread (in ticks) only
- Model B: Level-1 snapshot (spread, L1 imbalance, L1 OFI, micro-price deviation)
- Model C: Model B + lagged Level-1 signals (short history)
- Model D: Model C + Levels 2-3 (L3 depth imbalance, level-2/3 OFI)
- Model E: Every feature (adds levels 4-10 depth, book slopes, MLOFI PCA, VPIN)

Attribution uses LightGBM's total split gain on the Model E fit.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR
from src.experiments.common import PreparedData, load_prepared
from src.models.tabular_trees import LightGBMLOB
from src.utils.metrics import compute_classification_metrics
from src.utils.visualization import plot_ablation_progression

L1_SNAPSHOT = ["spread_ticks", "imbalance_l1", "ofi_l1", "micro_deviation"]
L1_LAGS = [f"{name}_lag_{lag}" for lag in (1, 2, 5) for name in ("ofi", "imbalance", "micro_dev")]
LEVELS_2_3 = ["imbalance_l3", "mlofi_lvl_2", "mlofi_lvl_3"]


def ablation_feature_sets(feat_names: List[str]) -> List[Tuple[str, List[str]]]:
    return [
        ("Model A: Spread Only", ["spread_ticks"]),
        ("Model B: L1 Snapshot", L1_SNAPSHOT),
        ("Model C: L1 + Lags", L1_SNAPSHOT + L1_LAGS),
        ("Model D: L1-L3 + Lags", L1_SNAPSHOT + L1_LAGS + LEVELS_2_3),
        ("Model E: All Features", list(feat_names)),
    ]


def run_experiment_c(
    data: PreparedData,
    horizon_k: int = 20,
    save_artifacts: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Executes Experiment C: Ablation Study & Feature Importance."""
    print("\n=======================================================")
    print("   EXPERIMENT C: ABLATION AND FEATURE ATTRIBUTION       ")
    print("=======================================================")
    seed = data.session.seed
    names = data.feat_names
    y_all = data.labels(horizon_k)
    idx = data.splits(horizon_k)
    y_test = y_all[idx["test"]]

    results = []
    full_model = None
    for label, cols in ablation_feature_sets(names):
        col_idx = [names.index(c) for c in cols]
        X = data.tab_feat[:, col_idx]
        model = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=150, random_state=seed)
        model.fit(X[idx["train"]], y_all[idx["train"]], X[idx["val"]], y_all[idx["val"]], early_stopping_rounds=10)
        metrics = compute_classification_metrics(y_test, model.predict(X[idx["test"]]), model.predict_proba(X[idx["test"]]))
        print(f"    {label:<24} ({len(cols):>2} features) Macro-F1 {metrics['macro_f1']:.4f}")
        results.append({"seed": seed, "model_tier": label, "num_features": len(cols), **metrics})
        full_model = model

    df_ablation = pd.DataFrame(results)

    gains = full_model.feature_importances_.astype(np.float64)
    df_importance = pd.DataFrame({
        "seed": seed,
        "feature": names,
        "gain": gains,
        "gain_share_pct": 100.0 * gains / max(gains.sum(), 1e-12),
    }).sort_values("gain", ascending=False)

    if save_artifacts:
        df_ablation.to_csv(ARTIFACTS_DIR / "experiment_c_ablation_results.csv", index=False)
        df_importance.to_csv(ARTIFACTS_DIR / "experiment_c_feature_importance.csv", index=False)
        plot_ablation_progression(df_ablation, ARTIFACTS_DIR / "experiment_c_ablation_progression.png")

    return df_ablation, df_importance


if __name__ == "__main__":
    run_experiment_c(load_prepared())
