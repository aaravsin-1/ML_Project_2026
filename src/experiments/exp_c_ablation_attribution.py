"""Experiment C: Ablation and Feature Attribution (Information Isolation).

Progresses through strictly restricted feature sets to determine the exact source of alpha:
- Model A: Best Bid/Ask Price only (Mid-price & Spread)
- Model B: Best Bid/Ask Price + Top-of-book Volumes (Level 1 Depth & OFI)
- Model C: Top 3 Levels of Depth & Imbalances
- Model D: Full 10-Level Order Book Snapshot + MLOFI PCA + VPIN
- Model E: Full 10-Level Order Book + Temporal Sequence History (DeepLOB / TLOB)

Performs feature importance and SHAP analysis to isolate the percentage contribution
of Level 1 OFI vs Multi-Level Depth vs Flow Toxicity.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import shap

from src.config import ARTIFACTS_DIR
from src.data.synthetic_generator import SyntheticLOBGenerator
from src.data.lob_dataset import temporal_train_val_test_split, build_dataloaders
from src.microstructure.pipeline import MicrostructureFeaturePipeline
from src.models.tabular_trees import LightGBMLOB
from src.models.deeplob import DeepLOB
from src.models.registry import DeepLearningTrainer
from src.utils.metrics import compute_classification_metrics
from src.utils.visualization import plot_ablation_progression


def run_experiment_c(
    lob_40: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    trades_df: Optional[pd.DataFrame] = None,
    horizon_k: int = 20,
    n_synthetic_events: int = 30000,
    save_artifacts: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Executes Experiment C: Ablation Study & Feature Importance."""
    print("\n=======================================================")
    print("   EXPERIMENT C: ABLATION AND FEATURE ATTRIBUTION       ")
    print("=======================================================")

    if lob_40 is None or labels is None:
        print(f"[+] Generating {n_synthetic_events} events for ablation study...")
        gen = SyntheticLOBGenerator(base_price=100.0, tick_size=0.01, regime="large_tick", seed=42)
        lob_40, labels_dict, trades_df = gen.generate_session(n_events=n_synthetic_events)
        target_labels = labels_dict[horizon_k]
    else:
        target_labels = labels

    pipeline = MicrostructureFeaturePipeline(num_levels=10, pca_components=3)
    tab_feat, deep_feat, feat_names = pipeline.fit_transform(lob_40, trades_df=trades_df)

    # Define feature subsets for ablation models
    # Model A: Spread only
    idx_a = [feat_names.index("rel_spread")]

    # Model B: Spread + Level 1 Imbalance + Level 1 OFI
    idx_b = [
        feat_names.index("rel_spread"),
        feat_names.index("imbalance_l1"),
        feat_names.index("ofi_l1"),
        feat_names.index("micro_deviation")
    ]

    # Model C: Model B + Top 3 depth imbalances and slopes
    idx_c = idx_b + [
        feat_names.index("imbalance_l3"),
        feat_names.index("slope_bid"),
        feat_names.index("slope_ask")
    ]

    # Model D: Full 10 levels, MLOFI PCA, VPIN, multi-level depth
    idx_d = list(range(len(feat_names)))

    ablation_configs = [
        ("Model A: Spread Only", idx_a),
        ("Model B: Top-of-Book (L1)", idx_b),
        ("Model C: Top 3 Levels Depth", idx_c),
        ("Model D: Full 10-Level Book", idx_d),
    ]

    results = []

    # Chronological splits
    splits_full_x, splits_y = temporal_train_val_test_split(tab_feat, target_labels, 0.70, 0.15, 0.15)
    y_train, y_val, y_test = splits_y["train"], splits_y["val"], splits_y["test"]

    trained_model_d = None

    for name, indices in ablation_configs:
        print(f"[+] Training {name} ({len(indices)} features)...")
        X_tr = splits_full_x["train"][:, indices]
        X_v = splits_full_x["val"][:, indices]
        X_te = splits_full_x["test"][:, indices]

        model = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=150)
        model.fit(X_tr, y_train, X_v, y_val, early_stopping_rounds=10)

        preds = model.predict(X_te)
        probs = model.predict_proba(X_te)
        metrics = compute_classification_metrics(y_test, preds, probs)

        print(f"    {name} -> Macro-F1 = {metrics['macro_f1']:.4f} | Acc = {metrics['accuracy']:.4f}")
        results.append({"model_tier": name, "num_features": len(indices), **metrics})

        if name == "Model D: Full 10-Level Book":
            trained_model_d = model

    # ---------------------------------------------------------
    # Model E: Full 10-Level Book + Temporal Sequence (DeepLOB)
    # ---------------------------------------------------------
    print("[+] Training Model E: Full 10-Level Book + Temporal Sequence (DeepLOB)...")
    seq_len = 100
    splits_deep, _ = temporal_train_val_test_split(deep_feat, target_labels, 0.70, 0.15, 0.15)
    loaders_deep = build_dataloaders(splits_deep, splits_y, seq_len=seq_len, batch_size=128, add_channel_dim=True)
    y_test_seq = splits_y["test"][seq_len - 1:]

    deeplob_net = DeepLOB(num_classes=3, lstm_hidden=64)
    trainer = DeepLearningTrainer(deeplob_net, learning_rate=0.0005)
    trainer.fit(loaders_deep["train"], loaders_deep["val"], epochs=6, patience=2)
    _, _, _, dl_preds, dl_probs = trainer.evaluate(loaders_deep["test"])
    metrics_e = compute_classification_metrics(y_test_seq, dl_preds, dl_probs)
    print(f"    Model E (DeepLOB) -> Macro-F1 = {metrics_e['macro_f1']:.4f} | Acc = {metrics_e['accuracy']:.4f}")
    results.append({"model_tier": "Model E: DeepLOB (Temporal)", "num_features": 40 * seq_len, **metrics_e})

    df_ablation = pd.DataFrame(results)

    # ---------------------------------------------------------
    # Feature Attribution (LightGBM Importance & Gini Gain)
    # ---------------------------------------------------------
    print("\n[+] Computing Feature Attribution on Model D...")
    importances = trained_model_d.feature_importances_
    total_imp = np.sum(importances)
    norm_imp = importances / max(1e-8, total_imp)

    df_importance = pd.DataFrame({
        "feature": feat_names,
        "importance": importances,
        "relative_contribution_pct": norm_imp * 100.0
    }).sort_values(by="importance", ascending=False)

    print("\nTop 8 Most Influential Microstructure Features:")
    print(df_importance.head(8).to_string(index=False))

    if save_artifacts:
        csv_path = ARTIFACTS_DIR / "experiment_c_ablation_results.csv"
        df_ablation.to_csv(csv_path, index=False)
        imp_path = ARTIFACTS_DIR / "experiment_c_feature_importance.csv"
        df_importance.to_csv(imp_path, index=False)
        plot_path = ARTIFACTS_DIR / "experiment_c_ablation_progression.png"
        plot_ablation_progression(df_ablation, plot_path)
        print(f"\n[+] Saved ablation results to {csv_path} and plot to {plot_path}")

    return df_ablation, df_importance


if __name__ == "__main__":
    run_experiment_c()
