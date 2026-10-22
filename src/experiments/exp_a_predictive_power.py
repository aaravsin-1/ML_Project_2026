"""Experiment A: Establishing the Presence of Predictive Information in the LOB.

Evaluates out-of-sample classification performance across the entire model hierarchy:
- Tier 0: Naïve Momentum, Naïve Flat
- Tier 1: Logistic Regression (Linear baseline validating OFI)
- Tier 2: Random Forest (Non-linear baseline)
- Tier 3: LightGBM (State-of-the-art Tabular GBDT)
- Tier 4: LSTM & GRU (Sequential Temporal Deep Learning)
- Tier 5: DeepLOB & TLOB (State-of-the-art Spatio-Temporal Deep Learning)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

from src.config import CONFIG, ARTIFACTS_DIR
from src.data.synthetic_generator import SyntheticLOBGenerator
from src.data.lob_dataset import temporal_train_val_test_split, build_dataloaders
from src.microstructure.pipeline import MicrostructureFeaturePipeline
from src.models.baselines import NaiveMomentumBaseline, NaiveFlatBaseline, LogisticRegressionBaseline
from src.models.tabular_trees import RandomForestLOB, LightGBMLOB
from src.models.recurrent import LSTMLOBNet, GRULOBNet
from src.models.deeplob import DeepLOB
from src.models.tlob import TLOB
from src.models.registry import DeepLearningTrainer
from src.utils.metrics import compute_classification_metrics
from src.utils.visualization import plot_model_comparison


def run_experiment_a(
    lob_40: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    trades_df: Optional[pd.DataFrame] = None,
    horizon_k: int = 20,
    n_synthetic_events: int = 30000,
    epochs: int = 8,
    save_artifacts: bool = True
) -> pd.DataFrame:
    """Executes Experiment A across all model tiers."""
    print("\n=======================================================")
    print("   EXPERIMENT A: ESTABLISHING PREDICTIVE INFORMATION   ")
    print("=======================================================")

    # 1. Acquire or generate dataset
    if lob_40 is None or labels is None:
        print(f"[+] Generating {n_synthetic_events} high-frequency events in large-tick regime...")
        gen = SyntheticLOBGenerator(base_price=100.0, tick_size=0.01, regime="large_tick", seed=42)
        lob_40, labels_dict, trades_df = gen.generate_session(n_events=n_synthetic_events)
        target_labels = labels_dict[horizon_k]
    else:
        target_labels = labels

    print(f"[+] Total samples: {len(lob_40)} | Horizon k: {horizon_k} events")
    unique, counts = np.unique(target_labels, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"[+] Class distribution: {dist} (0: FLAT, 1: UP, 2: DOWN)")

    # 2. Extract Microstructural Features
    print("[+] Extracting microstructural features (OFI, MLOFI with PCA, VPIN, Micro-Price)...")
    pipeline = MicrostructureFeaturePipeline(num_levels=10, pca_components=3)
    tab_feat, deep_feat, feat_names = pipeline.fit_transform(lob_40, trades_df=trades_df)

    # 3. Chronological Train / Val / Test Partitioning (No temporal leakage)
    splits_tab, splits_y = temporal_train_val_test_split(tab_feat, target_labels, 0.70, 0.15, 0.15)
    splits_deep, _ = temporal_train_val_test_split(deep_feat, target_labels, 0.70, 0.15, 0.15)

    X_train_tab, y_train = splits_tab["train"], splits_y["train"]
    X_val_tab, y_val = splits_tab["val"], splits_y["val"]
    X_test_tab, y_test = splits_tab["test"], splits_y["test"]

    print(f"[+] Split sizes -> Train: {len(X_train_tab)}, Val: {len(X_val_tab)}, Test: {len(X_test_tab)}")

    results = []

    # ---------------------------------------------------------
    # Tier 0: Naïve Baselines
    # ---------------------------------------------------------
    print("\n[+] Training Tier 0: Naïve Baselines...")
    naive_mom = NaiveMomentumBaseline()
    mom_preds = naive_mom.predict(y_test)
    mom_probs = naive_mom.predict_proba(y_test)
    mom_metrics = compute_classification_metrics(y_test, mom_preds, mom_probs)
    results.append({"tier": "Tier 0", "model": "Naïve Momentum", **mom_metrics})

    naive_flat = NaiveFlatBaseline()
    flat_preds = naive_flat.predict(X_test_tab)
    flat_probs = naive_flat.predict_proba(X_test_tab)
    flat_metrics = compute_classification_metrics(y_test, flat_preds, flat_probs)
    results.append({"tier": "Tier 0", "model": "Naïve Flat", **flat_metrics})

    # ---------------------------------------------------------
    # Tier 1: Logistic Regression (Linear Microstructure)
    # ---------------------------------------------------------
    print("[+] Training Tier 1: Logistic Regression (Linear Baseline)...")
    lr = LogisticRegressionBaseline(C=1.0)
    lr.fit(X_train_tab, y_train)
    lr_preds = lr.predict(X_test_tab)
    lr_probs = lr.predict_proba(X_test_tab)
    lr_metrics = compute_classification_metrics(y_test, lr_preds, lr_probs)
    results.append({"tier": "Tier 1", "model": "Logistic Regression", **lr_metrics})

    # ---------------------------------------------------------
    # Tier 2: Random Forest
    # ---------------------------------------------------------
    print("[+] Training Tier 2: Random Forest...")
    rf = RandomForestLOB(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train_tab, y_train)
    rf_preds = rf.predict(X_test_tab)
    rf_probs = rf.predict_proba(X_test_tab)
    rf_metrics = compute_classification_metrics(y_test, rf_preds, rf_probs)
    results.append({"tier": "Tier 2", "model": "Random Forest", **rf_metrics})

    # ---------------------------------------------------------
    # Tier 3: LightGBM
    # ---------------------------------------------------------
    print("[+] Training Tier 3: LightGBM (State-of-the-Art Tabular)...")
    lgb_model = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=200)
    lgb_model.fit(X_train_tab, y_train, X_val_tab, y_val, early_stopping_rounds=15)
    lgb_preds = lgb_model.predict(X_test_tab)
    lgb_probs = lgb_model.predict_proba(X_test_tab)
    lgb_metrics = compute_classification_metrics(y_test, lgb_preds, lgb_probs)
    results.append({"tier": "Tier 3", "model": "LightGBM", **lgb_metrics})

    # ---------------------------------------------------------
    # Deep Learning Sequence Setup (Tiers 4 & 5)
    # ---------------------------------------------------------
    seq_len = 100
    batch_size = 128
    loaders_deep = build_dataloaders(splits_deep, splits_y, seq_len=seq_len, batch_size=batch_size, add_channel_dim=True)
    y_test_seq = splits_y["test"][seq_len - 1:]

    # ---------------------------------------------------------
    # Tier 4: LSTM Sequence Model
    # ---------------------------------------------------------
    print(f"\n[+] Training Tier 4: LSTM (SeqLen={seq_len}, Epochs={epochs})...")
    lstm_net = LSTMLOBNet(num_features=40, hidden_dim=64, num_layers=2)
    trainer_lstm = DeepLearningTrainer(lstm_net, learning_rate=0.001)
    trainer_lstm.fit(loaders_deep["train"], loaders_deep["val"], epochs=epochs, patience=3)
    _, lstm_acc, lstm_f1, lstm_preds, lstm_probs = trainer_lstm.evaluate(loaders_deep["test"])
    lstm_metrics = compute_classification_metrics(y_test_seq, lstm_preds, lstm_probs)
    results.append({"tier": "Tier 4", "model": "LSTM", **lstm_metrics})

    # ---------------------------------------------------------
    # Tier 5: DeepLOB (CNN + Inception + LSTM)
    # ---------------------------------------------------------
    print(f"[+] Training Tier 5: DeepLOB (Inception Modules + Causal Conv + LSTM)...")
    deeplob_net = DeepLOB(num_classes=3, lstm_hidden=64)
    trainer_deeplob = DeepLearningTrainer(deeplob_net, learning_rate=0.0005)
    trainer_deeplob.fit(loaders_deep["train"], loaders_deep["val"], epochs=epochs, patience=3)
    _, dl_acc, dl_f1, dl_preds, dl_probs = trainer_deeplob.evaluate(loaders_deep["test"])
    deeplob_metrics = compute_classification_metrics(y_test_seq, dl_preds, dl_probs)
    results.append({"tier": "Tier 5", "model": "DeepLOB", **deeplob_metrics})

    # ---------------------------------------------------------
    # Tier 5: TLOB (Transformer with Dual Attention & BiN)
    # ---------------------------------------------------------
    print(f"[+] Training Tier 5: TLOB (Dual Spatial/Temporal Attention + BiN)...")
    tlob_net = TLOB(num_features=40, seq_len=seq_len, d_model=64, num_heads=4, num_layers=2)
    trainer_tlob = DeepLearningTrainer(tlob_net, learning_rate=0.0005)
    trainer_tlob.fit(loaders_deep["train"], loaders_deep["val"], epochs=epochs, patience=3)
    _, tlob_acc, tlob_f1, tlob_preds, tlob_probs = trainer_tlob.evaluate(loaders_deep["test"])
    tlob_metrics = compute_classification_metrics(y_test_seq, tlob_preds, tlob_probs)
    results.append({"tier": "Tier 5", "model": "TLOB", **tlob_metrics})

    # Compile Results
    df_results = pd.DataFrame(results)
    print("\n================ EXPERIMENT A FINAL RESULTS ================")
    print(df_results[["tier", "model", "accuracy", "macro_f1", "weighted_f1", "cross_entropy"]].to_string(index=False))
    print("============================================================\n")

    if save_artifacts:
        csv_path = ARTIFACTS_DIR / "experiment_a_results.csv"
        df_results.to_csv(csv_path, index=False)
        plot_path = ARTIFACTS_DIR / "experiment_a_model_comparison.png"
        plot_model_comparison(df_results, plot_path)
        print(f"[+] Saved results to {csv_path} and plot to {plot_path}")

    return df_results


if __name__ == "__main__":
    run_experiment_a()
