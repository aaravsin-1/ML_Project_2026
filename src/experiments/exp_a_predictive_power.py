"""Experiment A: Establishing the Presence of Predictive Information in the LOB.

Evaluates out-of-sample classification performance across the entire model hierarchy:
- Tier 0: Naïve Momentum (last realized k-event move), Naïve Flat
- Tier 1: Logistic Regression (Linear baseline validating OFI)
- Tier 2: Random Forest (Non-linear baseline)
- Tier 3: LightGBM (State-of-the-art Tabular GBDT)
- Tier 4: LSTM (Sequential Temporal Deep Learning)
- Tier 5: DeepLOB & TLOB (State-of-the-art Spatio-Temporal Deep Learning)

All learned models use "balanced" class weights and are scored on the same purged test rows.
"""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import torch

from src.config import ARTIFACTS_DIR
from src.data.lob_dataset import build_indexed_dataloaders
from src.experiments.common import PreparedData, SEQ_LEN, load_prepared
from src.models.baselines import NaiveMomentumBaseline, NaiveFlatBaseline, LogisticRegressionBaseline
from src.models.tabular_trees import RandomForestLOB, LightGBMLOB
from src.models.recurrent import LSTMLOBNet
from src.models.deeplob import DeepLOB
from src.models.tlob import TLOB
from src.models.registry import DeepLearningTrainer, balanced_class_weights
from src.utils.metrics import compute_classification_metrics
from src.utils.visualization import plot_model_comparison


def run_experiment_a(
    data: PreparedData,
    horizon_k: int = 20,
    epochs: int = 8,
    save_artifacts: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Executes Experiment A across all model tiers.

    Returns the metrics table and the fitted LightGBM / DeepLOB models with their test
    predictions, which Experiment D reuses so execution is driven by the real models.
    """
    print("\n=======================================================")
    print("   EXPERIMENT A: ESTABLISHING PREDICTIVE INFORMATION   ")
    print("=======================================================")
    seed = data.session.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    y_all = data.labels(horizon_k)
    idx = data.splits(horizon_k)
    X = data.tab_feat
    X_train, y_train = X[idx["train"]], y_all[idx["train"]]
    X_val, y_val = X[idx["val"]], y_all[idx["val"]]
    X_test, y_test = X[idx["test"]], y_all[idx["test"]]

    print(f"[+] Seed {seed} | Horizon k={horizon_k} | Train {len(idx['train'])}, Val {len(idx['val'])}, Test {len(idx['test'])}")
    print(f"[+] Test class distribution (FLAT/UP/DOWN): {np.bincount(y_test, minlength=3) / len(y_test)}")

    results = []

    def record(tier: str, name: str, preds: np.ndarray, probs: np.ndarray):
        metrics = compute_classification_metrics(y_test, preds, probs)
        results.append({"tier": tier, "model": name, **metrics})
        print(f"    {name:<20} Macro-F1 {metrics['macro_f1']:.4f} | Acc {metrics['accuracy']:.4f}")

    # Tier 0
    mom = NaiveMomentumBaseline(horizon_k=horizon_k)
    record("Tier 0", "Naïve Momentum", mom.predict(y_all, idx["test"]), mom.predict_proba(y_all, idx["test"]))
    flat = NaiveFlatBaseline()
    record("Tier 0", "Naïve Flat", flat.predict(X_test), flat.predict_proba(X_test))

    # Tier 1
    lr = LogisticRegressionBaseline(C=1.0)
    lr.fit(X_train, y_train)
    record("Tier 1", "Logistic Regression", lr.predict(X_test), lr.predict_proba(X_test))

    # Tier 2
    rf = RandomForestLOB(n_estimators=100, max_depth=10, random_state=seed)
    rf.fit(X_train, y_train)
    record("Tier 2", "Random Forest", rf.predict(X_test), rf.predict_proba(X_test))

    # Tier 3
    lgb_model = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=200, random_state=seed)
    lgb_model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=15)
    lgb_preds = lgb_model.predict(X_test)
    record("Tier 3", "LightGBM", lgb_preds, lgb_model.predict_proba(X_test))

    # Tiers 4 & 5: sequence models on the same rows
    loaders = build_indexed_dataloaders(data.deep_feat, y_all, idx, seq_len=SEQ_LEN, batch_size=128)
    class_w = balanced_class_weights(y_train)

    def train_seq(name: str, tier: str, net: torch.nn.Module, lr_rate: float) -> Tuple[DeepLearningTrainer, np.ndarray]:
        print(f"[+] Training {name} ({epochs} epochs max)...")
        trainer = DeepLearningTrainer(net, learning_rate=lr_rate, class_weights=class_w)
        trainer.fit(loaders["train"], loaders["val"], epochs=epochs, patience=3)
        _, _, _, preds, probs = trainer.evaluate(loaders["test"])
        record(tier, name, preds, probs)
        return trainer, preds

    train_seq("LSTM", "Tier 4", LSTMLOBNet(num_features=40, hidden_dim=64, num_layers=2), 0.001)
    deeplob_trainer, deeplob_preds = train_seq("DeepLOB", "Tier 5", DeepLOB(num_classes=3, lstm_hidden=64), 0.0005)
    train_seq("TLOB", "Tier 5", TLOB(num_features=40, seq_len=SEQ_LEN, d_model=64, num_heads=4, num_layers=2), 0.0005)

    df_results = pd.DataFrame(results)
    df_results.insert(0, "seed", seed)

    if save_artifacts:
        df_results.to_csv(ARTIFACTS_DIR / "experiment_a_results.csv", index=False)
        plot_model_comparison(df_results, ARTIFACTS_DIR / "experiment_a_model_comparison.png")

    outputs = {
        "test_idx": idx["test"],
        "lgb_model": lgb_model,
        "lgb_preds": lgb_preds,
        "deeplob_trainer": deeplob_trainer,
        "deeplob_preds": deeplob_preds,
    }
    return df_results, outputs


if __name__ == "__main__":
    run_experiment_a(load_prepared())
