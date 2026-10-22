"""Script to generate and headlessly execute the quantitative Jupyter Notebook."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nbformat as nbf
from nbclient import NotebookClient


def create_and_run_notebook(output_path: Path):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12"
        }
    }

    cells = []

    # -------------------------------------------------------------
    # Title & Introduction
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""# High-Frequency Limit Order Book Alpha & Microstructure Prediction
### An End-to-End Research Pipeline: Features, Deep Learning, Purged CV & High-Fidelity Execution

This notebook provides a complete quantitative research pipeline for short-term Limit Order Book (LOB) price trend forecasting:
1. **Market Microstructure Signals**: Cont et al. Level-1 OFI, Multi-Level OFI with PCA, Easley et al. Volume-Clock VPIN, and Stoikov Micro-Price.
2. **Model Progression Hierarchy**: Tier 0 Naïve Baselines, Tier 1 Linear Models, Tier 2 Random Forest, Tier 3 LightGBM, Tier 5 DeepLOB and TLOB.
3. **Statistical Validation**: Purged Time-Series Cross Validation with Embargo and White's Reality Check for zero lookahead leakage.
4. **Economic Reality**: Continuous double-auction execution simulator measuring net realized Sharpe after spread crossing, slippage, and adverse selection.
5. **Submission Generation**: Generates probability predictions and execution actions for out-of-sample events.
"""))

    # -------------------------------------------------------------
    # Imports & Setup
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, log_loss

# Add project root to sys.path
notebook_dir = Path.cwd()
project_root = notebook_dir.parent if notebook_dir.name == "submission" else notebook_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.synthetic_generator import SyntheticLOBGenerator
from src.data.lob_dataset import temporal_train_val_test_split, build_dataloaders
from src.microstructure.pipeline import MicrostructureFeaturePipeline
from src.models.baselines import NaiveFlatBaseline, LogisticRegressionBaseline
from src.models.tabular_trees import RandomForestLOB, LightGBMLOB
from src.models.deeplob import DeepLOB
from src.models.tlob import TLOB
from src.models.registry import DeepLearningTrainer
from src.data.purged_cv import PurgedTimeSeriesSplit
from src.utils.validation_checks import OverfittingDiagnostics
from src.execution.simulator import ExecutionSimulator
from src.execution.strategies import MicrostructureStrategyRunner
from src.execution.metrics import PerformanceMetrics

# Device & seed setup
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)
print(f"[+] Python environment initialized | PyTorch Device: {device} | CUDA Available: {torch.cuda.is_available()}")
"""))

    # -------------------------------------------------------------
    # Data Ingestion
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 1. High-Frequency Market Data Simulation
We simulate 20,000 continuous double-auction events capturing 10 levels of bid/ask prices and depth queues, Poisson/Hawkes arrival dynamics, and aggressive market sweeps.
"""))

    cells.append(nbf.v4.new_code_cell("""# Generate high-frequency LOB session in large-tick asset regime
generator = SyntheticLOBGenerator(base_price=100.0, tick_size=0.01, regime="large_tick", seed=42)
lob_40, labels_dict, trades_df = generator.generate_session(n_events=20000)

target_horizon_k = 20  # Horizon k in events (~200ms)
y_all = labels_dict[target_horizon_k]

mid_prices = (lob_40[:, 0] + lob_40[:, 2]) / 2.0
spreads = lob_40[:, 0] - lob_40[:, 2]

print(f"[+] Generated {len(lob_40):,} snapshots with 10 price levels.")
print(f"[+] Base Mid-Price: ${mid_prices[0]:.2f} -> Final Mid-Price: ${mid_prices[-1]:.2f}")
print(f"[+] Average Spread: ${np.mean(spreads):.4f}")
print(f"[+] Class Distribution (k={target_horizon_k}):", dict(zip(*np.unique(y_all, return_counts=True))))
"""))

    # -------------------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 2. Microstructural Feature Engineering
We transform raw price-volume levels into stationary, microstructurally sound signals:
* **Order Flow Imbalance (OFI & MLOFI)**: Net order book pressure across 10 levels, compressed via PCA.
* **Volume-Synchronized Probability of Informed Trading (VPIN)**: Toxic flow tracking on a volume clock.
* **Stoikov Micro-Price Estimator**: Depth-weighted mid-price adjusting for order book imbalance.
"""))

    cells.append(nbf.v4.new_code_cell("""# Extract microstructural features using pipeline
pipeline = MicrostructureFeaturePipeline(num_levels=10, pca_components=3)
X_tabular, X_deep, feature_names = pipeline.fit_transform(lob_40, trades_df=trades_df)

# Chronological Train / Val / Test Partitioning (70% train / 15% val / 15% test; zero shuffling)
splits_x, splits_y = temporal_train_val_test_split(X_tabular, y_all, 0.70, 0.15, 0.15)
splits_deep, _ = temporal_train_val_test_split(X_deep, y_all, 0.70, 0.15, 0.15)
splits_lob, _ = temporal_train_val_test_split(lob_40, y_all, 0.70, 0.15, 0.15)

X_train, y_train = splits_x["train"], splits_y["train"]
X_val, y_val = splits_x["val"], splits_y["val"]
X_test, y_test = splits_x["test"], splits_y["test"]
test_lob = splits_lob["test"]

print(f"[+] Extracted {len(feature_names)} engineered features: {feature_names[:6]} ...")
print(f"[+] Split Sizes -> Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
"""))

    # -------------------------------------------------------------
    # Model Progression Hierarchy
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 3. Model Progression Hierarchy (Tiers 0 to 5)
We progress from simple statistical baselines to tree ensembles and state-of-the-art deep neural networks:
* **Tier 0**: Naïve Flat baseline (predicts majority class)
* **Tier 1**: Logistic Regression (Linear OFI price impact)
* **Tier 2**: Random Forest (Non-linear depth interactions)
* **Tier 3**: LightGBM (Leaf-wise histogram GBDT)
* **Tier 5**: DeepLOB (Causal Convolutions + Inception + LSTM)
* **Tier 5**: TLOB (Transformer with Dual Attention & Bilinear Normalization)
"""))

    cells.append(nbf.v4.new_code_cell("""results = []

# --- Tier 0: Naive Flat Baseline ---
flat_model = NaiveFlatBaseline()
preds_flat = flat_model.predict(X_test)
results.append({
    "Tier": "Tier 0", "Model": "Naïve Flat",
    "Accuracy": accuracy_score(y_test, preds_flat),
    "Macro F1": f1_score(y_test, preds_flat, average="macro", zero_division=0),
    "MCC": matthews_corrcoef(y_test, preds_flat)
})

# --- Tier 1: Logistic Regression ---
lr_model = LogisticRegressionBaseline(C=1.0)
lr_model.fit(X_train, y_train)
preds_lr = lr_model.predict(X_test)
results.append({
    "Tier": "Tier 1", "Model": "Logistic Regression",
    "Accuracy": accuracy_score(y_test, preds_lr),
    "Macro F1": f1_score(y_test, preds_lr, average="macro", zero_division=0),
    "MCC": matthews_corrcoef(y_test, preds_lr)
})

# --- Tier 2: Random Forest ---
rf_model = RandomForestLOB(n_estimators=100, max_depth=8, random_state=42)
rf_model.fit(X_train, y_train)
preds_rf = rf_model.predict(X_test)
results.append({
    "Tier": "Tier 2", "Model": "Random Forest",
    "Accuracy": accuracy_score(y_test, preds_rf),
    "Macro F1": f1_score(y_test, preds_rf, average="macro", zero_division=0),
    "MCC": matthews_corrcoef(y_test, preds_rf)
})

# --- Tier 3: LightGBM ---
lgb_model = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=200)
lgb_model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=15)
preds_lgb = lgb_model.predict(X_test)
probs_lgb = lgb_model.predict_proba(X_test)
results.append({
    "Tier": "Tier 3", "Model": "LightGBM",
    "Accuracy": accuracy_score(y_test, preds_lgb),
    "Macro F1": f1_score(y_test, preds_lgb, average="macro", zero_division=0),
    "MCC": matthews_corrcoef(y_test, preds_lgb)
})

# --- Tier 5: DeepLOB & TLOB (PyTorch Sequential) ---
seq_len = 100
loaders_deep = build_dataloaders(splits_deep, splits_y, seq_len=seq_len, batch_size=128, add_channel_dim=True)
y_test_seq = splits_y["test"][seq_len - 1:]

deeplob = DeepLOB(num_classes=3, lstm_hidden=64)
trainer_dl = DeepLearningTrainer(deeplob, learning_rate=0.0005)
trainer_dl.fit(loaders_deep["train"], loaders_deep["val"], epochs=4, patience=2)
_, _, _, preds_dl, _ = trainer_dl.evaluate(loaders_deep["test"])
results.append({
    "Tier": "Tier 5", "Model": "DeepLOB (Inception-LSTM)",
    "Accuracy": accuracy_score(y_test_seq, preds_dl),
    "Macro F1": f1_score(y_test_seq, preds_dl, average="macro", zero_division=0),
    "MCC": matthews_corrcoef(y_test_seq, preds_dl)
})

tlob = TLOB(num_features=40, seq_len=seq_len, d_model=64, num_heads=4, num_layers=2)
trainer_tlob = DeepLearningTrainer(tlob, learning_rate=0.0005)
trainer_tlob.fit(loaders_deep["train"], loaders_deep["val"], epochs=4, patience=2)
_, _, _, preds_tlob, _ = trainer_tlob.evaluate(loaders_deep["test"])
results.append({
    "Tier": "Tier 5", "Model": "TLOB (Dual Attention)",
    "Accuracy": accuracy_score(y_test_seq, preds_tlob),
    "Macro F1": f1_score(y_test_seq, preds_tlob, average="macro", zero_division=0),
    "MCC": matthews_corrcoef(y_test_seq, preds_tlob)
})

df_comparison = pd.DataFrame(results)
display(df_comparison)
"""))

    # -------------------------------------------------------------
    # Validation & Leakage Audit
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 4. Quantitative Validation: Overfitting & Zero Lookahead Audit
We verify that our model does not suffer from data leakage or serial correlation through two institutional checks:
1. **Purged Time-Series Cross Validation with Embargo**: Eliminates overlapping future label leakage.
2. **Target Permutation Reality Check (White's Test)**: Shuffling targets must collapse edge to zero.
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Purged Cross-Validation with Embargo (4 sequential folds)
df_cv = OverfittingDiagnostics.run_purged_cv_evaluation(
    LightGBMLOB, X_tabular, y_all, n_splits=5, horizon_k=20, embargo_pct=0.01
)
print("=== Purged Time-Series Cross-Validation Results ===")
display(df_cv)
print(f"Mean Purged Macro-F1: {df_cv['test_macro_f1'].mean():.4f} (Std: {df_cv['test_macro_f1'].std():.4f})")
print(f"Mean Purged MCC:      {df_cv['test_mcc'].mean():.4f}")

# 2. Target Permutation Test (Zero Leakage Proof)
perm_diag = OverfittingDiagnostics.run_permutation_reality_check(
    LightGBMLOB, X_train, y_train, X_test, y_test, n_permutations=3
)
print("\\n=== Permutation Reality Check (Lookahead Leakage Audit) ===")
print(f"Permuted Targets Macro-F1: {perm_diag['permuted_macro_f1']:.4f} (Expected ~0.333)")
print(f"Permuted Targets MCC:      {perm_diag['permuted_mcc']:.4f} (Expected ~0.000)")
print("Leakage Status:            ", "PASS (Zero Lookahead Leakage)" if not perm_diag["leakage_detected"] else "FAIL (Leakage Detected)")
"""))

    # -------------------------------------------------------------
    # Execution Simulator
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 5. Economic Reality: High-Fidelity Execution Simulation
We test whether our machine learning predictions translate into positive net PnL when passed through a double-auction simulator modeling:
* **Spread Crossing**: Aggressive orders buy at Ask, sell at Bid.
* **Walking the Book**: Slippage from consuming multi-level depth.
* **Exchange Fees**: Maker rebates (-0.5 bps) vs Taker fees (1.5 bps).
* **VPIN Adverse Selection Defense**: Dynamic quote cancellation during toxic informed flow.
"""))

    cells.append(nbf.v4.new_code_cell("""sim = ExecutionSimulator(
    initial_cash=1000000.0,
    maker_fee_bps=-0.5,
    taker_fee_bps=1.5,
    latency_ticks=1,
    lot_size=100.0
)
runner = MicrostructureStrategyRunner(sim)
test_mid_prices = (test_lob[:, 0] + test_lob[:, 2]) / 2.0
test_vpin = X_test[:, feature_names.index("vpin")]

# Mode A: Academic Fantasy (Mid-price, 0 fees, 0 slippage)
res_acad = runner.run_academic_midprice(preds_lgb, test_mid_prices)
perf_acad = PerformanceMetrics.calculate_metrics(res_acad["equity"])

# Mode B: Realistic Aggressive Taker (Spread + Slippage + Taker Fees)
res_taker = runner.run_realistic_aggressive(preds_lgb, test_lob)
perf_taker = PerformanceMetrics.calculate_metrics(res_taker["equity"])

# Mode C: Dynamic Passive Maker with VPIN Defense
res_maker = runner.run_dynamic_passive_vpin(preds_lgb, test_lob, test_vpin, toxic_vpin_thresh=0.65)
perf_maker = PerformanceMetrics.calculate_metrics(res_maker["equity"])

strat_df = pd.DataFrame([
    {"Strategy": "Academic Mid-Price (Fantasy)", "Net PnL ($)": perf_acad["net_pnl"], "Sharpe": perf_acad["sharpe_ratio"], "Fees ($)": 0.0, "Slippage ($)": 0.0},
    {"Strategy": "Realistic Aggressive Taker", "Net PnL ($)": perf_taker["net_pnl"], "Sharpe": perf_taker["sharpe_ratio"], "Fees ($)": res_taker["total_fees"], "Slippage ($)": res_taker["total_slippage"]},
    {"Strategy": "Dynamic Passive Maker (VPIN)", "Net PnL ($)": perf_maker["net_pnl"], "Sharpe": perf_maker["sharpe_ratio"], "Fees ($)": res_maker["total_fees"], "Slippage ($)": 0.0}
])
display(strat_df)
"""))

    # -------------------------------------------------------------
    # Submission Generation
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 6. Competition Submission File Generation
We generate formatted out-of-sample test predictions including predicted directional trend probabilities, discrete classes, and tactical execution actions.
"""))

    cells.append(nbf.v4.new_code_cell("""# Generate competition-grade submission dataframe
submission_df = pd.DataFrame({
    "event_id": np.arange(len(preds_lgb)),
    "prob_flat": probs_lgb[:, 0],
    "prob_up": probs_lgb[:, 1],
    "prob_down": probs_lgb[:, 2],
    "predicted_label": preds_lgb,
    "vpin_toxicity": test_vpin,
})

# Map recommended trading action
actions = []
for idx, row in submission_df.iterrows():
    if row["vpin_toxicity"] > 0.65:
        actions.append("CANCEL_VPIN_TOXIC")
    elif row["predicted_label"] == 1:
        actions.append("POST_PASSIVE_BID")
    elif row["predicted_label"] == 2:
        actions.append("POST_PASSIVE_ASK")
    else:
        actions.append("HOLD_FLAT")

submission_df["recommended_action"] = actions

# Save submission CSV
sub_path = Path("submission.csv") if Path.cwd().name == "submission" else Path("submission/submission.csv")
submission_df.to_csv(sub_path, index=False)
print(f"[+] Saved submission predictions ({len(submission_df):,} records) to: {sub_path}")
display(submission_df.head(10))
"""))

    nb.cells = cells

    # Save unexecuted notebook
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[+] Notebook created at: {output_path}")

    # Execute notebook headlessly
    print(f"[+] Executing notebook headlessly to populate all outputs and charts...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()

    # Save executed notebook with all outputs
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[+] Notebook successfully executed and updated with all cell outputs: {output_path}")


if __name__ == "__main__":
    out_file = ROOT / "submission" / "LOB_Alpha_Prediction_Pipeline.ipynb"
    create_and_run_notebook(out_file)
