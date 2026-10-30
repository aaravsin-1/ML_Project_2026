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
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }

    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# Limit Order Book Alpha & Execution: End-to-End Pipeline (Single Seed)

A walkthrough of the research pipeline on **one synthetic session** (seed 42). The data is
simulated; the generator embeds a hidden informed-trader state, so predictability here
reflects that construction, not a real market. Multi-seed results are in the main README.

1. **Signals**: Level-1 OFI, multi-level OFI with PCA, volume-clock VPIN, micro-price, depth imbalances.
2. **Models**: honest momentum baseline, logistic regression, random forest, LightGBM, DeepLOB, TLOB,
   all with balanced class weights and scored on the same purged test rows.
3. **Validation**: purged CV with embargo, target-permutation sanity check.
4. **Execution**: fantasy mid-price fills vs aggressive taker vs passive maker with trade-through fills.
5. **Submission**: probabilities and actions for the test window.
"""))

    cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

notebook_dir = Path.cwd()
project_root = notebook_dir.parent if notebook_dir.name == "submission" else notebook_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.experiments.common import generate_session, prepare_data, SEQ_LEN
from src.data.lob_dataset import build_indexed_dataloaders
from src.models.baselines import NaiveMomentumBaseline, NaiveFlatBaseline, LogisticRegressionBaseline
from src.models.tabular_trees import RandomForestLOB, LightGBMLOB
from src.models.deeplob import DeepLOB
from src.models.tlob import TLOB
from src.models.registry import DeepLearningTrainer, balanced_class_weights
from src.utils.validation_checks import OverfittingDiagnostics
from src.execution.simulator import ExecutionSimulator
from src.execution.strategies import MicrostructureStrategyRunner
from src.execution.metrics import PerformanceMetrics

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"[+] PyTorch device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Market Data Simulation
60,000 events of a large-tick book (1-tick spread, 10 levels, ~10 ms between events).
"""))

    cells.append(nbf.v4.new_code_cell("""session = generate_session(SEED, n_events=60000)
k = 20
y_all = session.labels_dict[k]
lob_40 = session.lob_40
mid_prices = (lob_40[:, 0] + lob_40[:, 2]) / 2.0

print(f"[+] {len(lob_40):,} snapshots over {session.timestamps_ms[-1] / 1000:.0f} s of simulated time")
print(f"[+] Mid-price ${mid_prices[0]:.2f} -> ${mid_prices[-1]:.2f}")
print(f"[+] Class distribution (k={k}):", dict(zip(*np.unique(y_all, return_counts=True))))
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Features and Purged Splits
The pipeline (PCA, normalizer) is fit on the training period only; every feature is causal.
Splits are 70/15/15 chronological with a purge gap of k rows between them.
"""))

    cells.append(nbf.v4.new_code_cell("""data = prepare_data(session)
idx = data.splits(k)
X = data.tab_feat
X_train, y_train = X[idx["train"]], y_all[idx["train"]]
X_val, y_val = X[idx["val"]], y_all[idx["val"]]
X_test, y_test = X[idx["test"]], y_all[idx["test"]]
print(f"[+] {len(data.feat_names)} features: {data.feat_names[:6]} ...")
print(f"[+] Train {len(X_train):,} | Val {len(X_val):,} | Test {len(X_test):,}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 3. Model Hierarchy
The momentum baseline repeats the last *realized* k-event move (label at t-k), the most
recent label actually known at decision time.
"""))

    cells.append(nbf.v4.new_code_cell("""results = []
def record(tier, name, preds):
    results.append({"Tier": tier, "Model": name,
                    "Accuracy": accuracy_score(y_test, preds),
                    "Macro F1": f1_score(y_test, preds, average="macro", zero_division=0),
                    "MCC": matthews_corrcoef(y_test, preds)})

record("Tier 0", "Naive Momentum (lag k)", NaiveMomentumBaseline(k).predict(y_all, idx["test"]))
record("Tier 0", "Naive Flat", NaiveFlatBaseline().predict(X_test))

lr_model = LogisticRegressionBaseline(C=1.0); lr_model.fit(X_train, y_train)
record("Tier 1", "Logistic Regression", lr_model.predict(X_test))

rf_model = RandomForestLOB(n_estimators=100, max_depth=10, random_state=SEED); rf_model.fit(X_train, y_train)
record("Tier 2", "Random Forest", rf_model.predict(X_test))

lgb_model = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=200, random_state=SEED)
lgb_model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=15)
preds_lgb = lgb_model.predict(X_test)
probs_lgb = lgb_model.predict_proba(X_test)
record("Tier 3", "LightGBM", preds_lgb)

loaders = build_indexed_dataloaders(data.deep_feat, y_all, idx, seq_len=SEQ_LEN, batch_size=128)
class_w = balanced_class_weights(y_train)
for name, net in [("DeepLOB", DeepLOB(num_classes=3, lstm_hidden=64)),
                  ("TLOB", TLOB(num_features=40, seq_len=SEQ_LEN, d_model=64, num_heads=4, num_layers=2))]:
    trainer = DeepLearningTrainer(net, learning_rate=0.0005, class_weights=class_w)
    trainer.fit(loaders["train"], loaders["val"], epochs=8, patience=3)
    _, _, _, preds, _ = trainer.evaluate(loaders["test"])
    record("Tier 5", name, preds)

display(pd.DataFrame(results))
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4. Validation
* **Purged CV with embargo**: fold-to-fold stability of out-of-sample skill.
* **Target permutation**: training on shuffled targets must give no edge. This catches
  label/row misalignment bugs; it cannot prove the absence of lookahead in features.
"""))

    cells.append(nbf.v4.new_code_cell("""df_cv = OverfittingDiagnostics.run_purged_cv_evaluation(LightGBMLOB, X, y_all, n_splits=5, horizon_k=k, embargo_pct=0.01)
display(df_cv)
print(f"Purged CV Macro-F1: {df_cv['test_macro_f1'].mean():.4f} +/- {df_cv['test_macro_f1'].std():.4f}")

perm = OverfittingDiagnostics.run_permutation_reality_check(LightGBMLOB, X_train, y_train, X_test, y_test, n_permutations=3)
print(f"Permuted-target Macro-F1 {perm['permuted_macro_f1']:.4f}, MCC {perm['permuted_mcc']:.4f}",
      "-> FAIL" if perm["leakage_detected"] else "-> PASS")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 5. Execution Simulation (LightGBM signals, 1 ms latency)
* **Academic**: fills at mid, no fees (upper bound, not achievable).
* **Aggressive taker**: crosses the spread, walks the book, pays 1.5 bps.
* **Passive maker**: joins the touch, fills only when the market trades through the order
  (so fills are adversely selected), earns a 0.5 bps rebate, stands down when VPIN exceeds
  its 90th training percentile.
"""))

    cells.append(nbf.v4.new_code_cell("""test_idx = idx["test"]
test_lob = lob_40[test_idx]
test_ts = session.timestamps_ms[test_idx]
vpin = X[:, data.feat_names.index("vpin")]
vpin_thresh = float(np.quantile(vpin[:data.train_end], 0.90))
test_vpin = vpin[test_idx]

sim = ExecutionSimulator(initial_cash=1000000.0, maker_fee_bps=-0.5, taker_fee_bps=1.5,
                         lot_size=100.0, latency_ms=1.0, timestamps_ms=test_ts)
runner = MicrostructureStrategyRunner(sim)
runs = {
    "Academic Mid-Price (Fantasy)": runner.run_academic_midprice(preds_lgb, (test_lob[:, 0] + test_lob[:, 2]) / 2.0),
    "Aggressive Taker": runner.run_realistic_aggressive(preds_lgb, test_lob),
    "Passive Maker (VPIN gate)": runner.run_dynamic_passive_vpin(preds_lgb, test_lob, test_vpin, vpin_thresh),
}
rows = []
for name, res in runs.items():
    perf = PerformanceMetrics.calculate_metrics(res["equity"], 1000000.0, timestamps_ms=test_ts, bar_ms=100.0)
    rows.append({"Strategy": name, "Net PnL ($)": perf["net_pnl"], "Trades": res["num_trades"],
                 "Fees ($)": res.get("total_fees", 0.0), "Slippage ($)": res.get("total_slippage", 0.0)})
print(f"Test window: {(test_ts[-1] - test_ts[0]) / 1000:.0f} s of simulated trading")
display(pd.DataFrame(rows))
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 6. Submission File
Test-window probabilities, predicted labels, VPIN, and the action the passive maker would take.
"""))

    cells.append(nbf.v4.new_code_cell("""submission_df = pd.DataFrame({
    "event_id": test_idx,
    "prob_flat": probs_lgb[:, 0],
    "prob_up": probs_lgb[:, 1],
    "prob_down": probs_lgb[:, 2],
    "predicted_label": preds_lgb,
    "vpin_toxicity": test_vpin,
})
submission_df["recommended_action"] = np.select(
    [submission_df["vpin_toxicity"] > vpin_thresh, submission_df["predicted_label"] == 1, submission_df["predicted_label"] == 2],
    ["CANCEL_VPIN_TOXIC", "POST_PASSIVE_BID", "POST_PASSIVE_ASK"],
    default="HOLD_FLAT",
)
sub_path = Path("submission.csv") if Path.cwd().name == "submission" else Path("submission/submission.csv")
submission_df.to_csv(sub_path, index=False)
print(f"[+] Saved {len(submission_df):,} rows to {sub_path} (VPIN toxic threshold = {vpin_thresh:.3f})")
display(submission_df.head(10))
"""))

    nb.cells = cells

    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[+] Notebook created at: {output_path}")

    print("[+] Executing notebook headlessly...")
    client = NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()

    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[+] Notebook executed and saved: {output_path}")


if __name__ == "__main__":
    out_file = ROOT / "submission" / "LOB_Alpha_Prediction_Pipeline.ipynb"
    create_and_run_notebook(out_file)
