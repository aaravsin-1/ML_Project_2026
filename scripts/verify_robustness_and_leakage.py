"""Execution Script for Overfitting Diagnostics, Purged CV, and Leakage Verification."""

import sys
from pathlib import Path

# Ensure root directory is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from src.config import ARTIFACTS_DIR
from src.experiments.common import generate_session, prepare_data
from src.models.baselines import NaiveMomentumBaseline
from src.models.tabular_trees import LightGBMLOB
from src.utils.validation_checks import OverfittingDiagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--events", type=int, default=30000)
    args = parser.parse_args()
    k = 20

    print("================================================================================")
    print("     QUANTITATIVE VALIDATION SUITE: OVERFITTING, PURGED CV & LEAKAGE AUDIT      ")
    print("================================================================================")

    print(f"\n[+] 1. Generating large-tick session (seed {args.seed}, {args.events} events)...")
    data = prepare_data(generate_session(args.seed, args.events))
    y_large = data.labels(k)
    X_large = data.tab_feat
    idx = data.splits(k)
    X_tr, y_tr = X_large[idx["train"]], y_large[idx["train"]]
    X_va, y_va = X_large[idx["val"]], y_large[idx["val"]]
    X_te, y_te = X_large[idx["test"]], y_large[idx["test"]]
    pipeline = data.pipeline

    # 2. Train Model and Audit Generalization Gap
    print("\n[+] 2. Auditing Generalization Gap (Train vs Validation vs Test)...")
    model = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=200)
    model.fit(X_tr, y_tr, X_va, y_va, early_stopping_rounds=15)

    gap_diag = OverfittingDiagnostics.evaluate_generalization_gap(
        model, X_tr, y_tr, X_va, y_va, X_te, y_te
    )
    print("    Train Accuracy:   {:.2f}% | Train Macro-F1: {:.4f}".format(gap_diag["train_accuracy"]*100, gap_diag["train_macro_f1"]))
    print("    Val Accuracy:     {:.2f}% | Val Macro-F1:   {:.4f}".format(gap_diag["val_accuracy"]*100, gap_diag["val_macro_f1"]))
    print("    Test Accuracy:    {:.2f}% | Test Macro-F1:  {:.4f}".format(gap_diag["test_accuracy"]*100, gap_diag["test_macro_f1"]))
    print("    Accuracy Gap:     {:.2f}% | F1 Gap:         {:.4f}".format(gap_diag["accuracy_gap"]*100, gap_diag["f1_gap"]))
    print("    Test MCC:         {:.4f}  | Test Kappa:     {:.4f}".format(gap_diag["test_mcc"], gap_diag["test_kappa"]))
    print("    Overfitting Flag: {}".format("FAIL (Overfitted)" if gap_diag["is_overfitted"] else "PASS (Robust)"))

    # 3. Purged Time-Series Cross Validation with Embargo
    print("\n[+] 3. Running Purged Time-Series Cross Validation (5 Folds with Embargo Buffer)...")
    df_purged_cv = OverfittingDiagnostics.run_purged_cv_evaluation(
        LightGBMLOB, X_large, y_large, n_splits=5, horizon_k=20, embargo_pct=0.01
    )
    print(df_purged_cv.to_string(index=False))
    print("    Purged CV Mean Macro-F1: {:.4f} (Std: {:.4f})".format(
        df_purged_cv["test_macro_f1"].mean(), df_purged_cv["test_macro_f1"].std()
    ))
    print("    Purged CV Mean MCC:      {:.4f} (Std: {:.4f})".format(
        df_purged_cv["test_mcc"].mean(), df_purged_cv["test_mcc"].std()
    ))

    # 4. Target Permutation Check. This only catches label/feature misalignment bugs (e.g. labels leaking into features
    # by row position); it cannot detect features that peek at future prices, because
    # shuffling the targets breaks that link too.
    print("\n[+] 4. Executing Target Permutation Check (label-alignment sanity check)...")
    perm_diag = OverfittingDiagnostics.run_permutation_reality_check(
        LightGBMLOB, X_tr, y_tr, X_te, y_te, n_permutations=5
    )
    print("    Permuted Targets Macro-F1: {:.4f} (Expected: ~0.333)".format(perm_diag["permuted_macro_f1"]))
    print("    Permuted Targets MCC:      {:.4f} (Expected: ~0.000)".format(perm_diag["permuted_mcc"]))
    print("    Status: {}".format("FAIL" if perm_diag["leakage_detected"] else "PASS (no label-alignment leak)"))

    # 4b. Honest-baseline comparison: lift over the last realized k-event move
    mom_f1 = f1_score(y_te, NaiveMomentumBaseline(k).predict(y_large, idx["test"]), average="macro", zero_division=0)
    print("    Naïve momentum (lag k) Macro-F1: {:.4f} vs LightGBM {:.4f}".format(mom_f1, gap_diag["test_macro_f1"]))

    # 5. Out-of-Distribution Cross-Regime Generalization
    print("\n[+] 5. Cross-Regime Generalization Test (Trained on Large-Tick -> Evaluated on Small-Tick)...")
    small = generate_session(777, 15000, regime="small_tick")
    valid = np.arange(len(small.lob_40) - k)  # last k labels are undefined
    y_small = small.labels_dict[k][valid]
    X_small, _ = pipeline.transform(small.lob_40, small.trades_df, small.timestamps_ms)
    X_small = X_small[valid]

    preds_small = model.predict(X_small)
    acc_cross = accuracy_score(y_small, preds_small)
    f1_cross = f1_score(y_small, preds_small, average="macro", zero_division=0)
    mcc_cross = matthews_corrcoef(y_small, preds_small)

    print("    Small-Tick Regime Out-of-Sample Accuracy: {:.2f}%".format(acc_cross * 100))
    print("    Small-Tick Regime Out-of-Sample Macro-F1: {:.4f}".format(f1_cross))
    print("    Small-Tick Regime Out-of-Sample MCC:      {:.4f}".format(mcc_cross))

    # 6. Save Report: status is computed from the values, not hard-coded
    cv_f1 = df_purged_cv["test_macro_f1"]
    cv_mcc = df_purged_cv["test_mcc"]
    checks = [
        ("Generalization Gap (F1 Train - Test)", gap_diag["f1_gap"], gap_diag["f1_gap"] < 0.25, "< 0.25"),
        ("Test Matthews Correlation Coef (MCC)", gap_diag["test_mcc"], gap_diag["test_mcc"] > 0.0, "> 0"),
        ("Test Cohen's Kappa", gap_diag["test_kappa"], gap_diag["test_kappa"] > 0.0, "> 0"),
        ("Test Macro-F1 lift over naive momentum (lag k)", gap_diag["test_macro_f1"] - mom_f1, gap_diag["test_macro_f1"] > mom_f1, "> 0"),
        ("Purged CV Mean Macro-F1", float(cv_f1.mean()), float(cv_f1.std()) < 0.05, "fold std < 0.05"),
        ("Purged CV Min MCC", float(cv_mcc.min()), float(cv_mcc.min()) > 0.0, "every fold > 0"),
        ("Permutation Check Macro-F1", perm_diag["permuted_macro_f1"], perm_diag["permuted_macro_f1"] <= 0.45, "<= 0.45"),
        ("Permutation Check MCC", perm_diag["permuted_mcc"], abs(perm_diag["permuted_mcc"]) <= 0.10, "|MCC| <= 0.10"),
        ("Cross-Regime Small-Tick Transfer Macro-F1", float(f1_cross), float(mcc_cross) > 0.0, "MCC > 0"),
        ("Cross-Regime Small-Tick Transfer MCC", float(mcc_cross), float(mcc_cross) > 0.0, "> 0"),
    ]
    summary_report = {
        "check": [c[0] for c in checks],
        "value": [c[1] for c in checks],
        "criterion": [c[3] for c in checks],
        "status": ["PASS" if c[2] else "FAIL" for c in checks],
    }
    df_summary = pd.DataFrame(summary_report)
    csv_out = ARTIFACTS_DIR / "overfitting_verification_report.csv"
    df_summary.to_csv(csv_out, index=False)
    print("\n================ VERIFICATION SUMMARY ================")
    print(df_summary.to_string(index=False))
    print("======================================================\n")
    print(f"[+] Saved validation report to: {csv_out}")


if __name__ == "__main__":
    main()
