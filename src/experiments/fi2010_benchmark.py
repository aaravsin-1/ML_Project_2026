"""FI-2010 benchmark (Setup 2) to check our models against published results.

Data: FI-2010 "NoAuction, Z-score" files (Ntakaris et al., 2018), 5 Nasdaq Nordic
stocks over 10 days, one row per 10 order book events. Setup 2 trains on days 1-7
(Train_Dst_NoAuction_ZScore_CF_7) and tests on days 8-10 (Test_..._CF_7/8/9), the same
files the DeepLOB authors' public code uses. The last 20% of the training days (minus a
k-row purge) is held out for early stopping.

To match the benchmark convention, every model is trained without class weights, and
the paper's headline "F1" is the support-weighted F1 (its recall equals accuracy).
Macro-F1 is reported as well.
"""

import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.config import ARTIFACTS_DIR, DATA_DIR
from src.data.fi2010_loader import FI2010Loader
from src.data.lob_dataset import build_indexed_dataloaders
from src.models.baselines import NaiveMomentumBaseline, LogisticRegressionBaseline
from src.models.tabular_trees import LightGBMLOB
from src.models.deeplob import DeepLOB
from src.models.registry import DeepLearningTrainer

FI_DIR = DATA_DIR / "raw" / "fi2010" / "BenchmarkDatasets" / "NoAuction" / "1.NoAuction_Zscore"
SEQ_LEN = 100

# Zhang, Zohren & Roberts (2019), Table II (Setup 2): accuracy %, weighted F1 %
PUBLISHED_DEEPLOB = {10: (84.47, 83.40), 20: (74.85, 72.82), 50: (80.51, 80.35)}


def load_setup2():
    train_lob, train_extra, train_y = FI2010Loader.parse_file_with_extras(
        FI_DIR / "NoAuction_Zscore_Training" / "Train_Dst_NoAuction_ZScore_CF_7.txt")
    parts = [FI2010Loader.parse_file_with_extras(
        FI_DIR / "NoAuction_Zscore_Testing" / f"Test_Dst_NoAuction_ZScore_CF_{i}.txt") for i in (7, 8, 9)]
    test_lob = np.concatenate([p[0] for p in parts])
    test_extra = np.concatenate([p[1] for p in parts])
    test_y = {k: np.concatenate([p[2][k] for p in parts]) for k in train_y}
    return train_lob, train_extra, train_y, test_lob, test_extra, test_y


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def run_fi2010_benchmark(horizons: List[int], epochs: int, seed: int = 0) -> pd.DataFrame:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_lob, train_extra, train_y, test_lob, test_extra, test_y = load_setup2()
    n_tr = len(train_lob)
    print(f"[+] FI-2010 Setup 2: {n_tr:,} training rows (days 1-7), {len(test_lob):,} test rows (days 8-10)")

    # One continuous array so sequence windows for the first test rows can use the end of
    # training as context, exactly like the tabular rows (which need no context).
    all_lob = np.concatenate([train_lob, test_lob])
    all_extra = np.concatenate([train_extra, test_extra])
    tab = np.concatenate([all_lob, all_extra], axis=1)  # 144 features

    out_dir = ARTIFACTS_DIR / "fi2010"
    out_dir.mkdir(exist_ok=True)
    all_rows = []
    for k in horizons:
        # Per-horizon checkpoint so an interrupted run resumes at the next horizon
        ckpt = out_dir / f"setup2_k{k}_e{epochs}.csv"
        if ckpt.exists():
            print(f"[+] k={k}: loading checkpoint {ckpt.name}")
            all_rows.extend(pd.read_csv(ckpt).to_dict("records"))
            continue
        rows = []
        y_all = np.concatenate([train_y[k], test_y[k]])
        val_start = int(n_tr * 0.8)
        idx = {
            "train": np.arange(SEQ_LEN - 1, val_start - k),
            "val": np.arange(val_start, n_tr - k),
            "test": np.arange(n_tr, len(y_all)),
        }
        y_test = y_all[idx["test"]]
        dist = np.bincount(y_test, minlength=3) / len(y_test)
        print(f"\n[+] k={k}: test class shares FLAT/UP/DOWN = {np.round(dist, 3)}")

        def add(name: str, preds: np.ndarray):
            s = scores(y_test, preds)
            rows.append({"horizon_k": k, "model": name, **s})
            print(f"    {name:<22} acc {s['accuracy']:.4f} | weighted-F1 {s['weighted_f1']:.4f} | macro-F1 {s['macro_f1']:.4f}")

        add("Naive Flat", np.zeros(len(y_test), dtype=np.int64))
        add("Naive Momentum (lag k)", NaiveMomentumBaseline(k).predict(y_all, idx["test"]))

        lr = LogisticRegressionBaseline(C=1.0)
        lr.model.set_params(class_weight=None)
        lr.fit(tab[idx["train"]], y_all[idx["train"]])
        add("Logistic Regression", lr.predict(tab[idx["test"]]))

        lgb_model = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=500,
                                random_state=seed, class_weight=None)
        lgb_model.fit(tab[idx["train"]], y_all[idx["train"]], tab[idx["val"]], y_all[idx["val"]], early_stopping_rounds=30)
        add("LightGBM (144 feats)", lgb_model.predict(tab[idx["test"]]))

        loaders = build_indexed_dataloaders(all_lob, y_all, idx, seq_len=SEQ_LEN, batch_size=128)
        trainer = DeepLearningTrainer(DeepLOB(num_classes=3, lstm_hidden=64), learning_rate=0.0005)
        trainer.fit(loaders["train"], loaders["val"], epochs=epochs, patience=3)
        _, _, _, dl_preds, _ = trainer.evaluate(loaders["test"])
        add("DeepLOB (ours)", dl_preds)

        if k in PUBLISHED_DEEPLOB:
            acc, wf1 = PUBLISHED_DEEPLOB[k]
            rows.append({"horizon_k": k, "model": "DeepLOB (published)", "accuracy": acc / 100,
                         "weighted_f1": wf1 / 100, "macro_f1": np.nan})
        pd.DataFrame(rows).to_csv(ckpt, index=False)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "fi2010_setup2_results.csv", index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 50, 100])
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    out = run_fi2010_benchmark(args.horizons, args.epochs)
    print(out.to_string(index=False))
