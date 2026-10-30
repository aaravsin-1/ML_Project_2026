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


HISTORY_LAGS = (1, 5, 10, 20, 50, 100)


def history_features(lob: np.ndarray) -> np.ndarray:
    """Changes of the 40 raw LOB columns over the last 1..100 rows (strictly past).

    DeepLOB sees a 100-row window; this gives the tabular models comparable history so the
    comparison is about architecture, not about which model was allowed to see the past.
    """
    out = []
    for lag in HISTORY_LAGS:
        d = np.zeros_like(lob)
        d[lag:] = lob[lag:] - lob[:-lag]
        out.append(d)
    return np.concatenate(out, axis=1).astype(np.float32)


def run_fi2010_benchmark(horizons: List[int], epochs: int, seeds: List[int] = (0,)) -> pd.DataFrame:
    train_lob, train_extra, train_y, test_lob, test_extra, test_y = load_setup2()
    n_tr = len(train_lob)
    print(f"[+] FI-2010 Setup 2: {n_tr:,} training rows (days 1-7), {len(test_lob):,} test rows (days 8-10)")

    # One continuous array so sequence windows for the first test rows can use the end of
    # training as context, exactly like the tabular rows (which need no context).
    all_lob = np.concatenate([train_lob, test_lob])
    all_extra = np.concatenate([train_extra, test_extra])
    tab = np.concatenate([all_lob, all_extra], axis=1)  # 144 features
    tab_hist = np.concatenate([tab, history_features(all_lob)], axis=1)  # + 240 history features

    out_dir = ARTIFACTS_DIR / "fi2010"
    out_dir.mkdir(exist_ok=True)
    all_rows = []
    for seed in seeds:
        for k in horizons:
            y_all = np.concatenate([train_y[k], test_y[k]])
            val_start = int(n_tr * 0.8)
            idx = {
                "train": np.arange(SEQ_LEN - 1, val_start - k),
                "val": np.arange(val_start, n_tr - k),
                "test": np.arange(n_tr, len(y_all)),
            }
            y_test = y_all[idx["test"]]

            def add(rows, name: str, preds: np.ndarray):
                s = scores(y_test, preds)
                rows.append({"seed": seed, "horizon_k": k, "model": name, **s})
                print(f"    seed {seed} k={k} {name:<32} acc {s['accuracy']:.4f} | weighted-F1 {s['weighted_f1']:.4f} | macro-F1 {s['macro_f1']:.4f}")

            # Checkpoint 1: original model set (seed 0 keeps its original file name)
            suffix = "" if seed == 0 else f"_s{seed}"
            ckpt = out_dir / f"setup2_k{k}_e{epochs}{suffix}.csv"
            if ckpt.exists():
                print(f"[+] seed {seed} k={k}: loading checkpoint {ckpt.name}")
                prev = pd.read_csv(ckpt)
                prev["seed"] = seed
                all_rows.extend(prev.to_dict("records"))
            else:
                torch.manual_seed(seed)
                np.random.seed(seed)
                rows = []
                print(f"\n[+] seed {seed} k={k}: test class shares FLAT/UP/DOWN = {np.round(np.bincount(y_test, minlength=3) / len(y_test), 3)}")
                add(rows, "Naive Flat", np.zeros(len(y_test), dtype=np.int64))
                add(rows, "Naive Momentum (lag k)", NaiveMomentumBaseline(k).predict(y_all, idx["test"]))

                lr = LogisticRegressionBaseline(C=1.0)
                lr.model.set_params(class_weight=None)
                lr.fit(tab[idx["train"]], y_all[idx["train"]])
                add(rows, "Logistic Regression", lr.predict(tab[idx["test"]]))

                lgb_model = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=500,
                                        random_state=seed, class_weight=None)
                lgb_model.fit(tab[idx["train"]], y_all[idx["train"]], tab[idx["val"]], y_all[idx["val"]], early_stopping_rounds=30)
                add(rows, "LightGBM (144 feats)", lgb_model.predict(tab[idx["test"]]))

                loaders = build_indexed_dataloaders(all_lob, y_all, idx, seq_len=SEQ_LEN, batch_size=128)
                trainer = DeepLearningTrainer(DeepLOB(num_classes=3, lstm_hidden=64), learning_rate=0.0005)
                trainer.fit(loaders["train"], loaders["val"], epochs=epochs, patience=3)
                _, _, _, dl_preds, _ = trainer.evaluate(loaders["test"])
                add(rows, "DeepLOB (ours)", dl_preds)

                if k in PUBLISHED_DEEPLOB:
                    acc, wf1 = PUBLISHED_DEEPLOB[k]
                    rows.append({"seed": seed, "horizon_k": k, "model": "DeepLOB (published)", "accuracy": acc / 100,
                                 "weighted_f1": wf1 / 100, "macro_f1": np.nan})
                pd.DataFrame(rows).to_csv(ckpt, index=False)
                all_rows.extend(rows)

            # Checkpoint 2: tabular models given the same 100-row history DeepLOB sees
            hckpt = out_dir / f"setup2_k{k}_history_s{seed}.csv"
            if hckpt.exists():
                print(f"[+] seed {seed} k={k}: loading checkpoint {hckpt.name}")
                all_rows.extend(pd.read_csv(hckpt).to_dict("records"))
                continue
            np.random.seed(seed)
            rows = []
            lr = LogisticRegressionBaseline(C=1.0)
            lr.model.set_params(class_weight=None)
            lr.fit(tab_hist[idx["train"]], y_all[idx["train"]])
            add(rows, "Logistic Regression + history", lr.predict(tab_hist[idx["test"]]))
            lgb_model = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=500,
                                    random_state=seed, class_weight=None)
            lgb_model.fit(tab_hist[idx["train"]], y_all[idx["train"]], tab_hist[idx["val"]], y_all[idx["val"]],
                          early_stopping_rounds=30)
            add(rows, "LightGBM (144 feats + history)", lgb_model.predict(tab_hist[idx["test"]]))
            pd.DataFrame(rows).to_csv(hckpt, index=False)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "fi2010_setup2_results.csv", index=False)
    summary = (df.groupby(["horizon_k", "model"], sort=False)[["accuracy", "weighted_f1", "macro_f1"]]
                 .agg(["mean", "std", "count"]).round(4))
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(out_dir / "fi2010_setup2_summary.csv", index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 50, 100])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    out = run_fi2010_benchmark(args.horizons, args.epochs, args.seeds)
    print(out.to_string(index=False))
