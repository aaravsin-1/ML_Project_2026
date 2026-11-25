"""Extra classification and speed metrics for the report (BTC, tabular models).

Retrains Experiment A/J's tabular models on each BTC segment with the same settings
and keeps their class probabilities, so that probability-based metrics (ROC-AUC,
log-loss, Brier score, calibration, hit rate vs confidence) and timing can be
reported. Nothing in artifacts/ is changed; outputs go to report/data/.

Run from the repository root:  python report/scripts/btc_extra_metrics.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             cohen_kappa_score, f1_score, log_loss, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from src.experiments.common import btc_sessions, prepare_data
from src.models.baselines import LogisticRegressionBaseline, NaiveMomentumBaseline
from src.models.tabular_trees import LightGBMLOB, RandomForestLOB

OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(parents=True, exist_ok=True)
HORIZONS = [10, 20, 50]
CLASSES = ["FLAT", "UP", "DOWN"]


def latency_us(fn, X, n_calls=2000):
    """Median and P99 single-row latency in microseconds (Python call included)."""
    rows = X[np.random.default_rng(0).integers(0, len(X), n_calls)]
    for r in rows[:50]:
        fn(r[None, :])
    t = np.empty(n_calls)
    for i, r in enumerate(rows):
        t0 = time.perf_counter_ns()
        fn(r[None, :])
        t[i] = (time.perf_counter_ns() - t0) / 1e3
    return float(np.median(t)), float(np.percentile(t, 99))


def metrics(y, proba, pred):
    onehot = np.eye(3)[y]
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "weighted_f1": f1_score(y, pred, average="weighted"),
        "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y, pred, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "log_loss": log_loss(y, np.clip(proba, 1e-7, 1), labels=[0, 1, 2]),
        "brier": float(np.mean(np.sum((proba - onehot) ** 2, axis=1))),
        "roc_auc_macro_ovr": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
    }
    for c, name in enumerate(CLASSES):
        out[f"auc_{name}"] = roc_auc_score(y == c, proba[:, c])
    # UP vs DOWN only, on rows that actually moved: can the model tell the direction?
    moved = y != 0
    score = proba[moved, 1] - proba[moved, 2]
    out["auc_up_vs_down_moved"] = roc_auc_score(y[moved] == 1, score)
    return out


def main():
    rows, rocs, conf_rows, calib_rows = [], [], [], []
    for session in btc_sessions(4):
        data = prepare_data(session)
        X = data.tab_feat
        for k in HORIZONS:
            y = data.labels(k)
            idx = data.splits(k)
            tr, va, te = idx["train"], idx["val"], idx["test"]
            seg = session.seed
            models = {}

            mom = NaiveMomentumBaseline(horizon_k=k)
            models["Naive Momentum"] = (None, mom.predict_proba(y, te), 0.0, None)

            t0 = time.perf_counter()
            lr = LogisticRegressionBaseline(C=1.0)
            lr.fit(X[tr], y[tr])
            models["Logistic Regression"] = (lr, lr.predict_proba(X[te]), time.perf_counter() - t0, lr.predict_proba)

            t0 = time.perf_counter()
            rf = RandomForestLOB(n_estimators=100, max_depth=10, random_state=seg)
            rf.fit(X[tr], y[tr])
            models["Random Forest"] = (rf, rf.predict_proba(X[te]), time.perf_counter() - t0, rf.predict_proba)

            t0 = time.perf_counter()
            lgbm = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=200, random_state=seg)
            lgbm.fit(X[tr], y[tr], X[va], y[va], early_stopping_rounds=15)
            booster = lgbm.model.booster_
            models["LightGBM"] = (lgbm, lgbm.predict_proba(X[te]), time.perf_counter() - t0,
                                  lambda r: booster.predict(r, num_threads=1))

            yt = y[te]
            for name, (_, proba, fit_s, fn) in models.items():
                pred = proba.argmax(1)
                row = {"segment": seg, "horizon_s": k, "model": name, "n_train": len(tr), "n_test": len(te),
                       "fit_seconds": fit_s, **metrics(yt, proba, pred)}
                if fn is not None and k == 20:
                    t0 = time.perf_counter()
                    fn(X[te])
                    row["batch_us_per_row"] = (time.perf_counter() - t0) / len(te) * 1e6
                    row["single_row_p50_us"], row["single_row_p99_us"] = latency_us(fn, X[te])
                if name == "LightGBM":
                    row["n_trees"] = booster.num_trees()
                rows.append(row)

                if k == 20 and name != "Naive Momentum":
                    # hit rate on directional calls vs confidence (coverage), threshold from test quantiles
                    conf = proba.max(1)
                    directional = pred != 0
                    for cov in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02]:
                        thr = np.quantile(conf, 1 - cov)
                        sel = (conf >= thr) & directional
                        moved = sel & (yt != 0)
                        calib_rows.append({"segment": seg, "model": name, "coverage": cov,
                                           "n_calls": int(sel.sum()),
                                           "precision_directional": float(np.mean(pred[sel] == yt[sel])) if sel.any() else np.nan,
                                           "hit_rate_on_moved": float(np.mean(pred[moved] == yt[moved])) if moved.any() else np.nan})
                    for c, cname in enumerate(CLASSES):
                        fpr, tpr, _ = roc_curve(yt == c, proba[:, c])
                        sel = np.linspace(0, len(fpr) - 1, 200).astype(int)
                        rocs.append(pd.DataFrame({"segment": seg, "model": name, "class": cname,
                                                  "fpr": fpr[sel], "tpr": tpr[sel]}))
                    # reliability: predicted max prob vs empirical accuracy, 10 bins
                    bins = np.clip((proba.max(1) * 10).astype(int), 0, 9)
                    for b in range(10):
                        m = bins == b
                        if m.sum() >= 50:
                            conf_rows.append({"segment": seg, "model": name, "bin": b,
                                              "mean_conf": float(proba.max(1)[m].mean()),
                                              "accuracy": float(np.mean(pred[m] == yt[m])), "n": int(m.sum())})
            print(f"segment {seg} k={k} done", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "btc_extra_metrics_per_segment.csv", index=False)
    pd.concat(rocs).to_csv(OUT / "btc_roc_curves.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(OUT / "btc_reliability.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(OUT / "btc_hit_rate_vs_coverage.csv", index=False)
    df = pd.DataFrame(rows)
    summ = df.groupby(["horizon_s", "model"]).agg(["mean", "std"]).drop(columns="segment", level=0)
    summ.to_csv(OUT / "btc_extra_metrics_summary.csv")
    print(df.groupby(["horizon_s", "model"])[["accuracy", "macro_f1", "roc_auc_macro_ovr", "log_loss",
                                             "auc_up_vs_down_moved", "mcc"]].mean().round(3))


if __name__ == "__main__":
    main()
