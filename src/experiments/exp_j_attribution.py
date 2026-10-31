"""Experiment J: Attribution and statistical significance on Coinbase BTC (research spec 5.1, 5.3).

The spec asks for SHAP-style attribution of LightGBM and DeepLOB, confusion matrices, and
whether improvements over the baselines are statistically significant. Experiments A and C
reported gain importance and mean +/- std over 4 segments, which is neither.

Part 1 (CPU, laptop): per BTC segment, retrains Experiment A's tabular models at k = 20 s with
the same settings and saves per-row test predictions. Then:
- TreeSHAP for LightGBM (exact, via LightGBM's pred_contrib): mean |SHAP| per feature and per
  feature group, as a share of the total, next to Experiment C's gain shares.
- Row-normalized confusion matrices for every model.
- Significance: macro-F1 difference between LightGBM and each other model, with a 95% CI from a
  moving-block bootstrap over test rows (5-minute blocks, segments resampled separately and
  pooled), and an exact McNemar test on which rows each model gets right.
DeepLOB is included in all of this if its predictions exist (Part 2).

Part 2 (GPU, Colab: `--deeplob`): retrains Experiment A's DeepLOB per segment, saves its
per-row test predictions, and computes gradient x input attribution on a sample of test windows,
aggregated by book level, side, price vs size, and how far back in the 100-step window.

    python -m src.experiments.exp_j_attribution             # part 1
    python -m src.experiments.exp_j_attribution --deeplob   # part 2 (GPU), then rerun part 1
"""

import argparse
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score

from src.config import ARTIFACTS_DIR
from src.experiments.common import SEQ_LEN, btc_sessions, prepare_data
from src.models.baselines import LogisticRegressionBaseline, NaiveMomentumBaseline
from src.models.tabular_trees import LightGBMLOB, RandomForestLOB

K = 20
OUT = ARTIFACTS_DIR / "btc" / "attribution"
SHAP_ROWS = 20000
BLOCK_ROWS = 300          # 5 minutes of 1-second snapshots
BOOTSTRAP_DRAWS = 2000
CLASSES = ["FLAT", "UP", "DOWN"]
FEATURE_GROUPS = {
    "L1 snapshot": ["spread_ticks", "imbalance_l1", "ofi_l1", "micro_deviation"],
    "L1 lags": ["ofi_lag_", "imbalance_lag_", "micro_dev_lag_"],
    "Levels 2-3": ["imbalance_l3", "mlofi_lvl_2", "mlofi_lvl_3"],
    "Deeper book": ["imbalance_l5", "imbalance_l10", "slope_bid", "slope_ask", "mlofi_lvl_4", "mlofi_lvl_5", "ofi_pca"],
    "VPIN": ["vpin"],
}


def feature_group(name: str) -> str:
    for group, keys in FEATURE_GROUPS.items():
        if any(name == k or (k.endswith("_") and name.startswith(k)) or (k == "ofi_pca" and name.startswith(k))
               for k in keys):
            return group
    return "Other"


# --------------------------------------------------------------------------- part 1

def tabular_segment(session) -> Dict:
    data = prepare_data(session)
    X, y = data.tab_feat, data.labels(K)
    idx = data.splits(K)
    tr, va, te = idx["train"], idx["val"], idx["test"]
    seed = session.seed
    preds = {"y": y[te], "row": te}
    preds["Naive Momentum"] = NaiveMomentumBaseline(horizon_k=K).predict(y, te)
    lr = LogisticRegressionBaseline(C=1.0)
    lr.fit(X[tr], y[tr])
    preds["Logistic Regression"] = lr.predict(X[te])
    rf = RandomForestLOB(n_estimators=100, max_depth=10, random_state=seed)
    rf.fit(X[tr], y[tr])
    preds["Random Forest"] = rf.predict(X[te])
    lgbm = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=200, random_state=seed)
    lgbm.fit(X[tr], y[tr], X[va], y[va], early_stopping_rounds=15)
    preds["LightGBM"] = lgbm.predict(X[te])

    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(te, size=min(SHAP_ROWS, len(te)), replace=False))
    contrib = lgbm.model.booster_.predict(X[sample], pred_contrib=True)  # (n, 3 * (F + 1)), per class: F + bias
    nf = X.shape[1]
    phi = np.abs(contrib.reshape(len(sample), 3, nf + 1)[:, :, :nf]).sum(axis=1).mean(axis=0)
    shap = pd.DataFrame({"segment": seed, "feature": data.feat_names, "mean_abs_shap": phi})
    return {"preds": preds, "shap": shap}


def block_bootstrap_diff(segments, model_a: str, model_b: str, rng) -> np.ndarray:
    """Bootstrap draws of macro-F1(a) - macro-F1(b), resampling 5-minute blocks within each segment."""
    draws = np.empty(BOOTSTRAP_DRAWS)
    plans = []
    for p in segments:
        n = len(p["y"])
        n_blocks = int(np.ceil(n / BLOCK_ROWS))
        plans.append((p, n, n_blocks))
    for b in range(BOOTSTRAP_DRAWS):
        ys, pa, pb = [], [], []
        for p, n, n_blocks in plans:
            starts = rng.integers(0, n - BLOCK_ROWS + 1, size=n_blocks)
            ix = (starts[:, None] + np.arange(BLOCK_ROWS)[None, :]).ravel()[:n]
            ys.append(p["y"][ix])
            pa.append(p[model_a][ix])
            pb.append(p[model_b][ix])
        y, a, c = np.concatenate(ys), np.concatenate(pa), np.concatenate(pb)
        draws[b] = f1_score(y, a, average="macro", zero_division=0) - f1_score(y, c, average="macro", zero_division=0)
    return draws


def significance(segments, reference: str = "LightGBM") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    models = [m for m in segments[0] if m not in ("y", "row", reference)]
    rows = []
    for other in models:
        if not all(other in p for p in segments):
            continue
        y = np.concatenate([p["y"] for p in segments])
        a = np.concatenate([p[reference] for p in segments])
        c = np.concatenate([p[other] for p in segments])
        diff = f1_score(y, a, average="macro") - f1_score(y, c, average="macro")
        draws = block_bootstrap_diff(segments, reference, other, rng)
        a_right, c_right = a == y, c == y
        only_a, only_c = int(np.sum(a_right & ~c_right)), int(np.sum(~a_right & c_right))
        mcnemar_p = binomtest(only_a, only_a + only_c, 0.5).pvalue if only_a + only_c else np.nan
        per_seg = [f1_score(p["y"], p[reference], average="macro") - f1_score(p["y"], p[other], average="macro")
                   for p in segments]
        rows.append({"reference": reference, "other": other, "macro_f1_diff": diff,
                     "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
                     "share_draws_le_0": float(np.mean(draws <= 0)), "segments_reference_better": int(np.sum(np.array(per_seg) > 0)),
                     "rows_only_reference_right": only_a, "rows_only_other_right": only_c, "mcnemar_p": mcnemar_p})
    return pd.DataFrame(rows)


def confusion_table(segments) -> pd.DataFrame:
    rows = []
    for model in [m for m in segments[0] if m not in ("y", "row")]:
        if not all(model in p for p in segments):
            continue
        y = np.concatenate([p["y"] for p in segments])
        pr = np.concatenate([p[model] for p in segments])
        cm = confusion_matrix(y, pr, labels=[0, 1, 2], normalize="true")
        for i, true in enumerate(CLASSES):
            rows.append({"model": model, "true": true, **{f"pred_{c}": cm[i, j] for j, c in enumerate(CLASSES)}})
    return pd.DataFrame(rows)


def run_part1(n_segments: int = 4) -> None:
    (OUT / "sessions").mkdir(parents=True, exist_ok=True)
    segments, shaps = [], []
    for session in btc_sessions(n_segments):
        ck = OUT / "sessions" / f"{session.name}_tabular.npz"
        ck_shap = OUT / "sessions" / f"{session.name}_shap.csv"
        if ck.exists() and ck_shap.exists():
            print(f"[+] {session.name}: loading checkpoint")
            preds = dict(np.load(ck, allow_pickle=False))
            shap = pd.read_csv(ck_shap)
        else:
            print(f"\n######## {session.name} ########")
            res = tabular_segment(session)
            preds, shap = res["preds"], res["shap"]
            np.savez(ck, **preds)
            shap.to_csv(ck_shap, index=False)
        dl = OUT / "sessions" / f"{session.name}_deeplob.npz"
        if dl.exists():
            z = np.load(dl)
            assert np.array_equal(z["row"], preds["row"]), "DeepLOB predictions are on different rows"
            preds["DeepLOB"] = z["pred"]
        segments.append(preds)
        shaps.append(shap)

    shap = pd.concat(shaps)
    shap["share_pct"] = shap.groupby("segment")["mean_abs_shap"].transform(lambda s: 100 * s / s.sum())
    feat = shap.groupby("feature")["share_pct"].agg(["mean", "std"]).reset_index().sort_values("mean", ascending=False)
    gain = pd.read_csv(ARTIFACTS_DIR / "btc" / "experiment_c_feature_importance.csv")[["feature", "gain_share_pct"]]
    feat = feat.rename(columns={"mean": "shap_share_pct", "std": "shap_share_std"}).merge(gain, on="feature", how="left")
    feat["group"] = feat["feature"].map(feature_group)
    feat.to_csv(OUT / "shap_lightgbm_features.csv", index=False)
    groups = feat.groupby("group")[["shap_share_pct", "gain_share_pct"]].sum().sort_values("shap_share_pct", ascending=False)
    groups.reset_index().to_csv(OUT / "shap_lightgbm_groups.csv", index=False)

    sig = significance(segments)
    sig.to_csv(OUT / "significance_vs_lightgbm.csv", index=False)
    conf = confusion_table(segments)
    conf.to_csv(OUT / "confusion_matrices.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nLightGBM SHAP shares by feature group (vs gain shares from Experiment C):")
    print(groups.round(1).to_string())
    print("\nTop features by SHAP:")
    print(feat.head(10)[["feature", "group", "shap_share_pct", "gain_share_pct"]].round(1).to_string(index=False))
    print("\nSignificance (LightGBM minus other, macro-F1):")
    print(sig.round(4).to_string(index=False))
    print("\nConfusion matrices (rows = true class, row-normalized):")
    print(conf.round(3).to_string(index=False))


# --------------------------------------------------------------------------- part 2 (GPU)

def run_part2_deeplob(n_segments: int = 4, epochs: int = 8, attribution_windows: int = 2000) -> None:
    import torch
    from src.data.lob_dataset import build_indexed_dataloaders
    from src.models.deeplob import DeepLOB
    from src.models.registry import DeepLearningTrainer, balanced_class_weights

    (OUT / "sessions").mkdir(parents=True, exist_ok=True)
    attr_rows = []
    for session in btc_sessions(n_segments):
        ck = OUT / "sessions" / f"{session.name}_deeplob.npz"
        if ck.exists():
            print(f"[+] {session.name}: DeepLOB predictions already saved")
            continue
        print(f"\n######## {session.name}: DeepLOB ########")
        torch.manual_seed(session.seed)
        np.random.seed(session.seed)
        data = prepare_data(session)
        y = data.labels(K)
        idx = data.splits(K)
        loaders = build_indexed_dataloaders(data.deep_feat, y, idx, seq_len=SEQ_LEN, batch_size=128)
        trainer = DeepLearningTrainer(DeepLOB(num_classes=3, lstm_hidden=64), learning_rate=0.0005,
                                      class_weights=balanced_class_weights(y[idx["train"]]))
        trainer.fit(loaders["train"], loaders["val"], epochs=epochs, patience=3)
        _, _, _, preds, _ = trainer.evaluate(loaders["test"])

        # gradient x input on a sample of test windows, for the predicted class
        model, dev = trainer.model, trainer.device
        model.eval()
        rng = np.random.default_rng(session.seed)
        rows = np.sort(rng.choice(idx["test"], size=min(attribution_windows, len(idx["test"])), replace=False))
        feats = torch.tensor(data.deep_feat, dtype=torch.float32)
        total = np.zeros((SEQ_LEN, data.deep_feat.shape[1]))
        for start in range(0, len(rows), 128):
            batch = torch.stack([feats[r - SEQ_LEN + 1:r + 1] for r in rows[start:start + 128]]).unsqueeze(1).to(dev)
            batch.requires_grad_(True)
            out = model(batch)
            out.gather(1, out.argmax(1, keepdim=True)).sum().backward()
            total += (batch.grad * batch).abs().sum(dim=0).squeeze(0).detach().cpu().numpy()
        total /= total.sum()
        cols = data.deep_feat.shape[1]
        for c in range(cols):
            level, kind = c // 4, ["ask_price", "ask_size", "bid_price", "bid_size"][c % 4]
            attr_rows.append({"segment": session.seed, "level": level + 1, "kind": kind,
                              "share_last_10_steps": float(total[-10:, c].sum()),
                              "share_total": float(total[:, c].sum())})
        np.savez(ck, row=idx["test"], pred=np.asarray(preds))
        print(f"[+] saved {ck.name}")
    if attr_rows:
        path = OUT / "deeplob_attribution.csv"
        prev = pd.read_csv(path) if path.exists() else pd.DataFrame()
        pd.concat([prev, pd.DataFrame(attr_rows)]).to_csv(path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deeplob", action="store_true", help="Part 2: train DeepLOB and save predictions (GPU)")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    if args.deeplob:
        run_part2_deeplob(epochs=args.epochs)
    run_part1()
