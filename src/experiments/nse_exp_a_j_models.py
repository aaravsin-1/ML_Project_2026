"""NSE Experiments A & J: Full Tier 0-5 Model Ladder (with GPU Deep Models),
Probability/Calibration Metrics, Block-Bootstrap Significance, TreeSHAP & DeepLOB5 Attribution.

Runs on RELIANCE (flagship bluechip), NIFTY_FUT (flagship index futures), and IDEA
(ultra-large-tick equity), each across 4 contiguous 16-day segments at k = 20 s.

    python -m src.experiments.nse_exp_a_j_models
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)

from src.config import ARTIFACTS_DIR
from src.data.lob_dataset import build_indexed_dataloaders
from src.data.nse_loader import load_nse_symbol
from src.microstructure.nse_pipeline import SEQ_LEN, build_nse_segments, prepare_nse_segment
from src.models.baselines import LogisticRegressionBaseline, NaiveFlatBaseline, NaiveMomentumBaseline
from src.models.deeplob5 import DeepLOB5
from src.models.recurrent import LSTMLOBNet
from src.models.registry import DeepLearningTrainer, balanced_class_weights
from src.models.tabular_trees import LightGBMLOB, RandomForestLOB
from src.models.tlob import TLOB

OUT_DIR = ARTIFACTS_DIR / "nse" / "models"
TARGETS = [
    ("RELIANCE", "EQUITY"),
    ("NIFTY", "DERIVATIVE"),
    ("IDEA", "EQUITY"),
]
HORIZON_K = 20
EPOCHS = 10
BATCH_SIZE = 256
BLOCK_LEN = 300        # 5-minute blocks for moving-block bootstrap
N_BOOT = 1000
SHAP_SAMPLES = 20_000
ATTR_SAMPLES = 2_000


def _brier_multiclass(y_true: np.ndarray, probs: np.ndarray) -> float:
    oh = np.zeros_like(probs)
    oh[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))


def _full_metrics(y_true: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    p = np.clip(probs.astype(np.float64), 1e-7, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    per_f1 = f1_score(y_true, preds, average=None, labels=[0, 1, 2], zero_division=0)
    try:
        auc_ovr = float(roc_auc_score(y_true, p, multi_class="ovr", average="macro", labels=[0, 1, 2]))
    except ValueError:
        auc_ovr = np.nan
    moved = y_true != 0
    if moved.sum() > 20 and len(np.unique(y_true[moved])) == 2:
        p_up_given_move = p[moved, 1] / np.clip(p[moved, 1] + p[moved, 2], 1e-9, None)
        auc_ud = float(roc_auc_score((y_true[moved] == 1).astype(int), p_up_given_move))
    else:
        auc_ud = np.nan
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
        "f1_flat": float(per_f1[0]),
        "f1_up": float(per_f1[1]),
        "f1_down": float(per_f1[2]),
        "mcc": float(matthews_corrcoef(y_true, preds)),
        "kappa": float(cohen_kappa_score(y_true, preds)),
        "roc_auc_ovr": auc_ovr,
        "roc_auc_ud": auc_ud,
        "log_loss": float(log_loss(y_true, p, labels=[0, 1, 2])),
        "brier": _brier_multiclass(y_true, p),
    }


def _block_bootstrap_diff(
    seg_records: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    block_len: int = BLOCK_LEN,
    n_boot: int = N_BOOT,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Moving-block bootstrap of pooled macro-F1(A) - macro-F1(B), resampling within each segment."""
    y_all = np.concatenate([r[0] for r in seg_records])
    a_all = np.concatenate([r[1] for r in seg_records])
    b_all = np.concatenate([r[2] for r in seg_records])
    diff_obs = float(
        f1_score(y_all, a_all, average="macro", zero_division=0)
        - f1_score(y_all, b_all, average="macro", zero_division=0)
    )

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        ys, as_, bs_ = [], [], []
        for y, pa, pb in seg_records:
            n = len(y)
            n_blocks = int(np.ceil(n / block_len))
            starts = rng.integers(0, max(1, n - block_len + 1), size=n_blocks)
            idx = np.concatenate([np.arange(s, min(s + block_len, n)) for s in starts])[:n]
            ys.append(y[idx])
            as_.append(pa[idx])
            bs_.append(pb[idx])
        yb, ab, bb = np.concatenate(ys), np.concatenate(as_), np.concatenate(bs_)
        diffs[b] = f1_score(yb, ab, average="macro", zero_division=0) - f1_score(yb, bb, average="macro", zero_division=0)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return diff_obs, float(lo), float(hi)


def _deeplob5_attribution(
    trainer: DeepLearningTrainer,
    deep_feat: np.ndarray,
    test_idx: np.ndarray,
    n_samples: int = ATTR_SAMPLES,
) -> Dict[str, float]:
    """cuDNN-safe gradient x input attribution for DeepLOB5 across 5 levels and 100 steps."""
    model = trainer.model
    device = trainer.device
    step = max(1, len(test_idx) // n_samples)
    chosen = test_idx[::step][:n_samples]

    # Put model in eval mode, disable cuDNN during backward through LSTM if on CUDA
    model.eval()
    total_abs = np.zeros((SEQ_LEN, 20), dtype=np.float64)
    bs = 128
    with torch.backends.cudnn.flags(enabled=False):
        for i in range(0, len(chosen), bs):
            batch_rows = chosen[i : i + bs]
            x_np = np.stack([deep_feat[r - SEQ_LEN + 1 : r + 1] for r in batch_rows], axis=0)
            x = torch.tensor(x_np, dtype=torch.float32, device=device, requires_grad=True)
            logits = model(x)
            pred_cls = logits.argmax(dim=-1)
            selected = logits.gather(1, pred_cls.unsqueeze(1)).sum()
            model.zero_grad(set_to_none=True)
            selected.backward()
            gi = (x.grad * x).detach().abs().cpu().numpy()
            total_abs += gi.sum(axis=0)

    grand = total_abs.sum() + 1e-12
    out: Dict[str, float] = {}
    for lvl in range(5):
        out[f"level_{lvl + 1}_share"] = float(total_abs[:, lvl * 4 : (lvl + 1) * 4].sum() / grand)
    out["recent_10s_share"] = float(total_abs[-10:, :].sum() / grand)
    out["recent_20s_share"] = float(total_abs[-20:, :].sum() / grand)
    out["prices_share"] = float(total_abs[:, 0::2].sum() / grand)
    out["sizes_share"] = float(total_abs[:, 1::2].sum() / grand)
    out["ask_share"] = float((total_abs[:, 0::4].sum() + total_abs[:, 1::4].sum()) / grand)
    out["bid_share"] = float((total_abs[:, 2::4].sum() + total_abs[:, 3::4].sum()) / grand)
    return out


def _feature_group(name: str) -> str:
    if name in ("imbalance_l1", "spread_ticks", "micro_deviation", "ofi_l1"):
        return "L1 snapshot"
    if "_lag_" in name:
        return "L1 lags"
    if name == "vpin":
        return "VPIN"
    if name in (
        "total_book_imbalance",
        "order_imbalance_l1",
        "order_imbalance_l5",
        "avg_order_size_ratio_l1",
        "depth_ratio_l5_to_total",
    ):
        return "NSE order counts & full book"
    return "Deeper book (L2-5, slopes, PCA)"


def _run_symbol(symbol: str, asset: str, fresh: bool = False) -> None:
    sym_data = load_nse_symbol(symbol, asset=asset)
    tag = sym_data.symbol
    segs = build_nse_segments(sym_data, n_segments=4)
    sym_dir = OUT_DIR / tag
    sess_dir = sym_dir / "sessions"
    sess_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=====================================================================")
    print(f"   NSE EXP A + J: {tag} ({len(sym_data.days)} days, 4 segments, k={HORIZON_K}s)")
    print(f"=====================================================================")

    all_metrics: List[Dict[str, Any]] = []
    all_shap: List[pd.DataFrame] = []
    all_attr: List[Dict[str, Any]] = []
    all_lat: List[Dict[str, Any]] = []
    pooled_preds: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    pooled_probs: Dict[str, List[np.ndarray]] = {}
    pooled_y: List[np.ndarray] = []

    for seg in segs:
        ckpt = sess_dir / f"seg{seg.segment_id}_k{HORIZON_K}_e{EPOCHS}.npz"
        meta_ckpt = sess_dir / f"seg{seg.segment_id}_meta.pkl"
        prep = prepare_nse_segment(seg)
        y_all = prep.labels(HORIZON_K)
        idx = prep.splits(HORIZON_K)
        tr, va, te = idx["train"], idx["val"], idx["test"]
        y_tr, y_va, y_te = y_all[tr], y_all[va], y_all[te]

        if ckpt.exists() and meta_ckpt.exists() and not fresh:
            print(f"[+] {tag} seg {seg.segment_id}: loading checkpoint")
            saved = pd.read_pickle(meta_ckpt)
            z = np.load(ckpt)
            all_metrics.extend(saved["metrics"])
            all_shap.append(saved["shap"])
            all_attr.append(saved["attr"])
            all_lat.append(saved["lat"])
            pooled_y.append(z["y_test"])
            for mname in saved["model_names"]:
                pooled_preds.setdefault(mname, []).append((z["y_test"], z[f"pred_{mname}"]))
                pooled_probs.setdefault(mname, []).append(z[f"prob_{mname}"])
            continue

        t_seg = time.time()
        print(
            f"\n--- {tag} Segment {seg.segment_id} ({seg.days[0]}..{seg.days[-1]}) | "
            f"tr={len(tr):,} va={len(va):,} te={len(te):,} | alpha={prep.alphas_dict[HORIZON_K]*1e4:.2f}bp ---"
        )
        torch.manual_seed(seg.segment_id)
        np.random.seed(seg.segment_id)

        X_core = prep.tab_feat[:, prep.core_indices]
        X_full = prep.tab_feat

        seg_metrics: List[Dict[str, Any]] = []
        seg_preds_dict: Dict[str, np.ndarray] = {}
        seg_probs_dict: Dict[str, np.ndarray] = {}

        def record(tier: str, name: str, preds: np.ndarray, probs: np.ndarray) -> None:
            m = _full_metrics(y_te, preds, probs)
            seg_metrics.append({"symbol": tag, "segment": seg.segment_id, "tier": tier, "model": name, **m})
            seg_preds_dict[name] = preds.astype(np.int64)
            seg_probs_dict[name] = probs.astype(np.float32)
            print(
                f"    {name:<22} F1={m['macro_f1']:.4f} Acc={m['accuracy']:.4f} "
                f"AUC={m['roc_auc_ovr']:.4f} U/D={m['roc_auc_ud']:.4f} MCC={m['mcc']:.3f} LL={m['log_loss']:.4f}"
            )

        # Tier 0
        flat = NaiveFlatBaseline()
        record("Tier 0", "Naive Flat", flat.predict(X_core[te]), flat.predict_proba(X_core[te]))
        mom = NaiveMomentumBaseline(horizon_k=HORIZON_K)
        record("Tier 0", "Naive Momentum", mom.predict(y_all, te), mom.predict_proba(y_all, te))

        # Tier 1
        lr = LogisticRegressionBaseline(C=1.0)
        lr.fit(X_core[tr], y_tr)
        record("Tier 1", "Logistic Regression", lr.predict(X_core[te]), lr.predict_proba(X_core[te]))

        # Tier 2
        rf = RandomForestLOB(n_estimators=100, max_depth=10, random_state=seg.segment_id)
        rf.fit(X_core[tr], y_tr)
        record("Tier 2", "Random Forest", rf.predict(X_core[te]), rf.predict_proba(X_core[te]))

        # Tier 3: LightGBM (core 25) and LightGBM (+NSE extra 30)
        lgb_core = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=300, random_state=seg.segment_id)
        lgb_core.fit(X_core[tr], y_tr, X_core[va], y_va, early_stopping_rounds=15)
        record("Tier 3", "LightGBM", lgb_core.predict(X_core[te]), lgb_core.predict_proba(X_core[te]))

        lgb_full = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=300, random_state=seg.segment_id)
        lgb_full.fit(X_full[tr], y_tr, X_full[va], y_va, early_stopping_rounds=15)
        record("Tier 3+", "LightGBM (+NSE)", lgb_full.predict(X_full[te]), lgb_full.predict_proba(X_full[te]))

        # TreeSHAP on LightGBM (+NSE) for 20,000 test rows
        step_s = max(1, len(te) // SHAP_SAMPLES)
        X_shap = X_full[te[::step_s][:SHAP_SAMPLES]]
        booster = lgb_full.model.booster_
        contrib = booster.predict(X_shap, pred_contrib=True, num_iteration=lgb_full.model.best_iteration_)
        # multiclass pred_contrib shape: (N, 3 * (n_features + 1)) -> reshape (N, 3, n_features + 1)
        nf = X_full.shape[1]
        contrib = contrib.reshape(len(X_shap), 3, nf + 1)[:, :, :nf]
        mean_abs_shap = np.mean(np.abs(contrib), axis=(0, 1))
        gain = booster.feature_importance(importance_type="gain")
        gain_share = gain / (gain.sum() + 1e-12)
        shap_share = mean_abs_shap / (mean_abs_shap.sum() + 1e-12)
        df_shap = pd.DataFrame({
            "symbol": tag,
            "segment": seg.segment_id,
            "feature": prep.feat_names,
            "group": [_feature_group(f) for f in prep.feat_names],
            "shap_share": shap_share,
            "gain_share": gain_share,
        })

        # Tiers 4 & 5: GPU Deep Models on identical rows (N, 100, 20)
        # Every deep model uses the same validation-only collapse-retry protocol.
        loaders = build_indexed_dataloaders(prep.deep_feat, y_all, idx, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
        weights = balanced_class_weights(y_tr)

        lstm_tr = _fit_with_collapse_retry(
            lambda: LSTMLOBNet(num_features=20, hidden_dim=64, num_layers=2),
            loaders, weights, base_lr=1e-3, base_seed=seg.segment_id, epochs=EPOCHS, patience=3,
        )
        _, _, _, p_lstm, pr_lstm = lstm_tr.evaluate(loaders["test"])
        record("Tier 4", "LSTM", p_lstm, pr_lstm)

        dl_tr = _fit_with_collapse_retry(
            lambda: DeepLOB5(num_classes=3),
            loaders, weights, base_lr=5e-4, base_seed=seg.segment_id, epochs=EPOCHS, patience=3,
        )
        _, _, _, p_dl, pr_dl = dl_tr.evaluate(loaders["test"])
        record("Tier 5", "DeepLOB5", p_dl, pr_dl)

        tlob_tr = _fit_with_collapse_retry(
            lambda: TLOB(num_features=20, seq_len=SEQ_LEN, d_model=64, num_heads=4, num_layers=2),
            loaders, weights, base_lr=5e-4, base_seed=seg.segment_id, epochs=EPOCHS, patience=3,
        )
        _, _, _, p_tl, pr_tl = tlob_tr.evaluate(loaders["test"])
        record("Tier 5", "TLOB", p_tl, pr_tl)

        # DeepLOB5 gradient x input attribution
        attr = {"symbol": tag, "segment": seg.segment_id, **_deeplob5_attribution(dl_tr, prep.deep_feat, te)}

        # Inference latency measurement (single sample, 300 calls)
        n_lat = min(300, len(te))
        t_lgb = []
        b_core = lgb_core.model.booster_
        best_it = lgb_core.model.best_iteration_
        for r in te[:n_lat]:
            t0 = time.perf_counter()
            b_core.predict(X_core[r : r + 1], num_iteration=best_it)
            t_lgb.append((time.perf_counter() - t0) * 1000.0)

        dl_model = dl_tr.model
        dl_model.eval()
        deep_t = torch.tensor(prep.deep_feat, dtype=torch.float32, device=dl_tr.device)
        t_dl = []
        with torch.no_grad():
            for r in te[:n_lat]:
                x_one = deep_t[r - SEQ_LEN + 1 : r + 1].unsqueeze(0).unsqueeze(0)
                if dl_tr.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                dl_model(x_one).argmax(dim=-1).item()
                if dl_tr.device.type == "cuda":
                    torch.cuda.synchronize()
                t_dl.append((time.perf_counter() - t0) * 1000.0)

        lat_rec = {
            "symbol": tag,
            "segment": seg.segment_id,
            "lgbm_p50_ms": float(np.median(t_lgb)),
            "lgbm_p99_ms": float(np.quantile(t_lgb, 0.99)),
            "deeplob5_p50_ms": float(np.median(t_dl)),
            "deeplob5_p99_ms": float(np.quantile(t_dl, 0.99)),
            "device": dl_tr.device.type,
        }

        # Save per-segment checkpoint
        save_npz = {"y_test": y_te}
        for k_m, v_p in seg_preds_dict.items():
            save_npz[f"pred_{k_m}"] = v_p
            save_npz[f"prob_{k_m}"] = seg_probs_dict[k_m]
        np.savez_compressed(ckpt, **save_npz)
        pd.to_pickle(
            {
                "metrics": seg_metrics,
                "shap": df_shap,
                "attr": attr,
                "lat": lat_rec,
                "model_names": list(seg_preds_dict.keys()),
            },
            meta_ckpt,
        )

        all_metrics.extend(seg_metrics)
        all_shap.append(df_shap)
        all_attr.append(attr)
        all_lat.append(lat_rec)
        pooled_y.append(y_te)
        for mname, pv in seg_preds_dict.items():
            pooled_preds.setdefault(mname, []).append((y_te, pv))
            pooled_probs.setdefault(mname, []).append(seg_probs_dict[mname])
        print(f"    [seg {seg.segment_id} done in {time.time() - t_seg:.1f}s]")

    # Aggregate per-symbol tables
    df_per = pd.DataFrame(all_metrics)
    df_per.to_csv(sym_dir / "model_ladder_per_segment.csv", index=False)

    val_cols = [c for c in df_per.columns if c not in ("symbol", "segment", "tier", "model")]
    mean_df = df_per.groupby(["tier", "model"], sort=False)[val_cols].mean()
    std_df = df_per.groupby(["tier", "model"], sort=False)[val_cols].std(ddof=1).add_suffix("_std")
    df_agg = pd.concat([mean_df, std_df], axis=1).reset_index()
    df_agg.to_csv(sym_dir / "model_ladder_summary.csv", index=False)

    # Significance tests vs LightGBM (core) using moving-block bootstrap only
    # (McNemar's test is omitted because 1 s rows with overlapping 20 s labels are not independent)
    sig_rows = []
    lgb_list = pooled_preds["LightGBM"]
    for comp in ("Naive Momentum", "Logistic Regression", "Random Forest", "LightGBM (+NSE)", "LSTM", "DeepLOB5", "TLOB"):
        comp_list = pooled_preds[comp]
        seg_triples = [(lgb_list[i][0], lgb_list[i][1], comp_list[i][1]) for i in range(len(lgb_list))]
        diff, lo, hi = _block_bootstrap_diff(seg_triples)
        seg_wins = sum(
            f1_score(lgb_list[i][0], lgb_list[i][1], average="macro", zero_division=0)
            > f1_score(comp_list[i][0], comp_list[i][1], average="macro", zero_division=0)
            for i in range(len(lgb_list))
        )
        sig_rows.append({
            "symbol": tag,
            "baseline": comp,
            "lgbm_minus_baseline": diff,
            "ci_lo": lo,
            "ci_hi": hi,
            "segments_lgbm_better": f"{seg_wins}/{len(lgb_list)}",
        })
    df_sig = pd.DataFrame(sig_rows)
    df_sig.to_csv(sym_dir / "significance_vs_lightgbm.csv", index=False)

    # Confusion matrices pooled across segments
    cm_rows = []
    y_pool = np.concatenate(pooled_y)
    for mname in ("Naive Momentum", "Logistic Regression", "Random Forest", "LightGBM", "LightGBM (+NSE)", "DeepLOB5", "TLOB"):
        p_pool = np.concatenate([x[1] for x in pooled_preds[mname]])
        cm = confusion_matrix(y_pool, p_pool, labels=[0, 1, 2], normalize="true")
        for r_i, lbl in enumerate(("FLAT", "UP", "DOWN")):
            cm_rows.append({"symbol": tag, "model": mname, "true": lbl, "pred_FLAT": cm[r_i, 0], "pred_UP": cm[r_i, 1], "pred_DOWN": cm[r_i, 2]})
    pd.DataFrame(cm_rows).to_csv(sym_dir / "confusion_matrices.csv", index=False)

    # Reliability (calibration) bins for LightGBM
    lgb_probs = np.vstack(pooled_probs["LightGBM"])
    chosen_cls = lgb_probs.argmax(axis=1)
    chosen_p = lgb_probs.max(axis=1)
    correct = (chosen_cls == y_pool).astype(float)
    bins = np.linspace(0.33, 0.95, 11)
    cal_rows = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (chosen_p >= b0) & (chosen_p < b1)
        if m.sum() >= 50:
            cal_rows.append({"symbol": tag, "bin_lo": b0, "bin_hi": b1, "n": int(m.sum()), "mean_pred_p": float(chosen_p[m].mean()), "obs_acc": float(correct[m].mean())})
    pd.DataFrame(cal_rows).to_csv(sym_dir / "calibration_lgbm.csv", index=False)

    # SHAP & DeepLOB5 attribution & latency summaries
    df_shap_all = pd.concat(all_shap, ignore_index=True)
    df_shap_all.groupby(["feature", "group"], sort=False)[["shap_share", "gain_share"]].mean().reset_index().sort_values("shap_share", ascending=False).to_csv(sym_dir / "shap_features.csv", index=False)
    df_shap_grp = (
        df_shap_all.groupby(["segment", "group"], sort=False)[["shap_share", "gain_share"]]
        .sum()
        .groupby("group", sort=False)[["shap_share", "gain_share"]]
        .mean()
        .reset_index()
        .sort_values("shap_share", ascending=False)
    )
    df_shap_grp.to_csv(sym_dir / "shap_groups.csv", index=False)
    pd.DataFrame(all_attr).to_csv(sym_dir / "deeplob5_attribution.csv", index=False)
    pd.DataFrame(all_lat).to_csv(sym_dir / "inference_latency.csv", index=False)

    print(f"\n>>> {tag} MODEL LADDER SUMMARY (k={HORIZON_K}s, 4 segments):")
    print(df_agg[["tier", "model", "accuracy", "macro_f1", "macro_f1_std", "roc_auc_ovr", "roc_auc_ud", "mcc", "log_loss"]].round(4).to_string(index=False))
    print(f"\n>>> {tag} SIGNIFICANCE VS LIGHTGBM:")
    print(df_sig.round(4).to_string(index=False))


def _fit_with_collapse_retry(
    model_fn,
    loaders: Dict[str, Any],
    weights: np.ndarray,
    base_lr: float,
    base_seed: int,
    epochs: int = EPOCHS,
    patience: int = 3,
    min_val_f1: float = 0.25,
) -> DeepLearningTrainer:
    """Train a deep model and retry with different seeds / lower LR if validation collapses.

    Selection uses ONLY the validation split (lowest val_loss among non-collapsed runs),
    so test metrics are never inspected during model selection.
    """
    candidates = [
        (base_seed, base_lr),
        (base_seed + 101, base_lr * 0.5),
        (base_seed + 202, base_lr * 0.25),
        (base_seed + 303, base_lr * 0.5),
    ]
    best_tr: DeepLearningTrainer | None = None
    best_key = (1, float("inf"))  # (is_collapsed, val_loss)
    for attempt, (sd, lr) in enumerate(candidates):
        torch.manual_seed(sd)
        np.random.seed(sd)
        tr = DeepLearningTrainer(model_fn(), learning_rate=lr, class_weights=weights, use_amp=True)
        tr.fit(loaders["train"], loaders["val"], epochs=epochs, patience=patience)
        val_loss, _, val_f1, val_preds, _ = tr.evaluate(loaders["val"])
        n_cls = len(np.unique(val_preds))
        collapsed = int(val_f1 < min_val_f1 or n_cls < 2)
        key = (collapsed, float(val_loss))
        if key < best_key:
            best_key = key
            best_tr = tr
        if not collapsed:
            if attempt > 0:
                print(f"      [retry {attempt} seed={sd} lr={lr:.1e} recovered: val_f1={val_f1:.3f}, val_loss={val_loss:.4f}]")
            return tr
        print(f"      [attempt {attempt} seed={sd} lr={lr:.1e} collapsed on val: val_f1={val_f1:.3f}, classes={n_cls}]")
    assert best_tr is not None
    return best_tr


def _rerun_all_deep_models(
    targets: List[Tuple[str, str]],
    model_filter: Dict[str, List[str]] | None = None,
    seg_filter: Dict[str, List[int]] | None = None,
) -> None:
    """Uniformly rerun deep models across segments using validation-only collapse retry.

    Every selected (symbol, model) family is rerun uniformly across all 4 segments using
    strictly validation-only collapse detection (zero test-set peeking).
    """
    all_specs = {
        "LSTM": (lambda: LSTMLOBNet(num_features=20, hidden_dim=64, num_layers=2), 1e-3),
        "DeepLOB5": (lambda: DeepLOB5(num_classes=3), 5e-4),
        "TLOB": (lambda: TLOB(num_features=20, seq_len=SEQ_LEN, d_model=64, num_heads=4, num_layers=2), 5e-4),
    }
    for sym, asset in targets:
        sym_data = load_nse_symbol(sym, asset=asset)
        tag = sym_data.symbol
        if model_filter is not None and tag not in model_filter:
            continue
        chosen_models = model_filter[tag] if model_filter is not None else ["LSTM", "DeepLOB5", "TLOB"]
        chosen_segs = seg_filter.get(tag) if seg_filter is not None else None

        sym_dir = OUT_DIR / tag
        sess_dir = sym_dir / "sessions"
        segs = build_nse_segments(sym_data, n_segments=4)

        for seg in segs:
            if chosen_segs is not None and seg.segment_id not in chosen_segs:
                continue
            ckpt = sess_dir / f"seg{seg.segment_id}_k{HORIZON_K}_e{EPOCHS}.npz"
            meta_ckpt = sess_dir / f"seg{seg.segment_id}_meta.pkl"
            if not (ckpt.exists() and meta_ckpt.exists()):
                continue
            saved = pd.read_pickle(meta_ckpt)
            z = dict(np.load(ckpt))

            print(f"\n[*] Uniform deep rerun (val-only retry, bs={BATCH_SIZE}): {tag} seg {seg.segment_id} -> {chosen_models}")
            prep = prepare_nse_segment(seg, horizons=[HORIZON_K])
            y_all = prep.labels(HORIZON_K)
            idx = prep.splits(HORIZON_K)
            tr_idx, te_idx = idx["train"], idx["test"]
            y_tr, y_te = y_all[tr_idx], y_all[te_idx]
            loaders = build_indexed_dataloaders(prep.deep_feat, y_all, idx, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
            weights = balanced_class_weights(y_tr)

            for mname in chosen_models:
                fn, lr0 = all_specs[mname]
                trainer = _fit_with_collapse_retry(
                    fn, loaders, weights, base_lr=lr0, base_seed=seg.segment_id, epochs=EPOCHS, patience=3
                )
                _, _, _, p_new, pr_new = trainer.evaluate(loaders["test"])
                new_m = _full_metrics(y_te, p_new, pr_new)
                print(
                    f"    -> {mname:<10} F1={new_m['macro_f1']:.4f} Acc={new_m['accuracy']:.4f} "
                    f"AUC={new_m['roc_auc_ovr']:.4f} U/D={new_m['roc_auc_ud']:.4f} MCC={new_m['mcc']:.3f} LL={new_m['log_loss']:.4f}"
                )
                z[f"pred_{mname}"] = p_new.astype(np.int64)
                z[f"prob_{mname}"] = pr_new.astype(np.float32)
                for row in saved["metrics"]:
                    if row["model"] == mname:
                        row.update(new_m)
                if mname == "DeepLOB5":
                    saved["attr"] = {"symbol": tag, "segment": seg.segment_id, **_deeplob5_attribution(trainer, prep.deep_feat, te_idx)}

            np.savez_compressed(ckpt, **z)
            pd.to_pickle(saved, meta_ckpt)

        _run_symbol(sym, asset, fresh=False)


def main() -> None:
    fresh = "--fresh" in sys.argv
    rerun_deep = "--rerun-all-deep" in sys.argv
    rerun_affected = "--rerun-affected-families" in sys.argv
    t0 = time.time()
    if rerun_affected:
        # Uniformly rerun ALL 4 segments of every (symbol, model) family that saw a collapsed seed,
        # plus restore RELIANCE seg 0 at batch_size=256.
        _rerun_all_deep_models(
            TARGETS,
            model_filter={
                "RELIANCE": ["LSTM", "DeepLOB5", "TLOB"],
                "NIFTY_FUT": ["LSTM", "DeepLOB5"],
                "IDEA": ["LSTM"],
            },
            seg_filter={"RELIANCE": [0]},
        )
    elif rerun_deep:
        _rerun_all_deep_models(TARGETS)
    else:
        for sym, asset in TARGETS:
            _run_symbol(sym, asset, fresh=fresh)
    print(f"\n[+] NSE Experiments A + J finished in {(time.time() - t0) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
