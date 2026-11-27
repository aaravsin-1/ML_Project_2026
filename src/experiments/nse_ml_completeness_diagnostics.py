"""Comprehensive ML Evaluation & Completeness Diagnostics (Per-Class F1, Class Priors, Confusion Matrices, Calibration, Generalization Gap, SHAP, & 25-Stock Audit).

Computes and exports every standard ML reporting element on zero-leakage chronological splits:
  Part A (RELIANCE, NIFTY_FUT, IDEA — all model tiers + Flagship Hybrid):
    1. class_distribution_splits.csv     (Train / Val / Test / Seg0-3 priors & adaptive alpha_bp)
    2. per_class_classification_reports.csv (Precision, Recall, F1, Support per class, MCC, Kappa, Balanced Acc, PR-AUC, ROC-AUC, LogLoss, Brier, ECE, fold std)
    3. confusion_matrices_detailed.csv   (Full 3x3 count, recall %, & precision % matrices)
    4. train_val_test_generalization.csv (Train vs Val vs Test F1/LogLoss gaps + best_iteration stability)
    5. v2_ssf_feature_importance.csv     (Top-15 split-gain & TreeSHAP attribution for upgraded models)
  Part B (All 25 NSE Instruments — ~6.71M Test Rows):
    6. universe25_class_and_perclass_f1.csv (Target class priors & per-class F1 health across all 25 stocks)

Usage:
    python -u -m src.experiments.nse_ml_completeness_diagnostics
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS_DIR
from src.data.nse_loader import NSE_CACHE_DIR, load_nse_symbol
from src.experiments.nse_error_and_upgrade import compute_v2_features
from src.experiments.nse_exp_broad_universe import UNIVERSE_25
from src.experiments.nse_final_25stock_verification import (
    SSF_FEATURE_NAMES,
    _tick_bucket as _tick_regime,
    build_ssf_features,
)
from src.microstructure.nse_pipeline import build_nse_segments, prepare_nse_segment
from src.models.tabular_trees import LightGBMLOB

OUT_DIR = ARTIFACTS_DIR / "nse" / "final_verification"
BUNDLE_DIR = ROOT / "app" / "bundle"
CLASS_NAMES = ["FLAT", "UP", "DOWN"]
LOGLOSS_UNIFORM = float(np.log(3.0))


def _norm_probs(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    p = np.nan_to_num(p, nan=1.0 / 3.0, posinf=1.0, neginf=0.0)
    p = np.clip(p, 1e-6, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    return p


def _fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    cm = np.bincount(y_true.astype(np.int64) * 3 + y_pred.astype(np.int64), minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2.0 * tp + fp + fn
    f1s = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1.0), 0.0)
    return float(f1s.mean()), f1s


def _multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    oh = np.eye(3, dtype=np.float64)[y_true.astype(np.int64)]
    return float(np.mean(np.sum((oh - probs) ** 2, axis=1)))


def _ece_10bin(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    conf = probs.max(axis=1)
    corr = (probs.argmax(axis=1) == y_true).astype(np.float64)
    edges = np.linspace(0.30, 1.0001, n_bins + 1)
    bidx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(bidx, minlength=n_bins).astype(np.float64)
    sum_conf = np.bincount(bidx, weights=conf, minlength=n_bins)
    sum_corr = np.bincount(bidx, weights=corr, minlength=n_bins)
    mask = counts > 0
    return float(np.sum(np.abs(sum_corr[mask] - sum_conf[mask])) / max(len(y_true), 1))


def _class_dist_row(symbol: str, split_name: str, y: np.ndarray, alpha_bps: List[float]) -> Dict[str, Any]:
    n = int(len(y))
    cnt = np.bincount(y.astype(np.int64), minlength=3)
    return {
        "symbol": symbol,
        "split": split_name,
        "n_rows": n,
        "count_flat": int(cnt[0]),
        "count_up": int(cnt[1]),
        "count_down": int(cnt[2]),
        "pct_flat": round(float(cnt[0]) / max(n, 1) * 100.0, 2),
        "pct_up": round(float(cnt[1]) / max(n, 1) * 100.0, 2),
        "pct_down": round(float(cnt[2]) / max(n, 1) * 100.0, 2),
        "alpha_bp_mean": round(float(np.mean(alpha_bps)), 4),
        "alpha_bp_std": round(float(np.std(alpha_bps)), 4) if len(alpha_bps) > 1 else 0.0,
    }


def _full_report_row(
    symbol: str,
    tier: str,
    model: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    seg_slices: List[slice],
    y_override_pred: np.ndarray | None = None,
) -> Dict[str, Any]:
    p = _norm_probs(probs)
    y_pred = y_override_pred if y_override_pred is not None else p.argmax(1)

    rep = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    ll = float(log_loss(y_true, p, labels=[0, 1, 2]))
    brier = _multiclass_brier(y_true, p)
    ece = _ece_10bin(y_true, p)

    try:
        roc_auc = float(roc_auc_score(y_true, p, multi_class="ovr", average="macro"))
    except ValueError:
        roc_auc = 0.5

    oh = np.eye(3, dtype=np.int64)[y_true.astype(np.int64)]
    ap_c = [float(average_precision_score(oh[:, c], p[:, c])) for c in range(3)]
    pr_macro = float(np.mean(ap_c))

    seg_f1s: List[float] = []
    seg_mccs: List[float] = []
    for sl in seg_slices:
        ys_s, ps_s = y_true[sl], y_pred[sl]
        seg_f1s.append(_fast_macro_f1(ys_s, ps_s)[0])
        seg_mccs.append(float(matthews_corrcoef(ys_s, ps_s)))

    f1_vals = [rep[c]["f1-score"] for c in CLASS_NAMES]
    return {
        "symbol": symbol,
        "tier": tier,
        "model": model,
        "n_test": int(len(y_true)),
        "prec_flat": round(rep["FLAT"]["precision"], 5),
        "rec_flat": round(rep["FLAT"]["recall"], 5),
        "f1_flat": round(rep["FLAT"]["f1-score"], 5),
        "sup_flat": int(rep["FLAT"]["support"]),
        "prec_up": round(rep["UP"]["precision"], 5),
        "rec_up": round(rep["UP"]["recall"], 5),
        "f1_up": round(rep["UP"]["f1-score"], 5),
        "sup_up": int(rep["UP"]["support"]),
        "prec_down": round(rep["DOWN"]["precision"], 5),
        "rec_down": round(rep["DOWN"]["recall"], 5),
        "f1_down": round(rep["DOWN"]["f1-score"], 5),
        "sup_down": int(rep["DOWN"]["support"]),
        "min_class_f1": round(min(f1_vals), 5),
        "macro_f1": round(rep["macro avg"]["f1-score"], 5),
        "macro_f1_std": round(float(np.std(seg_f1s)), 5),
        "weighted_f1": round(rep["weighted avg"]["f1-score"], 5),
        "accuracy": round(rep["accuracy"], 5),
        "balanced_accuracy": round(bal_acc, 5),
        "mcc": round(mcc, 5),
        "mcc_std": round(float(np.std(seg_mccs)), 5),
        "cohen_kappa": round(kappa, 5),
        "roc_auc_ovr": round(roc_auc, 5),
        "pr_auc_macro": round(pr_macro, 5),
        "pr_auc_flat": round(ap_c[0], 5),
        "pr_auc_up": round(ap_c[1], 5),
        "pr_auc_down": round(ap_c[2], 5),
        "log_loss": round(ll, 5),
        "brier_score": round(brier, 5),
        "ece_10bin": round(ece, 5),
    }


def _confusion_rows(symbol: str, model: str, y_true: np.ndarray, probs: np.ndarray) -> List[Dict[str, Any]]:
    pred = _norm_probs(probs).argmax(1)
    cm = confusion_matrix(y_true, pred, labels=[0, 1, 2])
    row_sums = np.maximum(cm.sum(axis=1), 1)
    col_sums = np.maximum(cm.sum(axis=0), 1)
    out: List[Dict[str, Any]] = []
    for i, cname in enumerate(CLASS_NAMES):
        out.append({
            "symbol": symbol,
            "model": model,
            "true_class": cname,
            "support": int(cm[i].sum()),
            "pred_FLAT_cnt": int(cm[i, 0]),
            "pred_UP_cnt": int(cm[i, 1]),
            "pred_DOWN_cnt": int(cm[i, 2]),
            "recall_FLAT_pct": round(float(cm[i, 0]) / row_sums[i] * 100.0, 2),
            "recall_UP_pct": round(float(cm[i, 1]) / row_sums[i] * 100.0, 2),
            "recall_DOWN_pct": round(float(cm[i, 2]) / row_sums[i] * 100.0, 2),
            "class_precision_pct": round(float(cm[i, i]) / col_sums[i] * 100.0, 2),
        })
    return out


def _run_part_a() -> None:
    print("[Part A] Running deep ML completeness audit on RELIANCE, NIFTY_FUT, IDEA...", flush=True)
    fd = np.load(BUNDLE_DIR / "full_days.npz", allow_pickle=True)

    dist_rows: List[Dict[str, Any]] = []
    rep_rows: List[Dict[str, Any]] = []
    cm_rows: List[Dict[str, Any]] = []
    gen_rows: List[Dict[str, Any]] = []
    imp_rows: List[Dict[str, Any]] = []

    for sym_in, asset in [("RELIANCE", "EQUITY"), ("NIFTY", "DERIVATIVE"), ("IDEA", "EQUITY")]:
        sym_data = load_nse_symbol(sym_in, asset=asset)
        sym = sym_data.symbol
        ssf_path = NSE_CACHE_DIR / f"{sym}_SSF_1s_5lvl.npz"
        has_ssf = (asset == "EQUITY") and ssf_path.exists()
        ssf_npz = np.load(ssf_path, allow_pickle=True) if has_ssf else None

        segs = build_nse_segments(sym_data, n_segments=4)
        y_tr_l, y_va_l, y_te_l = [], [], []
        alphas: List[float] = []
        seg_slices: List[slice] = []
        cursor = 0

        # Per-model saved test probs & generalization collectors
        saved_models = [
            ("Tier 0", "Naive Flat"),
            ("Tier 0", "Naive Momentum"),
            ("Tier 1", "Logistic Regression"),
            ("Tier 2", "Random Forest"),
            ("Tier 3", "LightGBM"),
            ("Tier 4", "LSTM"),
            ("Tier 5", "DeepLOB5"),
            ("Tier 5", "TLOB"),
        ]
        saved_probs: Dict[str, List[np.ndarray]] = {m: [] for _, m in saved_models}

        gen_track: Dict[str, Dict[str, List[Any]]] = {
            name: {"tr_f1": [], "va_f1": [], "te_f1": [], "te_c": [], "tr_ll": [], "va_ll": [], "te_ll": [], "tr_acc": [], "va_acc": [], "te_acc": [], "iter": []}
            for name in (["LightGBM v1 (25f)", "LightGBM v2 (38f)"] + (["LightGBM v2 + SSF (46f)"] if has_ssf else []))
        }

        p_v1_te_l, p_v2_te_l, p_v2_10_l = [], [], []
        p_s20_te_l, p_s10_te_l = [], []
        gain_acc: np.ndarray | None = None
        shap_acc: np.ndarray | None = None
        imp_feat_names: List[str] = []

        for s in range(4):
            seg = segs[s]
            prep = prepare_nse_segment(seg, horizons=[10, 20])
            sp20 = prep.splits(20)
            sp10 = prep.splits(10)
            tr20, va20, te20 = sp20["train"], sp20["val"], sp20["test"]
            y_all20, y_all10 = prep.labels(20), prep.labels(10)
            alphas.append(float(prep.alphas_dict[20]) * 1e4)

            y_tr, y_va, y_te = y_all20[tr20].astype(np.int64), y_all20[va20].astype(np.int64), y_all20[te20].astype(np.int64)
            y_tr_l.append(y_tr)
            y_va_l.append(y_va)
            y_te_l.append(y_te)
            seg_slices.append(slice(cursor, cursor + len(y_te)))
            cursor += len(y_te)
            dist_rows.append(_class_dist_row(sym, f"Seg{s}_Test", y_te, [alphas[-1]]))

            # Saved baseline tiers
            z = np.load(ARTIFACTS_DIR / "nse" / "models" / sym / "sessions" / f"seg{s}_k20_e10.npz", allow_pickle=True)
            assert np.array_equal(z["y_test"], y_te), f"{sym} seg{s} y_test mismatch!"
            for _, mname in saved_models:
                saved_probs[mname].append(z[f"prob_{mname}"].astype(np.float64))

            core = prep.core_indices
            X0 = np.ascontiguousarray(prep.tab_feat[:, core], dtype=np.float32)
            cnames = [prep.feat_names[i] for i in core]
            X_v2, v2_names = compute_v2_features(seg, X0, cnames)
            X_v2 = np.ascontiguousarray(X_v2, dtype=np.float32)

            def _fit_and_log(tag: str, Xmat: np.ndarray, k_hz: int = 20) -> Tuple[LightGBMLOB, np.ndarray]:
                sp_use = sp20 if k_hz == 20 else sp10
                y_use = y_all20 if k_hz == 20 else y_all10
                m = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=250, random_state=42 + s, n_jobs=-1)
                m.fit(Xmat[sp_use["train"]], y_use[sp_use["train"]], Xmat[sp_use["val"]], y_use[sp_use["val"]], early_stopping_rounds=20)
                if tag in gen_track:
                    p_tr = _norm_probs(m.predict_proba(Xmat[tr20]))
                    p_va = _norm_probs(m.predict_proba(Xmat[va20]))
                    p_te_m = _norm_probs(m.predict_proba(Xmat[te20]))
                    f1_tr, _ = _fast_macro_f1(y_tr, p_tr.argmax(1))
                    f1_va, _ = _fast_macro_f1(y_va, p_va.argmax(1))
                    f1_te, f1_te_c = _fast_macro_f1(y_te, p_te_m.argmax(1))
                    gt = gen_track[tag]
                    gt["tr_f1"].append(f1_tr)
                    gt["va_f1"].append(f1_va)
                    gt["te_f1"].append(f1_te)
                    gt["te_c"].append(f1_te_c)
                    gt["tr_acc"].append(float(np.mean(p_tr.argmax(1) == y_tr)))
                    gt["va_acc"].append(float(np.mean(p_va.argmax(1) == y_va)))
                    gt["te_acc"].append(float(np.mean(p_te_m.argmax(1) == y_te)))
                    gt["tr_ll"].append(float(log_loss(y_tr, p_tr, labels=[0, 1, 2])))
                    gt["va_ll"].append(float(log_loss(y_va, p_va, labels=[0, 1, 2])))
                    gt["te_ll"].append(float(log_loss(y_te, p_te_m, labels=[0, 1, 2])))
                    gt["iter"].append(int(m.model.booster_.best_iteration or 250))
                    return m, p_te_m
                return m, _norm_probs(m.predict_proba(Xmat[te20]))

            _, pv1 = _fit_and_log("LightGBM v1 (25f)", X0, 20)
            p_v1_te_l.append(pv1)
            m_v2, pv2 = _fit_and_log("LightGBM v2 (38f)", X_v2, 20)
            p_v2_te_l.append(pv2)
            _, pv2_10 = _fit_and_log("_v2_10", X_v2, 10)
            p_v2_10_l.append(pv2_10)

            best_booster = m_v2.model.booster_
            best_X_te = X_v2[te20]
            imp_feat_names = v2_names

            if has_ssf:
                X_ssf, _ = build_ssf_features(seg, ssf_npz)
                X_aug = np.ascontiguousarray(np.hstack([X_v2, X_ssf]), dtype=np.float32)
                m_s20, ps20 = _fit_and_log("LightGBM v2 + SSF (46f)", X_aug, 20)
                p_s20_te_l.append(ps20)
                _, ps10 = _fit_and_log("_ssf_10", X_aug, 10)
                p_s10_te_l.append(ps10)
                best_booster = m_s20.model.booster_
                best_X_te = X_aug[te20]
                imp_feat_names = v2_names + SSF_FEATURE_NAMES

            # Accumulate split gain & 2,000-row TreeSHAP per segment (= 8,000 total)
            g = best_booster.feature_importance(importance_type="gain").astype(np.float64)
            gain_acc = g if gain_acc is None else (gain_acc + g)
            step = max(1, len(best_X_te) // 2000)
            X_sub = best_X_te[::step][:2000]
            sv = shap.TreeExplainer(best_booster).shap_values(X_sub)
            if isinstance(sv, list):
                sv_abs = sum(np.mean(np.abs(c), axis=0) for c in sv)
            else:
                # shap >=0.45 returns (n, feats, classes)
                sv_abs = np.mean(np.abs(sv), axis=(0, 2))
            shap_acc = sv_abs if shap_acc is None else (shap_acc + sv_abs)

        y_tr_cat, y_va_cat, y_te_cat = np.concatenate(y_tr_l), np.concatenate(y_va_l), np.concatenate(y_te_l)
        dist_rows.extend([
            _class_dist_row(sym, "Train", y_tr_cat, alphas),
            _class_dist_row(sym, "Validation", y_va_cat, alphas),
            _class_dist_row(sym, "Test", y_te_cat, alphas),
        ])

        # Sanity baselines: Uniform Random (1/3 each) and Stratified Random (matching Train priors)
        rng = np.random.default_rng(42)
        p_unif = np.full((len(y_te_cat), 3), 1.0 / 3.0, dtype=np.float64)
        pred_unif = rng.choice([0, 1, 2], size=len(y_te_cat), p=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        rep_rows.append(_full_report_row(sym, "Sanity", "Uniform Random", y_te_cat, p_unif, seg_slices, y_override_pred=pred_unif))

        tr_priors = np.bincount(y_tr_cat, minlength=3).astype(np.float64) / len(y_tr_cat)
        p_strat = np.tile(tr_priors, (len(y_te_cat), 1))
        pred_strat = rng.choice([0, 1, 2], size=len(y_te_cat), p=tr_priors)
        rep_rows.append(_full_report_row(sym, "Sanity", "Stratified Random", y_te_cat, p_strat, seg_slices, y_override_pred=pred_strat))

        # All saved Tiers 0..5
        probs_map: Dict[str, np.ndarray] = {}
        for tier_lbl, mname in saved_models:
            pm = np.vstack(saved_probs[mname])
            probs_map[mname] = pm
            rep_rows.append(_full_report_row(sym, tier_lbl, mname, y_te_cat, pm, seg_slices))

        # Upgraded models
        pv1_cat = np.vstack(p_v1_te_l)
        pv2_cat = np.vstack(p_v2_te_l)
        pv2_mh = 0.5 * np.vstack(p_v2_10_l) + 0.5 * pv2_cat
        rep_rows.append(_full_report_row(sym, "Upgraded", "LightGBM v2 (38f)", y_te_cat, pv2_cat, seg_slices))
        rep_rows.append(_full_report_row(sym, "Upgraded", "LightGBM v2 + MH(10,20)", y_te_cat, pv2_mh, seg_slices))

        if has_ssf:
            ps20_cat = np.vstack(p_s20_te_l)
            ps_mh = 0.5 * np.vstack(p_s10_te_l) + 0.5 * ps20_cat
            rep_rows.append(_full_report_row(sym, "Upgraded", "LightGBM v2 + SSF (46f)", y_te_cat, ps20_cat, seg_slices))
            rep_rows.append(_full_report_row(sym, "Upgraded", "LightGBM v2 + SSF + MH(10,20)", y_te_cat, ps_mh, seg_slices))

        # Deployed Flagship Hybrid (from full_days.npz)
        assert np.array_equal(y_te_cat, fd[f"{sym}__y_true"].astype(np.int64)), f"{sym} full_days mismatch!"
        p_hyb = _norm_probs(fd[f"{sym}__prob_hyb_u8"].astype(np.float64) / 255.0)
        rep_rows.append(_full_report_row(sym, "Flagship", "Hybrid (v2[+SSF]+MH+TLOB)", y_te_cat, p_hyb, seg_slices))

        # Confusion matrices for key architectures
        for cm_name, cm_probs in [
            ("Logistic Regression", probs_map["Logistic Regression"]),
            ("LightGBM v1", probs_map["LightGBM"]),
            ("LightGBM v2 (38f)", pv2_cat),
            ("TLOB", probs_map["TLOB"]),
            ("Flagship Hybrid", p_hyb),
        ]:
            cm_rows.extend(_confusion_rows(sym, cm_name, y_te_cat, cm_probs))

        # Generalization gap summary across 4 segments
        for mname, gt in gen_track.items():
            f1c = np.vstack(gt["te_c"])
            gen_rows.append({
                "symbol": sym,
                "model": mname,
                "train_macro_f1": round(float(np.mean(gt["tr_f1"])), 5),
                "val_macro_f1": round(float(np.mean(gt["va_f1"])), 5),
                "test_macro_f1": round(float(np.mean(gt["te_f1"])), 5),
                "test_macro_f1_std": round(float(np.std(gt["te_f1"])), 5),
                "train_minus_test_f1_gap": round(float(np.mean(gt["tr_f1"]) - np.mean(gt["te_f1"])), 5),
                "val_minus_test_f1_gap": round(float(np.mean(gt["va_f1"]) - np.mean(gt["te_f1"])), 5),
                "test_f1_flat": round(float(f1c[:, 0].mean()), 5),
                "test_f1_up": round(float(f1c[:, 1].mean()), 5),
                "test_f1_down": round(float(f1c[:, 2].mean()), 5),
                "train_acc": round(float(np.mean(gt["tr_acc"])), 5),
                "val_acc": round(float(np.mean(gt["va_acc"])), 5),
                "test_acc": round(float(np.mean(gt["te_acc"])), 5),
                "train_logloss": round(float(np.mean(gt["tr_ll"])), 5),
                "val_logloss": round(float(np.mean(gt["va_ll"])), 5),
                "test_logloss": round(float(np.mean(gt["te_ll"])), 5),
                "best_iter_mean": round(float(np.mean(gt["iter"])), 1),
                "best_iter_std": round(float(np.std(gt["iter"])), 1),
            })

        # Top-15 upgraded Feature Importance (Gain & TreeSHAP)
        g_norm = gain_acc / max(float(gain_acc.sum()), 1e-12) * 100.0
        s_norm = shap_acc / max(float(shap_acc.sum()), 1e-12) * 100.0
        order = np.argsort(-s_norm)
        best_tag = "LightGBM v2 + SSF (46f)" if has_ssf else "LightGBM v2 (38f)"
        for rank, idx in enumerate(order[:15], 1):
            imp_rows.append({
                "symbol": sym,
                "model": best_tag,
                "rank": rank,
                "feature": imp_feat_names[idx],
                "shap_share_pct": round(float(s_norm[idx]), 2),
                "gain_share_pct": round(float(g_norm[idx]), 2),
            })

    pd.DataFrame(dist_rows).to_csv(OUT_DIR / "class_distribution_splits.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(OUT_DIR / "per_class_classification_reports.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(OUT_DIR / "confusion_matrices_detailed.csv", index=False)
    pd.DataFrame(gen_rows).to_csv(OUT_DIR / "train_val_test_generalization.csv", index=False)
    pd.DataFrame(imp_rows).to_csv(OUT_DIR / "v2_ssf_feature_importance.csv", index=False)
    print("[Part A] Saved all 5 completeness CSVs.", flush=True)


def _eval_one_stock_classes(args: Tuple[int, str, str, str, int]) -> Dict[str, Any]:
    _, sym_in, asset, sector, lgb_jobs = args
    t0 = time.time()
    sym_data = load_nse_symbol(sym_in, asset=asset)
    sym = sym_data.symbol
    mid_med = float(np.median((sym_data.lob_20[:, 0] + sym_data.lob_20[:, 2]) / 2.0))
    tick_bp = float(sym_data.tick_size) / max(mid_med, 1e-8) * 1e4

    ssf_path = NSE_CACHE_DIR / f"{sym}_SSF_1s_5lvl.npz"
    has_ssf = (asset == "EQUITY") and ssf_path.exists()
    ssf_npz = np.load(ssf_path, allow_pickle=True) if has_ssf else None

    segs = build_nse_segments(sym_data, n_segments=4)
    ys_te: List[np.ndarray] = []
    p_v1_l, p_best_20_l, p_best_10_l = [], [], []
    alphas: List[float] = []

    for s in range(4):
        seg = segs[s]
        prep = prepare_nse_segment(seg, horizons=[10, 20])
        sp20, sp10 = prep.splits(20), prep.splits(10)
        te20 = sp20["test"]
        y_all20, y_all10 = prep.labels(20), prep.labels(10)
        alphas.append(float(prep.alphas_dict[20]) * 1e4)
        ys_te.append(y_all20[te20].astype(np.int64))

        core = prep.core_indices
        X0 = np.ascontiguousarray(prep.tab_feat[:, core], dtype=np.float32)
        cnames = [prep.feat_names[i] for i in core]
        X_v2, _ = compute_v2_features(seg, X0, cnames)
        X_v2 = np.ascontiguousarray(X_v2, dtype=np.float32)

        m_v1 = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=250, random_state=42 + s, n_jobs=lgb_jobs)
        m_v1.fit(X0[sp20["train"]], y_all20[sp20["train"]], X0[sp20["val"]], y_all20[sp20["val"]], early_stopping_rounds=20)
        p_v1_l.append(m_v1.predict_proba(X0[te20]).astype(np.float64))

        if has_ssf:
            X_ssf, _ = build_ssf_features(seg, ssf_npz)
            X_use = np.ascontiguousarray(np.hstack([X_v2, X_ssf]), dtype=np.float32)
        else:
            X_use = X_v2

        m_20 = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=250, random_state=42 + s, n_jobs=lgb_jobs)
        m_20.fit(X_use[sp20["train"]], y_all20[sp20["train"]], X_use[sp20["val"]], y_all20[sp20["val"]], early_stopping_rounds=20)
        p_best_20_l.append(m_20.predict_proba(X_use[te20]).astype(np.float64))

        m_10 = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=250, random_state=42 + s, n_jobs=lgb_jobs)
        m_10.fit(X_use[sp10["train"]], y_all10[sp10["train"]], X_use[sp10["val"]], y_all10[sp10["val"]], early_stopping_rounds=20)
        p_best_10_l.append(m_10.predict_proba(X_use[te20]).astype(np.float64))

    y = np.concatenate(ys_te)
    n = int(len(y))
    cnt = np.bincount(y, minlength=3)
    p_v1 = _norm_probs(np.vstack(p_v1_l))
    p_best = _norm_probs(0.5 * np.vstack(p_best_10_l) + 0.5 * np.vstack(p_best_20_l))

    pred_v1 = p_v1.argmax(1)
    pred_best = p_best.argmax(1)
    v1_macro, v1_c = _fast_macro_f1(y, pred_v1)
    b_macro, b_c = _fast_macro_f1(y, pred_best)

    rep_b = classification_report(y, pred_best, labels=[0, 1, 2], target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    bal_acc = float(balanced_accuracy_score(y, pred_best))
    mcc = float(matthews_corrcoef(y, pred_best))
    ll = float(log_loss(y, p_best, labels=[0, 1, 2]))

    rec = {
        "symbol": sym,
        "asset": asset,
        "sector": sector,
        "tick_bp": round(tick_bp, 4),
        "tick_regime": _tick_regime(tick_bp),
        "mean_alpha_bp": round(float(np.mean(alphas)), 4),
        "n_test": n,
        "pct_flat": round(float(cnt[0]) / n * 100.0, 2),
        "pct_up": round(float(cnt[1]) / n * 100.0, 2),
        "pct_down": round(float(cnt[2]) / n * 100.0, 2),
        "v1_f1_flat": round(float(v1_c[0]), 5),
        "v1_f1_up": round(float(v1_c[1]), 5),
        "v1_f1_down": round(float(v1_c[2]), 5),
        "v1_min_class_f1": round(float(v1_c.min()), 5),
        "best_model": "v2 + SSF + MH(10,20)" if has_ssf else "v2 + MH(10,20)",
        "best_f1_flat": round(float(b_c[0]), 5),
        "best_f1_up": round(float(b_c[1]), 5),
        "best_f1_down": round(float(b_c[2]), 5),
        "best_min_class_f1": round(float(b_c.min()), 5),
        "best_macro_f1": round(b_macro, 5),
        "best_weighted_f1": round(rep_b["weighted avg"]["f1-score"], 5),
        "best_balanced_acc": round(bal_acc, 5),
        "best_mcc": round(mcc, 5),
        "best_log_loss": round(ll, 5),
    }
    print(
        f"[Part B] {sym:13s} | Priors {rec['pct_flat']:5.1f}/{rec['pct_up']:4.1f}/{rec['pct_down']:4.1f}% "
        f"| Best F1 (F/U/D) {rec['best_f1_flat']:.3f}/{rec['best_f1_up']:.3f}/{rec['best_f1_down']:.3f} "
        f"(min={rec['best_min_class_f1']:.3f}) | MCC={rec['best_mcc']:.3f} ({time.time()-t0:.1f}s)",
        flush=True,
    )
    return rec


def run_all() -> None:
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _run_part_a()

    print(f"\n[Part B] Auditing class priors & per-class F1 across all {len(UNIVERSE_25)} NSE stocks...", flush=True)
    tasks = [(i, s, a, sec, 3) for i, (s, a, sec) in enumerate(UNIVERSE_25)]
    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(_eval_one_stock_classes, t) for t in tasks]
        for fut in as_completed(futs):
            results.append(fut.result())

    df = pd.DataFrame(results).sort_values("tick_bp", ascending=False).reset_index(drop=True)
    df.to_csv(OUT_DIR / "universe25_class_and_perclass_f1.csv", index=False)

    print(f"\n[DONE in {time.time() - t_start:.1f}s] 25-Stock Per-Class F1 Health Summary:", flush=True)
    print(f"  Global min per-class F1 across all 25 Best Deployed models: {df['best_min_class_f1'].min():.4f} (threshold healthy > 0.20)", flush=True)
    print(f"  Mean Best F1 (FLAT / UP / DOWN): {df['best_f1_flat'].mean():.4f} / {df['best_f1_up'].mean():.4f} / {df['best_f1_down'].mean():.4f}", flush=True)
    print(f"  Mean Best MCC: {df['best_mcc'].mean():.4f}, Mean Best LogLoss: {df['best_log_loss'].mean():.4f} (< uniform {LOGLOSS_UNIFORM:.4f})", flush=True)


if __name__ == "__main__":
    run_all()
