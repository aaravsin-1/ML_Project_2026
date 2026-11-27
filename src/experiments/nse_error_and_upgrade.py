"""Empirical Error Anatomy, TLOB-vs-LightGBM Disagreement Analysis, and Multi-Scale v2 Upgrade.

Strictly causal & day-boundary-safe:
- Every new feature uses only t and past timestamps [day_start, t], resetting cleanly at day boundaries.
- Models are trained strictly on `sp["train"]`, early-stopped on `sp["val"]`, and evaluated on `sp["test"]`.
- Ensemble weights are tuned Leave-One-Segment-Out (trained on j != s, evaluated on segment s).

Outputs to `artifacts/nse/error_diagnostics/`:
  - disagreement_summary.csv
  - microstructure_buckets.csv
  - upgrade_ablation_summary.csv
  - selective_conviction_curve.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS_DIR
from src.data.nse_loader import load_nse_symbol
from src.microstructure.nse_pipeline import (
    NSESegment,
    _day_smoothed_forward_returns,
    build_nse_segments,
    prepare_nse_segment,
)
from src.models.tabular_trees import LightGBMLOB

HORIZON_K = 20
OUT_DIR = ARTIFACTS_DIR / "nse" / "error_diagnostics"

V2_EXTRA_NAMES: List[str] = [
    "mid_drift_5s_ticks",
    "mid_drift_15s_ticks",
    "mid_drift_30s_ticks",
    "mid_drift_60s_ticks",
    "ofi_mean_10s",
    "ofi_mean_30s",
    "ofi_mean_60s",
    "imb1_mean_20s",
    "imb1_vel_20s",
    "vol_10s_ticks",
    "vol_30s_ticks",
    "spread_mean_20s",
    "day_progress",
]

V2_EXTRA_FEATURE_META: Dict[str, Dict[str, str]] = {
    "mid_drift_5s_ticks": {"label": "5s Price Drift (ticks)", "group": "Multi-Scale Drift", "desc": "Signed mid-price move over the past 5 seconds in ticks"},
    "mid_drift_15s_ticks": {"label": "15s Price Drift (ticks)", "group": "Multi-Scale Drift", "desc": "Signed mid-price move over the past 15 seconds in ticks"},
    "mid_drift_30s_ticks": {"label": "30s Price Drift (ticks)", "group": "Multi-Scale Drift", "desc": "Signed mid-price move over the past 30 seconds in ticks"},
    "mid_drift_60s_ticks": {"label": "60s Price Drift (ticks)", "group": "Multi-Scale Drift", "desc": "Signed mid-price move over the past 60 seconds in ticks"},
    "ofi_mean_10s": {"label": "10s Mean OFI", "group": "Multi-Scale OFI", "desc": "Trailing 10s within-day mean of L1 order flow"},
    "ofi_mean_30s": {"label": "30s Mean OFI", "group": "Multi-Scale OFI", "desc": "Trailing 30s within-day mean of L1 order flow"},
    "ofi_mean_60s": {"label": "60s Mean OFI", "group": "Multi-Scale OFI", "desc": "Trailing 60s within-day mean of L1 order flow"},
    "imb1_mean_20s": {"label": "20s Mean Imbalance", "group": "Multi-Scale OFI", "desc": "Trailing 20s within-day mean of L1 queue imbalance"},
    "imb1_vel_20s": {"label": "Imbalance Shock (vs 20s)", "group": "Multi-Scale OFI", "desc": "Current L1 imbalance minus its 20s trailing mean"},
    "vol_10s_ticks": {"label": "10s Realised Vol (ticks)", "group": "Volatility & Regime", "desc": "Trailing 10s mean absolute 1s mid step in ticks"},
    "vol_30s_ticks": {"label": "30s Realised Vol (ticks)", "group": "Volatility & Regime", "desc": "Trailing 30s mean absolute 1s mid step in ticks"},
    "spread_mean_20s": {"label": "20s Mean Spread (ticks)", "group": "Volatility & Regime", "desc": "Trailing 20s average bid-ask spread in ticks"},
    "day_progress": {"label": "Session Clock (IST)", "group": "Volatility & Regime", "desc": "Within-day progress from 09:15 (0.0) to 15:30 IST (1.0)"},
}


def compute_v2_features(
    seg: NSESegment, tab_core: np.ndarray, core_names: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Appends 13 strictly causal within-day features (Block A/B/C) to the 25 core features."""
    n = len(seg.lob_20)
    tick = max(float(seg.tick_size), 1e-8)
    mid = (seg.lob_20[:, 0] + seg.lob_20[:, 2]) / 2.0
    spread_ticks = (seg.lob_20[:, 0] - seg.lob_20[:, 2]) / tick
    imb1 = tab_core[:, core_names.index("imbalance_l1")].astype(np.float64)
    ofi1 = tab_core[:, core_names.index("ofi_l1")].astype(np.float64)

    extra = np.zeros((n, len(V2_EXTRA_NAMES)), dtype=np.float32)
    changes = np.concatenate([[0], np.nonzero(seg.day_id[1:] != seg.day_id[:-1])[0] + 1, [n]])

    for a, b in zip(changes[:-1], changes[1:]):
        m = mid[a:b]
        o1 = ofi1[a:b]
        i1 = imb1[a:b]
        sp = spread_ticks[a:b]
        idx = np.arange(b - a)

        # Block A: Multi-scale drift (5s, 15s, 30s, 60s), OFI means (10s, 30s, 60s), 20s imbalance state & velocity
        for col, w in enumerate((5, 15, 30, 60)):
            prev = np.maximum(0, idx - w)
            extra[a:b, col] = (m - m[prev]) / tick

        cs_o1 = np.cumsum(np.insert(o1, 0, 0.0))
        for j, w in enumerate((10, 30, 60)):
            win = np.minimum(idx + 1, w)
            extra[a:b, 4 + j] = (cs_o1[idx + 1] - cs_o1[idx + 1 - win]) / win

        cs_i1 = np.cumsum(np.insert(i1, 0, 0.0))
        win20 = np.minimum(idx + 1, 20)
        imb20 = (cs_i1[idx + 1] - cs_i1[idx + 1 - win20]) / win20
        extra[a:b, 7] = imb20
        extra[a:b, 8] = i1 - imb20

        # Block B: Realized volatility (10s, 30s) and trailing 20s spread
        dm1 = np.abs(np.diff(m, prepend=m[0])) / tick
        dm1[0] = 0.0
        cs_v = np.cumsum(np.insert(dm1, 0, 0.0))
        win10 = np.minimum(idx + 1, 10)
        win30 = np.minimum(idx + 1, 30)
        extra[a:b, 9] = (cs_v[idx + 1] - cs_v[idx + 1 - win10]) / win10
        extra[a:b, 10] = (cs_v[idx + 1] - cs_v[idx + 1 - win30]) / win30

        cs_sp = np.cumsum(np.insert(sp, 0, 0.0))
        extra[a:b, 11] = (cs_sp[idx + 1] - cs_sp[idx + 1 - win20]) / win20

        # Block C: Intraday session progress [0, ~1]
        extra[a:b, 12] = seg.pos_in_day[a:b] / 22500.0

    X_v2 = np.hstack([tab_core.astype(np.float32), extra])
    return X_v2, list(core_names) + V2_EXTRA_NAMES


def _metrics_row(symbol: str, stage: str, order: int, y: np.ndarray, probs: np.ndarray, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    pred = probs.argmax(1)
    f1_cls = f1_score(y, pred, labels=[0, 1, 2], average=None)
    p_clip = np.clip(probs, 1e-6, 1.0)
    p_clip /= p_clip.sum(axis=1, keepdims=True)
    row: Dict[str, Any] = {
        "symbol": symbol,
        "stage_order": order,
        "stage": stage,
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc_ovr": float(roc_auc_score(y, p_clip, multi_class="ovr")),
        "log_loss": float(log_loss(y, p_clip)),
        "f1_flat": float(f1_cls[0]),
        "f1_up": float(f1_cls[1]),
        "f1_down": float(f1_cls[2]),
    }
    if extra:
        row.update(extra)
    return row


def _loso_blend(ys: List[np.ndarray], p_lgb_list: List[np.ndarray], p_tlob_list: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """Leave-One-Segment-Out scalar weight search maximizing Macro-F1 with zero peeking."""
    oos_parts: List[np.ndarray] = []
    weights: List[float] = []
    for s in range(len(ys)):
        y_tr = np.concatenate([ys[j] for j in range(len(ys)) if j != s])
        pl_tr = np.concatenate([p_lgb_list[j] for j in range(len(ys)) if j != s])
        pt_tr = np.concatenate([p_tlob_list[j] for j in range(len(ys)) if j != s])
        best_w, best_f = 0.65, -1.0
        for w in np.linspace(0.0, 1.0, 21):
            f = f1_score(y_tr, (w * pl_tr + (1.0 - w) * pt_tr).argmax(1), average="macro")
            if f > best_f:
                best_f, best_w = float(f), float(w)
        weights.append(best_w)
        blend_s = best_w * p_lgb_list[s] + (1.0 - best_w) * p_tlob_list[s]
        blend_s = blend_s / np.maximum(blend_s.sum(axis=1, keepdims=True), 1e-8)
        oos_parts.append(blend_s)
    return np.vstack(oos_parts), float(np.mean(weights))


def run_diagnostics_and_upgrade() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    disagree_rows: List[Dict[str, Any]] = []
    bucket_rows: List[Dict[str, Any]] = []
    ablation_rows: List[Dict[str, Any]] = []
    conviction_rows: List[Dict[str, Any]] = []

    for sym_in, asset in [("RELIANCE", "EQUITY"), ("NIFTY", "DERIVATIVE"), ("IDEA", "EQUITY")]:
        print(f"[nse_error_and_upgrade] Loading {sym_in} ({asset})...")
        sym_data = load_nse_symbol(sym_in, asset=asset)
        sym = sym_data.symbol
        segs = build_nse_segments(sym_data, n_segments=4)

        ys: List[np.ndarray] = []
        p_base_l, p_tlob_l, p_ms_l, p_vol_l, p_v2_l = [], [], [], [], []
        p_dlob_l, p_rf_l, p_lr_l, p_lstm_l = [], [], [], []
        diag_feat_rows: List[Dict[str, np.ndarray]] = []
        close_flags: List[np.ndarray] = []

        for seg_id in range(4):
            seg = segs[seg_id]
            prep = prepare_nse_segment(seg, horizons=[HORIZON_K])
            sp = prep.splits(HORIZON_K)
            core = prep.core_indices
            X0 = prep.tab_feat[:, core]
            core_names = [prep.feat_names[i] for i in core]
            X_v2, v2_names = compute_v2_features(seg, X0, core_names)
            y_all = prep.labels(HORIZON_K)

            sess_path = ARTIFACTS_DIR / "nse" / "models" / sym / "sessions" / f"seg{seg_id}_k20_e10.npz"
            z = np.load(sess_path, allow_pickle=True)
            te = sp["test"]
            y_te = y_all[te]
            assert np.array_equal(y_te, z["y_test"]), f"Test alignment mismatch on {sym} seg {seg_id}"

            ys.append(y_te)
            p_base_l.append(z["prob_LightGBM"].astype(np.float64))
            p_tlob_l.append(z["prob_TLOB"].astype(np.float64))
            p_dlob_l.append(z["prob_DeepLOB5"].astype(np.float64))
            p_rf_l.append(z["prob_Random Forest"].astype(np.float64))
            p_lr_l.append(z["prob_Logistic Regression"].astype(np.float64))
            p_lstm_l.append(z["prob_LSTM"].astype(np.float64))

            # Train 34f (+Block A), 37f (+Block A+B), and 38f (+Block A+B+C = LightGBM v2)
            idx_a = list(range(25 + 9))
            idx_ab = list(range(25 + 12))
            for bucket_list, X_sub in [
                (p_ms_l, X_v2[:, idx_a]),
                (p_vol_l, X_v2[:, idx_ab]),
                (p_v2_l, X_v2),
            ]:
                m = LightGBMLOB(
                    num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=250, random_state=42 + seg_id
                )
                m.fit(X_sub[sp["train"]], y_all[sp["train"]], X_sub[sp["val"]], y_all[sp["val"]], early_stopping_rounds=20)
                bucket_list.append(m.predict_proba(X_sub[te]).astype(np.float64))

            fwd = _day_smoothed_forward_returns(
                (seg.lob_20[:, 0] + seg.lob_20[:, 2]) / 2.0, seg.day_id, HORIZON_K, smoothing=5
            )[te]
            alpha = float(prep.alphas_dict[HORIZON_K])
            Xv2_te = X_v2[te]
            diag_feat_rows.append({
                "abs_ret_over_alpha": np.abs(fwd) / max(alpha, 1e-8),
                "near_boundary": ((np.abs(fwd) > 0.5 * alpha) & (np.abs(fwd) < 1.5 * alpha)).astype(np.float64),
                "abs_imb1": np.abs(Xv2_te[:, v2_names.index("imbalance_l1")]),
                "spread_ticks": Xv2_te[:, v2_names.index("spread_ticks")],
                "abs_drift_30s_ticks": np.abs(Xv2_te[:, v2_names.index("mid_drift_30s_ticks")]),
                "abs_ofi_60s": np.abs(Xv2_te[:, v2_names.index("ofi_mean_60s")]),
                "vol_10s_ticks": Xv2_te[:, v2_names.index("vol_10s_ticks")],
                "imb_reversal": (
                    np.sign(Xv2_te[:, v2_names.index("imbalance_l1")])
                    != np.sign(Xv2_te[:, v2_names.index("imb1_mean_20s")])
                ).astype(np.float64),
            })
            # Late-day zero-volatility stall detection (~15:10-15:30 IST, pos >= 20,000 and flat vol)
            close_flags.append(
                (seg.pos_in_day[te] >= 20000) & (Xv2_te[:, v2_names.index("vol_10s_ticks")] <= 1e-6)
            )

        y = np.concatenate(ys)
        p_base = np.vstack(p_base_l)
        p_tlob = np.vstack(p_tlob_l)
        p_ms = np.vstack(p_ms_l)
        p_vol = np.vstack(p_vol_l)
        p_v2 = np.vstack(p_v2_l)
        p_old_blend, _ = _loso_blend(ys, p_base_l, p_tlob_l)
        p_hybrid, mean_w_lgb = _loso_blend(ys, p_v2_l, p_tlob_l)

        pred_l = p_base.argmax(1)
        pred_v2 = p_v2.argmax(1)
        pred_t = p_tlob.argmax(1)
        all6 = np.stack(
            [
                pred_l,
                pred_t,
                np.vstack(p_dlob_l).argmax(1),
                np.vstack(p_rf_l).argmax(1),
                np.vstack(p_lr_l).argmax(1),
                np.vstack(p_lstm_l).argmax(1),
            ],
            axis=1,
        )
        unanimous = np.all(all6 == all6[:, :1], axis=1)
        agree = pred_l == pred_t
        both_ok = (pred_l == y) & (pred_t == y)
        only_l = (pred_l == y) & (pred_t != y)
        only_t = (pred_l != y) & (pred_t == y)
        both_bad = (pred_l != y) & (pred_t != y)

        wrong_l = pred_l != y
        opp_rev = ((pred_l == 1) & (y == 2)) | ((pred_l == 2) & (y == 1))
        flat_conf = wrong_l & (~opp_rev)
        closing_mask = np.concatenate(close_flags)
        counts = np.bincount(y, minlength=3) / len(y)

        disagree_rows.append({
            "symbol": sym,
            "n_test": int(len(y)),
            "flat_pct": float(counts[0] * 100.0),
            "up_pct": float(counts[1] * 100.0),
            "down_pct": float(counts[2] * 100.0),
            "both_right_pct": float(both_ok.mean() * 100.0),
            "only_lgbm_pct": float(only_l.mean() * 100.0),
            "only_tlob_pct": float(only_t.mean() * 100.0),
            "both_wrong_pct": float(both_bad.mean() * 100.0),
            "oracle_union_pct": float((both_ok | only_l | only_t).mean() * 100.0),
            "agree_pct": float(agree.mean() * 100.0),
            "agree_acc": float(accuracy_score(y[agree], pred_l[agree])),
            "agree_f1": float(f1_score(y[agree], pred_l[agree], average="macro")),
            "all6_agree_pct": float(unanimous.mean() * 100.0),
            "all6_agree_acc": float(accuracy_score(y[unanimous], pred_l[unanimous])),
            "flat_boundary_err_share": float(flat_conf.sum() / max(int(wrong_l.sum()), 1) * 100.0),
            "opp_dir_err_share": float(opp_rev.sum() / max(int(wrong_l.sum()), 1) * 100.0),
            "closing_stall_n": int(closing_mask.sum()),
            "closing_stall_v1_acc": float(accuracy_score(y[closing_mask], pred_l[closing_mask])) if closing_mask.sum() > 0 else np.nan,
            "closing_stall_v2_acc": float(accuracy_score(y[closing_mask], pred_v2[closing_mask])) if closing_mask.sum() > 0 else np.nan,
            "oos_lgbm_weight": round(mean_w_lgb, 3),
        })

        # Microstructure bucket averages
        merged_diag = {k: np.concatenate([d[k] for d in diag_feat_rows]) for k in diag_feat_rows[0]}
        for bname, bmask in [
            ("BothRight", both_ok),
            ("OnlyLGBM", only_l),
            ("OnlyTLOB", only_t),
            ("BothWrong", both_bad),
        ]:
            bucket_rows.append({
                "symbol": sym,
                "bucket": bname,
                "n": int(bmask.sum()),
                "share_pct": float(bmask.mean() * 100.0),
                "abs_ret_over_alpha": float(np.nanmedian(merged_diag["abs_ret_over_alpha"][bmask])),
                "near_boundary_pct": float(np.mean(merged_diag["near_boundary"][bmask]) * 100.0),
                "abs_imb1": float(np.mean(merged_diag["abs_imb1"][bmask])),
                "spread_ticks": float(np.mean(merged_diag["spread_ticks"][bmask])),
                "abs_drift_30s_ticks": float(np.mean(merged_diag["abs_drift_30s_ticks"][bmask])),
                "abs_ofi_60s": float(np.mean(merged_diag["abs_ofi_60s"][bmask])),
                "vol_10s_ticks": float(np.mean(merged_diag["vol_10s_ticks"][bmask])),
                "imb_reversal_pct": float(np.mean(merged_diag["imb_reversal"][bmask]) * 100.0),
            })

        # 7-stage progression
        stages = [
            (1, "Baseline LightGBM (25f, 5s lags)", p_base, {"oos_lgbm_weight": 1.0}),
            (2, "TLOB (100s Transformer)", p_tlob, {"oos_lgbm_weight": 0.0}),
            (3, "Old LGBM + TLOB (OOS Blend)", p_old_blend, None),
            (4, "LGBM + Multi-Scale 10-60s (34f)", p_ms, None),
            (5, "LGBM + Volatility & Spread20s (37f)", p_vol, None),
            (6, "LightGBM v2 Full (38f)", p_v2, {"oos_lgbm_weight": 1.0}),
            (7, "Hybrid Ensemble (LGBM v2 + TLOB)", p_hybrid, {"oos_lgbm_weight": round(mean_w_lgb, 3)}),
        ]
        for order, sname, probs, ext in stages:
            ablation_rows.append(_metrics_row(sym, sname, order, y, probs, ext))

        # Selective conviction curve on Hybrid Ensemble
        max_p = p_hybrid.max(axis=1)
        pred_h = p_hybrid.argmax(axis=1)
        for tau in [0.34, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
            m_cov = max_p >= tau
            if int(m_cov.sum()) >= 50:
                conviction_rows.append({
                    "symbol": sym,
                    "threshold": float(tau),
                    "coverage_pct": float(m_cov.mean() * 100.0),
                    "accuracy": float(accuracy_score(y[m_cov], pred_h[m_cov])),
                    "macro_f1": float(f1_score(y[m_cov], pred_h[m_cov], average="macro")),
                    "n_rows": int(m_cov.sum()),
                })

    pd.DataFrame(disagree_rows).to_csv(OUT_DIR / "disagreement_summary.csv", index=False)
    pd.DataFrame(bucket_rows).to_csv(OUT_DIR / "microstructure_buckets.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(OUT_DIR / "upgrade_ablation_summary.csv", index=False)
    pd.DataFrame(conviction_rows).to_csv(OUT_DIR / "selective_conviction_curve.csv", index=False)
    print(f"[nse_error_and_upgrade] Saved 4 CSVs to {OUT_DIR}")


if __name__ == "__main__":
    run_diagnostics_and_upgrade()
