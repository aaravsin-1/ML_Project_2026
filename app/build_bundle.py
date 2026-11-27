"""Builds the self-contained deployment bundle in `app/bundle/`.

Outputs (under ~5 MB total, pickle-free):
  - `app/bundle/models.json`    : Portable LightGBM booster strings, Logistic Regression weights,
                                  5-level MLOFI Block PCA parameters, and training baselines
                                  for RELIANCE, NIFTY_FUT (seg 0 & seg 3), and IDEA.
  - `app/bundle/replays.npz`    : 900-second held-out test replays from 64-day NSE data with exact
                                  saved predictions from all 6 ML tiers (LightGBM, TLOB, DeepLOB5,
                                  LSTM, Random Forest, Logistic Regression, Momentum).
  - `app/bundle/summaries.json` : All verified experiment summary tables for the UI views.

Run once from repository root:
    python -m app.build_bundle
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS_DIR
from src.data.nse_loader import NSE_CACHE_DIR, load_nse_symbol
from src.experiments.nse_error_and_upgrade import (
    OUT_DIR as DIAG_DIR,
    V2_EXTRA_FEATURE_META,
    compute_v2_features,
)
from src.experiments.nse_final_25stock_verification import (
    OUT_DIR as FINAL_VERIF_DIR,
    SSF_FEATURE_META,
    SSF_FEATURE_NAMES,
    build_ssf_features,
)
from src.microstructure.nse_pipeline import (
    _day_smoothed_forward_returns,
    build_nse_segments,
    prepare_nse_segment,
)
from src.microstructure.ofi import OrderFlowImbalance
from src.models.baselines import LogisticRegressionBaseline
from src.models.tabular_trees import LightGBMLOB

BUNDLE_DIR = ROOT / "app" / "bundle"
IST = timezone(timedelta(hours=5, minutes=30))
HORIZON_K = 20
WINDOW_LEN = 900

FEATURE_META: Dict[str, Dict[str, str]] = {
    "imbalance_l1": {"label": "L1 Queue Imbalance", "group": "Queue Imbalance", "desc": "(BidQ1 - AskQ1) / (BidQ1 + AskQ1)"},
    "imbalance_l3": {"label": "L1-3 Queue Imbalance", "group": "Queue Imbalance", "desc": "Cumulative depth ratio across top 3 levels"},
    "imbalance_l5": {"label": "L1-5 Queue Imbalance", "group": "Queue Imbalance", "desc": "Cumulative depth ratio across all 5 levels"},
    "spread_ticks": {"label": "Spread (ticks)", "group": "Spread & Micro", "desc": "Best Ask - Best Bid measured in integer price ticks"},
    "micro_deviation": {"label": "Micro-Price Gap", "group": "Spread & Micro", "desc": "Imbalance-weighted mid minus plain mid price"},
    "slope_bid": {"label": "Bid Slope", "group": "Book Slopes", "desc": "Price drop per unit of cumulative bid depth"},
    "slope_ask": {"label": "Ask Slope", "group": "Book Slopes", "desc": "Price rise per unit of cumulative ask depth"},
    "vpin": {"label": "VPIN Toxicity", "group": "Flow Toxicity", "desc": "Order-flow one-sidedness over recent volume buckets"},
    "ofi_l1": {"label": "L1 Order Flow (OFI)", "group": "Order Flow (OFI)", "desc": "Net change in resting L1 buy vs sell demand"},
    "ofi_pca1": {"label": "MLOFI Block PC1", "group": "Block PCA (OFI)", "desc": "1st orthogonal order-flow factor across 5 levels (train PCA)"},
    "ofi_pca2": {"label": "MLOFI Block PC2", "group": "Block PCA (OFI)", "desc": "2nd orthogonal order-flow factor (top vs deep rotation)"},
    "ofi_pca3": {"label": "MLOFI Block PC3", "group": "Block PCA (OFI)", "desc": "3rd orthogonal order-flow factor (curvature)"},
    "mlofi_lvl_2": {"label": "L2 Order Flow", "group": "Order Flow (OFI)", "desc": "Net queue flow at level 2"},
    "mlofi_lvl_3": {"label": "L3 Order Flow", "group": "Order Flow (OFI)", "desc": "Net queue flow at level 3"},
    "mlofi_lvl_4": {"label": "L4 Order Flow", "group": "Order Flow (OFI)", "desc": "Net queue flow at level 4"},
    "mlofi_lvl_5": {"label": "L5 Order Flow", "group": "Order Flow (OFI)", "desc": "Net queue flow at level 5"},
    "ofi_lag_1": {"label": "L1 OFI (t-1s)", "group": "Causal L1 Lags", "desc": "1-second lagged L1 order flow"},
    "imbalance_lag_1": {"label": "L1 Imbalance (t-1s)", "group": "Causal L1 Lags", "desc": "1-second lagged L1 queue imbalance"},
    "micro_dev_lag_1": {"label": "Micro Gap (t-1s)", "group": "Causal L1 Lags", "desc": "1-second lagged micro-price deviation"},
    "ofi_lag_2": {"label": "L1 OFI (t-2s)", "group": "Causal L1 Lags", "desc": "2-second lagged L1 order flow"},
    "imbalance_lag_2": {"label": "L1 Imbalance (t-2s)", "group": "Causal L1 Lags", "desc": "2-second lagged L1 queue imbalance"},
    "micro_dev_lag_2": {"label": "Micro Gap (t-2s)", "group": "Causal L1 Lags", "desc": "2-second lagged micro-price deviation"},
    "ofi_lag_5": {"label": "L1 OFI (t-5s)", "group": "Causal L1 Lags", "desc": "5-second lagged L1 order flow"},
    "imbalance_lag_5": {"label": "L1 Imbalance (t-5s)", "group": "Causal L1 Lags", "desc": "5-second lagged L1 queue imbalance"},
    "micro_dev_lag_5": {"label": "Micro Gap (t-5s)", "group": "Causal L1 Lags", "desc": "5-second lagged micro-price deviation"},
}

REPLAY_TARGETS = [
    {
        "key": "RELIANCE",
        "symbol": "RELIANCE",
        "asset": "EQUITY",
        "segment_id": 0,
        "title": "RELIANCE Industries",
        "subtitle": "Medium-tick flagship equity (0.80 bp tick) - Balanced tree vs transformer regime",
        "badge": "Medium Tick",
    },
    {
        "key": "NIFTY_FUT",
        "symbol": "NIFTY",
        "asset": "DERIVATIVE",
        "segment_id": 0,
        "title": "NIFTY 50 Index Futures",
        "subtitle": "Tiny-tick index futures (0.043 bp tick) - Deeper L2-5 queues drive prediction",
        "badge": "Tiny Tick",
    },
    {
        "key": "NIFTY_FUT_SEG3",
        "symbol": "NIFTY",
        "asset": "DERIVATIVE",
        "segment_id": 3,
        "title": "NIFTY Seg 3 (+10.7% Rally Shift)",
        "subtitle": "Covariate-shift stress test: static LSTM/DeepLOB5 fail while LightGBM & TLOB hold steady",
        "badge": "Distribution Shift",
    },
    {
        "key": "IDEA",
        "symbol": "IDEA",
        "asset": "EQUITY",
        "segment_id": 0,
        "title": "Vodafone Idea (IDEA)",
        "subtitle": "Ultra-large-tick equity (7.41 bp tick, 69% FLAT) - Exposes the Accuracy Trap",
        "badge": "Ultra-Large Tick",
    },
]


def _clean_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Converts a DataFrame into JSON-safe records (NaN -> None)."""
    clean = df.where(pd.notnull(df), None)
    records = clean.to_dict(orient="records")
    for r in records:
        for k, v in list(r.items()):
            if isinstance(v, (np.floating, float)) and not np.isfinite(v):
                r[k] = None
            elif isinstance(v, (np.integer,)):
                r[k] = int(v)
            elif isinstance(v, (np.floating,)):
                r[k] = float(v)
    return records


def _select_active_window(mid: np.ndarray, p_lgb: np.ndarray, window: int = WINDOW_LEN) -> int:
    """Picks an informative 900 s replay slice with active price movement and high-confidence signals."""
    conf = np.maximum(p_lgb[:, 1], p_lgb[:, 2])
    ret20 = np.abs(np.roll(mid, -20) - mid) / np.maximum(mid, 1e-8)
    ret20[-20:] = 0.0
    score = conf + 150.0 * ret20
    cs = np.cumsum(np.insert(score, 0, 0.0))
    # Avoid very end where horizon tails zero out
    max_start = max(1, len(mid) - window - 30)
    win_scores = cs[window:max_start + window] - cs[:max_start]
    return int(np.argmax(win_scores))


def _pick_steps(mask: np.ndarray, score: np.ndarray, window: int = WINDOW_LEN, k: int = 12, min_gap: int = 15) -> List[int]:
    """Selects up to k spaced step indices inside [90, window - 30] where mask is True."""
    valid = mask.copy()
    valid[:90] = False
    valid[window - 30:] = False
    cand = np.where(valid)[0]
    if len(cand) == 0:
        return []
    order = cand[np.argsort(-score[cand])]
    chosen: List[int] = []
    for idx_c in order:
        if all(abs(int(idx_c) - prev) >= min_gap for prev in chosen):
            chosen.append(int(idx_c))
        if len(chosen) >= k:
            break
    return chosen


def _companion_futures_spec(sym: str, asset: str) -> tuple[Path | None, str]:
    """Returns companion cross-market 5-level futures cache path and human label."""
    if asset == "EQUITY":
        p = NSE_CACHE_DIR / f"{sym}_SSF_1s_5lvl.npz"
        if p.exists():
            return p, f"{sym} Single-Stock Futures (SSF)"
    elif sym == "NIFTY_FUT":
        p = NSE_CACHE_DIR / "BANKNIFTY_FUT_1s_5lvl.npz"
        if p.exists():
            return p, "BANKNIFTY Cross-Index Futures"
    return None, "None"


def build_bundle() -> None:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    models_out: Dict[str, Any] = {}
    replay_arrays: Dict[str, np.ndarray] = {}
    replay_catalogue: List[Dict[str, Any]] = []
    full_days_arrays: Dict[str, np.ndarray] = {}
    full_feature_meta = {**FEATURE_META, **V2_EXTRA_FEATURE_META, **SSF_FEATURE_META}

    # Load verified OOS ensemble blend weights computed in nse_error_and_upgrade.py
    w_map: Dict[str, float] = {"RELIANCE": 0.75, "NIFTY_FUT": 0.875, "IDEA": 0.85}
    disagree_csv = DIAG_DIR / "disagreement_summary.csv"
    if disagree_csv.exists():
        for _, r in pd.read_csv(disagree_csv).iterrows():
            w_map[str(r["symbol"])] = float(r["oos_lgbm_weight"])

    cache_sym: Dict[str, Any] = {}
    cache_segs: Dict[str, Any] = {}

    # Pre-compute upgraded flagship predictions (v2 + 8 Cross-Market Futures + Multi-Horizon 10s/20s + TLOB)
    # across all 12 held-out test days (4 segments x 3 test days) for Full-Day & Multi-Day Explorer.
    for sym_in, asset in [("RELIANCE", "EQUITY"), ("NIFTY", "DERIVATIVE"), ("IDEA", "EQUITY")]:
        cache_id = f"{sym_in}_{asset}"
        print(f"[build_bundle] Loading cached {cache_id} for 12-day bundle...")
        data_sym = load_nse_symbol(sym_in, asset=asset)
        sym = data_sym.symbol
        segs = build_nse_segments(data_sym, n_segments=4)
        cache_sym[cache_id] = data_sym
        cache_segs[cache_id] = segs

        fut_path, comp_label = _companion_futures_spec(sym, asset)
        fut_npz = np.load(fut_path, allow_pickle=True) if fut_path is not None else None
        w_blend = w_map.get(sym, 0.75)

        day_dates: List[str] = []
        bounds: List[int] = [0]
        sec_list, mid_list, y_list, u8_list = [], [], [], []

        for s_id, seg in enumerate(segs):
            prep = prepare_nse_segment(seg, horizons=[10, 20])
            sp20, sp10 = prep.splits(20), prep.splits(10)
            te20 = sp20["test"]
            core = prep.core_indices
            X_core = prep.tab_feat[:, core]
            cnames = [prep.feat_names[i] for i in core]
            X_v2, _ = compute_v2_features(seg, X_core, cnames)
            if fut_npz is not None:
                X_ssf, _ = build_ssf_features(seg, fut_npz)
                X_full = np.ascontiguousarray(np.hstack([X_v2, X_ssf]), dtype=np.float32)
            else:
                X_full = np.ascontiguousarray(X_v2, dtype=np.float32)

            m20 = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=250, random_state=42 + s_id)
            m20.fit(X_full[sp20["train"]], prep.labels(20)[sp20["train"]], X_full[sp20["val"]], prep.labels(20)[sp20["val"]], early_stopping_rounds=20)
            m10 = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=250, random_state=42 + s_id)
            m10.fit(X_full[sp10["train"]], prep.labels(10)[sp10["train"]], X_full[sp10["val"]], prep.labels(10)[sp10["val"]], early_stopping_rounds=20)

            p_mh = 0.5 * m10.predict_proba(X_full[te20]).astype(np.float64) + 0.5 * m20.predict_proba(X_full[te20]).astype(np.float64)
            sess_npz = np.load(ARTIFACTS_DIR / "nse" / "models" / sym / "sessions" / f"seg{s_id}_k20_e10.npz", allow_pickle=True)
            assert np.array_equal(prep.labels(20)[te20], sess_npz["y_test"]), f"Test alignment error {sym} seg{s_id}"
            p_hyb = w_blend * p_mh + (1.0 - w_blend) * sess_npz["prob_TLOB"].astype(np.float64)
            p_hyb /= np.maximum(p_hyb.sum(axis=1, keepdims=True), 1e-8)

            te_days = seg.day_id[te20]
            mid_seg = (seg.lob_20[te20, 0] + seg.lob_20[te20, 2]) / 2.0
            sec_seg = seg.pos_in_day[te20]
            y_seg = prep.labels(20)[te20]

            for d_loc in np.unique(te_days):
                dm = te_days == d_loc
                day_dates.append(str(seg.days[int(d_loc)]))
                sec_list.append(sec_seg[dm].astype(np.int16))
                mid_list.append(mid_seg[dm].astype(np.float32))
                y_list.append(y_seg[dm].astype(np.int8))
                u8_list.append(np.clip(np.round(p_hyb[dm] * 255.0), 0, 255).astype(np.uint8))
                bounds.append(bounds[-1] + int(dm.sum()))

        full_days_arrays[f"{sym}__dates"] = np.array(day_dates)
        full_days_arrays[f"{sym}__bounds"] = np.array(bounds, dtype=np.int32)
        full_days_arrays[f"{sym}__sec"] = np.concatenate(sec_list)
        full_days_arrays[f"{sym}__mid"] = np.concatenate(mid_list)
        full_days_arrays[f"{sym}__y_true"] = np.concatenate(y_list)
        full_days_arrays[f"{sym}__prob_hyb_u8"] = np.vstack(u8_list)

    for spec in REPLAY_TARGETS:
        key = spec["key"]
        sym_in = spec["symbol"]
        asset = spec["asset"]
        seg_id = spec["segment_id"]
        cache_id = f"{sym_in}_{asset}"
        if cache_id not in cache_sym:
            print(f"[build_bundle] Loading cached {cache_id}...")
            data_sym = load_nse_symbol(sym_in, asset=asset)
            cache_sym[cache_id] = data_sym
            cache_segs[cache_id] = build_nse_segments(data_sym, n_segments=4)

        data_sym = cache_sym[cache_id]
        seg = cache_segs[cache_id][seg_id]
        prep = prepare_nse_segment(seg, horizons=[10, 20])
        sp = prep.splits(HORIZON_K)
        sp10 = prep.splits(10)
        core = prep.core_indices
        X_core = prep.tab_feat[:, core]
        core_names = [prep.feat_names[i] for i in core]
        X_v2, v2_names = compute_v2_features(seg, X_core, core_names)
        y_all = prep.labels(HORIZON_K)
        y_all10 = prep.labels(10)

        X_tr, y_tr = X_core[sp["train"]], y_all[sp["train"]]
        X_va, y_va = X_core[sp["val"]], y_all[sp["val"]]
        X_te, y_te = X_core[sp["test"]], y_all[sp["test"]]

        Xv2_tr, Xv2_va, Xv2_te = X_v2[sp["train"]], X_v2[sp["val"]], X_v2[sp["test"]]

        fut_path, comp_label = _companion_futures_spec(data_sym.symbol, asset)
        if fut_path is not None:
            fut_npz = np.load(fut_path, allow_pickle=True)
            X_ssf, _ = build_ssf_features(seg, fut_npz)
        else:
            X_ssf = np.zeros((len(seg.lob_20), len(SSF_FEATURE_NAMES)), dtype=np.float32)
        X_flag = np.ascontiguousarray(np.hstack([X_v2, X_ssf]), dtype=np.float32)
        flag_names = list(v2_names) + list(SSF_FEATURE_NAMES)
        Xf_tr, Xf_va, Xf_te = X_flag[sp["train"]], X_flag[sp["val"]], X_flag[sp["test"]]

        # Verify exact alignment with saved GPU/benchmark session file
        sess_path = ARTIFACTS_DIR / "nse" / "models" / data_sym.symbol / "sessions" / f"seg{seg_id}_k20_e10.npz"
        sess_npz = np.load(sess_path, allow_pickle=True)
        assert np.array_equal(y_te, sess_npz["y_test"]), f"Label mismatch on {key}"

        # Fit portable LightGBM v1 (25f), v2 (38f), Flagship 46f (k=20 & k=10), & LogReg
        print(f"[build_bundle] Training portable v1/v2/Flagship-46f + LogReg for {key} ({len(X_tr):,} train rows, {comp_label})...")
        lgb_model = LightGBMLOB(
            num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=200, random_state=42 + seg_id
        )
        lgb_model.fit(X_tr, y_tr, X_va, y_va, early_stopping_rounds=15)

        lgb_v2 = LightGBMLOB(
            num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=250, random_state=42 + seg_id
        )
        lgb_v2.fit(Xv2_tr, y_tr, Xv2_va, y_va, early_stopping_rounds=20)
        p_v2_te = lgb_v2.predict_proba(Xv2_te).astype(np.float64)

        lgb_flag20 = LightGBMLOB(
            num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=250, random_state=42 + seg_id
        )
        lgb_flag20.fit(Xf_tr, y_tr, Xf_va, y_va, early_stopping_rounds=20)
        lgb_flag10 = LightGBMLOB(
            num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=250, random_state=42 + seg_id
        )
        lgb_flag10.fit(
            X_flag[sp10["train"]], y_all10[sp10["train"]],
            X_flag[sp10["val"]], y_all10[sp10["val"]],
            early_stopping_rounds=20,
        )
        p_flag_te = 0.5 * lgb_flag10.predict_proba(Xf_te).astype(np.float64) + 0.5 * lgb_flag20.predict_proba(Xf_te).astype(np.float64)

        w_blend = w_map.get(data_sym.symbol, 0.75)
        p_hybrid_te = w_blend * p_flag_te + (1.0 - w_blend) * sess_npz["prob_TLOB"].astype(np.float64)
        p_hybrid_te /= np.maximum(p_hybrid_te.sum(axis=1, keepdims=True), 1e-8)

        lr_model = LogisticRegressionBaseline(C=1.0, max_iter=1000, random_state=42 + seg_id)
        lr_model.fit(X_tr, y_tr)

        # Extract exact MLOFI Block PCA parameters fitted on training days only
        mlofi_tr = OrderFlowImbalance.compute_multi_level_ofi(seg.lob_20[:prep.train_end], num_levels=5)
        mlofi_tr[seg.pos_in_day[:prep.train_end] == 0] = 0.0
        _, pca_model = OrderFlowImbalance.reduce_mlofi_pca(mlofi_tr, n_components=3, pca_model=None)

        # Compute empirical training distribution stats for the interactive What-If Sandbox
        lob_tr = seg.lob_20[sp["train"]]
        mid_tr = (lob_tr[:, 0] + lob_tr[:, 2]) / 2.0
        ask_q_med = np.median(lob_tr[:, 1::4], axis=0)
        bid_q_med = np.median(lob_tr[:, 3::4], axis=0)
        ofi_idx = core_names.index("ofi_l1")
        vpin_idx = core_names.index("vpin")
        ofi_std = float(np.std(X_tr[:, ofi_idx])) or 1.0
        vpin_q = [float(np.quantile(X_tr[:, vpin_idx], q)) for q in (0.1, 0.5, 0.9)]
        vol10_idx = v2_names.index("vol_10s_ticks")
        vol30_idx = v2_names.index("vol_30s_ticks")
        vol10_q = [float(np.quantile(Xv2_tr[:, vol10_idx], q)) for q in (0.1, 0.5, 0.9)]
        vol30_q = [float(np.quantile(Xv2_tr[:, vol30_idx], q)) for q in (0.1, 0.5, 0.9)]
        slope_q = {
            "bid": float(np.median(X_tr[:, core_names.index("slope_bid")])),
            "ask": float(np.median(X_tr[:, core_names.index("slope_ask")])),
        }
        ssf_ret5_std = float(np.std(Xf_tr[:, flag_names.index("ssf_ret_5s_bp")])) or 2.0
        lead5_std = float(np.std(Xf_tr[:, flag_names.index("lead_5s_bp")])) or 1.5
        basis_dev_std = float(np.std(Xf_tr[:, flag_names.index("basis_dev_bp")])) or 1.5

        models_out[key] = {
            "key": key,
            "symbol": data_sym.symbol,
            "asset": asset,
            "segment_id": seg_id,
            "title": spec["title"],
            "subtitle": spec["subtitle"],
            "badge": spec["badge"],
            "companion_source": comp_label,
            "feat_names": core_names,
            "feat_names_v2": v2_names,
            "feat_names_flagship": flag_names,
            "lgbm_model_str": lgb_model.model.booster_.model_to_string(),
            "lgbm_best_iter": int(getattr(lgb_model.model, "best_iteration_", 0) or 0),
            "lgbm_v2_model_str": lgb_v2.model.booster_.model_to_string(),
            "lgbm_v2_best_iter": int(getattr(lgb_v2.model, "best_iteration_", 0) or 0),
            "lgbm_flagship_20_model_str": lgb_flag20.model.booster_.model_to_string(),
            "lgbm_flagship_10_model_str": lgb_flag10.model.booster_.model_to_string(),
            "lgbm_flagship_20_best_iter": int(getattr(lgb_flag20.model, "best_iteration_", 0) or 0),
            "lgbm_flagship_10_best_iter": int(getattr(lgb_flag10.model, "best_iteration_", 0) or 0),
            "hybrid_lgbm_weight": float(w_blend),
            "logreg": {
                "mean": lr_model.mean_.astype(float).tolist(),
                "std": lr_model.std_.astype(float).tolist(),
                "coef": lr_model.model.coef_.astype(float).tolist(),
                "intercept": lr_model.model.intercept_.astype(float).tolist(),
            },
            "pca": {
                "mean": pca_model.mean_.astype(float).tolist(),
                "components": pca_model.components_.astype(float).tolist(),
                "explained_variance_ratio": pca_model.explained_variance_ratio_.astype(float).tolist(),
            },
            "stats": {
                "median_mid": float(np.median(mid_tr)),
                "tick_size": float(seg.tick_size),
                "alpha_20s": float(prep.alphas_dict[HORIZON_K]),
                "ask_q_med": [float(v) for v in ask_q_med],
                "bid_q_med": [float(v) for v in bid_q_med],
                "ofi_l1_std": ofi_std,
                "vpin_p10": vpin_q[0],
                "vpin_p50": vpin_q[1],
                "vpin_p90": vpin_q[2],
                "vol10_p10": vol10_q[0],
                "vol10_p50": vol10_q[1],
                "vol10_p90": vol10_q[2],
                "vol30_p10": vol30_q[0],
                "vol30_p50": vol30_q[1],
                "vol30_p90": vol30_q[2],
                "drift30_std": float(np.std(Xv2_tr[:, v2_names.index("mid_drift_30s_ticks")])) or 1.0,
                "ofi60_std": float(np.std(Xv2_tr[:, v2_names.index("ofi_mean_60s")])) or 1.0,
                "ssf_ret5_std": ssf_ret5_std,
                "lead5_std": lead5_std,
                "basis_dev_std": basis_dev_std,
                "slope_bid_med": slope_q["bid"],
                "slope_ask_med": slope_q["ask"],
                "train_days": seg.days[:seg.train_days],
                "test_days": seg.days[seg.train_days + seg.val_days:],
            },
        }

        # Select 900 s active test slice + save full aligned predictions
        te_idx = sp["test"]
        lob_te = seg.lob_20[te_idx]
        mid_te = (lob_te[:, 0] + lob_te[:, 2]) / 2.0
        p_lgb_saved = sess_npz["prob_LightGBM"]
        s0 = _select_active_window(mid_te, p_lgb_saved, WINDOW_LEN)
        s1 = s0 + WINDOW_LEN

        win_raw_idx = te_idx[s0:s1]
        fwd_ret_full = _day_smoothed_forward_returns(
            (seg.lob_20[:, 0] + seg.lob_20[:, 2]) / 2.0, seg.day_id, HORIZON_K, smoothing=5
        )
        fwd_ret_bp = np.nan_to_num(fwd_ret_full[win_raw_idx] * 1e4, nan=0.0).astype(np.float32)

        ts_ms_win = seg.timestamps_ms[win_raw_idx]
        time_labels = np.array([
            datetime.fromtimestamp(ms / 1000.0, tz=IST).strftime("%H:%M:%S")
            for ms in ts_ms_win
        ])
        date_str = datetime.fromtimestamp(ts_ms_win[0] / 1000.0, tz=IST).strftime("%d %b %Y")

        replay_arrays[f"{key}__lob_20"] = lob_te[s0:s1].astype(np.float32)
        replay_arrays[f"{key}__tab_feat"] = X_te[s0:s1].astype(np.float32)
        replay_arrays[f"{key}__tab_feat_v2"] = Xv2_te[s0:s1].astype(np.float32)
        replay_arrays[f"{key}__tab_feat_flagship"] = Xf_te[s0:s1].astype(np.float32)
        replay_arrays[f"{key}__mid"] = mid_te[s0:s1].astype(np.float32)
        replay_arrays[f"{key}__y_true"] = y_te[s0:s1].astype(np.int8)
        replay_arrays[f"{key}__fwd_ret_bp"] = fwd_ret_bp
        replay_arrays[f"{key}__time"] = time_labels
        for mname in ["LightGBM", "TLOB", "DeepLOB5", "LSTM", "Random Forest", "Logistic Regression", "Naive Momentum"]:
            clean_m = mname.replace(" ", "_")
            replay_arrays[f"{key}__prob_{clean_m}"] = sess_npz[f"prob_{mname}"][s0:s1].astype(np.float32)
        replay_arrays[f"{key}__prob_LightGBM_v2"] = p_v2_te[s0:s1].astype(np.float32)
        replay_arrays[f"{key}__prob_LightGBM_Flagship"] = p_flag_te[s0:s1].astype(np.float32)
        replay_arrays[f"{key}__prob_Hybrid_Ensemble"] = p_hybrid_te[s0:s1].astype(np.float32)

        # Curated diagnostic jump targets within the 900s replay slice
        alpha_bp = round(float(prep.alphas_dict[HORIZON_K] * 1e4), 3)
        y_win = y_te[s0:s1]
        p_win_lgb = sess_npz["prob_LightGBM"][s0:s1]
        p_win_tlob = sess_npz["prob_TLOB"][s0:s1]
        p_win_v2 = p_v2_te[s0:s1]
        p_win_flag = p_flag_te[s0:s1]
        p_win_hyb = p_hybrid_te[s0:s1]

        pred_v1 = p_win_lgb.argmax(1)
        pred_v2 = p_win_v2.argmax(1)
        pred_flag = p_win_flag.argmax(1)
        pred_hyb = p_win_hyb.argmax(1)
        pred_tlob = p_win_tlob.argmax(1)
        idx_all = np.arange(WINDOW_LEN)

        dir_score = np.maximum(p_win_hyb[:, 1], p_win_hyb[:, 2])
        chosen = _pick_steps(dir_score >= 0.0, dir_score, WINDOW_LEN, k=12, min_gap=30)
        tlob_wins = _pick_steps(
            (pred_tlob == y_win) & (pred_v1 != y_win),
            p_win_tlob[idx_all, y_win] - p_win_lgb[idx_all, y_win],
            WINDOW_LEN,
        )
        lgbm_wins = _pick_steps(
            (pred_v1 == y_win) & (pred_tlob != y_win),
            p_win_lgb[idx_all, y_win] - p_win_tlob[idx_all, y_win],
            WINDOW_LEN,
        )
        v2_rescues = _pick_steps(
            (pred_hyb == y_win) & (pred_v2 == y_win) & (pred_v1 != y_win),
            p_win_hyb[idx_all, y_win] - p_win_lgb[idx_all, y_win],
            WINDOW_LEN,
        )
        ssf_rescues = _pick_steps(
            ((pred_flag == y_win) | (pred_hyb == y_win)) & (pred_v2 != y_win),
            p_win_flag[idx_all, y_win] - p_win_v2[idx_all, y_win],
            WINDOW_LEN,
        )
        if len(ssf_rescues) < 3:
            ssf_rescues = v2_rescues or chosen
        boundary_traps = _pick_steps(
            (pred_v1 != y_win) & (np.abs(fwd_ret_bp) <= max(1.35 * alpha_bp, 1.0)),
            -np.abs(np.abs(fwd_ret_bp) - alpha_bp),
            WINDOW_LEN,
        )

        replay_catalogue.append({
            "key": key,
            "symbol": data_sym.symbol,
            "title": spec["title"],
            "subtitle": spec["subtitle"],
            "badge": spec["badge"],
            "companion_source": comp_label,
            "segment_id": seg_id,
            "date_ist": date_str,
            "start_time": str(time_labels[0]),
            "end_time": str(time_labels[-1]),
            "n_steps": WINDOW_LEN,
            "tick_size": float(seg.tick_size),
            "alpha_bp": alpha_bp,
            "hybrid_lgbm_weight": float(w_blend),
            "median_mid": float(np.median(mid_te[s0:s1])),
            "train_mid": float(np.median(mid_tr)),
            "high_conf_steps": chosen,
            "tlob_wins_steps": tlob_wins,
            "lgbm_wins_steps": lgbm_wins,
            "v2_rescues_steps": v2_rescues,
            "ssf_rescues_steps": ssf_rescues,
            "boundary_trap_steps": boundary_traps,
        })

    summaries = _load_summaries_payload(
        full_feature_meta,
        models_out["RELIANCE"]["feat_names"],
        models_out["RELIANCE"]["feat_names_v2"],
        models_out["RELIANCE"]["feat_names_flagship"],
        replay_catalogue,
    )
    (BUNDLE_DIR / "models.json").write_text(json.dumps(models_out), encoding="utf-8")
    np.savez_compressed(BUNDLE_DIR / "replays.npz", **replay_arrays)
    np.savez_compressed(BUNDLE_DIR / "full_days.npz", **full_days_arrays)
    (BUNDLE_DIR / "summaries.json").write_text(json.dumps(summaries), encoding="utf-8")

    for fn in ("models.json", "replays.npz", "full_days.npz", "summaries.json"):
        sz_kb = (BUNDLE_DIR / fn).stat().st_size / 1024.0
        print(f"[build_bundle] Wrote {BUNDLE_DIR / fn} ({sz_kb:.1f} KB)")


def _load_summaries_payload(
    feature_meta: Dict[str, Any],
    feat_names: List[str],
    feat_names_v2: List[str],
    feat_names_flagship: List[str],
    replay_catalogue: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "feature_meta": feature_meta,
        "feat_names": feat_names,
        "feat_names_v2": feat_names_v2,
        "feat_names_flagship": feat_names_flagship,
        "replays": replay_catalogue,
        "error_diagnostics": {
            "disagreement": _clean_records(pd.read_csv(DIAG_DIR / "disagreement_summary.csv")),
            "buckets": _clean_records(pd.read_csv(DIAG_DIR / "microstructure_buckets.csv")),
            "upgrade_ablation": _clean_records(pd.read_csv(DIAG_DIR / "upgrade_ablation_summary.csv")),
            "conviction_curve": _clean_records(pd.read_csv(DIAG_DIR / "selective_conviction_curve.csv")),
        },
        "final_verification": {
            "universe25": _clean_records(pd.read_csv(FINAL_VERIF_DIR / "universe25_upgrade_verification.csv")),
            "tick_regimes": _clean_records(pd.read_csv(FINAL_VERIF_DIR / "tick_regime_verification.csv")),
            "stat_tests": _clean_records(pd.read_csv(FINAL_VERIF_DIR / "statistical_tests_summary.csv")),
            "averaging_1_to_100": _clean_records(pd.read_csv(FINAL_VERIF_DIR / "averaging_1_to_100_summary.csv")),
            "ensemble_k1_to_k8": _clean_records(pd.read_csv(FINAL_VERIF_DIR / "ensemble_k1_to_k8_summary.csv")),
        },
        "ml_completeness": {
            "class_distributions": _clean_records(pd.read_csv(FINAL_VERIF_DIR / "class_distribution_splits.csv")),
            "per_class_reports": _clean_records(pd.read_csv(FINAL_VERIF_DIR / "per_class_classification_reports.csv")),
            "confusion_matrices": _clean_records(pd.read_csv(FINAL_VERIF_DIR / "confusion_matrices_detailed.csv")),
            "generalization_gap": _clean_records(pd.read_csv(FINAL_VERIF_DIR / "train_val_test_generalization.csv")),
            "v2_feature_importance": _clean_records(pd.read_csv(FINAL_VERIF_DIR / "v2_ssf_feature_importance.csv")),
            "universe25_per_class": _clean_records(pd.read_csv(FINAL_VERIF_DIR / "universe25_class_and_perclass_f1.csv")),
            "inference_latency": _clean_records(pd.read_csv(ARTIFACTS_DIR / "nse" / "models" / "RELIANCE" / "inference_latency.csv")),
        },
        "nse_universe_25": _clean_records(pd.read_csv(ARTIFACTS_DIR / "nse" / "universe_25" / "nse_universe_25_summary.csv")),
        "nse_universe_corr": _clean_records(pd.read_csv(ARTIFACTS_DIR / "nse" / "universe_25" / "nse_universe_25_correlations.csv")),
        "nse_eda": _clean_records(pd.read_csv(ARTIFACTS_DIR / "nse" / "nse_dataset_summary.csv")),
        "nse_ladders": {
            sym: _clean_records(pd.read_csv(ARTIFACTS_DIR / "nse" / "models" / sym / "model_ladder_summary.csv"))
            for sym in ("RELIANCE", "NIFTY_FUT", "IDEA")
        },
        "nse_sig": {
            sym: _clean_records(pd.read_csv(ARTIFACTS_DIR / "nse" / "models" / sym / "significance_vs_lightgbm.csv"))
            for sym in ("RELIANCE", "NIFTY_FUT", "IDEA")
        },
        "nse_shap_groups": {
            sym: _clean_records(pd.read_csv(ARTIFACTS_DIR / "nse" / "models" / sym / "shap_groups.csv"))
            for sym in ("RELIANCE", "NIFTY_FUT", "IDEA")
        },
        "nse_deeplob5_attr": {
            sym: _clean_records(pd.read_csv(ARTIFACTS_DIR / "nse" / "models" / sym / "deeplob5_attribution.csv"))
            for sym in ("RELIANCE", "NIFTY_FUT", "IDEA")
        },
        "nse_ablation": _clean_records(pd.read_csv(ARTIFACTS_DIR / "nse" / "decay_ablation" / "nse_ablation_summary.csv")),
        "nse_decay": _clean_records(pd.read_csv(ARTIFACTS_DIR / "nse" / "decay_ablation" / "nse_decay_summary.csv")),
        "nse_edge_decomp": _clean_records(pd.read_csv(ARTIFACTS_DIR / "nse" / "execution" / "nse_edge_decomposition.csv")),
        "btc_ladder": _clean_records(pd.read_csv(ARTIFACTS_DIR / "btc" / "experiment_a_results.csv")),
        "synthetic_ladder": _clean_records(pd.read_csv(ARTIFACTS_DIR / "experiment_a_results.csv")),
        "fi2010": _clean_records(pd.read_csv(ARTIFACTS_DIR / "fi2010" / "fi2010_setup2_summary.csv")),
        "cpp_bench": _clean_records(pd.read_csv(ARTIFACTS_DIR / "cpp_trained_model_benchmark.csv")),
    }


def refresh_summaries_only() -> None:
    """Fast summary-only bundle refresh preserving models.json, replays.npz, and full_days.npz."""
    s_path = BUNDLE_DIR / "summaries.json"
    if not s_path.exists():
        raise FileNotFoundError(f"Missing {s_path}; run full build_bundle() first.")
    existing = json.loads(s_path.read_text(encoding="utf-8"))
    summaries = _load_summaries_payload(
        existing["feature_meta"],
        existing["feat_names"],
        existing["feat_names_v2"],
        existing.get("feat_names_flagship", []),
        existing["replays"],
    )
    s_path.write_text(json.dumps(summaries), encoding="utf-8")
    print(f"[build_bundle] Refreshed {s_path} ({s_path.stat().st_size / 1024.0:.1f} KB)")


if __name__ == "__main__":
    if "--summaries-only" in sys.argv:
        refresh_summaries_only()
    else:
        build_bundle()
