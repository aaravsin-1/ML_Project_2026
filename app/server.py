"""FastAPI live ML inference, TreeSHAP explanation, and order-book replay server.

Runs cleanly on CPU anywhere using only the self-contained `app/bundle/` artefacts:
    python -m app.server --port 8050
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import lightgbm as lgb
import numpy as np
import shap
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "app"
BUNDLE_DIR = APP_DIR / "bundle"
STATIC_DIR = APP_DIR / "static"

CLASS_NAMES = ["FLAT", "UP", "DOWN"]

app = FastAPI(title="LOB Alpha Studio · Live ML Order-Book Deployment", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class State:
    models_meta: Dict[str, Any] = {}
    boosters: Dict[str, lgb.Booster] = {}
    explainers: Dict[str, shap.TreeExplainer] = {}
    boosters_v2: Dict[str, lgb.Booster] = {}
    explainers_v2: Dict[str, shap.TreeExplainer] = {}
    boosters_flag20: Dict[str, lgb.Booster] = {}
    boosters_flag10: Dict[str, lgb.Booster] = {}
    explainers_flag20: Dict[str, shap.TreeExplainer] = {}
    explainers_flag10: Dict[str, shap.TreeExplainer] = {}
    replays: Any = None
    full_days: Any = None
    summaries: Dict[str, Any] = {}


STATE = State()


def _load_bundle() -> None:
    if not (BUNDLE_DIR / "models.json").exists():
        raise RuntimeError(
            "Bundle not found at app/bundle/. Run `python -m app.build_bundle` first."
        )
    STATE.models_meta = json.loads((BUNDLE_DIR / "models.json").read_text(encoding="utf-8"))
    STATE.summaries = json.loads((BUNDLE_DIR / "summaries.json").read_text(encoding="utf-8"))
    STATE.replays = np.load(BUNDLE_DIR / "replays.npz", allow_pickle=False)
    fd_path = BUNDLE_DIR / "full_days.npz"
    STATE.full_days = np.load(fd_path, allow_pickle=False) if fd_path.exists() else None

    for key, spec in STATE.models_meta.items():
        booster = lgb.Booster(model_str=spec["lgbm_model_str"])
        STATE.boosters[key] = booster
        STATE.explainers[key] = shap.TreeExplainer(booster)
        if "lgbm_v2_model_str" in spec:
            booster_v2 = lgb.Booster(model_str=spec["lgbm_v2_model_str"])
            STATE.boosters_v2[key] = booster_v2
            STATE.explainers_v2[key] = shap.TreeExplainer(booster_v2)
        if "lgbm_flagship_20_model_str" in spec and "lgbm_flagship_10_model_str" in spec:
            b20 = lgb.Booster(model_str=spec["lgbm_flagship_20_model_str"])
            b10 = lgb.Booster(model_str=spec["lgbm_flagship_10_model_str"])
            STATE.boosters_flag20[key] = b20
            STATE.boosters_flag10[key] = b10
            STATE.explainers_flag20[key] = shap.TreeExplainer(b20)
            STATE.explainers_flag10[key] = shap.TreeExplainer(b10)


_load_bundle()


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def _logreg_predict_one(lr: Dict[str, Any], x: np.ndarray) -> np.ndarray:
    mean = np.asarray(lr["mean"], dtype=np.float64)
    std = np.asarray(lr["std"], dtype=np.float64)
    coef = np.asarray(lr["coef"], dtype=np.float64)
    intercept = np.asarray(lr["intercept"], dtype=np.float64)
    z = coef @ ((x - mean) / std) + intercept
    return _softmax(z)


def _shap_matrix(explainer: shap.TreeExplainer, row_2d: np.ndarray) -> np.ndarray:
    """Normalises SHAP TreeExplainer output to shape (n_features, 3)."""
    sv = explainer.shap_values(row_2d)
    if isinstance(sv, list):
        return np.stack([np.asarray(c)[0] for c in sv], axis=1)
    arr = np.asarray(sv)
    if arr.ndim == 3:
        return arr[0]
    raise ValueError(f"Unexpected SHAP shape: {arr.shape}")


def _top_shap(shap_25x3: np.ndarray, feat_names: List[str], x_vals: np.ndarray, focus_cls: int, top_k: int = 8) -> List[Dict[str, Any]]:
    meta = STATE.summaries.get("feature_meta", {})
    col = shap_25x3[:, focus_cls]
    order = np.argsort(-np.abs(col))[:top_k]
    out: List[Dict[str, Any]] = []
    for idx in order:
        fname = feat_names[int(idx)]
        finfo = meta.get(fname, {})
        out.append({
            "feature": fname,
            "label": finfo.get("label", fname),
            "group": finfo.get("group", "Other"),
            "value": round(float(x_vals[int(idx)]), 4),
            "shap": round(float(col[int(idx)]), 4),
            "shap_up": round(float(shap_25x3[int(idx), 1]), 4),
            "shap_down": round(float(shap_25x3[int(idx), 2]), 4),
        })
    return out


class ExplainStepReq(BaseModel):
    replay_key: str
    step: int = Field(ge=0, lt=900)
    ver: str = "flagship"
    use_v2: bool | None = None


class SandboxPredictReq(BaseModel):
    model_key: str = "RELIANCE"
    bid_l1_mult: float = Field(1.0, ge=0.1, le=5.0)
    ask_l1_mult: float = Field(1.0, ge=0.1, le=5.0)
    deep_mult: float = Field(0.0, ge=-0.8, le=1.5)
    spread_ticks: int = Field(1, ge=1, le=6)
    ofi_sigma: float = Field(0.0, ge=-3.0, le=3.0)
    lag_persist: float = Field(0.7, ge=0.0, le=1.0)
    vpin_pct: float = Field(0.5, ge=0.05, le=0.95)
    drift_30s_ticks: float = Field(0.0, ge=-12.0, le=12.0)
    vol_pct: float = Field(0.5, ge=0.0, le=0.98)
    day_progress: float = Field(0.45, ge=0.02, le=0.99)
    ssf_imb: float = Field(0.0, ge=-0.95, le=0.95)
    ssf_lead_sigma: float = Field(0.0, ge=-3.0, le=3.0)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "models_loaded": list(STATE.boosters.keys()),
        "models_v2_loaded": list(STATE.boosters_v2.keys()),
        "models_flagship_loaded": list(STATE.boosters_flag20.keys()),
        "num_features": len(STATE.summaries.get("feat_names", [])),
        "num_features_v2": len(STATE.summaries.get("feat_names_v2", [])),
        "num_features_flagship": len(STATE.summaries.get("feat_names_flagship", [])),
        "cpp_p50_us": 10.5,
        "gpu_deeplob5_us": 5400.0,
    }


@app.get("/api/bootstrap")
def bootstrap() -> Dict[str, Any]:
    model_profiles = {}
    for k, spec in STATE.models_meta.items():
        model_profiles[k] = {
            "key": k,
            "symbol": spec["symbol"],
            "title": spec["title"],
            "subtitle": spec["subtitle"],
            "badge": spec["badge"],
            "companion_source": spec.get("companion_source", "Single-Stock Futures"),
            "stats": spec["stats"],
            "pca_explained_var": spec["pca"]["explained_variance_ratio"],
            "lgbm_best_iter": spec["lgbm_best_iter"],
            "lgbm_v2_best_iter": spec.get("lgbm_v2_best_iter"),
            "lgbm_flagship_20_best_iter": spec.get("lgbm_flagship_20_best_iter"),
            "lgbm_flagship_10_best_iter": spec.get("lgbm_flagship_10_best_iter"),
            "hybrid_lgbm_weight": spec.get("hybrid_lgbm_weight", 0.75),
        }
    return {
        "summaries": STATE.summaries,
        "model_profiles": model_profiles,
    }


def _fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = np.bincount(y_true.astype(np.int64) * 3 + y_pred.astype(np.int64), minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2.0 * tp + fp + fn
    f1s = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1.0), 0.0)
    return float(f1s.mean())


def _trailing_smooth_probs(probs: np.ndarray, n_win: int) -> np.ndarray:
    if n_win <= 1:
        return probs
    cs = np.cumsum(np.vstack([np.zeros((1, 3), dtype=np.float64), probs]), axis=0)
    idx = np.arange(1, len(probs) + 1)
    win = np.minimum(idx, n_win)
    return (cs[idx] - cs[idx - win]) / win[:, None]


def _day_stats_and_batches(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    pred = probs.argmax(1)
    n_sec = int(len(y_true))
    acc = float(np.mean(pred == y_true)) if n_sec else 0.0
    f1 = _fast_macro_f1(y_true, pred) if n_sec else 0.0
    dm = (pred != 0) & (y_true != 0)
    hits = (pred[dm] == y_true[dm]).astype(np.float64)
    dir_hit = float(hits.mean()) if len(hits) else 0.0

    max_p = probs.max(axis=1)
    hi_m = dm & (max_p >= 0.55)
    hi_hit = float((pred[hi_m] == y_true[hi_m]).mean()) if hi_m.sum() >= 10 else None

    batch_wins: Dict[str, float | None] = {"1": round(dir_hit * 100.0, 2) if len(hits) else None}
    for bs in (10, 20, 30, 50, 100):
        if len(hits) >= bs:
            nb = len(hits) // bs
            means = hits[:nb * bs].reshape(nb, bs).mean(axis=1)
            win_pct = float(np.mean((means > 0.5) + 0.5 * (means == 0.5)) * 100.0)
            batch_wins[str(bs)] = round(win_pct, 2)
        else:
            batch_wins[str(bs)] = None

    return {
        "n_seconds": n_sec,
        "n_directional": int(len(hits)),
        "macro_f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "dir_hit": round(dir_hit, 4),
        "high_conf_hit": round(hi_hit, 4) if hi_hit is not None else None,
        "high_conf_n": int(hi_m.sum()),
        "batch_wins": batch_wins,
    }


def _sec_to_ist(sec: int) -> str:
    tot_s = 9 * 3600 + 15 * 60 + int(sec)
    hh = tot_s // 3600
    mm = (tot_s % 3600) // 60
    return f"{hh:02d}:{mm:02d}"


@app.get("/api/day_overview/{symbol}")
def day_overview(
    symbol: str,
    day_idx: int = -1,
    window_sec: int = 22500,
    offset_sec: int = 0,
    smooth_n: int = 1,
) -> Dict[str, Any]:
    if STATE.full_days is None:
        raise HTTPException(status_code=503, detail="Full-day archive not loaded in bundle.")
    sym = "NIFTY_FUT" if symbol == "NIFTY_FUT_SEG3" else symbol
    if sym not in {"RELIANCE", "NIFTY_FUT", "IDEA"}:
        raise HTTPException(status_code=404, detail=f"Unknown symbol for full-day view: {symbol}")

    fd = STATE.full_days
    dates = [str(d) for d in fd[f"{sym}__dates"].tolist()]
    bounds = fd[f"{sym}__bounds"]
    sec_all = fd[f"{sym}__sec"]
    mid_all = fd[f"{sym}__mid"]
    y_all = fd[f"{sym}__y_true"].astype(np.int64)
    p_all_u8 = fd[f"{sym}__prob_hyb_u8"]

    n_days = len(dates)
    smooth_n = max(1, min(int(smooth_n), 100))

    # Build quick 12-day summary table (always returned)
    daily_summary: List[Dict[str, Any]] = []
    smoothed_days: List[np.ndarray] = []
    for d_i in range(n_days):
        a, b = int(bounds[d_i]), int(bounds[d_i + 1])
        p_d = p_all_u8[a:b].astype(np.float64) / 255.0
        p_d = _trailing_smooth_probs(p_d, smooth_n)
        smoothed_days.append(p_d)
        yd = y_all[a:b]
        prd = p_d.argmax(1)
        dm = (prd != 0) & (yd != 0)
        dh = float((prd[dm] == yd[dm]).mean()) if dm.sum() else 0.0
        daily_summary.append({
            "day_idx": d_i,
            "date": dates[d_i],
            "n_seconds": b - a,
            "open_mid": round(float(mid_all[a]), 2),
            "close_mid": round(float(mid_all[b - 1]), 2),
            "dir_hit": round(dh, 4),
            "macro_f1": round(_fast_macro_f1(yd, prd), 4),
        })

    if day_idx < 0:
        p_cat = np.vstack(smoothed_days)
        stats = _day_stats_and_batches(y_all, p_cat)
        # Aggregate per-day batch_wins across days properly
        all_hits = []
        for d_i in range(n_days):
            a, b = int(bounds[d_i]), int(bounds[d_i + 1])
            yd, prd = y_all[a:b], smoothed_days[d_i].argmax(1)
            dm = (prd != 0) & (yd != 0)
            all_hits.append((prd[dm] == yd[dm]).astype(np.float64))
        for bs in (10, 20, 30, 50, 100):
            blk_scores = []
            for h_arr in all_hits:
                if len(h_arr) >= bs:
                    nb = len(h_arr) // bs
                    means = h_arr[:nb * bs].reshape(nb, bs).mean(axis=1)
                    blk_scores.extend(((means > 0.5) + 0.5 * (means == 0.5)).tolist())
            stats["batch_wins"][str(bs)] = round(float(np.mean(blk_scores) * 100.0), 2) if blk_scores else None

        series = [
            {
                "label": r["date"][5:],  # MM-DD
                "mid": r["close_mid"],
                "dir_hit_pct": round(r["dir_hit"] * 100.0, 2),
                "macro_f1": r["macro_f1"],
            }
            for r in daily_summary
        ]
        return {
            "symbol": sym,
            "mode": "all_days",
            "day_idx": -1,
            "date": f"All {n_days} Held-Out Test Days",
            "smooth_n": smooth_n,
            "window_sec": 22500 * n_days,
            "max_day_sec": 22500,
            "dates": dates,
            "daily_summary": daily_summary,
            "stats": stats,
            "series": series,
        }

    d_clamped = max(0, min(day_idx, n_days - 1))
    a, b = int(bounds[d_clamped]), int(bounds[d_clamped + 1])
    day_len = b - a
    w_sec = max(300, min(int(window_sec), day_len))
    off = max(0, min(int(offset_sec), max(0, day_len - w_sec)))
    hi = min(day_len, off + w_sec)

    y_win = y_all[a + off:a + hi]
    p_win = smoothed_days[d_clamped][off:hi]
    mid_win = mid_all[a + off:a + hi]
    sec_win = sec_all[a + off:a + hi]

    stats = _day_stats_and_batches(y_win, p_win)

    # Downsample to <= 320 chart points for fast crisp plotting
    n_pts = len(mid_win)
    step = max(1, n_pts // 300)
    prd_w = p_win.argmax(1)
    dm_w = (prd_w != 0) & (y_win != 0)
    cum_ok = np.cumsum((prd_w == y_win) & dm_w)
    cum_dir = np.cumsum(dm_w)

    idxs = list(range(0, n_pts, step))
    if idxs[-1] != n_pts - 1:
        idxs.append(n_pts - 1)
    series = []
    for k_i in idxs:
        cd = int(cum_dir[k_i])
        hit_pct = round(int(cum_ok[k_i]) / cd * 100.0, 2) if cd > 0 else 50.0
        series.append({
            "label": _sec_to_ist(int(sec_win[k_i])),
            "mid": round(float(mid_win[k_i]), 2),
            "dir_hit_pct": hit_pct,
            "net_prob": round(float(p_win[k_i, 1] - p_win[k_i, 2]), 3),
        })

    return {
        "symbol": sym,
        "mode": "single_day",
        "day_idx": d_clamped,
        "date": dates[d_clamped],
        "start_ist": _sec_to_ist(int(sec_win[0])),
        "end_ist": _sec_to_ist(int(sec_win[-1])),
        "smooth_n": smooth_n,
        "window_sec": hi - off,
        "offset_sec": off,
        "max_day_sec": day_len,
        "dates": dates,
        "daily_summary": daily_summary,
        "stats": stats,
        "series": series,
    }


@app.get("/api/replay/{key}")
def get_replay(key: str) -> Dict[str, Any]:
    if key not in STATE.models_meta:
        raise HTTPException(status_code=404, detail=f"Unknown replay key: {key}")
    z = STATE.replays
    lob = np.round(z[f"{key}__lob_20"], 4).tolist()
    mid = np.round(z[f"{key}__mid"], 4).tolist()
    y_true = z[f"{key}__y_true"].astype(int).tolist()
    fwd_ret = np.round(z[f"{key}__fwd_ret_bp"], 3).tolist()
    times = [str(t) for t in z[f"{key}__time"].tolist()]

    probs: Dict[str, List[List[float]]] = {}
    for mname in [
        "Hybrid Ensemble",
        "LightGBM Flagship",
        "LightGBM v2",
        "LightGBM",
        "TLOB",
        "DeepLOB5",
        "LSTM",
        "Random Forest",
        "Logistic Regression",
        "Naive Momentum",
    ]:
        clean_m = mname.replace(" ", "_")
        arr_key = f"{key}__prob_{clean_m}"
        if arr_key in z.files:
            probs[mname] = np.round(z[arr_key], 4).tolist()

    ssf_telemetry: Dict[str, Any] | None = None
    if f"{key}__tab_feat_flagship" in z.files and "feat_names_flagship" in STATE.models_meta[key]:
        fn_f = STATE.models_meta[key]["feat_names_flagship"]
        Xf = z[f"{key}__tab_feat_flagship"]
        ssf_telemetry = {
            "source": STATE.models_meta[key].get("companion_source", "SSF"),
            "imb_l1": np.round(Xf[:, fn_f.index("ssf_imb_l1")], 3).tolist(),
            "ret_5s_bp": np.round(Xf[:, fn_f.index("ssf_ret_5s_bp")], 2).tolist(),
            "lead_5s_bp": np.round(Xf[:, fn_f.index("lead_5s_bp")], 2).tolist(),
            "basis_dev_bp": np.round(Xf[:, fn_f.index("basis_dev_bp")], 2).tolist(),
        }

    meta = next(r for r in STATE.summaries["replays"] if r["key"] == key)
    return {
        "meta": meta,
        "time": times,
        "mid": mid,
        "lob_20": lob,
        "y_true": y_true,
        "fwd_ret_bp": fwd_ret,
        "probs": probs,
        "ssf_telemetry": ssf_telemetry,
    }


def _diagnose_step(
    y_true: int,
    fwd_ret_bp: float,
    alpha_bp: float,
    p_v1: np.ndarray,
    p_v2: np.ndarray,
    p_flag: np.ndarray,
    p_tlob: np.ndarray,
    p_hyb: np.ndarray,
    vol_10s: float,
    drift_30s: float,
    day_prog: float,
    lead_5s_bp: float = 0.0,
) -> Dict[str, Any]:
    pred_v1 = int(np.argmax(p_v1))
    pred_v2 = int(np.argmax(p_v2))
    pred_flag = int(np.argmax(p_flag))
    pred_tlob = int(np.argmax(p_tlob))
    pred_hyb = int(np.argmax(p_hyb))

    v1_ok = pred_v1 == y_true
    v2_ok = pred_v2 == y_true
    flag_ok = pred_flag == y_true
    tlob_ok = pred_tlob == y_true
    hyb_ok = pred_hyb == y_true
    near_alpha = alpha_bp > 0 and abs(fwd_ret_bp) <= 1.35 * alpha_bp

    if (hyb_ok or flag_ok) and not v2_ok:
        badge, tone = "Futures + MH Rescue", "up"
        headline = (
            f"38f cash-only v2 missed ({CLASS_NAMES[pred_v2]} vs {CLASS_NAMES[y_true]}), "
            f"but 46f Flagship (+SSF lead {lead_5s_bp:+.2f} bp + 10s/20s Multi-Horizon + TLOB) "
            f"caught the move ({CLASS_NAMES[y_true]})."
        )
    elif hyb_ok and v2_ok and not v1_ok:
        badge, tone = "v2 Multi-Scale Rescue", "up"
        headline = (
            f"Baseline 25f LightGBM missed ({CLASS_NAMES[pred_v1]} vs {CLASS_NAMES[y_true]}), "
            f"but v2's 10s–60s flow/volatility context (30s drift {drift_30s:+.1f}t, 10s vol {vol_10s:.2f}t) "
            f"corrected the prediction to {CLASS_NAMES[pred_hyb]}."
        )
    elif v1_ok and tlob_ok:
        badge, tone = "Consensus Hit", "up"
        headline = (
            f"Both LightGBM and TLOB agree on {CLASS_NAMES[y_true]}: realised move "
            f"({fwd_ret_bp:+.2f} bp) cleared the ±{alpha_bp:.2f} bp dead-zone cleanly."
        )
    elif tlob_ok and not v1_ok:
        badge, tone = "TLOB 100s Context Win", "blue"
        headline = (
            f"TLOB's 100-second attention window captured calm/multi-step buildup ({CLASS_NAMES[y_true]}), "
            f"while baseline LightGBM only saw 5 seconds of lags ({CLASS_NAMES[pred_v1]})."
        )
    elif v1_ok and not tlob_ok:
        badge, tone = "LightGBM 1s Flip Win", "blue"
        headline = (
            f"Fast tree splits reacted immediately to an L1 queue/flow shock ({CLASS_NAMES[y_true]}), "
            f"whereas TLOB lagged due to 100-second sequence inertia ({CLASS_NAMES[pred_tlob]})."
        )
    elif not hyb_ok and near_alpha and ((pred_hyb == 0) != (y_true == 0)):
        badge, tone = "±α Dead-Zone Boundary Trap", "amber"
        headline = (
            f"Classic boundary noise: realised 20s move ({fwd_ret_bp:+.2f} bp) hovered right beside "
            f"the ±{alpha_bp:.2f} bp label threshold (same reason ~62–87% of errors occur)."
        )
    elif hyb_ok:
        badge, tone = "Flagship Hybrid Hit", "up"
        headline = (
            f"Flagship Hybrid (46f +SSF + 10s/20s + TLOB) nailed {CLASS_NAMES[y_true]} at {max(p_hyb)*100:.1f}% confidence "
            f"(day progress {day_prog*100:.0f}%)."
        )
    else:
        badge, tone = "Order-Flow Reversal", "down"
        headline = (
            f"Sudden opposing flow after t shifted the realised 20s return to {fwd_ret_bp:+.2f} bp "
            f"({CLASS_NAMES[y_true]}), reversing initial quote pressure."
        )

    return {
        "badge": badge,
        "tone": tone,
        "headline": headline,
        "alpha_bp": round(float(alpha_bp), 3),
        "fwd_ret_bp": round(float(fwd_ret_bp), 3),
        "y_true_name": CLASS_NAMES[y_true],
        "v1_correct": v1_ok,
        "v2_correct": v2_ok,
        "flagship_correct": flag_ok,
        "tlob_correct": tlob_ok,
        "hybrid_correct": hyb_ok,
    }


@app.post("/api/explain_step")
def explain_step(req: ExplainStepReq) -> Dict[str, Any]:
    key = req.replay_key
    if key not in STATE.boosters:
        raise HTTPException(status_code=404, detail=f"Unknown replay key: {key}")
    z = STATE.replays
    x = z[f"{key}__tab_feat"][req.step].astype(np.float64)
    row_2d = x.reshape(1, -1)

    t0 = time.perf_counter_ns()
    p_v1 = STATE.boosters[key].predict(row_2d)[0]
    t_v1_us = (time.perf_counter_ns() - t0) / 1000.0
    p_lr = _logreg_predict_one(STATE.models_meta[key]["logreg"], x)

    has_v2 = key in STATE.boosters_v2 and f"{key}__tab_feat_v2" in z.files
    if has_v2:
        x_v2 = z[f"{key}__tab_feat_v2"][req.step].astype(np.float64)
        row_v2 = x_v2.reshape(1, -1)
        t0 = time.perf_counter_ns()
        p_v2 = STATE.boosters_v2[key].predict(row_v2)[0]
        t_v2_us = (time.perf_counter_ns() - t0) / 1000.0
    else:
        x_v2, row_v2, p_v2, t_v2_us = x, row_2d, p_v1, t_v1_us

    has_flag = key in STATE.boosters_flag20 and f"{key}__tab_feat_flagship" in z.files
    if has_flag:
        x_flag = z[f"{key}__tab_feat_flagship"][req.step].astype(np.float64)
        row_flag = x_flag.reshape(1, -1)
        t0 = time.perf_counter_ns()
        p_f20 = STATE.boosters_flag20[key].predict(row_flag)[0]
        p_f10 = STATE.boosters_flag10[key].predict(row_flag)[0]
        p_flag = 0.5 * p_f10 + 0.5 * p_f20
        t_flag_us = (time.perf_counter_ns() - t0) / 1000.0
    else:
        x_flag, row_flag, p_flag, t_flag_us = x_v2, row_v2, p_v2, t_v2_us

    ver = req.ver.lower().strip()
    if req.use_v2 is False:
        ver = "v1"
    if ver == "flagship" and has_flag:
        active_ver = "flagship"
        p_active, x_active = p_flag, x_flag
        feat_names = STATE.models_meta[key]["feat_names_flagship"]
        shap_mat = 0.5 * _shap_matrix(STATE.explainers_flag10[key], row_flag) + 0.5 * _shap_matrix(STATE.explainers_flag20[key], row_flag)
        t_us = t_flag_us
    elif ver in ("flagship", "v2") and has_v2:
        active_ver = "v2"
        p_active, x_active = p_v2, x_v2
        feat_names = STATE.models_meta[key]["feat_names_v2"]
        shap_mat = _shap_matrix(STATE.explainers_v2[key], row_v2)
        t_us = t_v2_us
    else:
        active_ver = "v1"
        p_active, x_active = p_v1, x
        feat_names = STATE.models_meta[key]["feat_names"]
        shap_mat = _shap_matrix(STATE.explainers[key], row_2d)
        t_us = t_v1_us

    pred_cls = int(np.argmax(p_active))
    focus_cls = pred_cls if pred_cls != 0 else (1 if p_active[1] >= p_active[2] else 2)

    y_true = int(z[f"{key}__y_true"][req.step])
    fwd_ret_bp = float(z[f"{key}__fwd_ret_bp"][req.step])
    meta_rep = next(r for r in STATE.summaries["replays"] if r["key"] == key)
    alpha_bp = float(meta_rep.get("alpha_bp", 1.5))
    p_tlob = z[f"{key}__prob_TLOB"][req.step].astype(np.float64)
    p_hyb = (
        z[f"{key}__prob_Hybrid_Ensemble"][req.step].astype(np.float64)
        if f"{key}__prob_Hybrid_Ensemble" in z.files
        else p_flag
    )
    v2_names = STATE.models_meta[key].get("feat_names_v2", feat_names)
    vol_10s = float(x_v2[v2_names.index("vol_10s_ticks")]) if "vol_10s_ticks" in v2_names else 0.0
    drift_30s = float(x_v2[v2_names.index("mid_drift_30s_ticks")]) if "mid_drift_30s_ticks" in v2_names else 0.0
    day_prog = float(x_v2[v2_names.index("day_progress")]) if "day_progress" in v2_names else 0.5
    flag_names = STATE.models_meta[key].get("feat_names_flagship", feat_names)
    lead_5s_bp = float(x_flag[flag_names.index("lead_5s_bp")]) if has_flag and "lead_5s_bp" in flag_names else 0.0

    diagnosis = _diagnose_step(
        y_true=y_true,
        fwd_ret_bp=fwd_ret_bp,
        alpha_bp=alpha_bp,
        p_v1=p_v1,
        p_v2=p_v2,
        p_flag=p_flag,
        p_tlob=p_tlob,
        p_hyb=p_hyb,
        vol_10s=vol_10s,
        drift_30s=drift_30s,
        day_prog=day_prog,
        lead_5s_bp=lead_5s_bp,
    )

    return {
        "replay_key": key,
        "step": req.step,
        "active_ver": active_ver,
        "use_v2": active_ver != "v1",
        "companion_source": STATE.models_meta[key].get("companion_source", "SSF"),
        "lgbm_probs": [round(float(v), 4) for v in p_active],
        "lgbm_pred": CLASS_NAMES[pred_cls],
        "lgbm_flagship_probs": [round(float(v), 4) for v in p_flag],
        "lgbm_flagship_pred": CLASS_NAMES[int(np.argmax(p_flag))],
        "lgbm_v1_probs": [round(float(v), 4) for v in p_v1],
        "lgbm_v1_pred": CLASS_NAMES[int(np.argmax(p_v1))],
        "lgbm_v2_probs": [round(float(v), 4) for v in p_v2],
        "lgbm_v2_pred": CLASS_NAMES[int(np.argmax(p_v2))],
        "hybrid_probs": [round(float(v), 4) for v in p_hyb],
        "hybrid_pred": CLASS_NAMES[int(np.argmax(p_hyb))],
        "logreg_probs": [round(float(v), 4) for v in p_lr],
        "focus_class": CLASS_NAMES[focus_cls],
        "inference_us": round(t_us, 1),
        "diagnosis": diagnosis,
        "top_shap": _top_shap(shap_mat, feat_names, x_active, focus_cls=focus_cls, top_k=8),
        "features": {name: round(float(val), 4) for name, val in zip(feat_names, x_active)},
    }


@app.post("/api/predict")
def sandbox_predict(req: SandboxPredictReq) -> Dict[str, Any]:
    key = req.model_key
    if key not in STATE.boosters:
        raise HTTPException(status_code=404, detail=f"Unknown model key: {key}")

    spec = STATE.models_meta[key]
    st = spec["stats"]
    tick = float(st["tick_size"])
    base_mid = float(st["median_mid"])

    # Construct a valid 5-level limit order book around median_mid & tick_size
    bp1 = round(base_mid / tick) * tick - (req.spread_ticks // 2) * tick
    ap1 = bp1 + req.spread_ticks * tick
    mid = 0.5 * (ap1 + bp1)

    ask_med = np.asarray(st["ask_q_med"], dtype=np.float64)
    bid_med = np.asarray(st["bid_q_med"], dtype=np.float64)

    aq = ask_med.copy()
    bq = bid_med.copy()
    bq[0] = max(1.0, round(bid_med[0] * req.bid_l1_mult))
    aq[0] = max(1.0, round(ask_med[0] * req.ask_l1_mult))
    deep_scale_bid = max(0.2, 1.0 + req.deep_mult)
    deep_scale_ask = max(0.2, 1.0 - 0.55 * req.deep_mult)
    bq[1:] = np.maximum(1.0, np.round(bid_med[1:] * deep_scale_bid))
    aq[1:] = np.maximum(1.0, np.round(ask_med[1:] * deep_scale_ask))

    ap = np.array([ap1 + i * tick for i in range(5)], dtype=np.float64)
    bp = np.array([bp1 - i * tick for i in range(5)], dtype=np.float64)
    lob_20 = np.empty(20, dtype=np.float64)
    for lvl in range(5):
        lob_20[lvl * 4 + 0] = round(ap[lvl], 4)
        lob_20[lvl * 4 + 1] = bq_val = aq[lvl]
        lob_20[lvl * 4 + 2] = round(bp[lvl], 4)
        lob_20[lvl * 4 + 3] = bq[lvl]

    # Exact causal features matching nse_pipeline.py
    imb_1 = (bq[0] - aq[0]) / (bq[0] + aq[0] + 1e-8)
    imb_3 = (bq[:3].sum() - aq[:3].sum()) / (bq[:3].sum() + aq[:3].sum() + 1e-8)
    imb_5 = (bq.sum() - aq.sum()) / (bq.sum() + aq.sum() + 1e-8)
    spread = ap[0] - bp[0]
    micro = (bq[0] * ap[0] + aq[0] * bp[0]) / (bq[0] + aq[0] + 1e-8)
    micro_dev = micro - mid
    slope_bid = (bp[0] - bp[-1]) / (bq.sum() + 1e-8)
    slope_ask = (ap[-1] - ap[0]) / (aq.sum() + 1e-8)

    # Interpolate VPIN across training quantiles (10%, 50%, 90%)
    vpin_val = float(np.interp(req.vpin_pct, [0.1, 0.5, 0.9], [st["vpin_p10"], st["vpin_p50"], st["vpin_p90"]]))

    # Multi-level OFI impulse vector across levels 1..5 projected through trained Block PCA
    ofi_l1 = req.ofi_sigma * float(st["ofi_l1_std"])
    mlofi_vec = ofi_l1 * np.array([1.0, 0.55, 0.35, 0.22, 0.14], dtype=np.float64)
    pca_mean = np.asarray(spec["pca"]["mean"], dtype=np.float64)
    pca_comp = np.asarray(spec["pca"]["components"], dtype=np.float64)
    ofi_pca = pca_comp @ (mlofi_vec - pca_mean)

    # 1s, 2s, 5s causal lags scaled by lag_persist decay
    lag_scales = {1: 0.85 * req.lag_persist, 2: 0.65 * req.lag_persist, 5: 0.40 * req.lag_persist}

    fmap = {
        "imbalance_l1": imb_1,
        "imbalance_l3": imb_3,
        "imbalance_l5": imb_5,
        "spread_ticks": float(req.spread_ticks),
        "micro_deviation": micro_dev,
        "slope_bid": slope_bid,
        "slope_ask": slope_ask,
        "vpin": vpin_val,
        "ofi_l1": ofi_l1,
        "ofi_pca1": float(ofi_pca[0]),
        "ofi_pca2": float(ofi_pca[1]),
        "ofi_pca3": float(ofi_pca[2]),
        "mlofi_lvl_2": float(mlofi_vec[1]),
        "mlofi_lvl_3": float(mlofi_vec[2]),
        "mlofi_lvl_4": float(mlofi_vec[3]),
        "mlofi_lvl_5": float(mlofi_vec[4]),
    }
    for lag, dec in lag_scales.items():
        fmap[f"ofi_lag_{lag}"] = ofi_l1 * dec
        fmap[f"imbalance_lag_{lag}"] = imb_1 * dec
        fmap[f"micro_dev_lag_{lag}"] = micro_dev * dec

    # 13 v2 Multi-Scale, Volatility & Session-Clock features
    vol10_val = float(
        np.interp(req.vol_pct, [0.0, 0.1, 0.5, 0.9], [0.0, st.get("vol10_p10", 0.05), st.get("vol10_p50", 0.25), st.get("vol10_p90", 0.8)])
    )
    vol30_val = float(
        np.interp(req.vol_pct, [0.0, 0.1, 0.5, 0.9], [0.0, st.get("vol30_p10", 0.05), st.get("vol30_p50", 0.25), st.get("vol30_p90", 0.8)])
    )
    fmap["mid_drift_5s_ticks"] = float(req.drift_30s_ticks * 0.25)
    fmap["mid_drift_15s_ticks"] = float(req.drift_30s_ticks * 0.60)
    fmap["mid_drift_30s_ticks"] = float(req.drift_30s_ticks)
    fmap["mid_drift_60s_ticks"] = float(req.drift_30s_ticks * 1.45)
    fmap["ofi_mean_10s"] = float(ofi_l1 * (0.35 + 0.45 * req.lag_persist))
    fmap["ofi_mean_30s"] = float(ofi_l1 * (0.20 + 0.35 * req.lag_persist) + req.drift_30s_ticks * 4.0)
    fmap["ofi_mean_60s"] = float(ofi_l1 * (0.12 + 0.25 * req.lag_persist) + req.drift_30s_ticks * 5.0)
    fmap["imb1_mean_20s"] = float(imb_1 * req.lag_persist)
    fmap["imb1_vel_20s"] = float(imb_1 * (1.0 - req.lag_persist))
    fmap["vol_10s_ticks"] = vol10_val
    fmap["vol_30s_ticks"] = vol30_val
    fmap["spread_mean_20s"] = float(req.spread_ticks)
    fmap["day_progress"] = float(req.day_progress)

    # 8 Cross-Market Futures (SSF / Companion Futures) features
    ssf_ret5_bp = float(req.ssf_lead_sigma * float(st.get("ssf_ret5_std", 2.0)))
    lead5_bp = float(req.ssf_lead_sigma * float(st.get("lead5_std", 1.5)))
    basis_bp = float(req.ssf_lead_sigma * float(st.get("basis_dev_std", 1.5)))
    fmap["ssf_imb_l1"] = float(req.ssf_imb)
    fmap["ssf_imb_l5"] = float(req.ssf_imb * 0.80)
    fmap["ssf_imb1_5s"] = float(req.ssf_imb * 0.72)
    fmap["basis_dev_bp"] = basis_bp
    fmap["ssf_ret_1s_bp"] = ssf_ret5_bp * 0.35
    fmap["ssf_ret_5s_bp"] = ssf_ret5_bp
    fmap["lead_5s_bp"] = lead5_bp
    fmap["lead_15s_bp"] = lead5_bp * 1.45

    feat_names = spec["feat_names"]
    x = np.array([fmap[name] for name in feat_names], dtype=np.float64)
    row_2d = x.reshape(1, -1)

    t0 = time.perf_counter_ns()
    p_lgb = STATE.boosters[key].predict(row_2d)[0]
    t_v1_us = (time.perf_counter_ns() - t0) / 1000.0
    p_lr = _logreg_predict_one(spec["logreg"], x)
    shap_v1 = _shap_matrix(STATE.explainers[key], row_2d)

    if key in STATE.boosters_v2:
        feat_names_v2 = spec["feat_names_v2"]
        x_v2 = np.array([fmap[name] for name in feat_names_v2], dtype=np.float64)
        row_v2 = x_v2.reshape(1, -1)
        t0 = time.perf_counter_ns()
        p_v2 = STATE.boosters_v2[key].predict(row_v2)[0]
        t_v2_us = (time.perf_counter_ns() - t0) / 1000.0
        shap_v2 = _shap_matrix(STATE.explainers_v2[key], row_v2)
    else:
        feat_names_v2, x_v2, p_v2, t_v2_us, shap_v2 = feat_names, x, p_lgb, t_v1_us, shap_v1

    if key in STATE.boosters_flag20:
        feat_names_flag = spec["feat_names_flagship"]
        x_flag = np.array([fmap[name] for name in feat_names_flag], dtype=np.float64)
        row_flag = x_flag.reshape(1, -1)
        t0 = time.perf_counter_ns()
        p_f20 = STATE.boosters_flag20[key].predict(row_flag)[0]
        p_f10 = STATE.boosters_flag10[key].predict(row_flag)[0]
        p_flag = 0.5 * p_f10 + 0.5 * p_f20
        t_flag_us = (time.perf_counter_ns() - t0) / 1000.0
        shap_flag = 0.5 * _shap_matrix(STATE.explainers_flag10[key], row_flag) + 0.5 * _shap_matrix(STATE.explainers_flag20[key], row_flag)
    else:
        feat_names_flag, x_flag, p_flag, t_flag_us, shap_flag = feat_names_v2, x_v2, p_v2, t_v2_us, shap_v2

    pred_v1 = int(np.argmax(p_lgb))
    pred_v2 = int(np.argmax(p_v2))
    pred_flag = int(np.argmax(p_flag))
    focus_cls = 0 if (pred_flag == 0 and p_flag[0] >= 0.70) else (pred_flag if pred_flag != 0 else (1 if p_flag[1] >= p_flag[2] else 2))
    focus_v2 = 0 if (pred_v2 == 0 and p_v2[0] >= 0.70) else (pred_v2 if pred_v2 != 0 else (1 if p_v2[1] >= p_v2[2] else 2))
    focus_v1 = 0 if (pred_v1 == 0 and p_lgb[0] >= 0.70) else (pred_v1 if pred_v1 != 0 else (1 if p_lgb[1] >= p_lgb[2] else 2))

    return {
        "model_key": key,
        "companion_source": spec.get("companion_source", "SSF"),
        "lgbm_probs": [round(float(v), 4) for v in p_lgb],
        "lgbm_pred": CLASS_NAMES[pred_v1],
        "lgbm_v2_probs": [round(float(v), 4) for v in p_v2],
        "lgbm_v2_pred": CLASS_NAMES[pred_v2],
        "lgbm_flagship_probs": [round(float(v), 4) for v in p_flag],
        "lgbm_flagship_pred": CLASS_NAMES[pred_flag],
        "logreg_probs": [round(float(v), 4) for v in p_lr],
        "logreg_pred": CLASS_NAMES[int(np.argmax(p_lr))],
        "focus_class": CLASS_NAMES[focus_cls],
        "inference_us": round(t_flag_us, 1),
        "mid": round(mid, 4),
        "spread_bp": round((spread / mid) * 1e4, 3),
        "lob_20": lob_20.tolist(),
        "top_shap": _top_shap(shap_flag, feat_names_flag, x_flag, focus_cls=focus_cls, top_k=8),
        "top_shap_v2": _top_shap(shap_v2, feat_names_v2, x_v2, focus_cls=focus_v2, top_k=8),
        "top_shap_v1": _top_shap(shap_v1, feat_names, x, focus_cls=focus_v1, top_k=8),
        "features": {name: round(float(val), 4) for name, val in zip(feat_names_flag, x_flag)},
    }


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOB Alpha Studio live ML web server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8050")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
