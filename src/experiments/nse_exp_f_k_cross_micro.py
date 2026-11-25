"""NSE Experiments F & K: Cross-Asset Lead-Lag (NIFTY_FUT -> Equities & Bank Peers)
and Micro-Price as Predictor vs Training Target across all 8 NSE Instruments.

    python -m src.experiments.nse_exp_f_k_cross_micro
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore", message="X does not have valid feature names")

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src.config import ARTIFACTS_DIR
from src.data.nse_loader import load_nse_symbol
from src.microstructure.nse_pipeline import build_nse_segments, prepare_nse_segment
from src.models.tabular_trees import LightGBMLOB

OUT_DIR = ARTIFACTS_DIR / "nse"
CROSS_PAIRS = [
    (("RELIANCE", "EQUITY"), ("NIFTY", "DERIVATIVE")),
    (("HDFCBANK", "EQUITY"), ("NIFTY", "DERIVATIVE")),
    (("ICICIBANK", "EQUITY"), ("NIFTY", "DERIVATIVE")),
    (("HDFCBANK", "EQUITY"), ("ICICIBANK", "EQUITY")),
    (("ICICIBANK", "EQUITY"), ("HDFCBANK", "EQUITY")),
]
CROSS_HORIZONS = [5, 10, 20, 50]

MICRO_INSTRUMENTS = [
    ("IDEA", "EQUITY", "ultra_large_tick"),
    ("YESBANK", "EQUITY", "large_tick"),
    ("ITC", "EQUITY", "large_tick"),
    ("RELIANCE", "EQUITY", "medium_tick"),
    ("HDFCBANK", "EQUITY", "medium_tick"),
    ("ICICIBANK", "EQUITY", "medium_tick"),
    ("TCS", "EQUITY", "small_tick"),
    ("NIFTY", "DERIVATIVE", "tiny_tick_fut"),
]
MICRO_HORIZONS = [5, 10, 20, 50]
CONTROL_REPS = 3
N_BOOT = 1000
BLOCK_LEN = 300


def _day_forward_change_bp(price: np.ndarray, day_id: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(price), np.nan, dtype=np.float64)
    if k < len(price):
        same = day_id[:-k] == day_id[k:]
        idx = np.nonzero(same)[0]
        out[idx] = (price[idx + k] - price[idx]) / price[idx] * 1e4
    return out


def _day_past_return_bp(price: np.ndarray, day_id: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(price), np.nan, dtype=np.float64)
    if k < len(price):
        same = day_id[k:] == day_id[:-k]
        idx = np.nonzero(same)[0] + k
        out[idx] = (price[idx] - price[idx - k]) / price[idx - k] * 1e4
    return out


def _mid_and_micro(lob_20: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ask, ask_q, bid, bid_q = lob_20[:, 0], lob_20[:, 1], lob_20[:, 2], lob_20[:, 3]
    mid = (ask + bid) / 2.0
    depth = bid_q + ask_q
    micro = np.where(depth > 0, (bid_q * ask + ask_q * bid) / np.where(depth > 0, depth, 1.0), mid)
    return mid, micro


def _fit_beta(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    denom = float(np.dot(x[ok], x[ok]))
    return float(np.dot(x[ok], y[ok]) / denom) if denom > 0 else 0.0


def _r2_zero(y: np.ndarray, pred: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(pred)
    sst = float(np.sum(y[ok] ** 2))
    return 1.0 - float(np.sum((y[ok] - pred[ok]) ** 2)) / sst if sst > 0 else np.nan


def _score_reg(y: np.ndarray, pred: np.ndarray, calib: float = 1.0) -> Dict[str, float]:
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    const = np.std(pred) == 0 or np.std(y) == 0
    moved = y != 0
    return {
        "ic_pearson": np.nan if const else float(np.corrcoef(y, pred)[0, 1]),
        "ic_spearman": np.nan if const else float(spearmanr(y, pred).statistic),
        "r2": _r2_zero(y, calib * pred),
        "hit_rate": float(np.mean(np.sign(pred[moved]) == np.sign(y[moved]))) if moved.any() else np.nan,
    }


def _bootstrap_ic_diff(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, seed: int = 42) -> Tuple[float, float]:
    ok = np.isfinite(y) & np.isfinite(pred_a) & np.isfinite(pred_b)
    y, pa, pb = y[ok], pred_a[ok], pred_b[ok]
    n = len(y)
    if n < 50:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    bl = min(BLOCK_LEN, max(1, n // 10))
    nb = int(np.ceil(n / bl))
    diffs = np.empty(N_BOOT, dtype=np.float64)
    for b in range(N_BOOT):
        starts = rng.integers(0, max(1, n - bl + 1), size=nb)
        idx = np.concatenate([np.arange(s, min(s + bl, n)) for s in starts])[:n]
        yb, pab, pbb = y[idx], pa[idx], pb[idx]
        ica = 0.0 if np.std(pab) == 0 else np.corrcoef(yb, pab)[0, 1]
        icb = 0.0 if np.std(pbb) == 0 else np.corrcoef(yb, pbb)[0, 1]
        diffs[b] = ica - icb
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(lo), float(hi)


def run_cross_asset() -> None:
    print("\n=====================================================================")
    print("   NSE EXP F: CROSS-ASSET LEAD-LAG (NIFTY_FUT -> EQUITIES & BANKS)")
    print("=====================================================================")
    ca_dir = OUT_DIR / "cross_asset"
    ca_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for (t_sym, t_asset), (l_sym, l_asset) in CROSS_PAIRS:
        td = load_nse_symbol(t_sym, asset=t_asset)
        ld = load_nse_symbol(l_sym, asset=l_asset)
        t_segs = build_nse_segments(td, n_segments=4)
        l_segs = build_nse_segments(ld, n_segments=4)
        pair_name = f"{ld.symbol}->{td.symbol}"
        print(f"\n--- Pair: {pair_name} ---")

        for seg_id in range(4):
            t_prep = prepare_nse_segment(t_segs[seg_id], horizons=CROSS_HORIZONS)
            l_prep = prepare_nse_segment(l_segs[seg_id], horizons=CROSS_HORIZONS)

            # Causal alignment: for each target timestamp, take latest leader row with ts_leader <= ts_target on same day
            t_ts = t_segs[seg_id].timestamps_ms
            l_ts = l_segs[seg_id].timestamps_ms
            pos = np.searchsorted(l_ts, t_ts, side="right") - 1
            valid_align = pos >= 0
            pos_safe = np.maximum(pos, 0)
            # Also require same calendar day (within 5 seconds, no overnight stale carry)
            stale = (~valid_align) | ((t_ts - l_ts[pos_safe]) > 5000.0)
            X_leader = l_prep.tab_feat[:, l_prep.core_indices][pos_safe].copy()
            X_leader[stale] = 0.0

            X_base = t_prep.tab_feat[:, t_prep.core_indices]
            X_both = np.hstack([X_base, X_leader])
            n_base = X_base.shape[1]

            for k in CROSS_HORIZONS:
                y_all = t_prep.labels(k)
                sp = t_prep.splits(k)
                tr, va, te = sp["train"], sp["val"], sp["test"]

                m_base = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=300, random_state=seg_id)
                m_base.fit(X_base[tr], y_all[tr], X_base[va], y_all[va], early_stopping_rounds=15)
                f1_base = float(f1_score(y_all[te], m_base.predict(X_base[te]), average="macro", zero_division=0))

                m_both = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=300, random_state=seg_id)
                m_both.fit(X_both[tr], y_all[tr], X_both[va], y_all[va], early_stopping_rounds=15)
                f1_both = float(f1_score(y_all[te], m_both.predict(X_both[te]), average="macro", zero_division=0))

                gain = m_both.model.booster_.feature_importance(importance_type="gain")
                leader_gain_share = float(gain[n_base:].sum() / (gain.sum() + 1e-12))

                rows.append({
                    "pair": pair_name,
                    "target": td.symbol,
                    "leader": ld.symbol,
                    "segment": seg_id,
                    "horizon_s": k,
                    "base_f1": f1_base,
                    "with_leader_f1": f1_both,
                    "delta_f1": f1_both - f1_base,
                    "leader_gain_share": leader_gain_share,
                })

    df = pd.DataFrame(rows)
    df.to_csv(ca_dir / "nse_cross_asset_per_segment.csv", index=False)
    summary = (
        df.groupby(["pair", "horizon_s"], sort=False)
        .agg(
            base_f1=("base_f1", "mean"),
            with_leader_f1=("with_leader_f1", "mean"),
            delta_f1=("delta_f1", "mean"),
            delta_f1_std=("delta_f1", "std"),
            segs_improved=("delta_f1", lambda s: int((s > 0).sum())),
            leader_gain_share=("leader_gain_share", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(ca_dir / "nse_cross_asset_summary.csv", index=False)
    print("\n>>> NSE CROSS-ASSET SUMMARY:")
    print(summary.round(4).to_string(index=False))


def run_microprice() -> None:
    print("\n=====================================================================")
    print("   NSE EXP K: MICRO-PRICE AS PREDICTOR VS TARGET (8 INSTRUMENTS)")
    print("=====================================================================")
    mp_dir = OUT_DIR / "microprice"
    mp_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    beta_rows = []

    for sym, asset, tick_class in MICRO_INSTRUMENTS:
        sym_data = load_nse_symbol(sym, asset=asset)
        tag = sym_data.symbol
        segs = build_nse_segments(sym_data, n_segments=4)
        print(f"\n--- Micro-price: {tag} ({tick_class}) ---")

        for seg in segs:
            prep = prepare_nse_segment(seg, horizons=MICRO_HORIZONS)
            X = prep.tab_feat[:, prep.core_indices]
            mid, micro = _mid_and_micro(seg.lob_20)
            gap = (micro - mid) / mid * 1e4

            for k in MICRO_HORIZONS:
                sp = prep.splits(k)
                tr, va, te = sp["train"], sp["val"], sp["test"]
                y_mid = _day_forward_change_bp(mid, seg.day_id, k)
                y_mic = _day_forward_change_bp(micro, seg.day_id, k)
                past_r = _day_past_return_bp(mid, seg.day_id, k)

                # Winsorize training targets at train 0.1 / 99.9 percentiles (exact Exp K protocol)
                lo_m, hi_m = np.nanquantile(y_mid[tr], [0.001, 0.999])
                y_mid_tr = np.clip(y_mid[tr], lo_m, hi_m)
                y_mid_va = np.clip(y_mid[va], lo_m, hi_m)

                lo_c, hi_c = np.nanquantile(y_mic[tr], [0.001, 0.999])
                y_mic_tr = np.clip(y_mic[tr], lo_c, hi_c)
                y_mic_va = np.clip(y_mic[va], lo_c, hi_c)

                # 1. Beta convergence slope on train -> apply to test
                beta_tr = _fit_beta(gap[tr], y_mid_tr)
                beta_te = _fit_beta(gap[te], y_mid[te])
                beta_rows.append({
                    "symbol": tag,
                    "tick_class": tick_class,
                    "segment": seg.segment_id,
                    "horizon_s": k,
                    "beta_train": beta_tr,
                    "beta_test": beta_te,
                    "mean_abs_gap_bp": float(np.mean(np.abs(gap[te]))),
                })

                # 1-param rule & momentum
                pred_rule = beta_tr * gap[te]
                mom_beta = _fit_beta(past_r[tr], y_mid_tr)
                pred_mom = mom_beta * past_r[te]

                # Ridge regression
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X[tr])
                X_te_s = scaler.transform(X[te])
                ridge = Ridge(alpha=1.0).fit(X_tr_s, y_mid_tr)
                pred_ridge = ridge.predict(X_te_s)

                # LightGBM trained on mid change
                lgb_params = dict(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=400, random_state=seg.segment_id, n_jobs=-1, verbose=-1)
                m_mid = lgb.LGBMRegressor(**lgb_params)
                m_mid.fit(X[tr], y_mid_tr, eval_set=[(X[va], y_mid_va)], callbacks=[lgb.early_stopping(15, verbose=False)])
                pred_lgb_mid = m_mid.predict(X[te])

                # LightGBM trained on micro-price change, calibrated to mid change on val
                m_mic = lgb.LGBMRegressor(**lgb_params)
                m_mic.fit(X[tr], y_mic_tr, eval_set=[(X[va], y_mic_va)], callbacks=[lgb.early_stopping(15, verbose=False)])
                pred_lgb_mic = m_mic.predict(X[te])
                calib_mic_to_mid = _fit_beta(m_mic.predict(X[va]), y_mid_va)

                # Shuffled-target negative control (3 shuffles)
                ctrl_ics = []
                rng = np.random.default_rng(1000 + seg.segment_id * 100 + k)
                for rep in range(CONTROL_REPS):
                    y_shuf_tr = rng.permutation(y_mid_tr)
                    y_shuf_va = rng.permutation(y_mid_va)
                    m_ctrl = lgb.LGBMRegressor(**{**lgb_params, "random_state": seg.segment_id * 10 + rep})
                    m_ctrl.fit(X[tr], y_shuf_tr, eval_set=[(X[va], y_shuf_va)], callbacks=[lgb.early_stopping(15, verbose=False)])
                    sc = _score_reg(y_mid[te], m_ctrl.predict(X[te]))
                    ctrl_ics.append(sc["ic_pearson"])

                # Bootstrap CI for IC(lgb_mic -> mid) - IC(lgb_mid -> mid)
                diff_lo, diff_hi = _bootstrap_ic_diff(y_mid[te], pred_lgb_mic, pred_lgb_mid, seed=seg.segment_id + k)

                for mname, pred_v, calib_v in [
                    ("microprice_rule", pred_rule, 1.0),
                    ("momentum", pred_mom, 1.0),
                    ("ridge_mid", pred_ridge, 1.0),
                    ("lgbm_mid", pred_lgb_mid, 1.0),
                    ("lgbm_micro_to_mid", pred_lgb_mic, calib_mic_to_mid),
                ]:
                    sc = _score_reg(y_mid[te], pred_v, calib=calib_v)
                    all_rows.append({
                        "symbol": tag,
                        "tick_class": tick_class,
                        "segment": seg.segment_id,
                        "horizon_s": k,
                        "model": mname,
                        **sc,
                        "mic_vs_mid_ci_lo": diff_lo if mname == "lgbm_micro_to_mid" else np.nan,
                        "mic_vs_mid_ci_hi": diff_hi if mname == "lgbm_micro_to_mid" else np.nan,
                    })
                # Also record shuffled control & own-target R2 of micro-price
                sc_mic_own = _score_reg(y_mic[te], pred_lgb_mic, calib=1.0)
                all_rows.append({
                    "symbol": tag,
                    "tick_class": tick_class,
                    "segment": seg.segment_id,
                    "horizon_s": k,
                    "model": "lgbm_micro_on_micro",
                    **sc_mic_own,
                    "mic_vs_mid_ci_lo": np.nan,
                    "mic_vs_mid_ci_hi": np.nan,
                })
                all_rows.append({
                    "symbol": tag,
                    "tick_class": tick_class,
                    "segment": seg.segment_id,
                    "horizon_s": k,
                    "model": "shuffled_control",
                    "ic_pearson": float(np.nanmean(ctrl_ics)),
                    "ic_spearman": np.nan,
                    "r2": np.nan,
                    "hit_rate": np.nan,
                    "mic_vs_mid_ci_lo": np.nan,
                    "mic_vs_mid_ci_hi": np.nan,
                })

    df_b = pd.DataFrame(beta_rows)
    df_b.to_csv(mp_dir / "nse_microprice_beta_per_segment.csv", index=False)
    df_b_sum = df_b.groupby(["symbol", "tick_class", "horizon_s"], sort=False)[["beta_train", "beta_test", "mean_abs_gap_bp"]].mean().reset_index()
    df_b_sum.to_csv(mp_dir / "nse_microprice_beta_summary.csv", index=False)

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(mp_dir / "nse_microprice_per_segment.csv", index=False)
    df_sum = df_all.groupby(["symbol", "tick_class", "horizon_s", "model"], sort=False)[["ic_pearson", "ic_spearman", "r2", "hit_rate"]].mean().reset_index()
    df_sum.to_csv(mp_dir / "nse_microprice_summary.csv", index=False)

    print("\n>>> NSE MICRO-PRICE BETA SUMMARY:")
    print(df_b_sum.pivot(index=["symbol", "tick_class"], columns="horizon_s", values="beta_test").round(3).to_string())
    print("\n>>> NSE MICRO-PRICE IC SUMMARY (k = 10 s):")
    s10 = df_sum[df_sum["horizon_s"] == 10].pivot(index=["symbol", "tick_class"], columns="model", values="ic_pearson")
    cols = [c for c in ("microprice_rule", "momentum", "ridge_mid", "lgbm_mid", "lgbm_micro_to_mid", "shuffled_control") if c in s10.columns]
    print(s10[cols].round(4).to_string())


def main() -> None:
    t0 = time.time()
    run_cross_asset()
    run_microprice()
    print(f"\n[+] NSE Experiments F + K finished in {(time.time() - t0) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
