"""Final 25-Stock Out-of-Universe Verification, SSF Augmentation Audit, Averaging (1..100), and K=1..8 Ensemble Study.

Freezes the exact LightGBM v2 (38f), Multi-Horizon Avg(10s, 20s), and SSF->Cash (46f) recipes
discovered on RELIANCE/NIFTY_FUT/IDEA (with zero per-stock tuning) and evaluates them strictly
out-of-sample across:
  1. All 25 NSE instruments across 5 relative-tick regimes (22 unseen + 3 dev, ~6.71M held-out rows).
  2. All 12 NSE equities with cached Single-Stock Futures (SSF) order books (10 unseen + 2 dev).
  3. Nonparametric statistical verification (Wilcoxon signed-rank & exact Binomial sign tests).
  4. Permanent reproducible exports for K=1..8 ensemble subsets and N=1..100 averaging/batch win rates.

Usage:
    python -u -m src.experiments.nse_final_25stock_verification
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS_DIR
from src.data.nse_loader import NSE_CACHE_DIR, load_nse_symbol
from src.experiments.nse_error_and_upgrade import compute_v2_features
from src.experiments.nse_exp_broad_universe import UNIVERSE_25
from src.microstructure.nse_pipeline import NSESegment, build_nse_segments, prepare_nse_segment
from src.models.tabular_trees import LightGBMLOB

OUT_DIR = ARTIFACTS_DIR / "nse" / "final_verification"
DEV_SYMS = {"RELIANCE", "NIFTY_FUT", "IDEA"}

SSF_FEATURE_NAMES: List[str] = [
    "ssf_imb_l1",
    "ssf_imb_l5",
    "ssf_imb1_5s",
    "basis_dev_bp",
    "ssf_ret_1s_bp",
    "ssf_ret_5s_bp",
    "lead_5s_bp",
    "lead_15s_bp",
]

SSF_FEATURE_META: Dict[str, Dict[str, str]] = {
    "ssf_imb_l1": {"label": "Futures L1 Imbalance", "group": "Cross-Market SSF", "desc": "Single-Stock Futures best bid/ask queue imbalance"},
    "ssf_imb_l5": {"label": "Futures L1-5 Imbalance", "group": "Cross-Market SSF", "desc": "Single-Stock Futures 5-level depth imbalance"},
    "ssf_imb1_5s": {"label": "Futures 5s Mean Imb", "group": "Cross-Market SSF", "desc": "Trailing 5s mean of SSF L1 queue imbalance"},
    "basis_dev_bp": {"label": "Futures-Cash Basis Dev", "group": "Cross-Market SSF", "desc": "30-min causally demeaned Futures vs Cash basis (bp)"},
    "ssf_ret_1s_bp": {"label": "Futures 1s Return (bp)", "group": "Cross-Market SSF", "desc": "1-second SSF mid-price return in basis points"},
    "ssf_ret_5s_bp": {"label": "Futures 5s Return (bp)", "group": "Cross-Market SSF", "desc": "5-second SSF mid-price return in basis points"},
    "lead_5s_bp": {"label": "SSF->Cash 5s Lead (bp)", "group": "Cross-Market SSF", "desc": "5s SSF return minus 5s Cash equity return"},
    "lead_15s_bp": {"label": "SSF->Cash 15s Lead (bp)", "group": "Cross-Market SSF", "desc": "15s SSF return minus 15s Cash equity return"},
}


def build_ssf_features(seg: NSESegment, ssf_npz: Any) -> Tuple[np.ndarray, int]:
    """Aligns cached 1s 5-level Single-Stock Futures LOB strictly causally by timestamps_ms."""
    n = len(seg.lob_20)
    c_mid = (seg.lob_20[:, 0] + seg.lob_20[:, 2]) / 2.0
    out = np.zeros((n, len(SSF_FEATURE_NAMES)), dtype=np.float32)

    ssf_ts = ssf_npz["timestamps_ms"]
    ssf_lob = ssf_npz["lob_20"]
    ssf_days = [str(d) for d in ssf_npz["days"].tolist()]
    ssf_day_id = ssf_npz["day_id"]

    changes = np.concatenate([[0], np.nonzero(seg.day_id[1:] != seg.day_id[:-1])[0] + 1, [n]])
    matched_days = 0

    for a, b in zip(changes[:-1], changes[1:]):
        d_str = seg.days[int(seg.day_id[a])]
        if d_str not in ssf_days:
            continue
        matched_days += 1
        smask = ssf_day_id == ssf_days.index(d_str)
        f_ts = ssf_ts[smask]
        f_lob = ssf_lob[smask]
        if len(f_ts) == 0:
            continue

        pos = np.clip(np.searchsorted(f_ts, seg.timestamps_ms[a:b], side="right") - 1, 0, len(f_ts) - 1)
        fl = f_lob[pos]
        f_mid = (fl[:, 0] + fl[:, 2]) / 2.0
        cm = c_mid[a:b]

        f_imb1 = (fl[:, 3] - fl[:, 1]) / (fl[:, 3] + fl[:, 1] + 1.0)
        bq5 = fl[:, 3::4].sum(axis=1)
        aq5 = fl[:, 1::4].sum(axis=1)
        f_imb5 = (bq5 - aq5) / (bq5 + aq5 + 1.0)

        basis_bp = (np.log(np.maximum(f_mid, 1e-6)) - np.log(np.maximum(cm, 1e-6))) * 1e4
        cs_b = np.cumsum(np.insert(basis_bp, 0, 0.0))
        L = b - a
        idx = np.arange(1, L + 1)
        win1800 = np.minimum(idx, 1800)
        basis_dev = basis_bp - (cs_b[idx] - cs_b[idx - win1800]) / win1800

        i0 = np.arange(L)
        f_ret1 = (f_mid - f_mid[np.maximum(0, i0 - 1)]) / np.maximum(f_mid, 1e-6) * 1e4
        f_ret5 = (f_mid - f_mid[np.maximum(0, i0 - 5)]) / np.maximum(f_mid, 1e-6) * 1e4
        c_ret5 = (cm - cm[np.maximum(0, i0 - 5)]) / np.maximum(cm, 1e-6) * 1e4
        f_ret15 = (f_mid - f_mid[np.maximum(0, i0 - 15)]) / np.maximum(f_mid, 1e-6) * 1e4
        c_ret15 = (cm - cm[np.maximum(0, i0 - 15)]) / np.maximum(cm, 1e-6) * 1e4

        cs_fi = np.cumsum(np.insert(f_imb1, 0, 0.0))
        win5 = np.minimum(idx, 5)
        f_imb1_5s = (cs_fi[idx] - cs_fi[idx - win5]) / win5

        out[a:b] = np.column_stack([
            f_imb1,
            f_imb5,
            f_imb1_5s,
            basis_dev,
            f_ret1,
            f_ret5,
            f_ret5 - c_ret5,
            f_ret15 - c_ret15,
        ]).astype(np.float32)

    return out, matched_days


def _fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = np.bincount(y_true * 3 + y_pred, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2.0 * tp + fp + fn
    f1s = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1.0), 0.0)
    return float(f1s.mean())


def _eval_probs(y: np.ndarray, probs: np.ndarray) -> Tuple[float, float, float, float]:
    pr = probs.argmax(1)
    pc = np.clip(probs.astype(np.float64), 1e-6, 1.0)
    pc /= pc.sum(axis=1, keepdims=True)
    f1 = _fast_macro_f1(y, pr)
    acc = float(accuracy_score(y, pr))
    auc = float(roc_auc_score(y, pc, multi_class="ovr", average="macro", labels=[0, 1, 2]))
    dm = (pr != 0) & (y != 0)
    dh = float((pr[dm] == y[dm]).mean()) if dm.sum() > 0 else np.nan
    return f1, acc, auc, dh


def _tick_bucket(tick_bp: float) -> str:
    if tick_bp < 0.10:
        return "1. Tiny Tick (<0.10 bp)"
    if tick_bp < 0.65:
        return "2. Small Tick (0.20-0.65 bp)"
    if tick_bp < 1.30:
        return "3. Medium Tick (0.65-1.30 bp)"
    if tick_bp < 2.50:
        return "4. Large Tick (1.30-2.50 bp)"
    return "5. Ultra-Large Tick (>2.50 bp)"


def _run_stat_test(name: str, subset_name: str, a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    diff = b - a
    wins = int(np.sum(diff > 0))
    n = int(len(diff))
    w_stat, p_wil_1s = wilcoxon(diff, alternative="greater")
    _, p_wil_2s = wilcoxon(diff, alternative="two-sided")
    p_binom = float(binomtest(wins, n, p=0.5, alternative="greater").pvalue)
    return {
        "comparison": name,
        "cohort": subset_name,
        "n_instruments": n,
        "wins": wins,
        "win_rate_pct": round(wins / max(n, 1) * 100.0, 2),
        "mean_before": round(float(np.mean(a)), 5),
        "mean_after": round(float(np.mean(b)), 5),
        "mean_delta": round(float(np.mean(diff)), 5),
        "median_delta": round(float(np.median(diff)), 5),
        "min_delta": round(float(np.min(diff)), 5),
        "max_delta": round(float(np.max(diff)), 5),
        "wilcoxon_stat": float(w_stat),
        "p_wilcoxon_onesided": float(p_wil_1s),
        "p_wilcoxon_twosided": float(p_wil_2s),
        "p_binom_signtest": p_binom,
    }


def _export_averaging_and_k_studies() -> None:
    """Writes clean CSVs for averaging (N=1..100), batch majority win-rates, and K=1..8 subsets."""
    grid_n = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    avg_rows: List[Dict[str, Any]] = []
    k_rows: List[Dict[str, Any]] = []

    pool7 = ["NaiveMom", "LogReg", "RF", "LGBM_v1", "LSTM", "DeepLOB5", "TLOB"]

    for sym_in, asset in [("RELIANCE", "EQUITY"), ("NIFTY", "DERIVATIVE"), ("IDEA", "EQUITY")]:
        print(f"[final_verification] Generating averaging & K=1..7 tables for {sym_in}...", flush=True)
        sym_data = load_nse_symbol(sym_in, asset=asset)
        sym = sym_data.symbol
        segs = build_nse_segments(sym_data, n_segments=4)

        ys: List[np.ndarray] = []
        trail_p: Dict[int, List[np.ndarray]] = {n_w: [] for n_w in grid_n}
        block_wins: Dict[int, List[float]] = {n_w: [] for n_w in grid_n}
        p_seg: Dict[str, List[np.ndarray]] = {m: [] for m in pool7 + ["LGBM_v2"]}

        for s in range(4):
            seg = segs[s]
            prep = prepare_nse_segment(seg, horizons=[20])
            sp20 = prep.splits(20)
            te20 = sp20["test"]
            y_all20 = prep.labels(20)
            y_te20 = y_all20[te20].astype(np.int64)
            ys.append(y_te20)

            core = prep.core_indices
            X0 = prep.tab_feat[:, core]
            cnames = [prep.feat_names[i] for i in core]
            X_v2, _ = compute_v2_features(seg, X0, cnames)

            z = np.load(ARTIFACTS_DIR / "nse" / "models" / sym / "sessions" / f"seg{s}_k20_e10.npz", allow_pickle=True)
            p_seg["NaiveMom"].append(z["prob_Naive Momentum"].astype(np.float32))
            p_seg["LogReg"].append(z["prob_Logistic Regression"].astype(np.float32))
            p_seg["RF"].append(z["prob_Random Forest"].astype(np.float32))
            p_seg["LGBM_v1"].append(z["prob_LightGBM"].astype(np.float32))
            p_seg["LSTM"].append(z["prob_LSTM"].astype(np.float32))
            p_seg["DeepLOB5"].append(z["prob_DeepLOB5"].astype(np.float32))
            p_seg["TLOB"].append(z["prob_TLOB"].astype(np.float32))

            m20 = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=250, random_state=42 + s)
            m20.fit(X_v2[sp20["train"]], y_all20[sp20["train"]], X_v2[sp20["val"]], y_all20[sp20["val"]], early_stopping_rounds=20)
            p_full = m20.predict_proba(X_v2).astype(np.float32)
            p_te = p_full[te20]
            p_seg["LGBM_v2"].append(p_te)

            changes = np.concatenate([[0], np.nonzero(seg.day_id[1:] != seg.day_id[:-1])[0] + 1, [len(seg.day_id)]])
            for n_w in grid_n:
                if n_w == 1:
                    trail_p[1].append(p_te)
                else:
                    out_n = np.empty_like(p_full)
                    for a, b in zip(changes[:-1], changes[1:]):
                        sub = p_full[a:b]
                        cs = np.cumsum(np.vstack([np.zeros((1, 3), dtype=np.float32), sub]), axis=0)
                        idx = np.arange(1, len(sub) + 1)
                        win = np.minimum(idx, n_w)
                        out_n[a:b] = (cs[idx] - cs[idx - win]) / win[:, None]
                    trail_p[n_w].append(out_n[te20])

            te_days = seg.day_id[te20]
            pr_te = p_te.argmax(1)
            for d_loc in np.unique(te_days):
                dm = te_days == d_loc
                yd, prd = y_te20[dm], pr_te[dm]
                dir_m = (prd != 0) & (yd != 0)
                hits_seq = (prd[dir_m] == yd[dir_m]).astype(np.float64)
                for n_w in grid_n:
                    if n_w == 1:
                        if len(hits_seq):
                            block_wins[1].append(float(hits_seq.mean()))
                    elif len(hits_seq) >= n_w:
                        n_blk = len(hits_seq) // n_w
                        blks = hits_seq[:n_blk * n_w].reshape(n_blk, n_w).mean(axis=1)
                        block_wins[n_w].extend(((blks > 0.5) + 0.5 * (blks == 0.5)).tolist())

        y = np.concatenate(ys)
        for n_w in grid_n:
            pn = np.vstack(trail_p[n_w])
            f1, acc, auc, dh = _eval_probs(y, pn)
            hi_m = (pn.max(1) >= 0.50) & (pn.argmax(1) != 0) & (y != 0)
            dh_hi = float((pn.argmax(1)[hi_m] == y[hi_m]).mean()) if hi_m.sum() > 50 else np.nan
            avg_rows.append({
                "symbol": sym,
                "n_window": n_w,
                "trail_f1": round(f1, 5),
                "trail_acc": round(acc, 5),
                "trail_auc": round(auc, 5),
                "trail_dir_hit": round(dh, 5),
                "trail_conf50_dir_hit": round(dh_hi, 5) if np.isfinite(dh_hi) else np.nan,
                "batch_majority_win_pct": round(float(np.nanmean(block_wins[n_w]) * 100.0), 2),
            })

        # Evaluate Original 7 K=1,2,3,4,7 + Upgraded Pool (with LGBM_v2) K=1,2,3,4,8
        yt_folds = [np.concatenate([ys[j] for j in range(4) if j != s]) for s in range(4)]

        def best_for_pool(pool_name: str, pool_list: List[str], k_sizes: List[int], must_include: str | None) -> None:
            m_tr = {m: [np.vstack([p_seg[m][j] for j in range(4) if j != s]) for s in range(4)] for m in pool_list}
            for K in k_sizes:
                subs = list(itertools.combinations(pool_list, K))
                if must_include and K in (3, 4):
                    subs = [sb for sb in subs if must_include in sb]
                best_rec = None
                for sb in subs:
                    if K == 1:
                        po = np.vstack(p_seg[sb[0]])
                        ws = [1.0]
                    else:
                        oos = []
                        w_avg = np.zeros(K)
                        for s in range(4):
                            yt = yt_folds[s][::4]
                            mats = [m_tr[m][s][::4] for m in sb]
                            w = np.ones(K, dtype=np.float32) / K
                            cur_p = sum(w[i] * mats[i] for i in range(K))
                            best_f = _fast_macro_f1(yt, cur_p.argmax(1))
                            for _ in range(2):
                                for i in range(K):
                                    other_p = cur_p - w[i] * mats[i]
                                    rem_w = float(w.sum() - w[i])
                                    norm_other = other_p / max(rem_w, 1e-6) if rem_w > 1e-5 else mats[i]
                                    for alpha in (0.0, 0.08, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 0.90):
                                        cand_p = np.float32(alpha) * mats[i] + np.float32(1.0 - alpha) * norm_other
                                        f = _fast_macro_f1(yt, cand_p.argmax(1))
                                        if f > best_f + 1e-5:
                                            best_f = f
                                            w = w * ((1.0 - alpha) / rem_w) if rem_w > 1e-5 else np.zeros_like(w)
                                            w[i] = alpha
                                            w /= max(float(w.sum()), 1e-8)
                                            cur_p = cand_p
                            w /= max(float(w.sum()), 1e-8)
                            w_avg += w / 4.0
                            oos.append(sum(np.float32(w[i]) * p_seg[sb[i]][s] for i in range(K)))
                        po = np.vstack(oos)
                        po /= np.maximum(po.sum(axis=1, keepdims=True), 1e-8)
                        ws = np.round(w_avg, 3).tolist()
                    f1_cand = _fast_macro_f1(y, po.argmax(1))
                    if best_rec is None or f1_cand > best_rec[0]:
                        f1, acc, auc, dh = _eval_probs(y, po)
                        best_rec = (f1, acc, auc, dh, sb, ws)
                f1, acc, auc, dh, sb, ws = best_rec
                k_rows.append({
                    "symbol": sym,
                    "pool": pool_name,
                    "k": K,
                    "macro_f1": round(f1, 5),
                    "accuracy": round(acc, 5),
                    "roc_auc_ovr": round(auc, 5),
                    "dir_hit": round(dh, 5),
                    "models": " + ".join(sb),
                    "weights": str(ws),
                })

        best_for_pool("Original-7", pool7, [1, 2, 3, 4, 7], None)
        best_for_pool("Upgraded-8 (with LGBM_v2)", pool7 + ["LGBM_v2"], [1, 2, 3, 4, 8], "LGBM_v2")

    pd.DataFrame(avg_rows).to_csv(OUT_DIR / "averaging_1_to_100_summary.csv", index=False)
    pd.DataFrame(k_rows).to_csv(OUT_DIR / "ensemble_k1_to_k8_summary.csv", index=False)


def _eval_one_instrument(args: Tuple[int, str, str, str, int]) -> Dict[str, Any]:
    idx_sym, sym_in, asset, sector, lgb_jobs = args
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
    p_v1_l, p_v2_l, p_v2_10_l = [], [], []
    p_ssf_20_l, p_ssf_10_l = [], []
    flat_err_tot, all_err_tot = 0, 0
    matched_days_tot = 0
    day_keys: List[str] = []

    for s in range(4):
        seg = segs[s]
        prep = prepare_nse_segment(seg, horizons=[10, 20])
        sp20 = prep.splits(20)
        sp10 = prep.splits(10)
        te20 = sp20["test"]
        y_all20 = prep.labels(20)
        y_all10 = prep.labels(10)
        y_te = y_all20[te20].astype(np.int64)
        ys_te.append(y_te)

        core = prep.core_indices
        X0 = np.ascontiguousarray(prep.tab_feat[:, core], dtype=np.float32)
        cnames = [prep.feat_names[i] for i in core]
        X_v2, _ = compute_v2_features(seg, X0, cnames)
        X_v2 = np.ascontiguousarray(X_v2, dtype=np.float32)

        # 1. Baseline v1 (25f, k=20)
        m_v1 = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=250, random_state=42 + s, n_jobs=lgb_jobs)
        m_v1.fit(X0[sp20["train"]], y_all20[sp20["train"]], X0[sp20["val"]], y_all20[sp20["val"]], early_stopping_rounds=20)
        pv1 = m_v1.predict_proba(X0[te20]).astype(np.float64)
        p_v1_l.append(pv1)

        pred_v1 = pv1.argmax(1)
        wrong = pred_v1 != y_te
        opp = ((pred_v1 == 1) & (y_te == 2)) | ((pred_v1 == 2) & (y_te == 1))
        all_err_tot += int(wrong.sum())
        flat_err_tot += int((wrong & (~opp)).sum())

        # 2. Upgraded v2 (38f, k=20 & k=10)
        m_v2 = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=250, random_state=42 + s, n_jobs=lgb_jobs)
        m_v2.fit(X_v2[sp20["train"]], y_all20[sp20["train"]], X_v2[sp20["val"]], y_all20[sp20["val"]], early_stopping_rounds=20)
        p_v2_l.append(m_v2.predict_proba(X_v2[te20]).astype(np.float64))

        m_v2_10 = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=250, random_state=42 + s, n_jobs=lgb_jobs)
        m_v2_10.fit(X_v2[sp10["train"]], y_all10[sp10["train"]], X_v2[sp10["val"]], y_all10[sp10["val"]], early_stopping_rounds=20)
        p_v2_10_l.append(m_v2_10.predict_proba(X_v2[te20]).astype(np.float64))

        # 3. Cross-Market SSF Augmentation (46f, k=20 & k=10) if SSF exists
        if has_ssf:
            X_ssf, md = build_ssf_features(seg, ssf_npz)
            matched_days_tot += md
            X_aug = np.ascontiguousarray(np.hstack([X_v2, X_ssf]), dtype=np.float32)
            m_s20 = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=250, random_state=42 + s, n_jobs=lgb_jobs)
            m_s20.fit(X_aug[sp20["train"]], y_all20[sp20["train"]], X_aug[sp20["val"]], y_all20[sp20["val"]], early_stopping_rounds=20)
            p_ssf_20_l.append(m_s20.predict_proba(X_aug[te20]).astype(np.float64))

            m_s10 = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=250, random_state=42 + s, n_jobs=lgb_jobs)
            m_s10.fit(X_aug[sp10["train"]], y_all10[sp10["train"]], X_aug[sp10["val"]], y_all10[sp10["val"]], early_stopping_rounds=20)
            p_ssf_10_l.append(m_s10.predict_proba(X_aug[te20]).astype(np.float64))

        for d_loc in seg.day_id[te20]:
            day_keys.append(seg.days[int(d_loc)])

    y = np.concatenate(ys_te)
    p_v1 = np.vstack(p_v1_l)
    p_v2 = np.vstack(p_v2_l)
    p_v2_mh = 0.5 * np.vstack(p_v2_10_l) + 0.5 * p_v2

    v1_f1, v1_acc, v1_auc, v1_dh = _eval_probs(y, p_v1)
    v2_f1, v2_acc, v2_auc, v2_dh = _eval_probs(y, p_v2)
    mh_f1, mh_acc, mh_auc, mh_dh = _eval_probs(y, p_v2_mh)

    if has_ssf:
        p_ssf20 = np.vstack(p_ssf_20_l)
        p_ssf_mh = 0.5 * np.vstack(p_ssf_10_l) + 0.5 * p_ssf20
        s20_f1, s20_acc, s20_auc, s20_dh = _eval_probs(y, p_ssf20)
        smh_f1, smh_acc, smh_auc, smh_dh = _eval_probs(y, p_ssf_mh)
        p_best = p_ssf_mh
        best_f1, best_acc, best_auc, best_dh = smh_f1, smh_acc, smh_auc, smh_dh
        best_name = "v2 + SSF + MH(10,20)"
    else:
        s20_f1 = s20_acc = s20_auc = s20_dh = np.nan
        smh_f1 = smh_acc = smh_auc = smh_dh = np.nan
        p_best = p_v2_mh
        best_f1, best_acc, best_auc, best_dh = mh_f1, mh_acc, mh_auc, mh_dh
        best_name = "v2 + MH(10,20)"

    day_arr = np.asarray(day_keys)
    pred_best = p_best.argmax(1)
    day_dhs: List[float] = []
    blk_10, blk_50, blk_100 = [], [], []
    for d_u in sorted(set(day_keys)):
        dm = day_arr == d_u
        yd, prd = y[dm], pred_best[dm]
        m_dir = (prd != 0) & (yd != 0)
        if m_dir.sum() > 20:
            hits = (prd[m_dir] == yd[m_dir]).astype(np.float64)
            day_dhs.append(float(hits.mean()))
            for bs, bl in [(10, blk_10), (50, blk_50), (100, blk_100)]:
                if len(hits) >= bs:
                    nb = len(hits) // bs
                    means = hits[:nb * bs].reshape(nb, bs).mean(axis=1)
                    bl.extend(((means > 0.5) + 0.5 * (means == 0.5)).tolist())

    rec = {
        "symbol": sym,
        "asset": asset,
        "sector": sector,
        "is_dev_stock": sym in DEV_SYMS,
        "has_ssf": has_ssf,
        "ssf_matched_days": matched_days_tot if has_ssf else 0,
        "mid_inr": round(mid_med, 2),
        "tick_inr": float(sym_data.tick_size),
        "tick_bp": round(tick_bp, 4),
        "tick_regime": _tick_bucket(tick_bp),
        "n_test": int(len(y)),
        "flat_boundary_err_share_pct": round(flat_err_tot / max(all_err_tot, 1) * 100.0, 2),
        "v1_f1": round(v1_f1, 5),
        "v1_acc": round(v1_acc, 5),
        "v1_auc": round(v1_auc, 5),
        "v1_dir_hit": round(v1_dh, 5),
        "v2_f1": round(v2_f1, 5),
        "v2_auc": round(v2_auc, 5),
        "v2_dir_hit": round(v2_dh, 5),
        "v2_mh_f1": round(mh_f1, 5),
        "v2_mh_acc": round(mh_acc, 5),
        "v2_mh_auc": round(mh_auc, 5),
        "v2_mh_dir_hit": round(mh_dh, 5),
        "ssf_20s_f1": round(s20_f1, 5) if has_ssf else np.nan,
        "ssf_20s_auc": round(s20_auc, 5) if has_ssf else np.nan,
        "ssf_mh_f1": round(smh_f1, 5) if has_ssf else np.nan,
        "ssf_mh_auc": round(smh_auc, 5) if has_ssf else np.nan,
        "ssf_mh_dir_hit": round(smh_dh, 5) if has_ssf else np.nan,
        "best_deployed_model": best_name,
        "best_f1": round(best_f1, 5),
        "best_acc": round(best_acc, 5),
        "best_auc": round(best_auc, 5),
        "best_dir_hit": round(best_dh, 5),
        "delta_v2mh_vs_v1_f1": round(mh_f1 - v1_f1, 5),
        "delta_v2mh_vs_v1_auc": round(mh_auc - v1_auc, 5),
        "delta_ssf_vs_v2mh_f1": round(smh_f1 - mh_f1, 5) if has_ssf else np.nan,
        "delta_best_vs_v1_f1": round(best_f1 - v1_f1, 5),
        "delta_best_vs_v1_auc": round(best_auc - v1_auc, 5),
        "days_evaluated": len(day_dhs),
        "days_beating_55pct_dirhit": int(sum(d > 0.55 for d in day_dhs)),
        "min_day_dir_hit_pct": round(min(day_dhs) * 100.0, 2) if day_dhs else np.nan,
        "max_day_dir_hit_pct": round(max(day_dhs) * 100.0, 2) if day_dhs else np.nan,
        "batch10_win_pct": round(float(np.mean(blk_10) * 100.0), 2) if blk_10 else np.nan,
        "batch50_win_pct": round(float(np.mean(blk_50) * 100.0), 2) if blk_50 else np.nan,
        "batch100_win_pct": round(float(np.mean(blk_100) * 100.0), 2) if blk_100 else np.nan,
        "_elapsed_s": round(time.time() - t0, 1),
    }
    return rec


def run_25stock_verification(workers: int = 4) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    t_start = time.time()
    lgb_jobs = max(4, 32 // max(workers, 1))
    tasks = [(i + 1, sym_in, asset, sector, lgb_jobs) for i, (sym_in, asset, sector) in enumerate(UNIVERSE_25)]

    print(f"[final_verification] Running 25 instruments across {workers} workers ({lgb_jobs} LGB threads each)...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_eval_one_instrument, t) for t in tasks]
        for done_cnt, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            el = rec.pop("_elapsed_s")
            rows.append(rec)
            print(
                f"[{done_cnt:2d}/25] {rec['symbol']:13s} ({rec['tick_bp']:5.3f}bp) | "
                f"v1 F1={rec['v1_f1']:.4f} -> v2+MH F1={rec['v2_mh_f1']:.4f} ({rec['delta_v2mh_vs_v1_f1']:+.4f})"
                + (f" -> +SSF F1={rec['ssf_mh_f1']:.4f} ({rec['delta_ssf_vs_v2mh_f1']:+.4f})" if rec["has_ssf"] else "")
                + f" | AUC {rec['v1_auc']:.4f}->{rec['best_auc']:.4f} | B100={rec['batch100_win_pct']}% ({el:.1f}s)",
                flush=True,
            )

    df = pd.DataFrame(rows).sort_values("tick_bp", ascending=False).reset_index(drop=True)
    df.to_csv(OUT_DIR / "universe25_upgrade_verification.csv", index=False)

    # Tick-regime aggregation
    reg_df = df.groupby("tick_regime", as_index=False).agg(
        n_instruments=("symbol", "count"),
        mean_tick_bp=("tick_bp", "mean"),
        mean_boundary_err_pct=("flat_boundary_err_share_pct", "mean"),
        mean_v1_f1=("v1_f1", "mean"),
        mean_v2_mh_f1=("v2_mh_f1", "mean"),
        mean_best_f1=("best_f1", "mean"),
        mean_delta_f1=("delta_best_vs_v1_f1", "mean"),
        mean_v1_auc=("v1_auc", "mean"),
        mean_best_auc=("best_auc", "mean"),
        mean_delta_auc=("delta_best_vs_v1_auc", "mean"),
        mean_batch100_win_pct=("batch100_win_pct", "mean"),
    )
    reg_df.to_csv(OUT_DIR / "tick_regime_verification.csv", index=False)

    # Nonparametric Statistical Tests
    stat_rows: List[Dict[str, Any]] = []
    unseen = df[~df["is_dev_stock"]]
    ssf_all = df[df["has_ssf"]]
    ssf_unseen = ssf_all[~ssf_all["is_dev_stock"]]

    for cohort_name, sub_df in [("All 25 Instruments", df), ("Unseen 22 Instruments", unseen)]:
        stat_rows.append(_run_stat_test("Macro-F1: v2 (38f, 20s) vs v1 (25f, 20s)", cohort_name, sub_df["v1_f1"].to_numpy(), sub_df["v2_f1"].to_numpy()))
        stat_rows.append(_run_stat_test("Macro-F1: v2+MH(10,20) vs v1 (25f)", cohort_name, sub_df["v1_f1"].to_numpy(), sub_df["v2_mh_f1"].to_numpy()))
        stat_rows.append(_run_stat_test("ROC-AUC: v2+MH(10,20) vs v1 (25f)", cohort_name, sub_df["v1_auc"].to_numpy(), sub_df["v2_mh_auc"].to_numpy()))
        stat_rows.append(_run_stat_test("Macro-F1: Best Deployed (incl SSF) vs v1", cohort_name, sub_df["v1_f1"].to_numpy(), sub_df["best_f1"].to_numpy()))
        stat_rows.append(_run_stat_test("ROC-AUC: Best Deployed (incl SSF) vs v1", cohort_name, sub_df["v1_auc"].to_numpy(), sub_df["best_auc"].to_numpy()))

    for cohort_name, sub_df in [("All 12 SSF Equities", ssf_all), ("Unseen 10 SSF Equities", ssf_unseen)]:
        stat_rows.append(_run_stat_test("Macro-F1: v2+SSF+MH vs v2+MH (Pure SSF Lift)", cohort_name, sub_df["v2_mh_f1"].to_numpy(), sub_df["ssf_mh_f1"].to_numpy()))
        stat_rows.append(_run_stat_test("ROC-AUC: v2+SSF+MH vs v2+MH (Pure SSF Lift)", cohort_name, sub_df["v2_mh_auc"].to_numpy(), sub_df["ssf_mh_auc"].to_numpy()))
        stat_rows.append(_run_stat_test("Macro-F1: v2+SSF (20s) vs v2 (20s)", cohort_name, sub_df["v2_f1"].to_numpy(), sub_df["ssf_20s_f1"].to_numpy()))

    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(OUT_DIR / "statistical_tests_summary.csv", index=False)

    # Also write permanent CSVs for averaging 1..100 & K=1..8 subsets
    _export_averaging_and_k_studies()

    print("\n================ FINAL 25-STOCK VERIFICATION SUMMARY ================")
    print(reg_df.to_string(index=False))
    print("\n================ NONPARAMETRIC ANTI-OVERFITTING TESTS ================")
    print(stat_df[["comparison", "cohort", "wins", "n_instruments", "mean_delta", "p_wilcoxon_onesided", "p_binom_signtest"]].to_string(index=False))
    print(f"\nCompleted full verification in {time.time() - t_start:.1f}s -> {OUT_DIR}")


if __name__ == "__main__":
    run_25stock_verification()
