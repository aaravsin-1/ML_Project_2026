"""Broad 25-Instrument Indian NSE Cross-Sectional Study (64 Trading Days, 36M Snapshots).

Extends the core 8-instrument NSE suite to 25 liquid Indian instruments spanning
10 equity sectors plus NIFTY and BANKNIFTY front-month index futures (64 full
regular sessions, 1.44M 1-second 5-level snapshots per instrument = 36.0M rows).

Tests whether the project's three headline findings hold across the broader Indian market:
  1. Cross-sectional Tick-Size Predictability Law (relative tick -> 1-tick lock-in ->
     queue depth -> L1 imbalance correlation -> out-of-sample LightGBM ROC-AUC & F1).
  2. Universal Horizon Decay & Lift Over Momentum (5 s vs 20 s across 25 instruments).
  3. Market-Wide Taker Viability Boundary (observed spread crossing > decision-mid edge
     before Indian statutory STT & exchange fees on all 25 instruments).

    python -u -m src.experiments.nse_exp_broad_universe
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, roc_auc_score

from src.config import ARTIFACTS_DIR
from src.data.nse_loader import load_nse_symbol
from src.experiments.nse_exp_d_e_g_execution import _nse_round_trips
from src.microstructure.nse_pipeline import build_nse_segments, prepare_nse_segment
from src.models.baselines import NaiveMomentumBaseline
from src.models.tabular_trees import LightGBMLOB

OUT_DIR = ARTIFACTS_DIR / "nse" / "universe_25"

# 25 liquid NSE instruments across 10 sectors + 2 flagship index futures
UNIVERSE_25: List[Tuple[str, str, str]] = [
    ("IDEA", "EQUITY", "Telecom"),
    ("YESBANK", "EQUITY", "Banking"),
    ("SUZLON", "EQUITY", "Power/Renewables"),
    ("PNB", "EQUITY", "Banking"),
    ("CANBK", "EQUITY", "Banking"),
    ("ASHOKLEY", "EQUITY", "Auto"),
    ("ONGC", "EQUITY", "Oil & Gas"),
    ("ITC", "EQUITY", "FMCG"),
    ("BPCL", "EQUITY", "Oil & Gas"),
    ("NTPC", "EQUITY", "Power/Renewables"),
    ("BHEL", "EQUITY", "Capital Goods"),
    ("KOTAKBANK", "EQUITY", "Banking"),
    ("HDFCBANK", "EQUITY", "Banking"),
    ("SBIN", "EQUITY", "Banking"),
    ("BAJFINANCE", "EQUITY", "Financials/NBFC"),
    ("INFY", "EQUITY", "IT Services"),
    ("RELIANCE", "EQUITY", "Energy/Conglom."),
    ("AXISBANK", "EQUITY", "Banking"),
    ("ICICIBANK", "EQUITY", "Banking"),
    ("SUNPHARMA", "EQUITY", "Pharma"),
    ("BHARTIARTL", "EQUITY", "Telecom"),
    ("TCS", "EQUITY", "IT Services"),
    ("TITAN", "EQUITY", "Consumer"),
    ("NIFTY", "DERIVATIVE", "Index Futures"),
    ("BANKNIFTY", "DERIVATIVE", "Index Futures"),
]


def _infer_lot_size(lob: np.ndarray, asset: str, mid_med: float) -> float:
    if asset == "EQUITY":
        return float(max(1, int(round(100_000.0 / max(mid_med, 1.0)))))
    vals = np.unique(np.concatenate([lob[:100_000, 1], lob[:100_000, 3]]).astype(np.int64))
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 65.0
    g = int(np.gcd.reduce(vals))
    return float(max(1, g))


def _auc_ovr_and_ud(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    p = np.clip(probs.astype(np.float64), 1e-7, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    try:
        ovr = float(roc_auc_score(y_true, p, multi_class="ovr", average="macro", labels=[0, 1, 2]))
    except ValueError:
        ovr = np.nan
    mv = y_true != 0
    if mv.sum() > 20 and len(np.unique(y_true[mv])) == 2:
        p_up = p[mv, 1] / np.clip(p[mv, 1] + p[mv, 2], 1e-9, None)
        ud = float(roc_auc_score((y_true[mv] == 1).astype(int), p_up))
    else:
        ud = np.nan
    return ovr, ud


def _evaluate_one_instrument(sym: str, asset: str, sector: str) -> Dict[str, float | str | int]:
    t0 = time.time()
    sdata = load_nse_symbol(sym, asset=asset)
    tag = sdata.symbol
    lob = sdata.lob_20
    day_id = sdata.day_id

    ap1, aq1, bp1, bq1 = lob[:, 0], lob[:, 1], lob[:, 2], lob[:, 3]
    mid = 0.5 * (ap1 + bp1)
    spr = ap1 - bp1
    mid_med = float(np.median(mid))
    tick = float(sdata.tick_size)
    tick_bp = tick / mid_med * 1e4
    spr_bp_med = float(np.median(spr / mid * 1e4))
    one_tick_pct = float(np.mean(np.abs(spr - tick) < 0.25 * tick) * 100.0)
    bq1_med = float(np.median(bq1))

    same_day = day_id[1:] == day_id[:-1]
    unchanged_pct = float(np.all(lob[1:] == lob[:-1], axis=1)[same_day].mean() * 100.0)

    # Raw 20 s same-day L1 imbalance -> mid return correlation
    imb1 = (bq1 - aq1) / np.maximum(bq1 + aq1, 1e-9)
    same20 = day_id[:-20] == day_id[20:]
    ret20 = (mid[20:][same20] - mid[:-20][same20]) / mid[:-20][same20]
    corr_imb_20s = float(np.corrcoef(imb1[:-20][same20], ret20)[0, 1])

    lot_sz = _infer_lot_size(lob, asset=asset, mid_med=mid_med)
    stat_fee_side = 2.8 if asset == "DERIVATIVE" else 2.0

    segs = build_nse_segments(sdata, n_segments=4)
    f1_5_list, auc5_list = [], []
    f1_20_list, auc20_list, auc20_ud_list, mom20_list = [], [], [], []
    decomp_parts = []

    for seg in segs:
        prep = prepare_nse_segment(seg, horizons=[5, 20])
        X = prep.tab_feat[:, prep.core_indices]

        # k = 5 s
        y5 = prep.labels(5)
        sp5 = prep.splits(5)
        tr5, va5, te5 = sp5["train"], sp5["val"], sp5["test"]
        m5 = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=250, random_state=seg.segment_id)
        m5.fit(X[tr5], y5[tr5], X[va5], y5[va5], early_stopping_rounds=15)
        p5 = m5.predict(X[te5])
        pr5 = m5.predict_proba(X[te5])
        ovr5, _ = _auc_ovr_and_ud(y5[te5], pr5)
        f1_5_list.append(float(f1_score(y5[te5], p5, average="macro", zero_division=0)))
        auc5_list.append(ovr5)

        # k = 20 s
        y20 = prep.labels(20)
        sp20 = prep.splits(20)
        tr20, va20, te20 = sp20["train"], sp20["val"], sp20["test"]
        m20 = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=250, random_state=seg.segment_id)
        m20.fit(X[tr20], y20[tr20], X[va20], y20[va20], early_stopping_rounds=15)
        p20 = m20.predict(X[te20])
        pr20_va = m20.predict_proba(X[va20])
        pr20_te = m20.predict_proba(X[te20])
        ovr20, ud20 = _auc_ovr_and_ud(y20[te20], pr20_te)
        f1_20 = float(f1_score(y20[te20], p20, average="macro", zero_division=0))
        mom_p = NaiveMomentumBaseline(horizon_k=20).predict(y20, te20)
        mom_f1 = float(f1_score(y20[te20], mom_p, average="macro", zero_division=0))

        f1_20_list.append(f1_20)
        auc20_list.append(ovr20)
        auc20_ud_list.append(ud20)
        mom20_list.append(mom_f1)

        # Top-2% (or top-10% if ultra-large-tick has < 150 moved round trips per seg) taker round trips
        s_va = pr20_va[:, 1] - pr20_va[:, 2]
        s_te = pr20_te[:, 1] - pr20_te[:, 2]
        cov_used = 0.02
        tau = float(np.quantile(np.abs(s_va), 1.0 - cov_used))
        mask = np.abs(s_te) >= tau
        sides = np.sign(s_te[mask]).astype(int)
        trips = _nse_round_trips(seg.lob_20, seg.timestamps_ms, seg.day_id, te20[mask], sides, k=20, lot=lot_sz)
        moved = trips[trips["abs_move_bps"] > 1e-9] if len(trips) else pd.DataFrame()
        if len(moved) < 150:
            cov_used = 0.10
            tau = float(np.quantile(np.abs(s_va), 1.0 - cov_used))
            mask = np.abs(s_te) >= tau
            sides = np.sign(s_te[mask]).astype(int)
            trips = _nse_round_trips(seg.lob_20, seg.timestamps_ms, seg.day_id, te20[mask], sides, k=20, lot=lot_sz)
            moved = trips[trips["abs_move_bps"] > 1e-9] if len(trips) else pd.DataFrame()
        if len(moved):
            decomp_parts.append((
                cov_used,
                len(moved),
                float(moved["edge_decision_mid_bps"].mean()),
                float(moved["edge_after_delay_bps"].mean()),
                float(moved["gross_bps"].mean()),
                float((moved["edge_decision_mid_bps"] > 0).mean()),
            ))

    tot_tr = sum(x[1] for x in decomp_parts)
    w = np.array([x[1] for x in decomp_parts], dtype=np.float64)
    edge_mid = float(np.average([x[2] for x in decomp_parts], weights=w))
    edge_dly = float(np.average([x[3] for x in decomp_parts], weights=w))
    gross = float(np.average([x[4] for x in decomp_parts], weights=w))
    hit = float(np.average([x[5] for x in decomp_parts], weights=w))
    cov_rep = float(decomp_parts[0][0])

    rec = {
        "symbol": tag,
        "asset": asset,
        "sector": sector,
        "days": len(sdata.days),
        "rows": len(lob),
        "lot_size": lot_sz,
        "mid_inr": mid_med,
        "tick_inr": tick,
        "tick_bp": tick_bp,
        "spread_1tick_pct": one_tick_pct,
        "spread_bp_med": spr_bp_med,
        "bq1_med": bq1_med,
        "unchanged_1s_pct": unchanged_pct,
        "corr_imb1_20s": corr_imb_20s,
        "f1_5s": float(np.mean(f1_5_list)),
        "auc_ovr_5s": float(np.mean(auc5_list)),
        "f1_20s": float(np.mean(f1_20_list)),
        "f1_20s_std": float(np.std(f1_20_list, ddof=1)),
        "auc_ovr_20s": float(np.mean(auc20_list)),
        "auc_ud_20s": float(np.mean(auc20_ud_list)),
        "mom_f1_20s": float(np.mean(mom20_list)),
        "lift_mom_20s": float(np.mean(f1_20_list) - np.mean(mom20_list)),
        "top_cov": cov_rep,
        "top_trades": int(tot_tr),
        "hit_rate_20s": hit,
        "edge_mid_bps": edge_mid,
        "delay_1s_bps": edge_mid - edge_dly,
        "crossing_bps": edge_dly - gross,
        "gross_bps": gross,
        "statutory_rt_bps": 2.0 * stat_fee_side,
        "net_bps": gross - 2.0 * stat_fee_side,
    }
    print(
        f"[+] {tag:<14} ({sector:<16}) tick={tick_bp:5.2f}bp 1tk={one_tick_pct:5.1f}% "
        f"corr={corr_imb_20s:+.3f} | 5sAUC={rec['auc_ovr_5s']:.3f} 20sF1={rec['f1_20s']:.3f} "
        f"(UD={rec['auc_ud_20s']:.3f}) | mid={edge_mid:+.2f}bp cross={rec['crossing_bps']:.2f}bp "
        f"gross={gross:+.2f}bp [{time.time()-t0:.1f}s]"
    )
    return rec


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "nse_universe_25_summary.csv"
    done: Dict[str, Dict] = {}
    if out_csv.exists() and "--fresh" not in sys.argv:
        prev = pd.read_csv(out_csv)
        for r in prev.to_dict("records"):
            done[str(r["symbol"])] = r

    t0 = time.time()
    rows: List[Dict] = []
    for sym, asset, sector in UNIVERSE_25:
        tag = f"{sym}_FUT" if asset == "DERIVATIVE" else sym
        if tag in done:
            print(f"[=] {tag:<14} cached in summary CSV")
            rows.append(done[tag])
            continue
        rec = _evaluate_one_instrument(sym, asset, sector)
        rows.append(rec)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows).sort_values("tick_bp", ascending=False).reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    # Cross-sectional correlation summary against log10(tick_bp)
    lx = np.log10(df["tick_bp"].to_numpy())
    stat_rows = []
    for col, label in [
        ("spread_1tick_pct", "1-tick spread lock-in (%)"),
        ("corr_imb1_20s", "Raw Corr(L1 imb, 20 s move)"),
        ("auc_ovr_5s", "LightGBM 5 s OvR ROC-AUC"),
        ("auc_ovr_20s", "LightGBM 20 s OvR ROC-AUC"),
        ("auc_ud_20s", "LightGBM 20 s UP-vs-DOWN AUC"),
        ("f1_20s", "LightGBM 20 s Macro-F1"),
    ]:
        yv = df[col].to_numpy()
        r_p, p_p = pearsonr(lx, yv)
        r_s, p_s = spearmanr(df["tick_bp"].to_numpy(), yv)
        stat_rows.append({
            "metric": col,
            "description": label,
            "pearson_r_vs_log_tick": float(r_p),
            "pearson_p": float(p_p),
            "spearman_rho_vs_tick": float(r_s),
            "spearman_p": float(p_s),
        })
    df_corr = pd.DataFrame(stat_rows)
    df_corr.to_csv(OUT_DIR / "nse_universe_25_correlations.csv", index=False)

    print("\n>>> 25-INSTRUMENT NSE CROSS-SECTIONAL CORRELATIONS WITH RELATIVE TICK SIZE:")
    print(df_corr.round(4).to_string(index=False))
    print(f"\n>>> Positive pre-fee gross taker instruments: {(df['gross_bps'] > 0).sum()} / {len(df)}")
    print(f"[+] Saved 25-instrument NSE study to {OUT_DIR} in {(time.time() - t0) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
