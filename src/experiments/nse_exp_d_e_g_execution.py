"""NSE Experiments D, E, G & Edge Decomposition: Execution Realism, Real Indian Fees,
Confidence Filtering, Cost-Aware Labels, Latency Sweep, and Market Regimes.

Evaluates whether NSE LOB predictions survive spread crossing, 5-level book walking,
1 s reaction delay, and real Indian statutory + exchange transaction costs:
  - 0.0 bp/side  (frictionless / prop maker)
  - 2.0 bp/side  (4.0 bp round trip: NSE intraday equity statutory + exchange, ₹0 brokerage)
  - 2.8 bp/side  (5.6 bp round trip: NSE NIFTY futures statutory + exchange, post-Budget 2026 0.05% sell STT)
  - 4.0 bp/side  (8.0 bp round trip: retail brokerage / Coinbase BTC comparison)

    python -m src.experiments.nse_exp_d_e_g_execution
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR
from src.data.nse_loader import load_nse_symbol
from src.execution.simulator import ExecutionSimulator
from src.execution.strategies import MicrostructureStrategyRunner
from src.microstructure.nse_pipeline import build_nse_segments, prepare_nse_segment
from src.models.tabular_trees import LightGBMLOB

OUT_DIR = ARTIFACTS_DIR / "nse" / "execution"
INSTRUMENTS = [
    ("RELIANCE", "EQUITY"),
    ("NIFTY", "DERIVATIVE"),
    ("HDFCBANK", "EQUITY"),
    ("ITC", "EQUITY"),
    ("IDEA", "EQUITY"),
    ("TCS", "EQUITY"),
]
COVERAGES = [1.0, 0.20, 0.10, 0.05, 0.02, 0.01]
FEE_TIERS_BPS = [0.0, 2.0, 2.8, 4.0]
LATENCIES_MS = [0.0, 1000.0, 2000.0, 5000.0, 10000.0, 30000.0]


def _day_safe_cost_aware_labels(
    lob: np.ndarray,
    day_id: np.ndarray,
    k: int,
    fee_bps: float,
) -> np.ndarray:
    """UP if buying at ask[t+1] and selling at bid[t+k+1] within same day beats 2*fee_bps."""
    n = len(lob)
    y = np.zeros(n, dtype=np.int64)
    if k + 1 >= n:
        return y
    f = fee_bps / 1e4
    valid = (day_id[:-k - 1] == day_id[1:-k]) & (day_id[:-k - 1] == day_id[k + 1:])
    idx = np.nonzero(valid)[0]
    ask_in, bid_in = lob[idx + 1, 0], lob[idx + 1, 2]
    ask_out, bid_out = lob[idx + k + 1, 0], lob[idx + k + 1, 2]
    y[idx[bid_out * (1.0 - f) > ask_in * (1.0 + f)]] = 1
    y[idx[bid_in * (1.0 - f) > ask_out * (1.0 + f)]] = 2
    return y


def _nse_round_trips(
    lob: np.ndarray,
    ts: np.ndarray,
    day_id: np.ndarray,
    rows: np.ndarray,
    sides: np.ndarray,
    k: int,
    lot: float,
) -> pd.DataFrame:
    """Day-safe non-overlapping enter/exit taker round trips (1 s delay = next row in same day)."""
    sim = ExecutionSimulator(
        initial_cash=1e11,
        taker_fee_bps=0.0,
        lot_size=lot,
        max_inventory=1e11,
        latency_ms=1.0,
        timestamps_ms=ts,
    )
    mid = (lob[:, 0] + lob[:, 2]) / 2.0
    n = len(lob)
    out = []
    busy_until = -1
    for t, side in zip(rows, sides):
        t = int(t)
        side = int(side)
        if t <= busy_until or t + k + 1 >= n:
            continue
        # Ensure decision, entry (t+1), and exit (t+k+1) are all on the same trading day
        if day_id[t] != day_id[t + 1] or day_id[t] != day_id[t + k + 1]:
            continue
        entry = sim.execute_aggressive_order(t, side, lob)
        exit_ = sim.execute_aggressive_order(t + k, -side, lob)
        if entry is None or exit_ is None:
            continue
        # Verify fills did not jump across a day boundary
        if day_id[entry.timestamp_idx] != day_id[t] or day_id[exit_.timestamp_idx] != day_id[t]:
            continue
        mid_t = mid[t]
        mid_tk = mid[t + k]
        mid_in = mid[entry.timestamp_idx]
        mid_out = mid[exit_.timestamp_idx]
        out.append({
            "row": t,
            "side": side,
            "edge_decision_mid_bps": side * (mid_tk - mid_t) / mid_t * 1e4,
            "edge_after_delay_bps": side * (mid_out - mid_in) / mid_in * 1e4,
            "gross_bps": side * (exit_.fill_price - entry.fill_price) / entry.fill_price * 1e4,
            "abs_move_bps": abs(mid_tk - mid_t) / mid_t * 1e4,
            "beyond_book": int(entry.beyond_book_size > 0 or exit_.beyond_book_size > 0),
        })
        busy_until = exit_.timestamp_idx
    return pd.DataFrame(out)


def _block_bootstrap_mean(vals: np.ndarray, block_len: int = 20, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    if len(vals) < 10:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(vals)
    bl = min(block_len, max(1, n // 5))
    nb = int(np.ceil(n / bl))
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        starts = rng.integers(0, max(1, n - bl + 1), size=nb)
        idx = np.concatenate([np.arange(s, min(s + bl, n)) for s in starts])[:n]
        means[i] = vals[idx].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def _rolling_vol_60s(mid: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Causal 60-second realized volatility (std of 1 s log returns) reset per trading day."""
    n = len(mid)
    lr = np.zeros(n, dtype=np.float64)
    same = day_id[1:] == day_id[:-1]
    lr[1:][same] = np.log(mid[1:][same] / mid[:-1][same])
    out = np.zeros(n, dtype=np.float64)
    changes = np.concatenate([[0], np.nonzero(day_id[1:] != day_id[:-1])[0] + 1, [n]])
    for a, b in zip(changes[:-1], changes[1:]):
        s = pd.Series(lr[a:b]).rolling(60, min_periods=5).std().fillna(0.0).to_numpy()
        out[a:b] = s * 1e4
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    exp_d_rows: List[Dict] = []
    lat_rows: List[Dict] = []
    trad_rows: List[Dict] = []
    decomp_rows: List[Dict] = []
    regime_trade_frames: List[pd.DataFrame] = []

    for sym, asset in INSTRUMENTS:
        sym_data = load_nse_symbol(sym, asset=asset)
        tag = sym_data.symbol
        segs = build_nse_segments(sym_data, n_segments=4)
        ref_fee = 2.8 if asset == "DERIVATIVE" else 2.0
        print(f"\n######## NSE EXECUTION + TRADABILITY + REGIMES: {tag} (ref fee={ref_fee} bp/side) ########")

        for seg in segs:
            prep = prepare_nse_segment(seg, horizons=[20])
            X = prep.tab_feat[:, prep.core_indices]  # 25 clean core 5-level LOB features
            y20 = prep.labels(20)
            sp = prep.splits(20)
            tr, va, te = sp["train"], sp["val"], sp["test"]

            # Train LightGBM on direction labels (k=20s)
            m_dir = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=300, random_state=seg.segment_id)
            m_dir.fit(X[tr], y20[tr], X[va], y20[va], early_stopping_rounds=15)
            pred_te = m_dir.predict(X[te])
            pr_va = m_dir.predict_proba(X[va])
            pr_te = m_dir.predict_proba(X[te])
            s_va = pr_va[:, 1] - pr_va[:, 2]
            s_te = pr_te[:, 1] - pr_te[:, 2]

            # --- Part 1: Exp D continuous execution (run day-by-day on test days so no overnight inventory!) ---
            ex = seg.exec_params
            vpin_col = X[:, prep.feat_names.index("vpin")]
            vpin_thresh = float(np.quantile(vpin_col[: prep.train_end], 0.90))
            test_days = np.unique(seg.day_id[te])

            for regime_name in ("Academic Mid-Price (Fantasy)", "Aggressive Taker", "Passive Maker (VPIN gate)"):
                tot_pnl, tot_trades, tot_fees, tot_slip = 0.0, 0, 0.0, 0.0
                for d_id in test_days:
                    dm = seg.day_id[te] == d_id
                    d_rows = te[dm]
                    d_lob = seg.lob_20[d_rows]
                    d_ts = seg.timestamps_ms[d_rows]
                    d_mid = (d_lob[:, 0] + d_lob[:, 2]) / 2.0
                    d_sig = pred_te[dm]
                    d_vpin = vpin_col[d_rows]
                    sim = ExecutionSimulator(
                        initial_cash=ex["initial_cash"],
                        maker_fee_bps=ex["maker_fee_bps"],
                        taker_fee_bps=ex["taker_fee_bps"],
                        lot_size=ex["lot_size"],
                        max_inventory=ex["max_inventory"],
                        latency_ms=ex["base_latency_ms"],
                        timestamps_ms=d_ts,
                    )
                    runner = MicrostructureStrategyRunner(sim)
                    if regime_name.startswith("Academic"):
                        res = runner.run_academic_midprice(d_sig, d_mid)
                    elif regime_name.startswith("Aggressive"):
                        res = runner.run_realistic_aggressive(d_sig, d_lob)
                    else:
                        res = runner.run_dynamic_passive_vpin(d_sig, d_lob, d_vpin, vpin_thresh)
                    tot_pnl += float(res["equity"][-1] - ex["initial_cash"])
                    tot_trades += int(res["num_trades"])
                    tot_fees += float(res.get("total_fees", 0.0))
                    tot_slip += float(res.get("total_slippage", 0.0))
                exp_d_rows.append({
                    "symbol": tag,
                    "segment": seg.segment_id,
                    "strategy": regime_name,
                    "net_pnl_inr": tot_pnl,
                    "num_trades": tot_trades,
                    "pnl_per_trade_inr": tot_pnl / max(tot_trades, 1),
                    "total_fees_inr": tot_fees,
                    "slippage_inr": tot_slip,
                })

            # Latency sweep (Aggressive Taker on test days)
            for lat_ms in LATENCIES_MS:
                tot_pnl = 0.0
                for d_id in test_days:
                    dm = seg.day_id[te] == d_id
                    d_rows = te[dm]
                    sim = ExecutionSimulator(
                        initial_cash=ex["initial_cash"],
                        maker_fee_bps=ex["maker_fee_bps"],
                        taker_fee_bps=ex["taker_fee_bps"],
                        lot_size=ex["lot_size"],
                        max_inventory=ex["max_inventory"],
                        latency_ms=lat_ms,
                        timestamps_ms=seg.timestamps_ms[d_rows],
                    )
                    res = MicrostructureStrategyRunner(sim).run_realistic_aggressive(pred_te[dm], seg.lob_20[d_rows])
                    tot_pnl += float(res["equity"][-1] - ex["initial_cash"])
                lat_rows.append({"symbol": tag, "segment": seg.segment_id, "latency_ms": lat_ms, "net_pnl_inr": tot_pnl})

            # --- Part 2: Exp E confidence filtering & Edge Decomposition (direction labels, k=20s) ---
            mid_all = (seg.lob_20[:, 0] + seg.lob_20[:, 2]) / 2.0
            sprd_bp = (seg.lob_20[:, 0] - seg.lob_20[:, 2]) / mid_all * 1e4
            vol60 = _rolling_vol_60s(mid_all, seg.day_id)
            tr_mask_full = seg.day_id < seg.train_days
            vpin_q = np.quantile(vpin_col[tr_mask_full], [1 / 3, 2 / 3])
            sprd_q = np.quantile(sprd_bp[tr_mask_full], [1 / 3, 2 / 3])
            vol_q = np.quantile(vol60[tr_mask_full], [1 / 3, 2 / 3])

            def _tercile(val: np.ndarray, qs: np.ndarray) -> np.ndarray:
                return np.where(val <= qs[0], "low", np.where(val <= qs[1], "mid", "high"))

            for cov in COVERAGES:
                if cov == 1.0:
                    mask = pred_te != 0
                    sides = np.where(pred_te[mask] == 1, 1, -1)
                    tau = np.nan
                else:
                    tau = float(np.quantile(np.abs(s_va), 1.0 - cov))
                    mask = np.abs(s_te) >= tau
                    sides = np.sign(s_te[mask]).astype(int)
                trips = _nse_round_trips(seg.lob_20, seg.timestamps_ms, seg.day_id, te[mask], sides, k=20, lot=ex["lot_size"])
                if len(trips) == 0:
                    continue

                rec = {
                    "symbol": tag,
                    "segment": seg.segment_id,
                    "label": "direction",
                    "horizon_k": 20,
                    "coverage": cov,
                    "trades": len(trips),
                    "edge_decision_mid_bps": float(trips["edge_decision_mid_bps"].mean()),
                    "edge_after_delay_bps": float(trips["edge_after_delay_bps"].mean()),
                    "gross_bps": float(trips["gross_bps"].mean()),
                    "abs_move_bps": float(trips["abs_move_bps"].mean()),
                    "hit_rate_decision": float((trips["edge_decision_mid_bps"] > 0).mean()),
                    "hit_rate_gross": float((trips["gross_bps"] > 0).mean()),
                    "beyond_book_pct": float(trips["beyond_book"].mean() * 100.0),
                }
                for fee in FEE_TIERS_BPS:
                    rec[f"net_fee{fee}_bps"] = rec["gross_bps"] - 2.0 * fee
                trad_rows.append(rec)

                # Save edge decomposition rows on all / 10% / 2%
                if cov in (1.0, 0.10, 0.02):
                    moved = trips[trips["abs_move_bps"] > 1e-9]
                    if len(moved):
                        decomp_rows.append({
                            "symbol": tag,
                            "segment": seg.segment_id,
                            "coverage": cov,
                            "trades": len(moved),
                            "edge_decision_mid_bps": float(moved["edge_decision_mid_bps"].mean()),
                            "edge_after_delay_bps": float(moved["edge_after_delay_bps"].mean()),
                            "gross_bps": float(moved["gross_bps"].mean()),
                            "mean_abs_move_bps": float(moved["abs_move_bps"].mean()),
                            "hit_rate_decision": float((moved["edge_decision_mid_bps"] > 0).mean()),
                        })

                # Save regime tags on all / 2% trades for Exp G
                if cov in (1.0, 0.02):
                    r_idx = trips["row"].to_numpy()
                    tdf = trips[["gross_bps", "edge_decision_mid_bps"]].copy()
                    tdf["symbol"] = tag
                    tdf["segment"] = seg.segment_id
                    tdf["coverage"] = cov
                    tdf["vpin_reg"] = _tercile(vpin_col[r_idx], vpin_q)
                    tdf["spread_reg"] = _tercile(sprd_bp[r_idx], sprd_q)
                    tdf["vol_reg"] = _tercile(vol60[r_idx], vol_q)
                    regime_trade_frames.append(tdf)

            # --- Part 3: Cost-aware labels at k = 60 s (trained to clear ref_fee per side) ---
            y_ca60 = _day_safe_cost_aware_labels(seg.lob_20, seg.day_id, k=60, fee_bps=ref_fee)
            sp60 = prep.splits(60)
            tr60, va60, te60 = sp60["train"], sp60["val"], sp60["test"]
            if np.sum(y_ca60[tr60] != 0) >= 200:
                m_ca = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=300, random_state=seg.segment_id)
                m_ca.fit(X[tr60], y_ca60[tr60], X[va60], y_ca60[va60], early_stopping_rounds=15)
                pr_va60 = m_ca.predict_proba(X[va60])
                pr_te60 = m_ca.predict_proba(X[te60])
                s_va60 = pr_va60[:, 1] - pr_va60[:, 2]
                s_te60 = pr_te60[:, 1] - pr_te60[:, 2]
                for cov in (0.10, 0.05, 0.02, 0.01):
                    tau = float(np.quantile(np.abs(s_va60), 1.0 - cov))
                    mask = np.abs(s_te60) >= tau
                    sides = np.sign(s_te60[mask]).astype(int)
                    trips = _nse_round_trips(seg.lob_20, seg.timestamps_ms, seg.day_id, te60[mask], sides, k=60, lot=ex["lot_size"])
                    if len(trips) == 0:
                        continue
                    rec = {
                        "symbol": tag,
                        "segment": seg.segment_id,
                        "label": f"cost_aware_{ref_fee}bp",
                        "horizon_k": 60,
                        "coverage": cov,
                        "trades": len(trips),
                        "edge_decision_mid_bps": float(trips["edge_decision_mid_bps"].mean()),
                        "edge_after_delay_bps": float(trips["edge_after_delay_bps"].mean()),
                        "gross_bps": float(trips["gross_bps"].mean()),
                        "abs_move_bps": float(trips["abs_move_bps"].mean()),
                        "hit_rate_decision": float((trips["edge_decision_mid_bps"] > 0).mean()),
                        "hit_rate_gross": float((trips["gross_bps"] > 0).mean()),
                        "beyond_book_pct": float(trips["beyond_book"].mean() * 100.0),
                    }
                    for fee in FEE_TIERS_BPS:
                        rec[f"net_fee{fee}_bps"] = rec["gross_bps"] - 2.0 * fee
                    trad_rows.append(rec)

        # Quick summary for symbol
        d_sub = pd.DataFrame([r for r in exp_d_rows if r["symbol"] == tag])
        print(d_sub.groupby("strategy")[["net_pnl_inr", "num_trades", "total_fees_inr"]].mean().round(1).to_string())

    # Save Exp D
    df_d = pd.DataFrame(exp_d_rows)
    df_d.to_csv(OUT_DIR / "nse_exp_d_per_segment.csv", index=False)
    d_sum = df_d.groupby(["symbol", "strategy"], sort=False)[["net_pnl_inr", "num_trades", "pnl_per_trade_inr", "total_fees_inr", "slippage_inr"]].agg(["mean", "std"])
    d_sum.columns = [f"{a}_{b}" if b == "std" else a for a, b in d_sum.columns]
    d_sum.reset_index().to_csv(OUT_DIR / "nse_exp_d_summary.csv", index=False)

    # Save latency sweep
    df_lat = pd.DataFrame(lat_rows)
    df_lat.to_csv(OUT_DIR / "nse_latency_sweep_per_segment.csv", index=False)
    df_lat.groupby(["symbol", "latency_ms"], sort=False)["net_pnl_inr"].agg(["mean", "std"]).reset_index().to_csv(OUT_DIR / "nse_latency_sweep_summary.csv", index=False)

    # Save Exp E tradability
    df_trad = pd.DataFrame(trad_rows)
    df_trad.to_csv(OUT_DIR / "nse_tradability_per_segment.csv", index=False)
    t_cols = [c for c in df_trad.columns if c not in ("symbol", "segment", "label", "horizon_k", "coverage")]
    df_trad.groupby(["symbol", "label", "horizon_k", "coverage"], sort=False)[t_cols].mean().reset_index().to_csv(OUT_DIR / "nse_tradability_summary.csv", index=False)

    # Save Edge Decomposition (weighted by trades across 4 segments)
    df_dec = pd.DataFrame(decomp_rows)
    df_dec.to_csv(OUT_DIR / "nse_edge_decomp_per_segment.csv", index=False)
    d_cols = ["edge_decision_mid_bps", "edge_after_delay_bps", "gross_bps", "mean_abs_move_bps", "hit_rate_decision"]
    dec_sum = (
        df_dec.groupby(["symbol", "coverage"], sort=False)
        .apply(lambda g: pd.Series({"trades": g["trades"].sum(), **{c: np.average(g[c], weights=g["trades"]) for c in d_cols}}), include_groups=False)
        .reset_index()
    )
    dec_sum["delay_cost_bps"] = dec_sum["edge_decision_mid_bps"] - dec_sum["edge_after_delay_bps"]
    dec_sum["crossing_cost_bps"] = dec_sum["edge_after_delay_bps"] - dec_sum["gross_bps"]
    for fee in FEE_TIERS_BPS:
        cost = dec_sum["delay_cost_bps"] + dec_sum["crossing_cost_bps"] + 2.0 * fee
        dec_sum[f"breakeven_hit_fee{fee}"] = 0.5 + cost / (2.0 * dec_sum["mean_abs_move_bps"])
    dec_sum.to_csv(OUT_DIR / "nse_edge_decomposition.csv", index=False)

    # Save Exp G Regimes
    all_reg = pd.concat(regime_trade_frames, ignore_index=True)
    reg_out = []
    for (sym, cov), g in all_reg.groupby(["symbol", "coverage"], sort=False):
        for var in ("vpin_reg", "spread_reg", "vol_reg"):
            for level in ("low", "mid", "high"):
                sub = g[g[var] == level]
                if len(sub) < 20:
                    continue
                lo, hi = _block_bootstrap_mean(sub["gross_bps"].to_numpy())
                seg_pos = sum(sub[sub["segment"] == s]["gross_bps"].mean() > 0 for s in sub["segment"].unique())
                reg_out.append({
                    "symbol": sym,
                    "coverage": cov,
                    "regime_var": var.replace("_reg", ""),
                    "regime": level,
                    "trades": len(sub),
                    "gross_bps": float(sub["gross_bps"].mean()),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "segments_pos": f"{seg_pos}/{sub['segment'].nunique()}",
                })
    pd.DataFrame(reg_out).to_csv(OUT_DIR / "nse_regimes_summary.csv", index=False)

    print("\n>>> NSE EDGE DECOMPOSITION SUMMARY (20 s hold):")
    show = ["symbol", "coverage", "trades", "edge_decision_mid_bps", "delay_cost_bps", "crossing_cost_bps", "gross_bps", "mean_abs_move_bps", "hit_rate_decision", "breakeven_hit_fee0.0", "breakeven_hit_fee2.8", "breakeven_hit_fee2.0"]
    print(dec_sum[show].round(3).to_string(index=False))
    print(f"\n[+] NSE Experiments D + E + G saved to {OUT_DIR} in {(time.time() - t0) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
