"""Experiment E: Can the real BTC signal be traded after costs?

Experiments A-D show a small but real edge on Coinbase BTC that is several times smaller
than the taker fee. This experiment asks whether a subset of it is tradable:

1. Confidence filtering: score s = P(UP) - P(DOWN); go long when s >= tau, short when
   s <= -tau. tau is set from the VALIDATION distribution of |s| so that a chosen share
   (coverage) of rows would trade; the test set never influences it.
2. Cost-aware labels: instead of "the smoothed mid moved more than alpha", label UP only
   if buying at the ask now and selling at the bid k seconds later beats the round-trip
   fee (DOWN symmetric). The model then learns to spot tradable moves directly.

Every signal is a round trip: enter with a taker order (arriving at the next snapshot),
exit with a taker order k seconds later, walking the book both times; positions never
overlap. Gross P&L includes spread crossing and book walking; fees are applied per side
so results can be shown for several fee tiers from the same trades.
"""

from typing import Dict, List
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR
from src.experiments.common import PreparedData, btc_sessions, prepare_data
from src.execution.simulator import ExecutionSimulator
from src.models.tabular_trees import LightGBMLOB

COVERAGES = [1.0, 0.20, 0.10, 0.05, 0.02, 0.01, 0.005]
FEE_TIERS_BPS = [0.0, 1.0, 2.0, 4.0]
COST_AWARE_HORIZONS = [20, 60, 300]
REFERENCE_FEE_BPS = 4.0


def cost_aware_labels(lob: np.ndarray, ts: np.ndarray, k: int, fee_bps: float, latency_ms: float = 1.0) -> np.ndarray:
    """UP if long entry at the next snapshot's ask and exit k rows later at the bid clears fees; DOWN symmetric."""
    n = len(lob)
    entry = np.minimum(np.searchsorted(ts, ts + latency_ms, side="left"), n - 1)
    exit_ = np.minimum(entry + k, n - 1)
    f = fee_bps / 1e4
    ask_in, bid_in = lob[entry, 0], lob[entry, 2]
    ask_out, bid_out = lob[exit_, 0], lob[exit_, 2]
    y = np.zeros(n, dtype=np.int64)
    y[bid_out * (1 - f) > ask_in * (1 + f)] = 1
    y[bid_in * (1 - f) > ask_out * (1 + f)] = 2
    y[n - k - 1:] = 0
    return y


def round_trips(lob: np.ndarray, ts: np.ndarray, rows: np.ndarray, sides: np.ndarray, k: int,
                lot: float) -> pd.DataFrame:
    """Executes non-overlapping enter/exit taker round trips; fees are added later per tier."""
    sim = ExecutionSimulator(initial_cash=1e9, taker_fee_bps=0.0, lot_size=lot, max_inventory=1e9,
                             latency_ms=1.0, timestamps_ms=ts)
    out = []
    busy_until = -1
    for t, side in zip(rows, sides):
        if t <= busy_until or t + k >= len(lob) - 1:
            continue
        entry = sim.execute_aggressive_order(int(t), int(side), lob)
        exit_ = sim.execute_aggressive_order(int(t + k), -int(side), lob)
        mid_in = (lob[t, 0] + lob[t, 2]) / 2
        mid_out = (lob[t + k, 0] + lob[t + k, 2]) / 2
        out.append({
            "gross_bps": side * (exit_.fill_price - entry.fill_price) / entry.fill_price * 1e4,
            "mid_move_bps": side * (mid_out - mid_in) / mid_in * 1e4,
            "notional": entry.fill_price * lot,
            "row": int(t),
        })
        busy_until = exit_.timestamp_idx
    return pd.DataFrame(out, columns=["gross_bps", "mid_move_bps", "notional", "row"])


def evaluate_signal(data: PreparedData, X: np.ndarray, y: np.ndarray, k: int, label_type: str,
                    purge: int) -> List[Dict]:
    idx = data.splits(purge)
    model = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=300, random_state=data.session.seed)
    model.fit(X[idx["train"]], y[idx["train"]], X[idx["val"]], y[idx["val"]], early_stopping_rounds=15)
    s_val = np.diff(model.predict_proba(X[idx["val"]])[:, [2, 1]], axis=1).ravel()   # P(UP) - P(DOWN)
    s_test = np.diff(model.predict_proba(X[idx["test"]])[:, [2, 1]], axis=1).ravel()
    pred_test = model.predict(X[idx["test"]])
    lob, ts = data.session.lob_40, data.session.timestamps_ms
    lot = data.session.exec_params["lot_size"]

    rows = []
    for cov in COVERAGES:
        if cov == 1.0:
            # every UP/DOWN prediction, as in Experiment D
            mask = pred_test != 0
            tau = np.nan
        else:
            tau = float(np.quantile(np.abs(s_val), 1 - cov))
            mask = np.abs(s_test) >= tau
        sides = np.sign(s_test[mask]).astype(int)
        trips = round_trips(lob, ts, idx["test"][mask], sides, k, lot)
        rec = {"seed": data.session.seed, "label": label_type, "horizon_k": k, "coverage": cov, "tau": tau,
               "signals": int(mask.sum()), "trades": len(trips),
               "label_up_down_share_train": float(np.mean(y[idx["train"]] != 0))}
        if len(trips):
            rec.update({"mean_mid_move_bps": trips.mid_move_bps.mean(), "mean_gross_bps": trips.gross_bps.mean(),
                        "hit_rate_gross": float((trips.gross_bps > 0).mean())})
            for fee in FEE_TIERS_BPS:
                net = trips.gross_bps - 2 * fee
                rec[f"mean_net_bps_fee{fee:g}"] = net.mean()
                rec[f"total_net_usd_fee{fee:g}"] = float((net / 1e4 * trips.notional).sum())
        rows.append(rec)
        print(f"    {label_type:<11} k={k:<4} cov={cov:<6} trades={len(trips):<6} "
              f"gross={rec.get('mean_gross_bps', np.nan):7.2f} bp  net@4bp={rec.get('mean_net_bps_fee4', np.nan):7.2f} bp")
    return rows


def run_experiment_e(n_segments: int = 4) -> pd.DataFrame:
    out = ARTIFACTS_DIR / "btc" / "tradability"
    ckpt_dir = out / "sessions"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for session in btc_sessions(n_segments):
        ckpt = ckpt_dir / f"{session.name}.csv"
        if ckpt.exists():
            print(f"[+] {session.name}: loading checkpoint")
            all_rows.extend(pd.read_csv(ckpt).to_dict("records"))
            continue
        print(f"\n######## {session.name} ########")
        data = prepare_data(session)
        X = data.tab_feat
        rows = evaluate_signal(data, X, data.labels(20), 20, "direction", purge=20)
        for k in COST_AWARE_HORIZONS:
            y = cost_aware_labels(session.lob_40, session.timestamps_ms, k, REFERENCE_FEE_BPS)
            # labels look k + 1 rows ahead (entry at the next snapshot), so purge one extra row
            rows += evaluate_signal(data, X, y, k, "cost-aware", purge=k + 1)
        pd.DataFrame(rows).to_csv(ckpt, index=False)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "tradability_per_segment.csv", index=False)
    value_cols = [c for c in df.columns if c.startswith(("mean_", "total_", "hit_", "trades", "label_up"))]
    summary = df.groupby(["label", "horizon_k", "coverage"], sort=False)[value_cols].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / "tradability_summary.csv", index=False)
    return summary


if __name__ == "__main__":
    res = run_experiment_e()
    cols = ["label", "horizon_k", "coverage", "trades_mean", "mean_gross_bps_mean", "mean_net_bps_fee4_mean",
            "mean_net_bps_fee0_mean", "hit_rate_gross_mean", "total_net_usd_fee4_mean"]
    print(res[cols].round(3).to_string(index=False))
