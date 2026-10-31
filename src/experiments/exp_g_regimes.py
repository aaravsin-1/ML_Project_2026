"""Experiment G: Does the BTC edge survive costs in any market regime?

The research spec's success condition (sections 6.2 and 8) is to find regimes, such as
periods of toxic order flow measured by VPIN, where the predictive edge survives the
execution simulator. Experiment E filtered trades by model confidence; this experiment
splits them by market regime instead.

Regimes are terciles (low / mid / high) of three variables known at decision time:
- vpin:     order-flow toxicity (the pipeline's causal VPIN feature)
- spread:   quoted spread in bp of the mid
- vol_60s:  realized volatility, std of 1-second mid returns over the previous 60 seconds (bp)
Tercile cut-offs come from each segment's TRAINING rows, so the test period never
influences them.

For every regime the experiment reports:
1. Predictability: macro-F1 of the direction model on the test rows inside the regime.
2. Tradability: a strategy that trades ONLY inside the regime (Experiment E's taker round
   trips: enter at the next snapshot, exit k seconds later, walking the book both ways),
   at several confidence coverages, with net results at fees of 0 / 1 / 2 / 4 bp per side.
3. Uncertainty: 95% confidence intervals for mean gross bp per trade, from a moving-block
   bootstrap over the pooled trades of all 4 segments.

Models are the same LightGBM configurations as Experiment E: direction labels at k = 20 s
and cost-aware labels at k = 60 s (the best setting in E).

    python -m src.experiments.exp_g_regimes
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.config import ARTIFACTS_DIR
from src.experiments.common import PreparedData, btc_sessions, prepare_data
from src.experiments.exp_e_tradability import FEE_TIERS_BPS, REFERENCE_FEE_BPS, cost_aware_labels, round_trips
from src.models.tabular_trees import LightGBMLOB

REGIME_VARS = ["vpin", "spread", "vol_60s"]
TERCILES = ["low", "mid", "high"]
COVERAGES = [1.0, 0.10, 0.02]
SIGNALS = [("direction", 20), ("cost-aware", 60)]
VOL_WINDOW = 60
BOOTSTRAP_DRAWS = 2000
BLOCK_LEN = 10


def regime_variables(data: PreparedData) -> pd.DataFrame:
    """Causal regime variables for every row of the session."""
    lob = data.session.lob_40
    mid = (lob[:, 0] + lob[:, 2]) / 2.0
    ret_bp = np.zeros(len(mid))
    ret_bp[1:] = np.diff(np.log(mid)) * 1e4
    vol = pd.Series(ret_bp).rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std().to_numpy()
    return pd.DataFrame({
        "vpin": data.tab_feat[:, data.feat_names.index("vpin")],
        "spread": (lob[:, 0] - lob[:, 2]) / mid * 1e4,
        "vol_60s": vol,
    })


def tercile_labels(values: np.ndarray, train_rows: np.ndarray) -> np.ndarray:
    """0 / 1 / 2 for low / mid / high, with cut-offs from training rows only (-1 if undefined).

    If a variable has too few distinct values (the spread is often one tick), cut-offs can
    coincide; rows are then split by <= and > so the terciles stay well defined.
    """
    train_vals = values[train_rows]
    train_vals = train_vals[np.isfinite(train_vals)]
    q1, q2 = np.quantile(train_vals, [1 / 3, 2 / 3])
    out = np.full(len(values), -1, dtype=np.int64)
    ok = np.isfinite(values)
    out[ok & (values <= q1)] = 0
    out[ok & (values > q1) & (values <= q2)] = 1
    out[ok & (values > q2)] = 2
    return out


def train_signal(data: PreparedData, y: np.ndarray, purge: int):
    idx = data.splits(purge)
    X = data.tab_feat
    model = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=300,
                        random_state=data.session.seed)
    model.fit(X[idx["train"]], y[idx["train"]], X[idx["val"]], y[idx["val"]], early_stopping_rounds=15)
    s_val = np.diff(model.predict_proba(X[idx["val"]])[:, [2, 1]], axis=1).ravel()  # P(UP) - P(DOWN)
    s_test = np.diff(model.predict_proba(X[idx["test"]])[:, [2, 1]], axis=1).ravel()
    return idx, model.predict(X[idx["test"]]), s_val, s_test


def evaluate_segment(data: PreparedData) -> pd.DataFrame:
    """All trades for one segment, tagged with signal, coverage, regime variable and tercile."""
    session = data.session
    lob, ts, lot = session.lob_40, session.timestamps_ms, session.exec_params["lot_size"]
    regimes = regime_variables(data)
    trades, skill = [], []

    for label_type, k in SIGNALS:
        if label_type == "direction":
            y, purge = data.labels(k), k
        else:
            y, purge = cost_aware_labels(lob, ts, k, REFERENCE_FEE_BPS), k + 1
        idx, pred_test, s_val, s_test = train_signal(data, y, purge)
        test_rows = idx["test"]

        for var in REGIME_VARS:
            terc_all = tercile_labels(regimes[var].to_numpy(), idx["train"])
            terc = terc_all[test_rows]
            for r, name in enumerate(TERCILES):
                in_r = terc == r
                if in_r.sum() > 0:
                    skill.append({"segment": session.seed, "signal": label_type, "horizon_k": k, "regime_var": var,
                                  "tercile": name, "test_rows": int(in_r.sum()),
                                  "macro_f1": f1_score(y[test_rows][in_r], pred_test[in_r], average="macro",
                                                       zero_division=0)})
                for cov in COVERAGES:
                    if cov == 1.0:
                        mask = pred_test != 0
                    else:
                        tau = float(np.quantile(np.abs(s_val), 1 - cov))  # set on validation rows
                        mask = np.abs(s_test) >= tau
                    mask &= in_r
                    sides = np.sign(s_test[mask]).astype(int)
                    sides[sides == 0] = 1
                    trips = round_trips(lob, ts, test_rows[mask], sides, k, lot)
                    if len(trips):
                        trips = trips.assign(segment=session.seed, signal=label_type, horizon_k=k, coverage=cov,
                                             regime_var=var, tercile=name,
                                             regime_value=regimes[var].to_numpy()[trips["row"].to_numpy()])
                        trades.append(trips)
                print(f"    {label_type:<10} k={k:<3} {var:<8} {name:<4}: {int(in_r.sum()):>6} test rows")

    pd.DataFrame(skill).to_csv(ARTIFACTS_DIR / "btc" / "regimes" / "sessions" / f"{session.name}_skill.csv", index=False)
    return pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()


def block_bootstrap_ci(x: np.ndarray, rng: np.random.Generator, draws: int = BOOTSTRAP_DRAWS,
                       block: int = BLOCK_LEN) -> tuple:
    """95% CI of the mean with a moving-block bootstrap (keeps short-range dependence between trades)."""
    n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    block = max(1, min(block, n // 10))  # small samples: shorter blocks, so resampling still varies
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(draws, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(draws, -1)[:, :n]
    means = x[idx].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    keys = ["signal", "horizon_k", "coverage", "regime_var", "tercile"]
    for key, g in trades.sort_values(["segment", "row"]).groupby(keys, sort=False):
        gross = g["gross_bps"].to_numpy()
        lo, hi = block_bootstrap_ci(gross, rng)
        rec = dict(zip(keys, key))
        rec.update({"trades": len(g), "segments_with_trades": g["segment"].nunique(),
                    "mean_mid_move_bps": g["mid_move_bps"].mean(), "mean_gross_bps": gross.mean(),
                    "gross_ci_low": lo, "gross_ci_high": hi, "hit_rate_gross": float((gross > 0).mean())})
        for fee in FEE_TIERS_BPS:
            rec[f"mean_net_bps_fee{fee:g}"] = gross.mean() - 2 * fee
        # mean gross per segment, to see whether a result is driven by one segment
        per_seg = g.groupby("segment")["gross_bps"].mean()
        rec["segments_gross_positive"] = int((per_seg > 0).sum())
        rows.append(rec)
    out = pd.DataFrame(rows)
    order = {name: i for i, name in enumerate(TERCILES)}
    return out.sort_values(["signal", "coverage", "regime_var", "tercile"],
                           key=lambda c: c.map(order) if c.name == "tercile" else c,
                           ascending=[True, False, True, True]).reset_index(drop=True)


def run_experiment_g(n_segments: int = 4) -> pd.DataFrame:
    out = ARTIFACTS_DIR / "btc" / "regimes"
    ckpt_dir = out / "sessions"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    all_trades = []
    for session in btc_sessions(n_segments):
        ckpt = ckpt_dir / f"{session.name}_trades.csv"
        if ckpt.exists() and (ckpt_dir / f"{session.name}_skill.csv").exists():
            print(f"[+] {session.name}: loading checkpoint")
            all_trades.append(pd.read_csv(ckpt))
            continue
        print(f"\n######## {session.name} ########")
        trades = evaluate_segment(prepare_data(session))
        trades.to_csv(ckpt, index=False)
        all_trades.append(trades)

    trades = pd.concat(all_trades, ignore_index=True)
    summary = summarize(trades)
    summary.to_csv(out / "regime_tradability.csv", index=False)

    skill = pd.concat([pd.read_csv(ckpt_dir / f"btc_segment_{i}_skill.csv") for i in range(n_segments)])
    skill_summary = (skill.groupby(["signal", "horizon_k", "regime_var", "tercile"], sort=False)["macro_f1"]
                     .agg(["mean", "std", "count"]).reset_index())
    skill_summary.to_csv(out / "regime_skill.csv", index=False)
    return summary


if __name__ == "__main__":
    res = run_experiment_g()
    cols = ["signal", "coverage", "regime_var", "tercile", "trades", "mean_mid_move_bps", "mean_gross_bps",
            "gross_ci_low", "gross_ci_high", "mean_net_bps_fee1", "segments_gross_positive"]
    pd.set_option("display.width", 200)
    print(res[cols].round(3).to_string(index=False))
