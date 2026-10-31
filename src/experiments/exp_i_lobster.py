"""Experiment I: Nasdaq event-level data (LOBSTER) for large-tick and small-tick stocks.

Covers the parts of the research spec the 1-second crypto data could not:
- section 4.1 / 5.2: the decay curve at the spec's horizons (100 ms, 500 ms, 1 s, 5 s, 10 s, 30 s)
- section 1: large-tick (INTC, MSFT) vs small-tick (AAPL, AMZN, GOOG) predictability
- section 1 / 3: event-trajectory features (limit submissions, cancellations, executions)
- section 6.2: market making where the spread is actually worth earning (large-tick stocks)

Data: LOBSTER free sample files, 10 levels, 21 June 2012, 9:30-16:00, resampled onto a
100 ms clock (data/raw/lobster). One trading day per stock: every result is a single-day
sample, split 70/15/15 in time (the test period is roughly the last hour of the session).

Part A (prediction): per stock and horizon, naive baselines, logistic regression, LightGBM on
the microstructure features, and LightGBM with the order-flow features added. Labels compare
the mid price k steps ahead with now, with thresholds set on training rows (a third FLAT).

Part B (market making): Experiment H's two-sided quoting (always / spec rule / adverse-selection
model) under three fill rules, with Nasdaq-style per-share pricing in 2012: 0.30 cents per share
to take liquidity, a 0.20 cent rebate to provide it (assumed, not the exact fee schedule).

    python -m src.experiments.exp_i_lobster
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.config import ARTIFACTS_DIR, DATA_DIR
from src.data.labels import direction_labels
from src.data.lobster_loader import load_lobster_day_on_grid
from src.experiments.common import BTC_EXEC, TRAIN_RATIO, MarketSession, prepare_data
from src.experiments import exp_h_market_making as mm
from src.models.baselines import LogisticRegressionBaseline, NaiveMomentumBaseline
from src.models.tabular_trees import LightGBMLOB

LOBSTER_DIR = DATA_DIR / "raw" / "lobster"
TICKERS = {"INTC": "large", "MSFT": "large", "AAPL": "small", "AMZN": "small", "GOOG": "small"}
GRID_MS = 100
HORIZON_ROWS = {"100ms": 1, "500ms": 5, "1s": 10, "5s": 50, "10s": 100, "30s": 300}
ROWS_PER_DAY = 6.5 * 3600 * 1000 / GRID_MS
TAKER_FEE_PER_SHARE = 0.0030
MAKER_REBATE_PER_SHARE = 0.0020
LOT_SHARES = 100
MAX_INVENTORY_SHARES = 500


def lobster_session(ticker: str, seed: int):
    d = load_lobster_day_on_grid(LOBSTER_DIR, ticker, grid_ms=GRID_MS)
    lob, ts, trades = d["lob_40"], d["timestamps_ms"], d["trades_df"]
    train_end = int(len(lob) * TRAIN_RATIO)
    mid = (lob[:, 0] + lob[:, 2]) / 2.0
    labels = direction_labels(mid, list(HORIZON_ROWS.values()), fit_end=train_end, smoothing=1)
    train_trades = trades[trades["timestamp_ms"] < ts[train_end]]
    bucket = float(train_trades["size"].sum() / max(ts[train_end] / 60000.0, 1.0))  # avg volume per minute
    exec_params = dict(BTC_EXEC, lot_size=LOT_SHARES, max_inventory=MAX_INVENTORY_SHARES)
    session = MarketSession(lob, labels, trades, ts, seed=seed, name=ticker, tick_size=d["tick_size"],
                            vpin_bucket_volume=bucket, exec_params=exec_params)
    return session, d["flows"]


def predict_part(data, flows) -> List[Dict]:
    session = data.session
    X = data.tab_feat
    fx, _ = mm.flow_features(flows, session.lob_40)
    X_flow = np.hstack([X.astype(np.float32), fx])
    rows = []
    for name, k in HORIZON_ROWS.items():
        y = data.labels(k)
        idx = data.splits(k)
        tr, va, te = idx["train"], idx["val"], idx["test"]
        y_te = y[te]

        def add(model_name, pred):
            rows.append({"ticker": session.name, "tick_class": TICKERS[session.name], "horizon": name, "k_rows": k,
                         "model": model_name, "macro_f1": f1_score(y_te, pred, average="macro", zero_division=0),
                         "accuracy": accuracy_score(y_te, pred), "test_flat_share": float(np.mean(y_te == 0))})

        add("Naive Flat", np.zeros(len(te), dtype=np.int64))
        add("Naive Momentum", NaiveMomentumBaseline(k).predict(y, te))
        lr = LogisticRegressionBaseline()
        lr.fit(X[tr], y[tr])
        add("Logistic Regression", lr.predict(X[te]))
        for model_name, feats in (("LightGBM", X), ("LightGBM + order flow", X_flow)):
            m = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=300, random_state=session.seed)
            m.fit(feats[tr], y[tr], feats[va], y[va], early_stopping_rounds=15)
            add(model_name, m.predict(feats[te]))
        best = max(r["macro_f1"] for r in rows if r["horizon"] == name and r["ticker"] == session.name)
        print(f"    {session.name} {name:>6}: test FLAT {np.mean(y_te == 0):.2f} | best macro-F1 {best:.3f}")
    return rows


def run_experiment_i() -> pd.DataFrame:
    out = ARTIFACTS_DIR / "lobster"
    ckpt_dir = out / "sessions"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pred_all, mm_all, diag_all = [], [], []
    for seed, ticker in enumerate(TICKERS):
        ck = {kind: ckpt_dir / f"{ticker}_{kind}.csv" for kind in ("predict", "mm", "diag")}
        if all(p.exists() for p in ck.values()):
            print(f"[+] {ticker}: loading checkpoint")
            pred_all.append(pd.read_csv(ck["predict"]))
            mm_all.append(pd.read_csv(ck["mm"]))
            diag_all.append(pd.read_csv(ck["diag"]))
            continue
        print(f"\n######## {ticker} ({TICKERS[ticker]} tick) ########")
        session, flows = lobster_session(ticker, seed)
        data = prepare_data(session)
        pred = pd.DataFrame(predict_part(data, flows))

        lob = session.lob_40
        price = float(np.median((lob[:, 0] + lob[:, 2]) / 2.0))
        params = {"life": 300, "markout": 300, "lot": LOT_SHARES, "max_inventory": MAX_INVENTORY_SHARES,
                  "maker_fee_bps": -MAKER_REBATE_PER_SHARE / price * 1e4,
                  "taker_fee_bps": TAKER_FEE_PER_SHARE / price * 1e4,
                  "rows_per_day": ROWS_PER_DAY, "flow_book": np.vstack([lob[:1], lob[:-1]])}
        results, diag = mm.evaluate_segment(data, flows, params)
        res, dg = pd.DataFrame(results), pd.DataFrame(diag)
        for df in (res, dg):
            df["ticker"], df["tick_class"] = ticker, TICKERS[ticker]
        pred.to_csv(ck["predict"], index=False)
        res.to_csv(ck["mm"], index=False)
        dg.to_csv(ck["diag"], index=False)
        pred_all.append(pred)
        mm_all.append(res)
        diag_all.append(dg)

    pred = pd.concat(pred_all, ignore_index=True)
    pred.to_csv(out / "lobster_prediction.csv", index=False)
    mm_res = pd.concat(mm_all, ignore_index=True)
    mm_res.to_csv(out / "lobster_market_making.csv", index=False)
    pd.concat(diag_all, ignore_index=True).to_csv(out / "lobster_adverse_selection.csv", index=False)

    # Decay curve: best learned model per stock and horizon, next to naive momentum
    learned = pred[~pred.model.str.startswith("Naive")]
    best = learned.loc[learned.groupby(["ticker", "horizon"], sort=False)["macro_f1"].idxmax()]
    decay = best.merge(pred[pred.model == "Naive Momentum"][["ticker", "horizon", "macro_f1"]]
                       .rename(columns={"macro_f1": "momentum_macro_f1"}), on=["ticker", "horizon"])
    decay.to_csv(out / "lobster_decay.csv", index=False)
    return decay


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    res = run_experiment_i()
    print(res[["ticker", "tick_class", "horizon", "model", "macro_f1", "momentum_macro_f1", "test_flat_share"]]
          .round(3).to_string(index=False))
