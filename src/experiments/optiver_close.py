"""Optiver Trading at the Close (Kaggle 2023), re-run with a strict date split.

Task: during the last 10 minutes before the Nasdaq closing auction, predict each stock's
60-second future WAP move relative to a synthetic index (in bps). Metric: MAE.

Validation follows the 1st-place solution: train on date_id < 400, hold out the last 81
days. Every feature uses only information available at that second:
- book/auction features: imbalances, spreads, pairwise price imbalances, pressure terms;
- within stock-day history: lags and changes (shifted, so strictly past);
- 1st-place "magic" features: seconds_in_bucket_group (0-290 s / 300-470 s / 480-540 s)
  first-value ratios and expanding means, and cross-sectional mean-ratio / rank at the same
  (date_id, seconds_in_bucket) across stocks;
- previous-day target at the same stock and second (targets are revealed after the day).

Models: zero baseline, LightGBM (MAE objective) trained once, and LightGBM retrained every
12 holdout days on all data seen so far (the winner's online-learning schedule).
Post-processing subtracts the cross-sectional mean prediction (equal weights; the
competition's index weights are not in train.csv).
"""

import argparse
from typing import List
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import ARTIFACTS_DIR, DATA_DIR

CLOSE_PATH = DATA_DIR / "raw" / "optiver_close" / "train.csv"
SPLIT_DAY = 400
RETRAIN_EVERY = 12
BASE = ["matched_size", "imbalance_size", "reference_price", "bid_size", "ask_size", "wap", "liquidity_imbalance"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date_id", "seconds_in_bucket", "stock_id"]).reset_index(drop=True)
    eps = 1e-9
    df["volume"] = df.bid_size + df.ask_size
    df["mid"] = (df.ask_price + df.bid_price) / 2
    df["liquidity_imbalance"] = (df.bid_size - df.ask_size) / (df.volume + eps)
    df["matched_imbalance"] = (df.imbalance_size - df.matched_size) / (df.imbalance_size + df.matched_size + eps)
    df["signed_imbalance"] = df.imbalance_size * df.imbalance_buy_sell_flag
    df["imbalance_ratio"] = df.imbalance_size / (df.matched_size + eps)
    df["spread"] = df.ask_price - df.bid_price
    df["price_pressure"] = df.imbalance_size * df.spread
    df["market_urgency"] = df.spread * df.liquidity_imbalance
    df["depth_pressure"] = (df.ask_size - df.bid_size) * (df.far_price - df.near_price)
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    for a, b in itertools.combinations(prices, 2):
        df[f"{a}_{b}_imb"] = (df[a] - df[b]) / (df[a] + df[b] + eps)

    # Strictly past history within each stock-day
    g = df.groupby(["stock_id", "date_id"], sort=False)
    for col in ["matched_size", "imbalance_size", "reference_price", "wap", "signed_imbalance", "liquidity_imbalance"]:
        for lag in (1, 2, 3, 6):
            df[f"{col}_ret{lag}"] = df[col] / g[col].shift(lag) - 1
    df["spread_change"] = df.spread - g.spread.shift(1)

    # 1st-place magic: group of seconds, first-value ratio and expanding mean within group
    df["sib_group"] = np.select([df.seconds_in_bucket < 300, df.seconds_in_bucket < 480], [0, 1], 2)
    # (the winner's rolling_mean(100) spans the whole group, i.e. an expanding mean)
    gg = df.groupby(["stock_id", "date_id", "sib_group"], sort=False)
    count = gg.cumcount() + 1
    for col in BASE:
        df[f"{col}_group_first_ratio"] = gg[col].transform("first") / (df[col] + eps)
        df[f"{col}_group_expmean_ratio"] = (gg[col].cumsum() / count) / (df[col] + eps)

    # Cross-sectional context at the same second (all stocks are observed together)
    gx = df.groupby(["date_id", "seconds_in_bucket"], sort=False)
    for col in BASE:
        df[f"{col}_xs_mean_ratio"] = gx[col].transform("mean") / (df[col] + eps)
        df[f"{col}_xs_rank"] = gx[col].rank(pct=True)

    # Previous day's revealed target for the same stock and second
    prev = df[["stock_id", "date_id", "seconds_in_bucket", "target"]].copy()
    prev["date_id"] += 1
    df = df.merge(prev.rename(columns={"target": "target_prev_day"}),
                  on=["stock_id", "date_id", "seconds_in_bucket"], how="left")
    return df


def feature_columns(df: pd.DataFrame) -> List[str]:
    drop = {"target", "row_id", "time_id", "date_id"}
    return [c for c in df.columns if c not in drop and df[c].dtype != object]


def fit(X, y, X_va=None, y_va=None, n_estimators=1500, seed=0):
    model = lgb.LGBMRegressor(objective="mae", learning_rate=0.05, num_leaves=255, max_depth=10,
                              n_estimators=n_estimators, subsample=0.7, subsample_freq=1,
                              colsample_bytree=0.6, random_state=seed, verbosity=-1)
    if X_va is not None:
        model.fit(X, y, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        model.fit(X, y)
    return model


def demean(df: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    s = pd.Series(pred, index=df.index)
    return (s - s.groupby([df.date_id, df.seconds_in_bucket]).transform("mean")).to_numpy()


def run_close_experiment() -> pd.DataFrame:
    raw = pd.read_csv(CLOSE_PATH)
    raw = raw[raw.target.notna()]
    df = build_features(raw)
    cols = feature_columns(df)
    tr = df.date_id < SPLIT_DAY
    ho = ~tr
    y = df.target.to_numpy()
    X = df[cols].to_numpy(np.float32)
    print(f"[+] {len(cols)} features | train rows {tr.sum():,} | holdout rows {ho.sum():,} "
          f"({df.date_id[ho].nunique()} days)")

    # Use the last 40 training days for early stopping, then refit on all training days
    es = (df.date_id >= SPLIT_DAY - 40) & tr
    probe = fit(X[tr & ~es], y[tr & ~es], X[es], y[es])
    n_best = probe.best_iteration_ or 1500
    static = fit(X[tr], y[tr], n_estimators=n_best)
    pred_static = static.predict(X[ho])

    # Online learning: retrain every RETRAIN_EVERY holdout days on everything seen so far
    pred_online = np.empty(ho.sum())
    ho_days = np.sort(df.date_id[ho].unique())
    ho_idx = np.nonzero(ho.to_numpy())[0]
    model = static
    for start in range(0, len(ho_days), RETRAIN_EVERY):
        block = ho_days[start:start + RETRAIN_EVERY]
        if start > 0:
            seen = (df.date_id < block[0]).to_numpy()
            model = fit(X[seen], y[seen], n_estimators=n_best)
        in_block = df.date_id.iloc[ho_idx].isin(block).to_numpy()
        pred_online[in_block] = model.predict(X[ho_idx[in_block]])

    y_ho = y[ho]
    df_ho = df[ho]
    res = pd.DataFrame([
        {"model": "Zero prediction", "mae": np.mean(np.abs(y_ho))},
        {"model": "LightGBM, trained once", "mae": np.mean(np.abs(y_ho - pred_static))},
        {"model": "LightGBM, trained once + demean", "mae": np.mean(np.abs(y_ho - demean(df_ho, pred_static)))},
        {"model": f"LightGBM, retrained every {RETRAIN_EVERY} days + demean",
         "mae": np.mean(np.abs(y_ho - demean(df_ho, pred_online)))},
    ])
    imp = pd.Series(static.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
    out = ARTIFACTS_DIR / "optiver_close"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "close_holdout_results.csv", index=False)
    (100 * imp / imp.sum()).head(30).rename("gain_share_pct").to_csv(out / "close_feature_importance.csv")
    return res


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    print(run_close_experiment().to_string(index=False))
