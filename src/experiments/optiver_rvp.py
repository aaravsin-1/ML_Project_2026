"""Optiver Realized Volatility Prediction (Kaggle 2021), re-run under forward-only validation.

Task: for each (stock_id, time_id), predict the realized volatility of the WAP over the
next 10 minutes from the previous 10 minutes of 2-level book and trade data. Metric:
RMSPE. The competition shuffled time_ids, which let winning solutions use other (including
future) time windows. Here we:

1. Build the standard book/trade features (WAP1/WAP2 log-return realized volatility over
   the full and last 450/300/150 s, spreads, depth, imbalance, trade RV and activity), plus
   cross-sectional time_id means (all stocks' windows are observed at the same time).
2. Recover the chronological order of time_ids as the 1st-place solution did: the true
   price level is 0.01 / (smallest normalized price step), and a 1-D t-SNE of the
   time_id x stock price matrix orders the windows.
3. Validate forward in time: expanding-window folds on the recovered order.
4. Compare the 1st-place nearest-neighbor aggregates in two forms:
   - "transductive": neighbors searched among ALL time_ids, as in the winning solution
     (this reaches into the future: the next window's current RV is close to the target);
   - "past-only": neighbors restricted to time_ids earlier than the row being predicted.
"""

import argparse
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.config import ARTIFACTS_DIR, DATA_DIR

RVP_DIR = DATA_DIR / "raw" / "optiver_rvp"
WINDOWS = [0, 150, 300, 450]  # feature windows: seconds_in_bucket >= w


def rmspe(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square((y - pred) / y))))


def book_features(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["wap1"] = (df.bid_price1 * df.ask_size1 + df.ask_price1 * df.bid_size1) / (df.bid_size1 + df.ask_size1)
    df["wap2"] = (df.bid_price2 * df.ask_size2 + df.ask_price2 * df.bid_size2) / (df.bid_size2 + df.ask_size2)
    g = df.groupby("time_id")
    df["lr1"] = np.log(df.wap1).groupby(df.time_id).diff()
    df["lr2"] = np.log(df.wap2).groupby(df.time_id).diff()
    df["spread"] = (df.ask_price1 - df.bid_price1) / ((df.ask_price1 + df.bid_price1) / 2)
    df["depth"] = df.bid_size1 + df.ask_size1 + df.bid_size2 + df.ask_size2
    df["imbalance"] = (df.bid_size1 + df.bid_size2 - df.ask_size1 - df.ask_size2) / df.depth
    df["wap_balance"] = (df.wap1 - df.wap2).abs()

    out = []
    for w in WINDOWS:
        sub = df[df.seconds_in_bucket >= w]
        gs = sub.groupby("time_id")
        f = pd.DataFrame({
            f"book_rv1_{w}": np.sqrt((sub.lr1 ** 2).groupby(sub.time_id).sum()),
            f"book_rv2_{w}": np.sqrt((sub.lr2 ** 2).groupby(sub.time_id).sum()),
        })
        if w == 0:
            f["book_spread_mean"] = gs.spread.mean()
            f["book_depth_mean"] = gs.depth.mean()
            f["book_imbalance_mean"] = gs.imbalance.mean()
            f["book_imbalance_std"] = gs.imbalance.std()
            f["book_wap_balance_mean"] = gs.wap_balance.mean()
            f["book_n_updates"] = gs.size()
        out.append(f)
    feats = pd.concat(out, axis=1)

    # Price level recovery (1st-place trick): real price ~ 0.01 / smallest normalized tick
    def tick_price(x: pd.DataFrame) -> float:
        vals = np.unique(x[["bid_price1", "ask_price1", "bid_price2", "ask_price2"]].to_numpy().ravel())
        steps = np.diff(vals)
        steps = steps[steps > 0]
        return 0.01 / steps.min() if len(steps) else np.nan

    feats["recovered_price"] = g.apply(tick_price)
    feats["stock_id"] = int(Path(path).parent.name.split("=")[1])
    return feats.reset_index()


def trade_features(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["lr"] = np.log(df.price).groupby(df.time_id).diff()
    out = []
    for w in WINDOWS:
        sub = df[df.seconds_in_bucket >= w]
        gs = sub.groupby("time_id")
        f = pd.DataFrame({f"trade_rv_{w}": np.sqrt((sub.lr ** 2).groupby(sub.time_id).sum()), f"trade_n_{w}": gs.size()})
        if w == 0:
            f["trade_size_sum"] = gs["size"].sum()
            f["trade_order_count_sum"] = gs.order_count.sum()
        out.append(f)
    feats = pd.concat(out, axis=1)
    feats["stock_id"] = int(Path(path).parent.name.split("=")[1])
    return feats.reset_index()


def build_feature_table(cache: bool = True) -> pd.DataFrame:
    cache_path = RVP_DIR / "features_train.parquet"
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)
    book_paths = sorted(glob.glob(str(RVP_DIR / "book_train.parquet" / "stock_id=*" / "*.parquet")))
    trade_paths = sorted(glob.glob(str(RVP_DIR / "trade_train.parquet" / "stock_id=*" / "*.parquet")))
    book = pd.concat(Parallel(n_jobs=-1)(delayed(book_features)(p) for p in book_paths))
    trade = pd.concat(Parallel(n_jobs=-1)(delayed(trade_features)(p) for p in trade_paths))
    df = pd.read_csv(RVP_DIR / "train.csv").merge(book, on=["stock_id", "time_id"], how="left")
    df = df.merge(trade, on=["stock_id", "time_id"], how="left")

    # Contemporaneous cross-sectional context: every stock's current window is known
    for col in ["book_rv1_0", "book_rv1_300", "trade_rv_0", "book_spread_mean"]:
        df[f"{col}_time_mean"] = df.groupby("time_id")[col].transform("mean")
    # Activity levels drift over time; rank them within each time_id (1st-place processing)
    for col in ["trade_order_count_sum", "book_depth_mean", "trade_size_sum"]:
        df[f"{col}_time_rank"] = df.groupby("time_id")[col].rank(pct=True)
    if cache:
        df.to_parquet(cache_path)
    return df


def recover_time_order(df: pd.DataFrame, seed: int = 0) -> pd.Series:
    """Position (0..T-1) of each time_id in recovered chronological order."""
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import minmax_scale

    prices = df.pivot(index="time_id", columns="stock_id", values="recovered_price")
    prices = prices.fillna(prices.mean())
    emb = TSNE(n_components=1, perplexity=400, random_state=seed, max_iter=2000, init="pca").fit_transform(
        minmax_scale(prices.to_numpy()))
    order = np.argsort(emb[:, 0])
    ordered = prices.iloc[order]
    # Orientation: stock 61 is AMZN, whose price rose over the 2020-01 to 2021-03 sample
    if 61 in ordered.columns and ordered[61].iloc[0] > ordered[61].iloc[-1]:
        order = order[::-1]
    return pd.Series(np.arange(len(order)), index=prices.index[order], name="time_pos")


def nearest_neighbor_features(df: pd.DataFrame, time_pos: pd.Series, past_only: bool,
                              ns=(2, 5, 10, 20), max_candidates: int = 200) -> pd.DataFrame:
    """Mean of each stock's current RV over the n most similar time_ids (1st-place style).

    Similarity is L1 distance between time_ids on the min-max scaled matrix of recovered
    prices across stocks. With past_only=True, candidates later in time are dropped.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import minmax_scale

    prices = df.pivot(index="time_id", columns="stock_id", values="recovered_price")
    prices = prices.fillna(prices.mean())
    rv = df.pivot(index="time_id", columns="stock_id", values="book_rv1_0")
    rv = rv.reindex(prices.index).fillna(rv.mean())
    X = minmax_scale(prices.to_numpy())
    nn = NearestNeighbors(n_neighbors=min(max_candidates, len(X)), p=1).fit(X)
    _, cand = nn.kneighbors(X)
    pos = time_pos.reindex(prices.index).to_numpy()
    rv_vals = rv.to_numpy()

    feats = {n: np.full(rv_vals.shape, np.nan) for n in ns}
    for i in range(len(X)):
        c = cand[i, 1:]  # drop self
        if past_only:
            c = c[pos[c] < pos[i]]
        for n in ns:
            if len(c):
                feats[n][i] = rv_vals[c[:n]].mean(axis=0)

    frames = []
    for n in ns:
        f = pd.DataFrame(feats[n], index=prices.index, columns=prices.columns).stack().rename(f"nn{n}_rv1")
        frames.append(f)
    out = pd.concat(frames, axis=1).reset_index()
    return out


def forward_folds(time_pos: pd.Series, n_folds: int = 4, val_frac: float = 0.10) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds: validate on the last n_folds blocks of val_frac each."""
    T = len(time_pos)
    folds = []
    for f in range(n_folds):
        start = int(T * (1 - val_frac * (n_folds - f)))
        end = int(T * (1 - val_frac * (n_folds - f - 1)))
        folds.append((time_pos.index[time_pos < start].to_numpy(),
                      time_pos.index[(time_pos >= start) & (time_pos < end)].to_numpy()))
    return folds


def fit_lgb(X_tr, y_tr, X_va, y_va, seed=0):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(objective="rmse", learning_rate=0.05, num_leaves=63, n_estimators=3000,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbosity=-1)
    # Weight 1/y^2 makes squared error equal to squared percentage error (RMSPE)
    model.fit(X_tr, y_tr, sample_weight=1 / y_tr ** 2, eval_set=[(X_va, y_va)],
              eval_sample_weight=[1 / y_va ** 2], callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def run_rvp_experiment(n_folds: int = 4) -> pd.DataFrame:
    df = build_feature_table()
    time_pos = recover_time_order(df)
    df["stock_rv_target_ratio"] = np.nan  # filled per fold (train-only target encoding)

    nn_trans = nearest_neighbor_features(df, time_pos, past_only=False)
    nn_past = nearest_neighbor_features(df, time_pos, past_only=True)
    base_cols = [c for c in df.columns if c not in ("stock_id", "time_id", "target", "recovered_price",
                                                     "stock_rv_target_ratio")]
    variants = {
        "LightGBM (book/trade features)": (df, base_cols),
        "+ NN features, past-only": (df.merge(nn_past, on=["time_id", "stock_id"], how="left"),
                                     base_cols + [c for c in nn_past.columns if c.startswith("nn")]),
        "+ NN features, transductive (1st-place style, leaks future)": (
            df.merge(nn_trans, on=["time_id", "stock_id"], how="left"),
            base_cols + [c for c in nn_trans.columns if c.startswith("nn")]),
    }

    rows = []
    for fold, (tr_ids, va_ids) in enumerate(forward_folds(time_pos, n_folds)):
        tr_mask, va_mask = df.time_id.isin(tr_ids).to_numpy(), df.time_id.isin(va_ids).to_numpy()
        y_va = df.target.to_numpy()[va_mask]
        rows.append({"fold": fold, "model": "Naive (current RV)", "rmspe": rmspe(y_va, df.book_rv1_0.to_numpy()[va_mask])})

        # Train-only stock target encoding: average ratio target / current RV per stock
        ratio = (df.target / df.book_rv1_0)[tr_mask].groupby(df.stock_id[tr_mask]).median()
        for name, (frame, cols) in variants.items():
            frame = frame.copy()
            frame["stock_rv_target_ratio"] = frame.stock_id.map(ratio)
            use = cols + ["stock_rv_target_ratio"]
            X, y = frame[use].to_numpy(np.float32), frame.target.to_numpy()
            model = fit_lgb(X[tr_mask], y[tr_mask], X[va_mask], y[va_mask])
            rows.append({"fold": fold, "model": name, "rmspe": rmspe(y[va_mask], model.predict(X[va_mask]))})
        print(pd.DataFrame(rows[-len(variants) - 1:]).to_string(index=False))

    # Shuffled K-fold on the base model, to show how optimistic non-temporal CV is
    from sklearn.model_selection import KFold
    X, y = df[base_cols].to_numpy(np.float32), df.target.to_numpy()
    for fold, (tr, va) in enumerate(KFold(n_folds, shuffle=True, random_state=0).split(X)):
        model = fit_lgb(X[tr], y[tr], X[va], y[va])
        rows.append({"fold": fold, "model": "LightGBM, shuffled K-fold CV", "rmspe": rmspe(y[va], model.predict(X[va]))})

    res = pd.DataFrame(rows)
    out = ARTIFACTS_DIR / "optiver_rvp"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "rvp_folds.csv", index=False)
    summary = res.groupby("model", sort=False).rmspe.agg(["mean", "std"]).reset_index()
    summary.to_csv(out / "rvp_summary.csv", index=False)
    time_pos.to_csv(out / "recovered_time_order.csv")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args()
    print(run_rvp_experiment(args.folds).to_string(index=False))
