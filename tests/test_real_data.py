"""Tests for the real-data loaders, labels and Optiver experiment features (no downloads needed)."""

import numpy as np
import pandas as pd

from src.data.labels import calibrate_alpha, direction_labels
from src.data.crypto_loader import CryptoLOBLoader


def test_label_threshold_uses_training_rows_only():
    rng = np.random.default_rng(0)
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, 5000)))
    fit_end = 3000
    alpha = calibrate_alpha(mid, 10, fit_end)
    mid2 = mid.copy()
    mid2[fit_end:] *= np.exp(np.cumsum(rng.normal(0, 1e-2, 2000)))  # wild future
    assert np.isclose(alpha, calibrate_alpha(mid2, 10, fit_end))

    y = direction_labels(mid, [10], fit_end)[10]
    assert abs(np.mean(y[:fit_end - 10] == 0) - 1 / 3) < 0.02
    assert set(np.unique(y)) <= {0, 1, 2}


def test_coinbase_snapshot_conversion(tmp_path):
    n, levels = 4, 2
    data = {"system_time": pd.date_range("2021-04-07", periods=n, freq="1s", tz="UTC").astype(str),
            "midpoint": [100.005] * n, "buys": [10.0, 0.0, 20.0, 0.0], "sells": [0.0, 5.0, 0.0, 0.0]}
    for lvl in range(levels):
        data[f"bids_distance_{lvl}"] = [-(0.005 + 0.01 * lvl) / 100.005] * n
        data[f"asks_distance_{lvl}"] = [(0.005 + 0.01 * lvl) / 100.005] * n
        data[f"bids_notional_{lvl}"] = [1000.0] * n
        data[f"asks_notional_{lvl}"] = [500.0] * n
    path = tmp_path / "BTC_test.csv"
    pd.DataFrame(data).to_csv(path)

    lob, ts, trades = CryptoLOBLoader.load_coinbase_snapshots(path, num_levels=levels, cache=False)
    assert lob.shape == (n, 8)
    assert np.allclose(lob[0, [0, 2, 4, 6]], [100.01, 100.00, 100.02, 99.99])
    assert np.isclose(lob[0, 1], 500.0 / 100.01) and np.isclose(lob[0, 3], 1000.0 / 100.00)
    assert np.allclose(ts, [0, 1000, 2000, 3000])
    assert trades.direction.tolist() == [1, -1, 1]
    assert np.allclose(trades["size"], np.array([10.0, 5.0, 20.0]) / 100.005)


def _close_frame(n_days=3, n_stocks=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for s in range(0, 550, 10):
            for st in range(n_stocks):
                rows.append({"stock_id": st, "date_id": d, "seconds_in_bucket": s,
                             "imbalance_size": rng.uniform(1e5, 1e6), "imbalance_buy_sell_flag": rng.choice([-1, 1]),
                             "reference_price": 1 + rng.normal(0, 1e-3), "matched_size": rng.uniform(1e6, 1e7),
                             "far_price": 1 + rng.normal(0, 1e-3), "near_price": 1 + rng.normal(0, 1e-3),
                             "bid_price": 1 - 1e-4, "bid_size": rng.uniform(1e3, 1e5),
                             "ask_price": 1 + 1e-4, "ask_size": rng.uniform(1e3, 1e5),
                             "wap": 1 + rng.normal(0, 1e-4), "target": rng.normal(0, 5)})
    return pd.DataFrame(rows)


def test_close_features_are_causal():
    from src.experiments.optiver_close import build_features
    df = _close_frame()
    late = (df.date_id == 1) & (df.seconds_in_bucket >= 300)
    df2 = df.copy()
    df2.loc[late, ["matched_size", "imbalance_size", "wap", "bid_size", "reference_price"]] *= 3.0
    df2.loc[late, "target"] += 100.0  # same-day targets must never leak into features

    f1, f2 = build_features(df), build_features(df2)
    early = (f1.date_id == 1) & (f1.seconds_in_bucket < 300)
    cols = [c for c in f1.columns if c != "target"]
    pd.testing.assert_frame_equal(f1.loc[early, cols].reset_index(drop=True),
                                  f2.loc[early, cols].reset_index(drop=True))
    # The previous-day target is the only target-derived feature, and it is lagged a full day
    assert f1.loc[f1.date_id == 0, "target_prev_day"].isna().all()


def test_rvp_past_only_neighbors_and_forward_folds():
    from src.experiments.optiver_rvp import nearest_neighbor_features, forward_folds
    rng = np.random.default_rng(0)
    T, S = 60, 4
    df = pd.DataFrame([{"time_id": t, "stock_id": s, "recovered_price": 100 + t + rng.normal(0, 0.1),
                        "book_rv1_0": float(t)} for t in range(T) for s in range(S)])
    time_pos = pd.Series(np.arange(T), index=np.arange(T))

    past = nearest_neighbor_features(df, time_pos, past_only=True, ns=(2,), max_candidates=T)
    trans = nearest_neighbor_features(df, time_pos, past_only=False, ns=(2,), max_candidates=T)
    p = past.set_index(["time_id", "stock_id"])["nn2_rv1"]
    # RV equals the time position here, so past-only neighbours average strictly smaller values
    assert p.loc[0].isna().all()
    assert (p.drop(index=0, level=0) < p.drop(index=0, level=0).index.get_level_values(0)).all()
    # Transductive neighbours reach into the future (a later window is among the nearest)
    t = trans.set_index(["time_id", "stock_id"])["nn2_rv1"]
    assert np.isclose(t.loc[(10, 0)], 10.0, atol=1.0)

    for tr_ids, va_ids in forward_folds(time_pos, n_folds=3, val_frac=0.1):
        assert time_pos.loc[tr_ids].max() < time_pos.loc[va_ids].min()
