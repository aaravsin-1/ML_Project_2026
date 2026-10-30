"""Experiment F: Does Ethereum's order book help predict Bitcoin? (cross-asset lead-lag)

Cont, Cucuringu & Zhang (2023) find that other assets' order flow adds little to
contemporaneous price impact but can help forecast future returns at short horizons.
The Kaggle Coinbase dataset records BTC and ETH books on the same 1-second clock, so this
is a cheap, direct test:

- ETH features are computed on the ETH book and attached to each BTC snapshot using the
  latest ETH snapshot at or before that BTC timestamp (strictly causal);
- for each of the 4 BTC segments, the same LightGBM is trained with and without the ETH
  features on identical purged splits, at horizons k = 10, 20, 50 seconds;
- the per-segment difference in test macro-F1 is the effect of adding ETH.
"""

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR
from src.data.crypto_loader import CryptoLOBLoader
from src.experiments.common import BTC_PATH, btc_sessions, prepare_data
from src.microstructure.ofi import OrderFlowImbalance
from src.models.tabular_trees import LightGBMLOB
from src.utils.metrics import compute_classification_metrics

ETH_PATH = BTC_PATH.parent / "ETH_1sec.csv"
HORIZONS = [10, 20, 50]


def eth_feature_frame() -> pd.DataFrame:
    """Causal ETH features indexed by ETH snapshot epoch time (ms)."""
    lob, _, _ = CryptoLOBLoader.load_coinbase_snapshots(ETH_PATH)
    t_abs = CryptoLOBLoader.load_absolute_times_ms(ETH_PATH)
    mid = (lob[:, 0] + lob[:, 2]) / 2.0
    log_mid = np.log(mid)
    bid_v, ask_v = lob[:, 3], lob[:, 1]
    ofi = OrderFlowImbalance.compute_top_level_ofi(lob[:, 2], bid_v, lob[:, 0], ask_v).astype(np.float64)
    depth5_bid = lob[:, 3:20:4].sum(axis=1)
    depth5_ask = lob[:, 1:20:4].sum(axis=1)
    feats = {"eth_imbalance_l1": (bid_v - ask_v) / (bid_v + ask_v + 1e-12),
             "eth_imbalance_l5": (depth5_bid - depth5_ask) / (depth5_bid + depth5_ask + 1e-12)}
    for lag in (1, 5, 20, 60):
        r = np.zeros_like(log_mid)
        r[lag:] = log_mid[lag:] - log_mid[:-lag]
        feats[f"eth_ret_{lag}s"] = r * 1e4  # bps
    for w in (5, 20):
        feats[f"eth_ofi_sum_{w}s"] = pd.Series(ofi).rolling(w, min_periods=1).sum().to_numpy()
    return pd.DataFrame(feats, index=t_abs)


def run_experiment_f(n_segments: int = 4) -> pd.DataFrame:
    out = ARTIFACTS_DIR / "btc" / "cross_asset"
    out.mkdir(parents=True, exist_ok=True)
    eth = eth_feature_frame()
    btc_abs = CryptoLOBLoader.load_absolute_times_ms(BTC_PATH)
    bounds = np.linspace(0, len(btc_abs), n_segments + 1).astype(int)  # same split as btc_sessions
    rows = []
    for i, session in enumerate(btc_sessions(n_segments)):
        ckpt = out / f"{session.name}.csv"
        if ckpt.exists():
            print(f"[+] {session.name}: loading checkpoint")
            rows.extend(pd.read_csv(ckpt).to_dict("records"))
            continue
        data = prepare_data(session)
        seg_abs = btc_abs[bounds[i]:bounds[i + 1]]
        pos = np.searchsorted(eth.index.to_numpy(), seg_abs, side="right") - 1  # latest ETH row at or before
        valid = pos >= 0
        eth_x = np.zeros((len(seg_abs), eth.shape[1]), dtype=np.float32)
        eth_x[valid] = eth.to_numpy()[pos[valid]]
        stale_s = np.where(valid, (seg_abs - eth.index.to_numpy()[np.maximum(pos, 0)]) / 1000.0, np.nan)
        X_btc = data.tab_feat
        X_both = np.concatenate([X_btc, eth_x], axis=1)

        seg_rows = []
        for k in HORIZONS:
            y, idx = data.labels(k), data.splits(k)
            for name, X in (("BTC features", X_btc), ("BTC + ETH features", X_both)):
                m = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=300, random_state=session.seed)
                m.fit(X[idx["train"]], y[idx["train"]], X[idx["val"]], y[idx["val"]], early_stopping_rounds=15)
                met = compute_classification_metrics(y[idx["test"]], m.predict(X[idx["test"]]), m.predict_proba(X[idx["test"]]))
                rec = {"seed": session.seed, "horizon_k": k, "features": name, **met,
                       "eth_staleness_median_s": float(np.nanmedian(stale_s))}
                if name == "BTC + ETH features":
                    imp = m.feature_importances_
                    rec["eth_gain_share_pct"] = float(100 * imp[X_btc.shape[1]:].sum() / imp.sum())
                seg_rows.append(rec)
                print(f"    {session.name} k={k:<3} {name:<20} macro-F1 {met['macro_f1']:.4f}")
        pd.DataFrame(seg_rows).to_csv(ckpt, index=False)
        rows.extend(seg_rows)

    df = pd.DataFrame(rows)
    df.to_csv(out / "cross_asset_per_segment.csv", index=False)
    piv = df.pivot_table(index=["horizon_k", "seed"], columns="features", values="macro_f1")
    piv["delta"] = piv["BTC + ETH features"] - piv["BTC features"]
    summary = piv.groupby("horizon_k").agg(
        btc_only_macro_f1=("BTC features", "mean"), with_eth_macro_f1=("BTC + ETH features", "mean"),
        delta_mean=("delta", "mean"), delta_std=("delta", "std"),
        segments_improved=("delta", lambda d: int((d > 0).sum())), segments=("delta", "size")).reset_index()
    gain = df[df.features == "BTC + ETH features"].groupby("horizon_k").eth_gain_share_pct.mean().rename("eth_gain_share_pct")
    summary = summary.merge(gain, on="horizon_k")
    summary.to_csv(out / "cross_asset_summary.csv", index=False)
    return summary


if __name__ == "__main__":
    print(run_experiment_f().round(4).to_string(index=False))
