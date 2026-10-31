"""Experiment K: The micro-price as a prediction target (research spec, section 3.3).

The spec suggests that the gap between the mid price and the micro-price is "an additional,
highly effective target variable". Here the micro-price is the imbalance-weighted mid,
    micro = (q_bid * ask + q_ask * bid) / (q_bid + q_ask),   so   micro - mid = I * spread / 2
with I the level-1 queue imbalance (the project's `micro_deviation` feature; Stoikov's full
micro-price adds a Markov-chain correction on top of this). Three questions, per horizon k:

1. Does the mid price move toward the micro-price?  Regress the future mid change on the
   current gap with no intercept: beta = 1 means the gap closes completely by t + k,
   beta = 0 means the gap says nothing. beta is fitted on training rows, reported on test.
2. Is the micro-price easier to predict than the mid?  Out-of-sample R^2 of LightGBM trained
   and evaluated on future micro-price changes vs on future mid changes.
3. Is it a better TRAINING target?  What matters for trading is the realized mid move, so a
   model trained on the micro-price target is judged on future mid changes, against a model
   trained on mid changes directly (information coefficient, calibrated R^2, sign hit rate),
   with a moving-block bootstrap CI for the difference.

Models: zero forecast, the one-parameter micro-price rule (beta * gap), a momentum baseline (the past
k-step return: do moves continue or reverse?), ridge regression and
LightGBM on the microstructure + order-flow features. The control is LightGBM trained on shuffled
training and validation targets, averaged over 3 shuffles: it must score about zero, or something leaks.

Data: Coinbase BTC (4 segments, k = 10 / 20 / 50 s) and LOBSTER (5 stocks on a 100 ms clock,
k = 500 ms, 1 s, 5 s, 10 s, 30 s). Returns are in bp. Training targets are winsorized at the
training 0.1 / 99.9 percentiles; evaluation always uses the raw values.

    python -m src.experiments.exp_k_microprice
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.config import ARTIFACTS_DIR

BTC_HORIZONS = {"10s": 10, "20s": 20, "50s": 50}
LOBSTER_HORIZONS = {"500ms": 5, "1s": 10, "5s": 50, "10s": 100, "30s": 300}
BOOTSTRAP_DRAWS = 1000
CONTROL_REPS = 3
OUT = ARTIFACTS_DIR / "microprice"


# --------------------------------------------------------------------------- pure functions

def mid_and_micro(lob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mid price and imbalance-weighted micro-price from level 1 of a 40-column book."""
    ask, ask_q, bid, bid_q = lob[:, 0], lob[:, 1], lob[:, 2], lob[:, 3]
    mid = (ask + bid) / 2.0
    depth = bid_q + ask_q
    micro = np.where(depth > 0, (bid_q * ask + ask_q * bid) / np.where(depth > 0, depth, 1.0), mid)
    return mid, micro


def forward_change_bp(price: np.ndarray, k: int) -> np.ndarray:
    """(price[t+k] - price[t]) / price[t] in bp; NaN for the last k rows (no wrap-around)."""
    out = np.full(len(price), np.nan)
    out[:-k] = (price[k:] - price[:-k]) / price[:-k] * 1e4
    return out


def gap_bp(mid: np.ndarray, micro: np.ndarray) -> np.ndarray:
    """Current micro-price gap, (micro - mid) / mid in bp."""
    return (micro - mid) / mid * 1e4


def fit_beta(x: np.ndarray, y: np.ndarray) -> float:
    """No-intercept least squares slope of y on x, ignoring NaNs."""
    ok = np.isfinite(x) & np.isfinite(y)
    denom = float(np.dot(x[ok], x[ok]))
    return float(np.dot(x[ok], y[ok]) / denom) if denom > 0 else 0.0


def r2_vs_zero(y: np.ndarray, pred: np.ndarray) -> float:
    """Out-of-sample R^2 against the zero forecast (the natural benchmark for returns)."""
    ok = np.isfinite(y) & np.isfinite(pred)
    sst = float(np.sum(y[ok] ** 2))
    return 1.0 - float(np.sum((y[ok] - pred[ok]) ** 2)) / sst if sst > 0 else np.nan


def score(y: np.ndarray, pred: np.ndarray, calib: float = 1.0) -> Dict[str, float]:
    """IC (Pearson and Spearman), R^2 vs zero after scaling by `calib`, and sign hit rate on moves."""
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    const = np.std(pred) == 0 or np.std(y) == 0
    moved = y != 0
    return {
        "ic_pearson": np.nan if const else float(np.corrcoef(y, pred)[0, 1]),
        "ic_spearman": np.nan if const else float(spearmanr(y, pred).statistic),
        "r2": r2_vs_zero(y, calib * pred),
        "hit_rate": float(np.mean(np.sign(pred[moved]) == np.sign(y[moved]))) if moved.any() else np.nan,
        "n": int(len(y)),
    }


def winsorize(y: np.ndarray, ref: np.ndarray, q: float = 0.001) -> np.ndarray:
    lo, hi = np.nanquantile(ref, [q, 1 - q])
    return np.clip(y, lo, hi)


def block_bootstrap_ic_diff(y: np.ndarray, a: np.ndarray, b: np.ndarray, block: int, draws: int = BOOTSTRAP_DRAWS,
                            seed: int = 0) -> Tuple[float, float]:
    """95% CI of corr(y, a) - corr(y, b) with a moving-block bootstrap over rows."""
    ok = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    y, a, b = y[ok], a[ok], b[ok]
    n = len(y)
    block = max(1, min(block, n // 10))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(draws)
    for i in range(draws):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        ix = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        diffs[i] = np.corrcoef(y[ix], a[ix])[0, 1] - np.corrcoef(y[ix], b[ix])[0, 1]
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


# --------------------------------------------------------------------------- models

def fit_lgbm(X: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, seed: int) -> lgb.LGBMRegressor:
    ok_tr, ok_va = tr[np.isfinite(y[tr])], va[np.isfinite(y[va])]
    yt = winsorize(y[ok_tr], y[ok_tr])
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, max_depth=8,
                              min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              random_state=seed, verbosity=-1)
    model.fit(X[ok_tr], yt, eval_set=[(X[ok_va], winsorize(y[ok_va], y[ok_tr]))],
              callbacks=[lgb.early_stopping(30, verbose=False)])
    return model


def fit_ridge(X: np.ndarray, y: np.ndarray, tr: np.ndarray):
    ok = tr[np.isfinite(y[tr])]
    scaler = StandardScaler().fit(X[ok])
    model = Ridge(alpha=1.0).fit(scaler.transform(X[ok]), winsorize(y[ok], y[ok]))
    return lambda Z: model.predict(scaler.transform(Z))


def evaluate_session(name: str, dataset: str, lob: np.ndarray, X: np.ndarray, splits_fn, horizons: Dict[str, int],
                     seed: int, block: int) -> Tuple[List[Dict], List[Dict]]:
    """All metrics for one segment / stock. Returns (model rows, beta rows)."""
    mid, micro = mid_and_micro(lob)
    gap = gap_bp(mid, micro)
    X = np.column_stack([X, gap]).astype(np.float32)  # make sure the gap itself is a feature
    rows, betas = [], []
    for hname, k in horizons.items():
        idx = splits_fn(k)
        tr, va, te = idx["train"], idx["val"], idx["test"]
        y_mid = forward_change_bp(mid, k)
        y_micro = forward_change_bp(micro, k)
        y_eval = y_mid[te]

        def add(model: str, target: str, pred_te: np.ndarray, pred_va: np.ndarray):
            # calibrate the scale to the MID target on validation, so models trained on another target are comparable
            calib = fit_beta(pred_va, y_mid[va]) if pred_va is not None else 1.0
            s = score(y_eval, pred_te, calib)
            own = score((y_micro if target == "micro" else y_mid)[te], pred_te)
            rows.append({"dataset": dataset, "name": name, "horizon": hname, "k_rows": k, "model": model,
                         "train_target": target, **s, "r2_own_target": own["r2"],
                         "test_mid_moved_share": float(np.mean(y_eval[np.isfinite(y_eval)] != 0))})

        # 1. micro-price rule: beta * gap, beta fitted on training rows
        beta_tr = fit_beta(gap[tr], y_mid[tr])
        beta_te = fit_beta(gap[te], y_mid[te])
        beta_micro_tr = fit_beta(gap[tr], y_micro[tr])
        betas.append({"dataset": dataset, "name": name, "horizon": hname, "k_rows": k, "beta_train": beta_tr,
                      "beta_test": beta_te, "beta_micro_target_train": beta_micro_tr,
                      "gap_abs_mean_bp": float(np.nanmean(np.abs(gap[te]))),
                      "mid_move_abs_mean_bp": float(np.nanmean(np.abs(y_eval)))})
        add("Zero forecast", "-", np.zeros(len(te)), None)
        add("Micro-price rule (beta x gap)", "mid", beta_tr * gap[te], beta_tr * gap[va])
        # momentum baseline: the past k-step mid return (positive IC = moves continue, negative = they reverse)
        past = np.full(len(mid), np.nan)
        past[k:] = (mid[k:] - mid[:-k]) / mid[:-k] * 1e4
        add("Momentum (past k-step return)", "mid", np.nan_to_num(past[te]), np.nan_to_num(past[va]))

        preds = {}
        for target, y in (("mid", y_mid), ("micro", y_micro)):
            ridge = fit_ridge(X, y, tr)
            add("Ridge", target, ridge(X[te]), ridge(X[va]))
            m = fit_lgbm(X, y, tr, va, seed)
            p_te, p_va = m.predict(X[te]), m.predict(X[va])
            preds[target] = p_te
            add("LightGBM", target, p_te, p_va)

        # control: train AND validation targets shuffled (so no real label reaches the model, not even
        # through early stopping), averaged over CONTROL_REPS shuffles; its IC must be about zero
        ctrl = []
        for rep in range(CONTROL_REPS):
            y_shuf = y_mid.copy()
            rng = np.random.default_rng(1000 * seed + 10 * k + rep)
            y_shuf[tr] = rng.permutation(y_mid[tr])
            y_shuf[va] = rng.permutation(y_mid[va])
            m = fit_lgbm(X, y_shuf, tr, va, seed + rep)
            ctrl.append(score(y_eval, m.predict(X[te])))
        rows.append({"dataset": dataset, "name": name, "horizon": hname, "k_rows": k,
                     "model": "LightGBM, shuffled-target control", "train_target": "mid",
                     "ic_pearson": float(np.nanmean([c["ic_pearson"] for c in ctrl])),
                     "ic_spearman": float(np.nanmean([c["ic_spearman"] for c in ctrl])),
                     "r2": float(np.nanmean([c["r2"] for c in ctrl])),
                     "hit_rate": float(np.nanmean([c["hit_rate"] for c in ctrl])), "n": ctrl[0]["n"],
                     "r2_own_target": np.nan, "control_ic_sd": float(np.nanstd([c["ic_pearson"] for c in ctrl])),
                     "test_mid_moved_share": float(np.mean(y_eval[np.isfinite(y_eval)] != 0))})

        lo, hi = block_bootstrap_ic_diff(y_eval, preds["micro"], preds["mid"], block, seed=seed + k)
        for r in rows[-8:]:
            r["ic_diff_micro_minus_mid_ci_low"] = lo if r["model"] == "LightGBM" and r["train_target"] == "micro" else np.nan
            r["ic_diff_micro_minus_mid_ci_high"] = hi if r["model"] == "LightGBM" and r["train_target"] == "micro" else np.nan
        best = [r for r in rows[-8:] if r["model"] == "LightGBM"]
        print(f"    {name:<14} {hname:>6}: beta {beta_tr:5.2f} | IC on mid move: trained on mid {best[0]['ic_pearson']:.3f}, "
              f"on micro {best[1]['ic_pearson']:.3f} [diff CI {lo:+.3f}, {hi:+.3f}]")
    return rows, betas


# --------------------------------------------------------------------------- datasets

def run_btc(n_segments: int = 4) -> None:
    from src.data.crypto_loader import CryptoLOBLoader
    from src.experiments.common import BTC_PATH, btc_sessions, prepare_data
    from src.experiments.exp_h_market_making import flow_features

    flows_all = CryptoLOBLoader.load_order_flows(BTC_PATH, num_levels=5)
    n_all = len(next(iter(flows_all.values())))
    bounds = np.linspace(0, n_all, n_segments + 1).astype(int)
    for session in btc_sessions(n_segments):
        ck = OUT / "sessions" / f"btc_{session.name}.csv"
        ck_b = OUT / "sessions" / f"btc_{session.name}_beta.csv"
        if ck.exists() and ck_b.exists():
            print(f"[+] {session.name}: loading checkpoint")
            continue
        print(f"\n######## BTC {session.name} ########")
        lo, hi = bounds[session.seed], bounds[session.seed + 1]
        flows = {k: v[lo:hi] for k, v in flows_all.items()}
        data = prepare_data(session)
        fx, _ = flow_features(flows, session.lob_40)
        X = np.hstack([data.tab_feat.astype(np.float32), fx])
        rows, betas = evaluate_session(session.name, "BTC", session.lob_40, X, data.splits, BTC_HORIZONS,
                                       session.seed, block=300)
        pd.DataFrame(rows).to_csv(ck, index=False)
        pd.DataFrame(betas).to_csv(ck_b, index=False)


def run_lobster() -> None:
    from src.experiments.common import prepare_data
    from src.experiments.exp_h_market_making import flow_features
    from src.experiments.exp_i_lobster import LOBSTER_DIR, TICKERS, lobster_session

    for seed, ticker in enumerate(TICKERS):
        ck = OUT / "sessions" / f"lobster_{ticker}.csv"
        ck_b = OUT / "sessions" / f"lobster_{ticker}_beta.csv"
        if ck.exists() and ck_b.exists():
            print(f"[+] {ticker}: loading checkpoint")
            continue
        if not (LOBSTER_DIR / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv").exists():
            print(f"[!] {ticker}: LOBSTER file missing, skipped")
            continue
        print(f"\n######## LOBSTER {ticker} ({TICKERS[ticker]} tick) ########")
        session, flows = lobster_session(ticker, seed)
        data = prepare_data(session)
        fx, _ = flow_features(flows, session.lob_40)
        X = np.hstack([data.tab_feat.astype(np.float32), fx])
        rows, betas = evaluate_session(ticker, f"LOBSTER-{TICKERS[ticker]}", session.lob_40, X, data.splits,
                                       LOBSTER_HORIZONS, seed, block=3000)
        pd.DataFrame(rows).to_csv(ck, index=False)
        pd.DataFrame(betas).to_csv(ck_b, index=False)


def summarize() -> Tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted((OUT / "sessions").glob("*.csv"))
    rows = pd.concat([pd.read_csv(f) for f in files if not f.stem.endswith("_beta")], ignore_index=True)
    betas = pd.concat([pd.read_csv(f) for f in files if f.stem.endswith("_beta")], ignore_index=True)
    rows.to_csv(OUT / "microprice_all.csv", index=False)
    betas.to_csv(OUT / "microprice_beta.csv", index=False)
    group = ["dataset", "horizon", "model", "train_target"]
    summary = (rows.groupby(group, sort=False)
               .agg(ic_pearson=("ic_pearson", "mean"), ic_spearman=("ic_spearman", "mean"), r2=("r2", "mean"),
                    r2_own_target=("r2_own_target", "mean"), hit_rate=("hit_rate", "mean"),
                    units=("name", "nunique"), ic_ci_low=("ic_diff_micro_minus_mid_ci_low", "mean"),
                    ic_ci_high=("ic_diff_micro_minus_mid_ci_high", "mean"))
               .reset_index())
    summary.to_csv(OUT / "microprice_summary.csv", index=False)
    beta_summary = betas.groupby(["dataset", "horizon"], sort=False)[
        ["beta_train", "beta_test", "beta_micro_target_train", "gap_abs_mean_bp", "mid_move_abs_mean_bp"]].mean().reset_index()
    beta_summary.to_csv(OUT / "microprice_beta_summary.csv", index=False)
    return summary, beta_summary


def run_experiment_k() -> Tuple[pd.DataFrame, pd.DataFrame]:
    (OUT / "sessions").mkdir(parents=True, exist_ok=True)
    run_btc()
    run_lobster()
    return summarize()


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    summary, beta = run_experiment_k()
    print(beta.round(3).to_string(index=False))
    print(summary[summary.model.isin(["LightGBM", "Micro-price rule (beta x gap)", "Momentum (past k-step return)",
                                      "LightGBM, shuffled-target control"])]
          [["dataset", "horizon", "model", "train_target", "ic_pearson", "r2", "r2_own_target", "hit_rate", "ic_ci_low", "ic_ci_high"]]
          .round(3).to_string(index=False))
