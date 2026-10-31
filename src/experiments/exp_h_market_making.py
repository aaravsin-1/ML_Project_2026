"""Experiment H: Market making with an adverse-selection model (research spec, section 6.2).

Instead of predicting direction and paying the spread, a market maker rests limit orders at
the best bid and ask and earns the spread, but its fills are adversely selected: a resting
bid tends to fill right before the price falls. The spec's "sophisticated iteration" is to
model that risk and quote only when it is worth it. This experiment:

1. Labels every snapshot t with the outcome of a hypothetical 0.1 BTC quote posted at the
   touch on each side, alive for QUOTE_LIFE seconds:
     - fill time under three fill rules, reported as bounds because 1-second snapshots
       cannot show queue position exactly:
         pessimistic  fills only when the price trades through the quote
         queue        fills once recorded market flow at the quote's price exceeds the
                      resting notional that was ahead of it (cancellations ahead ignored)
         optimistic   front of the queue: any market flow at the quote's price fills it
     - markout: mid price MARKOUT seconds after the fill relative to the fill price, in bp.
       Positive means the fill earned money (spread captured, no adverse move).
   The value of posting = markout if filled, else 0.
2. Trains LightGBM per side and fill rule to predict that value, from the usual microstructure
   features plus new order-flow features (limit / market / cancel notional per level, which
   the spec calls the event trajectory).
3. Simulates two-sided quoting on the test period with a 0.5 BTC inventory limit, marked to
   market, and any inventory left at the end closed with a taker order:
     always   quote both sides all the time (naive market maker)
     rule     the spec's Scenario A/B: quote when the spread is at least its training median
              and VPIN is below its training 67th percentile; cancel when VPIN exceeds its
              training 90th percentile
     model    quote a side only when the predicted value of posting exceeds a threshold
              chosen on the validation period
4. Reports how well adverse selection can be predicted (AUC of "toxic fill" given a fill),
   and how markouts vary with VPIN.

Fees: Coinbase top tier, April 2021: 0 bp maker, 4 bp taker.

    python -m src.experiments.exp_h_market_making
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.config import ARTIFACTS_DIR
from src.data.crypto_loader import CryptoLOBLoader
from src.experiments.common import BTC_PATH, PreparedData, btc_sessions, prepare_data

QUOTE_LIFE = 30          # seconds a quote rests before it is cancelled
MARKOUT = 30             # seconds after a fill at which it is marked
LOT_BTC = 0.1
MAX_INVENTORY_BTC = 0.5
MAKER_FEE_BPS = 0.0
TAKER_FEE_BPS = 4.0
FLOW_LEVELS = 5
FILL_RULES = ["pessimistic", "queue", "optimistic"]
SIDES = {"bid": 1, "ask": -1}
STRATEGIES = ["always", "rule", "model"]


# --------------------------------------------------------------------------- features

def flow_features(flows: Dict[str, np.ndarray], lob: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Order-flow features at each snapshot (flows accumulated since the previous one, so causal)."""
    feats, names = [], []
    l0_notional = {"bids": lob[:, 2] * lob[:, 3], "asks": lob[:, 0] * lob[:, 1]}
    for side in ("bids", "asks"):
        depth = l0_notional[side] + 1.0
        for kind in ("limit", "market", "cancel"):
            feats.append(flows[f"{side}_{kind}"][:, 0] / depth)
            names.append(f"{side}_{kind}_l0_rel")
            feats.append(np.log1p(flows[f"{side}_{kind}"].sum(axis=1)))
            names.append(f"{side}_{kind}_l0to4_log")
        feats.append(flows[f"{side}_cancel"][:, 0] / (flows[f"{side}_limit"][:, 0] + 1.0))
        names.append(f"{side}_cancel_to_limit_l0")
    # recent market-order imbalance: buys hit asks, sells hit bids (10-second sums)
    buy = pd.Series(flows["asks_market"].sum(axis=1)).rolling(10, min_periods=1).sum().to_numpy()
    sell = pd.Series(flows["bids_market"].sum(axis=1)).rolling(10, min_periods=1).sum().to_numpy()
    feats.append((buy - sell) / (buy + sell + 1.0))
    names.append("market_flow_imbalance_10s")
    return np.column_stack(feats).astype(np.float32), names


# --------------------------------------------------------------------------- quote outcomes

def quote_outcomes(lob: np.ndarray, flows: Dict[str, np.ndarray], side: int, rule: str,
                   life: int = QUOTE_LIFE, markout: int = MARKOUT, lot: float = LOT_BTC,
                   maker_fee_bps: float = MAKER_FEE_BPS, flow_book: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """For a quote posted at every row t at the touch: fill offset (0 = no fill) and markout in bp.

    Vectorized over t, looping over the offset d = 1..life. Prices are compared in integer cents.
    `flow_book` is the book whose level prices the flow columns refer to (default: `lob`, as in
    the Coinbase data; LOBSTER flows refer to the book at the start of each interval).
    """
    n = len(lob)
    p_col, q_col, flow_key = (2, 3, "bids_market") if side > 0 else (0, 1, "asks_market")
    cents = lambda a: np.rint(a * 100).astype(np.int64)
    price = cents(lob[:, p_col])
    queue_ahead = lob[:, p_col] * lob[:, q_col]          # USD resting at our price when we join
    need = queue_ahead + lot * lob[:, p_col]
    flow_book = lob if flow_book is None else flow_book
    level_prices = [cents(flow_book[:, 4 * lvl + p_col]) for lvl in range(FLOW_LEVELS)]
    mkt = flows[flow_key]

    fill_d = np.zeros(n, dtype=np.int64)
    consumed = np.zeros(n)
    rows = np.arange(n)
    for d in range(1, life + 1):
        j = np.minimum(rows + d, n - 1)
        open_ = (fill_d == 0) & (rows + d < n)
        through = (price[j] < price) if side > 0 else (price[j] > price)
        at_our_price = np.zeros(n)
        for lvl in range(FLOW_LEVELS):
            at_our_price += np.where(level_prices[lvl][j] == price, mkt[j, lvl], 0.0)
        consumed += at_our_price * open_
        if rule == "pessimistic":
            filled = through
        elif rule == "queue":
            filled = through | (consumed >= need)
        else:
            filled = through | (consumed > 0)
        fill_d[open_ & filled] = d

    mid = (lob[:, 0] + lob[:, 2]) / 2.0
    fill_row = np.minimum(rows + fill_d, n - 1)
    mark_row = np.minimum(fill_row + markout, n - 1)
    fill_price = lob[:, p_col]
    mk = side * (mid[mark_row] - fill_price) / fill_price * 1e4 - maker_fee_bps
    mk[fill_d == 0] = np.nan
    return fill_d, mk


# --------------------------------------------------------------------------- simulation

def simulate(lob: np.ndarray, test_rows: np.ndarray, fills: Dict[str, np.ndarray], allow: Dict[str, np.ndarray],
             cancel: np.ndarray, lot: float = LOT_BTC, max_inventory: float = MAX_INVENTORY_BTC,
             maker_fee_bps: float = MAKER_FEE_BPS, taker_fee_bps: float = TAKER_FEE_BPS,
             life: int = QUOTE_LIFE, rows_per_day: float = 86400.0) -> Dict[str, float]:
    """Two-sided quoting with one working order per side, inventory limits and mark-to-market P&L.

    fills[side][t]: fill offset if a quote posted at row t fills (0 = never within its life).
    allow[side][i]: whether the strategy may post that side at test_rows[i].
    cancel[i]: cancel all working orders at test_rows[i].
    """
    mid = (lob[:, 0] + lob[:, 2]) / 2.0
    cash, inv = 0.0, 0.0
    working = {s: None for s in SIDES}       # (fill_row or -1, expiry_row, price)
    n_fills = {s: 0 for s in SIDES}
    equity, invs = [], []
    for i, t in enumerate(test_rows):
        # 1. fills and expiries due at t
        for s, sign in SIDES.items():
            w = working[s]
            if w is None:
                continue
            fill_row, expiry, price = w
            if fill_row == t:
                cash -= sign * lot * price + lot * price * maker_fee_bps / 1e4
                inv += sign * lot
                n_fills[s] += 1
                working[s] = None
            elif t >= expiry or cancel[i]:
                working[s] = None
        # 2. new quotes
        for s, sign in SIDES.items():
            if working[s] is not None or not allow[s][i] or cancel[i]:
                continue
            if abs(inv + sign * lot) > max_inventory + 1e-9:
                continue
            d = fills[s][t]
            price = lob[t, 2] if sign > 0 else lob[t, 0]
            working[s] = (t + d if d > 0 else -1, t + life, price)
        equity.append(cash + inv * mid[t])
        invs.append(inv)
    # close any inventory with a taker order at the touch
    t_end = test_rows[-1]
    exit_price = lob[t_end, 2] if inv > 0 else lob[t_end, 0]
    close_cost = abs(inv) * exit_price * taker_fee_bps / 1e4
    final = cash + inv * exit_price - close_cost
    eq = np.array(equity)
    days = len(test_rows) / rows_per_day
    return {"net_pnl_usd": final, "pnl_per_day_usd": final / days, "fills_total": n_fills["bid"] + n_fills["ask"],
            "fills_bid": n_fills["bid"],
            "fills_ask": n_fills["ask"], "max_drawdown_usd": float(np.max(np.maximum.accumulate(eq) - eq)),
            "mean_abs_inventory_btc": float(np.mean(np.abs(invs))), "test_days": days}


# --------------------------------------------------------------------------- per segment

BTC_MM_PARAMS = {"life": QUOTE_LIFE, "markout": MARKOUT, "lot": LOT_BTC, "max_inventory": MAX_INVENTORY_BTC,
                 "maker_fee_bps": MAKER_FEE_BPS, "taker_fee_bps": TAKER_FEE_BPS, "rows_per_day": 86400.0,
                 "flow_book": None}


def evaluate_segment(data: PreparedData, flows: Dict[str, np.ndarray], params: Dict = None) -> Tuple[List[Dict], List[Dict]]:
    p = dict(BTC_MM_PARAMS, **(params or {}))
    session = data.session
    lob = session.lob_40
    fx, fx_names = flow_features(flows, lob)
    X = np.hstack([data.tab_feat.astype(np.float32), fx])
    idx = data.splits(p["life"] + p["markout"])      # purge: labels look this far ahead
    tr, va, te = idx["train"], idx["val"], idx["test"]
    vpin = data.tab_feat[:, data.feat_names.index("vpin")]
    spread_bp = (lob[:, 0] - lob[:, 2]) / ((lob[:, 0] + lob[:, 2]) / 2) * 1e4

    # spec Scenario A/B thresholds, from training rows only
    spread_med = np.median(spread_bp[tr])
    vpin_q67, vpin_q90 = np.quantile(vpin[tr], [0.67, 0.90])
    rule_allow = (spread_bp >= spread_med) & (vpin <= vpin_q67)
    rule_cancel = vpin > vpin_q90

    results, diag = [], []
    for rule in FILL_RULES:
        fills, values, preds = {}, {}, {}
        for s, sign in SIDES.items():
            fill_d, mk = quote_outcomes(lob, flows, sign, rule, life=p["life"], markout=p["markout"], lot=p["lot"],
                                        maker_fee_bps=p["maker_fee_bps"], flow_book=p["flow_book"])
            fills[s] = fill_d
            value = np.where(fill_d > 0, mk, 0.0)
            values[s] = value
            model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=6,
                                      min_child_samples=200, subsample=0.8, subsample_freq=1,
                                      random_state=session.seed, verbosity=-1)
            model.fit(X[tr], value[tr], eval_set=[(X[va], value[va])],
                      callbacks=[lgb.early_stopping(20, verbose=False)])
            preds[s] = model.predict(X)

            # adverse selection: given a fill, can we tell toxic (markout < 0) from good?
            filled_te = te[fill_d[te] > 0]
            filled_tr = tr[fill_d[tr] > 0]
            auc = np.nan
            if len(filled_tr) > 500 and len(filled_te) > 100:
                clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                         min_child_samples=200, random_state=session.seed, verbosity=-1)
                clf.fit(X[filled_tr], (mk[filled_tr] < 0).astype(int))
                y_te = (mk[filled_te] < 0).astype(int)
                if 0 < y_te.mean() < 1:
                    auc = roc_auc_score(y_te, clf.predict_proba(X[filled_te])[:, 1])
            vt = vpin[te]
            cuts = np.quantile(vpin[tr], [1 / 3, 2 / 3])
            terc = np.digitize(vt, cuts)
            rec = {"segment": session.seed, "fill_rule": rule, "side": s,
                   "fill_rate_test": float(np.mean(fill_d[te] > 0)),
                   "mean_markout_bp_test": float(np.nanmean(mk[te])) if np.any(fill_d[te] > 0) else np.nan,
                   "toxic_share_test": float(np.nanmean(mk[te] < 0)) if np.any(fill_d[te] > 0) else np.nan,
                   "toxic_auc_test": auc}
            for r, name in enumerate(["low", "mid", "high"]):
                sel = (terc == r) & (fill_d[te] > 0)
                rec[f"markout_bp_vpin_{name}"] = float(np.mean(mk[te][sel])) if sel.any() else np.nan
            diag.append(rec)

        # model threshold per side: maximise total validation value over a quantile grid of predictions
        thresholds = {}
        for s in SIDES:
            grid = np.unique(np.quantile(preds[s][va], [0.0, 0.5, 0.75, 0.9, 0.95, 0.99]))
            totals = [values[s][va][preds[s][va] > g].sum() for g in grid]
            thresholds[s] = float(grid[int(np.argmax(totals))])

        no_cancel = np.zeros(len(te), dtype=bool)
        strategies = {
            "always": ({s: np.ones(len(te), dtype=bool) for s in SIDES}, no_cancel),
            "rule": ({s: rule_allow[te] for s in SIDES}, rule_cancel[te]),
            "model": ({s: preds[s][te] > thresholds[s] for s in SIDES}, no_cancel),
        }
        for name, (allow, cancel) in strategies.items():
            res = simulate(lob, te, fills, allow, cancel, lot=p["lot"], max_inventory=p["max_inventory"],
                           maker_fee_bps=p["maker_fee_bps"], taker_fee_bps=p["taker_fee_bps"], life=p["life"],
                           rows_per_day=p["rows_per_day"])
            res.update({"segment": session.seed, "fill_rule": rule, "strategy": name,
                        "quote_share_bid": float(np.mean(allow["bid"])), "quote_share_ask": float(np.mean(allow["ask"]))})
            results.append(res)
            print(f"    {rule:<11} {name:<7} net ${res['net_pnl_usd']:>10,.0f}  fills {res['fills_bid']:>5}/{res['fills_ask']:<5}"
                  f" |inv| {res['mean_abs_inventory_btc']:.2f}")
    return results, diag


def run_experiment_h(n_segments: int = 4) -> pd.DataFrame:
    out = ARTIFACTS_DIR / "btc" / "market_making"
    ckpt_dir = out / "sessions"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    flows_all = CryptoLOBLoader.load_order_flows(BTC_PATH, num_levels=FLOW_LEVELS)
    n_all = len(next(iter(flows_all.values())))
    bounds = np.linspace(0, n_all, n_segments + 1).astype(int)   # same segmentation as btc_sessions

    res_all, diag_all = [], []
    for session in btc_sessions(n_segments):
        ck_res, ck_diag = ckpt_dir / f"{session.name}_results.csv", ckpt_dir / f"{session.name}_diag.csv"
        if ck_res.exists() and ck_diag.exists():
            print(f"[+] {session.name}: loading checkpoint")
            res_all.append(pd.read_csv(ck_res))
            diag_all.append(pd.read_csv(ck_diag))
            continue
        print(f"\n######## {session.name} ########")
        lo, hi = bounds[session.seed], bounds[session.seed + 1]
        flows = {k: v[lo:hi] for k, v in flows_all.items()}
        assert len(next(iter(flows.values()))) == len(session.lob_40)
        results, diag = evaluate_segment(prepare_data(session), flows)
        pd.DataFrame(results).to_csv(ck_res, index=False)
        pd.DataFrame(diag).to_csv(ck_diag, index=False)
        res_all.append(pd.DataFrame(results))
        diag_all.append(pd.DataFrame(diag))

    res = pd.concat(res_all, ignore_index=True)
    diag = pd.concat(diag_all, ignore_index=True)
    res.to_csv(out / "mm_per_segment.csv", index=False)
    diag.to_csv(out / "adverse_selection_per_segment.csv", index=False)
    summary = (res.groupby(["fill_rule", "strategy"], sort=False)
               .agg(net_pnl_usd=("net_pnl_usd", "mean"), net_pnl_std=("net_pnl_usd", "std"),
                    pnl_per_day_usd=("pnl_per_day_usd", "mean"), segments_positive=("net_pnl_usd", lambda x: int((x > 0).sum())),
                    fills=("fills_total", "mean"), max_drawdown_usd=("max_drawdown_usd", "mean"),
                    mean_abs_inventory_btc=("mean_abs_inventory_btc", "mean"))
               .reset_index())
    summary.to_csv(out / "mm_summary.csv", index=False)
    diag_summary = diag.groupby(["fill_rule", "side"], sort=False).mean(numeric_only=True).drop(columns="segment").reset_index()
    diag_summary.to_csv(out / "adverse_selection_summary.csv", index=False)
    return summary


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    print(run_experiment_h().round(2).to_string(index=False))
    print(pd.read_csv(ARTIFACTS_DIR / "btc" / "market_making" / "adverse_selection_summary.csv").round(3).to_string(index=False))
