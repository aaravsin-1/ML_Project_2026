"""NSE Experiments B & C: Signal Decay across ALL 8 NSE Instruments (large-tick to tiny-tick
futures) and 5-Stage Feature Ablation (testing NSE-native order counts & full-book qty).

    python -m src.experiments.nse_exp_b_c_decay_ablation
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.config import ARTIFACTS_DIR
from src.data.nse_loader import load_nse_symbol
from src.microstructure.nse_pipeline import NSE_HORIZONS, build_nse_segments, prepare_nse_segment
from src.models.baselines import LogisticRegressionBaseline, NaiveMomentumBaseline
from src.models.tabular_trees import LightGBMLOB

OUT_DIR = ARTIFACTS_DIR / "nse" / "decay_ablation"
INSTRUMENTS = [
    ("IDEA", "EQUITY", "ultra_large_tick"),
    ("YESBANK", "EQUITY", "large_tick"),
    ("ITC", "EQUITY", "large_tick"),
    ("RELIANCE", "EQUITY", "medium_tick"),
    ("HDFCBANK", "EQUITY", "medium_tick"),
    ("ICICIBANK", "EQUITY", "medium_tick"),
    ("TCS", "EQUITY", "small_tick"),
    ("NIFTY", "DERIVATIVE", "tiny_tick_fut"),
]
ABLATION_TARGETS = [
    ("RELIANCE", "EQUITY"),
    ("NIFTY", "DERIVATIVE"),
    ("IDEA", "EQUITY"),
    ("ITC", "EQUITY"),
    ("TCS", "EQUITY"),
]


def _ablation_sets(feat_names: List[str]) -> Dict[str, List[int]]:
    idx = {n: i for i, n in enumerate(feat_names)}
    a = [idx["spread_ticks"]]
    b = [idx[c] for c in ("spread_ticks", "imbalance_l1", "ofi_l1", "micro_deviation")]
    c = b + [idx[c] for c in feat_names if "_lag_" in c]
    d = c + [idx[c] for c in ("imbalance_l3", "imbalance_l5", "slope_bid", "slope_ask",
                              "mlofi_lvl_2", "mlofi_lvl_3", "mlofi_lvl_4", "mlofi_lvl_5",
                              "ofi_pca1", "ofi_pca2", "ofi_pca3", "vpin")]
    e = list(range(len(feat_names)))  # adds the 5 NSE-native order-count / full-book features
    return {
        "A: Spread only (1)": a,
        "B: L1 snapshot (4)": b,
        "C: B + L1 lags (13)": c,
        "D: C + L2-5 depth, PCA, VPIN (25)": d,
        "E: D + NSE order counts & full book (30)": e,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    decay_rows = []
    ablation_rows = []

    ablation_sym_set = {f"{s}_FUT" if a == "DERIVATIVE" else s for s, a in ABLATION_TARGETS}

    for sym, asset, tick_class in INSTRUMENTS:
        sym_data = load_nse_symbol(sym, asset=asset)
        tag = sym_data.symbol
        segs = build_nse_segments(sym_data, n_segments=4)
        print(f"\n######## {tag} ({tick_class}) — 4 segments × {len(NSE_HORIZONS)} horizons ########")

        for seg in segs:
            prep = prepare_nse_segment(seg, horizons=NSE_HORIZONS)
            X_core = prep.tab_feat[:, prep.core_indices]
            X_full = prep.tab_feat

            for k in NSE_HORIZONS:
                y_all = prep.labels(k)
                sp = prep.splits(k)
                tr, va, te = sp["train"], sp["val"], sp["test"]
                y_tr, y_va, y_te = y_all[tr], y_all[va], y_all[te]
                flat_share = float(np.mean(y_te == 0))

                mom_p = NaiveMomentumBaseline(k).predict(y_all, te)
                mom_f1 = float(f1_score(y_te, mom_p, average="macro", zero_division=0))

                lr = LogisticRegressionBaseline(C=1.0)
                lr.fit(X_core[tr], y_tr)
                lr_p = lr.predict(X_core[te])
                lr_f1 = float(f1_score(y_te, lr_p, average="macro", zero_division=0))

                lgb_c = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=300, random_state=seg.segment_id)
                lgb_c.fit(X_core[tr], y_tr, X_core[va], y_va, early_stopping_rounds=15)
                lgb_p = lgb_c.predict(X_core[te])
                lgb_pr = lgb_c.predict_proba(X_core[te])
                lgb_f1 = float(f1_score(y_te, lgb_p, average="macro", zero_division=0))
                lgb_acc = float(accuracy_score(y_te, lgb_p))
                try:
                    lgb_auc = float(roc_auc_score(y_te, lgb_pr, multi_class="ovr", average="macro", labels=[0, 1, 2]))
                except ValueError:
                    lgb_auc = np.nan

                lgb_f = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=300, random_state=seg.segment_id)
                lgb_f.fit(X_full[tr], y_tr, X_full[va], y_va, early_stopping_rounds=15)
                lgbf_p = lgb_f.predict(X_full[te])
                lgbf_f1 = float(f1_score(y_te, lgbf_p, average="macro", zero_division=0))

                decay_rows.append({
                    "symbol": tag,
                    "tick_class": tick_class,
                    "segment": seg.segment_id,
                    "horizon_s": k,
                    "alpha_bp": prep.alphas_dict[k] * 1e4,
                    "test_flat_share": flat_share,
                    "momentum_f1": mom_f1,
                    "logreg_f1": lr_f1,
                    "lgbm_core_f1": lgb_f1,
                    "lgbm_nse_f1": lgbf_f1,
                    "lgbm_minus_mom": lgb_f1 - mom_f1,
                    "nse_extra_lift": lgbf_f1 - lgb_f1,
                    "lgbm_core_acc": lgb_acc,
                    "lgbm_core_auc": lgb_auc,
                })

            # Feature ablation at k = 20 s
            if tag in ablation_sym_set:
                y20 = prep.labels(20)
                sp20 = prep.splits(20)
                tr, va, te = sp20["train"], sp20["val"], sp20["test"]
                for set_name, cols in _ablation_sets(prep.feat_names).items():
                    m = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=300, random_state=seg.segment_id)
                    m.fit(X_full[tr][:, cols], y20[tr], X_full[va][:, cols], y20[va], early_stopping_rounds=15)
                    pred = m.predict(X_full[te][:, cols])
                    pr = m.predict_proba(X_full[te][:, cols])
                    try:
                        auc = float(roc_auc_score(y20[te], pr, multi_class="ovr", average="macro", labels=[0, 1, 2]))
                    except ValueError:
                        auc = np.nan
                    ablation_rows.append({
                        "symbol": tag,
                        "segment": seg.segment_id,
                        "feature_set": set_name,
                        "n_features": len(cols),
                        "macro_f1": float(f1_score(y20[te], pred, average="macro", zero_division=0)),
                        "accuracy": float(accuracy_score(y20[te], pred)),
                        "roc_auc_ovr": auc,
                    })

        # Print quick summary for this symbol
        sub = pd.DataFrame([r for r in decay_rows if r["symbol"] == tag])
        g = sub.groupby("horizon_s")[["momentum_f1", "logreg_f1", "lgbm_core_f1", "lgbm_nse_f1", "lgbm_minus_mom", "lgbm_core_auc"]].mean()
        print(g.round(4).to_string())

    df_dec = pd.DataFrame(decay_rows)
    df_dec.to_csv(OUT_DIR / "nse_decay_per_segment.csv", index=False)
    val_cols = ["alpha_bp", "test_flat_share", "momentum_f1", "logreg_f1", "lgbm_core_f1", "lgbm_nse_f1", "lgbm_minus_mom", "nse_extra_lift", "lgbm_core_acc", "lgbm_core_auc"]
    dec_sum = df_dec.groupby(["symbol", "tick_class", "horizon_s"], sort=False)[val_cols].agg(["mean", "std"])
    dec_sum.columns = [f"{a}_{b}" if b == "std" else a for a, b in dec_sum.columns]
    dec_sum.reset_index().to_csv(OUT_DIR / "nse_decay_summary.csv", index=False)

    df_abl = pd.DataFrame(ablation_rows)
    df_abl.to_csv(OUT_DIR / "nse_ablation_per_segment.csv", index=False)
    abl_sum = df_abl.groupby(["symbol", "feature_set", "n_features"], sort=False)[["macro_f1", "accuracy", "roc_auc_ovr"]].agg(["mean", "std"])
    abl_sum.columns = [f"{a}_{b}" if b == "std" else a for a, b in abl_sum.columns]
    abl_sum.reset_index().to_csv(OUT_DIR / "nse_ablation_summary.csv", index=False)

    print(f"\n[+] NSE Experiments B + C saved to {OUT_DIR} in {(time.time() - t0) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
