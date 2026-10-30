"""Why does LightGBM + history score 56.7% on FI-2010 at k=20 but 77.4% at k=10?

Retrains the seed-0 "LightGBM (144 feats + history)" model from the FI-2010 benchmark at
k = 10 and k = 20 with the same splits and settings, and records:

- the early-stopping iteration it chose, and validation / test accuracy at that point;
- test accuracy at a fixed grid of boosting rounds (no early stopping), to see whether the
  problem is stopping too early, overfitting, or the model itself;
- label shares in train / validation / test next to the predicted shares, to spot a
  model that collapses onto one class or a validation slice unlike the test days.

Output: artifacts/fi2010/diagnose_lightgbm_history.csv, plus a printed summary.

    python scripts/diagnose_fi2010_lightgbm.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS_DIR  # noqa: E402
from src.experiments.fi2010_benchmark import SEQ_LEN, history_features, load_setup2  # noqa: E402
from src.models.tabular_trees import LightGBMLOB  # noqa: E402

HORIZONS = (10, 20)
ROUND_GRID = (10, 25, 50, 100, 200, 300, 500)
SEED = 0


def shares(y: np.ndarray) -> str:
    s = np.bincount(y.astype(np.int64), minlength=3) / max(len(y), 1)
    return "/".join(f"{v:.2f}" for v in s)


def main():
    train_lob, train_extra, train_y, test_lob, test_extra, test_y = load_setup2()
    n_tr = len(train_lob)
    all_lob = np.concatenate([train_lob, test_lob])
    tab = np.concatenate([all_lob, np.concatenate([train_extra, test_extra])], axis=1)
    tab_hist = np.concatenate([tab, history_features(all_lob)], axis=1)

    rows = []
    for k in HORIZONS:
        y_all = np.concatenate([train_y[k], test_y[k]])
        val_start = int(n_tr * 0.8)
        idx = {
            "train": np.arange(SEQ_LEN - 1, val_start - k),
            "val": np.arange(val_start, n_tr - k),
            "test": np.arange(n_tr, len(y_all)),
        }
        X = {name: tab_hist[i] for name, i in idx.items()}
        y = {name: y_all[i] for name, i in idx.items()}
        print(f"\n=== k={k}: label shares FLAT/UP/DOWN  train {shares(y['train'])}  "
              f"val {shares(y['val'])}  test {shares(y['test'])}")

        # 1. Same settings as the benchmark (early stopping on validation log-loss)
        es = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=500,
                         random_state=SEED, class_weight=None)
        es.fit(X["train"], y["train"], X["val"], y["val"], early_stopping_rounds=30)
        best = int(es.model.best_iteration_ or es.model.n_estimators)
        pred_val, pred_test = es.predict(X["val"]), es.predict(X["test"])
        rec = {"horizon_k": k, "setting": "early stopping (benchmark)", "rounds": best,
               "val_accuracy": float((pred_val == y["val"]).mean()),
               "test_accuracy": float((pred_test == y["test"]).mean()),
               "test_pred_shares": shares(pred_test), "test_label_shares": shares(y["test"])}
        rows.append(rec)
        print(f"    early stopping chose {best} rounds: val acc {rec['val_accuracy']:.4f}, "
              f"test acc {rec['test_accuracy']:.4f}, predicted shares {rec['test_pred_shares']}")

        # 2. No early stopping: accuracy along a grid of rounds from one 500-round model
        full = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=max(ROUND_GRID),
                           random_state=SEED, class_weight=None)
        full.fit(X["train"], y["train"])
        for r in ROUND_GRID:
            pv = full.model.predict(X["val"], num_iteration=r)
            pt = full.model.predict(X["test"], num_iteration=r)
            rec = {"horizon_k": k, "setting": "fixed rounds", "rounds": r,
                   "val_accuracy": float((pv == y["val"]).mean()),
                   "test_accuracy": float((pt == y["test"]).mean()),
                   "test_pred_shares": shares(pt), "test_label_shares": shares(y["test"])}
            rows.append(rec)
            print(f"    {r:>4} rounds: val acc {rec['val_accuracy']:.4f}, test acc {rec['test_accuracy']:.4f}, "
                  f"predicted shares {rec['test_pred_shares']}")

    out = ARTIFACTS_DIR / "fi2010" / "diagnose_lightgbm_history.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n[+] {out}")


if __name__ == "__main__":
    main()
