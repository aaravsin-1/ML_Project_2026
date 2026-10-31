"""Automatic checks on Experiment K's results. Exits with status 1 if any check fails.

    python scripts/check_microprice_results.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS_DIR  # noqa: E402
from src.experiments.exp_k_microprice import BTC_HORIZONS, LOBSTER_HORIZONS  # noqa: E402

OUT = ARTIFACTS_DIR / "microprice"
failures, notes = [], []


def check(ok: bool, msg: str):
    (notes if ok else failures).append(("PASS " if ok else "FAIL ") + msg)


rows = pd.read_csv(OUT / "microprice_all.csv")
betas = pd.read_csv(OUT / "microprice_beta.csv")

# 1. coverage: every segment / stock and horizon is present
btc = rows[rows.dataset == "BTC"]
lob = rows[rows.dataset.str.startswith("LOBSTER")]
check(btc.groupby(["name", "horizon"]).ngroups == 4 * len(BTC_HORIZONS),
      f"BTC: {btc.groupby(['name', 'horizon']).ngroups} segment-horizons (expected {4 * len(BTC_HORIZONS)})")
check(lob.groupby(["name", "horizon"]).ngroups == 5 * len(LOBSTER_HORIZONS),
      f"LOBSTER: {lob.groupby(['name', 'horizon']).ngroups} stock-horizons (expected {5 * len(LOBSTER_HORIZONS)})")
check(len(betas) == 4 * len(BTC_HORIZONS) + 5 * len(LOBSTER_HORIZONS), f"beta rows: {len(betas)}")
models = set(rows.model)
check(models == {"Zero forecast", "Micro-price rule (beta x gap)", "Momentum (past k-step return)", "Ridge", "LightGBM",
                 "LightGBM, shuffled-target control"},
      f"model set: {sorted(models)}")

# 2. no missing or impossible values
real = rows[rows.model != "Zero forecast"]
for col in ("ic_pearson", "r2", "hit_rate"):
    check(real[col].notna().all(), f"no NaN in {col} for non-zero models")
check(((rows.hit_rate >= 0) & (rows.hit_rate <= 1)).all(), "hit rates within [0, 1]")
check((rows.n >= 1000).all(), f"every evaluation has at least 1000 test rows (min {rows.n.min()})")
check(np.allclose(rows.loc[rows.model == "Zero forecast", "r2"], 0.0), "zero forecast has R^2 exactly 0")
check(np.isfinite(betas[["beta_train", "beta_test"]].to_numpy()).all(), "all betas finite")
ci = rows.dropna(subset=["ic_diff_micro_minus_mid_ci_low"])
check(len(ci) == 4 * len(BTC_HORIZONS) + 5 * len(LOBSTER_HORIZONS), f"one bootstrap CI per unit-horizon ({len(ci)})")
check((ci.ic_diff_micro_minus_mid_ci_low <= ci.ic_diff_micro_minus_mid_ci_high).all(), "bootstrap CIs ordered")

# 3. the shuffled-target control must be centred on zero and well below the real model
ctrl = rows[rows.model == "LightGBM, shuffled-target control"].set_index(["dataset", "name", "horizon"])
lgbm = rows[(rows.model == "LightGBM") & (rows.train_target == "mid")].set_index(["dataset", "name", "horizon"])
pooled = ctrl.ic_pearson.mean()
check(abs(pooled) < 0.02, f"control IC pooled over all {len(ctrl)} cases is ~0 ({pooled:+.4f})")
for ds, g in ctrl.groupby(level="dataset"):
    check(abs(g.ic_pearson.mean()) < 0.03, f"control IC mean for {ds}: {g.ic_pearson.mean():+.4f}")
joined = lgbm[["ic_pearson"]].join(ctrl[["ic_pearson"]].rename(columns={"ic_pearson": "ic_pearson_ctrl"}))
strong = joined[joined.ic_pearson > 0.05]
beaten = (strong.ic_pearson_ctrl.abs() >= strong.ic_pearson).sum()
check(beaten == 0, f"control never matches a real model with IC > 0.05 ({beaten} of {len(strong)} cases)")
check(ctrl.r2.mean() <= 0.005, f"control R^2 averages <= 0 ({ctrl.r2.mean():+.4f})")

print("\n".join(notes + failures))
print(f"\n{len(notes)} passed, {len(failures)} failed")
sys.exit(1 if failures else 0)
