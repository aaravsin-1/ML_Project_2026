"""Regenerate experiment_d_equity_curves.png with the current (fixed) simulator.
Same LightGBM settings as Experiment A, seed 0, 60k events, k=20 (the README setup)."""
import sys
import numpy as np
sys.path.insert(0, ".")
from src.config import ARTIFACTS_DIR
from src.experiments.common import prepare_data
from src.experiments.common import generate_session
from src.models.tabular_trees import LightGBMLOB
from src.execution.simulator import ExecutionSimulator
from src.execution.strategies import MicrostructureStrategyRunner
from src.utils.visualization import plot_execution_equity_curves

out = sys.argv[1] if len(sys.argv) > 1 else str(ARTIFACTS_DIR / "experiment_d_equity_curves.png")
data = prepare_data(generate_session(0, 60000))
k = 20
y, idx, X = data.labels(k), data.splits(k), data.tab_feat
m = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=200, random_state=0)
m.fit(X[idx["train"]], y[idx["train"]], X[idx["val"]], y[idx["val"]], early_stopping_rounds=15)
sig = m.predict(X[idx["test"]])

t = idx["test"]; lob = data.session.lob_40[t]; ts = data.session.timestamps_ms[t]
mid = (lob[:, 0] + lob[:, 2]) / 2
vpin = X[:, data.feat_names.index("vpin")]
thr = float(np.quantile(vpin[:data.train_end], 0.90))
ex = data.session.exec_params
def runner():
    return MicrostructureStrategyRunner(ExecutionSimulator(
        initial_cash=ex["initial_cash"], maker_fee_bps=ex["maker_fee_bps"], taker_fee_bps=ex["taker_fee_bps"],
        lot_size=ex["lot_size"], max_inventory=ex["max_inventory"], latency_ms=ex["base_latency_ms"], timestamps_ms=ts))
curves = {
    "Academic Mid-Price (Fantasy)": runner().run_academic_midprice(sig, mid),
    "Aggressive Taker": runner().run_realistic_aggressive(sig, lob),
    "Passive Maker (VPIN gate)": runner().run_dynamic_passive_vpin(sig, lob, vpin[t], thr),
}
for n, r in curves.items():
    print(f"{n:32s} net {r['equity'][-1] - ex['initial_cash']:10.1f}  trades {r['num_trades']}")
plot_execution_equity_curves({n: r["equity"] for n, r in curves.items()}, out)
print("saved", out)
