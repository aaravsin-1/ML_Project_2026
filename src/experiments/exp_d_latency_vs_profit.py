"""Experiment D: Economic Reality - Execution Frictions and Latency vs Profitability.

1. Runs the real LightGBM and DeepLOB test-set signals (from Experiment A) through three
   execution regimes: fantasy mid-price fills, an aggressive taker that crosses the spread
   and walks the book, and a passive maker that joins the touch, fills only under a
   conservative trade-through rule, and stands down when VPIN is high.
2. Sweeps added latency (in wall-clock ms against snapshot timestamps) for both models'
   own signals, and measures each model's single-sample inference latency.

Lot size, fees and the latency grid come from the session (synthetic vs Coinbase BTC).
Sharpe is computed on 100 ms bars and annualized; short test windows make it noisy, so
compare it across regimes and models rather than reading it as an absolute number.
"""

import time
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import torch

from src.config import ARTIFACTS_DIR
from src.experiments.common import PreparedData, SEQ_LEN, load_prepared
from src.execution.simulator import ExecutionSimulator
from src.execution.strategies import MicrostructureStrategyRunner
from src.execution.metrics import PerformanceMetrics
from src.utils.visualization import plot_execution_equity_curves, plot_latency_vs_profitability

BAR_MS = 100.0
VPIN_TOXIC_QUANTILE = 0.90


def measure_inference_latency_ms(exp_a_outputs: Dict[str, Any], data: PreparedData, n_calls: int = 300) -> Dict[str, float]:
    """Median single-sample inference latency of the trained Python models."""
    test_idx = exp_a_outputs["test_idx"]
    # Call the raw booster: the sklearn wrapper adds input validation (and a warning per
    # call) that is not part of model inference.
    lgb_booster = exp_a_outputs["lgb_model"].model.booster_
    X = data.tab_feat
    rows = test_idx[:n_calls]

    t_lgb = []
    for i in rows:
        t0 = time.perf_counter()
        lgb_booster.predict(X[i:i + 1], num_iteration=exp_a_outputs["lgb_model"].model.best_iteration_)
        t_lgb.append(time.perf_counter() - t0)

    trainer = exp_a_outputs["deeplob_trainer"]
    model, device = trainer.model, trainer.device
    model.eval()
    deep = torch.tensor(data.deep_feat, dtype=torch.float32, device=device)
    t_dl = []
    with torch.no_grad():
        for i in rows:
            x = deep[i - SEQ_LEN + 1:i + 1].unsqueeze(0).unsqueeze(0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x).argmax(dim=-1).item()  # .item() forces the result back to the host
            t_dl.append(time.perf_counter() - t0)

    return {"lightgbm_python_ms": 1000 * float(np.median(t_lgb)), "deeplob_ms": 1000 * float(np.median(t_dl)),
            "deeplob_device": device.type}


def run_experiment_d(
    data: PreparedData,
    exp_a_outputs: Dict[str, Any],
    save_artifacts: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Executes Experiment D: Execution Simulation & Latency Sweep."""
    print("\n=======================================================")
    print("   EXPERIMENT D: ECONOMIC REALITY & LATENCY PROFIT     ")
    print("=======================================================")
    seed = data.session.seed
    test_idx = exp_a_outputs["test_idx"]
    test_lob = data.session.lob_40[test_idx]
    test_ts = data.session.timestamps_ms[test_idx]
    test_mid = (test_lob[:, 0] + test_lob[:, 2]) / 2.0

    vpin = data.tab_feat[:, data.feat_names.index("vpin")]
    vpin_thresh = float(np.quantile(vpin[:data.train_end], VPIN_TOXIC_QUANTILE))
    test_vpin = vpin[test_idx]
    print(f"[+] Test window: {len(test_idx)} events, {(test_ts[-1] - test_ts[0]) / 1000:.1f} s | VPIN toxic threshold (train q{VPIN_TOXIC_QUANTILE:.2f}) = {vpin_thresh:.3f}")

    signals = {"LightGBM": exp_a_outputs["lgb_preds"], "DeepLOB": exp_a_outputs["deeplob_preds"]}
    ex = data.session.exec_params

    def make_sim(latency_ms: float) -> ExecutionSimulator:
        return ExecutionSimulator(
            initial_cash=ex["initial_cash"], maker_fee_bps=ex["maker_fee_bps"], taker_fee_bps=ex["taker_fee_bps"],
            lot_size=ex["lot_size"], max_inventory=ex["max_inventory"], latency_ms=latency_ms, timestamps_ms=test_ts
        )

    def metrics(res: Dict[str, Any]) -> Dict[str, float]:
        return PerformanceMetrics.calculate_metrics(res["equity"], ex["initial_cash"], timestamps_ms=test_ts, bar_ms=BAR_MS)

    # Part 1: execution regimes at the session's base latency
    strat_rows = []
    equity_curves = {}
    for model_name, sig in signals.items():
        runner = MicrostructureStrategyRunner(make_sim(ex["base_latency_ms"]))
        regimes = {
            "Academic Mid-Price (Fantasy)": runner.run_academic_midprice(sig, test_mid),
            "Aggressive Taker": runner.run_realistic_aggressive(sig, test_lob),
            "Passive Maker (VPIN gate)": runner.run_dynamic_passive_vpin(sig, test_lob, test_vpin, vpin_thresh),
        }
        for regime, res in regimes.items():
            perf = metrics(res)
            strat_rows.append({
                "seed": seed, "model": model_name, "strategy": regime,
                "net_pnl": perf["net_pnl"], "sharpe_ratio": perf["sharpe_ratio"],
                "max_drawdown_pct": perf["max_drawdown_pct"], "num_trades": res["num_trades"],
                "pnl_per_trade": perf["net_pnl"] / max(res["num_trades"], 1),
                "total_fees": res.get("total_fees", 0.0), "slippage": res.get("total_slippage", 0.0),
                "n_bars": perf["n_bars"],
            })
            if model_name == "LightGBM":
                equity_curves[regime] = res["equity"]
    df_strat = pd.DataFrame(strat_rows)
    print(df_strat[["model", "strategy", "net_pnl", "num_trades", "sharpe_ratio"]].to_string(index=False))

    # Part 2: added-latency sweep, aggressive taker, each model with its own signals
    lat_rows = []
    for lat in ex["latencies_ms"]:
        row = {"seed": seed, "latency_ms": lat}
        for model_name, sig in signals.items():
            res = MicrostructureStrategyRunner(make_sim(lat)).run_realistic_aggressive(sig, test_lob)
            perf = metrics(res)
            row[f"pnl_{model_name}"] = perf["net_pnl"]
            row[f"sharpe_{model_name}"] = perf["sharpe_ratio"]
        lat_rows.append(row)
    df_latency = pd.DataFrame(lat_rows)
    print(df_latency.to_string(index=False))

    inference = measure_inference_latency_ms(exp_a_outputs, data)
    inference["seed"] = seed
    print(f"[+] Measured single-sample inference latency: {inference}")

    if save_artifacts:
        df_strat.to_csv(ARTIFACTS_DIR / "experiment_d_execution_strategies.csv", index=False)
        df_latency.to_csv(ARTIFACTS_DIR / "experiment_d_latency_sharpe.csv", index=False)
        plot_execution_equity_curves(equity_curves, ARTIFACTS_DIR / "experiment_d_equity_curves.png")
        plot_latency_vs_profitability(
            ex["latencies_ms"],
            {"LightGBM": df_latency["sharpe_LightGBM"].tolist(), "DeepLOB": df_latency["sharpe_DeepLOB"].tolist()},
            ARTIFACTS_DIR / "experiment_d_latency_vs_profit.png",
        )

    return df_strat, df_latency, inference


if __name__ == "__main__":
    from src.experiments.exp_a_predictive_power import run_experiment_a
    prepared = load_prepared()
    _, outputs = run_experiment_a(prepared)
    run_experiment_d(prepared, outputs)
