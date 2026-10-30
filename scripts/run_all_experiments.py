"""Master Experiment Orchestrator and Artifact Generator.

Runs Experiments A-D on several independent market sessions and reports mean +/- std:
- --source synthetic: one generated session per seed (artifacts/)
- --source btc: contiguous segments of real Coinbase BTC-USD 1-second data (artifacts/btc/)
Experiments:
1. Experiment A: Model Hierarchy Validation (Tier 0 to Tier 5)
2. Experiment B: Horizon of Signal Survival (Alpha Decay Curve)
3. Experiment C: Ablation & Feature Attribution (LightGBM split gain)
4. Experiment D: Execution Simulation & Latency vs Profitability
5. (optional) C++ Low-Latency Benchmark Execution

Per-seed rows go to artifacts/*_per_seed.csv; aggregated tables and plots to the
usual artifact names.
"""

import argparse
import subprocess
import time
import sys
from pathlib import Path

# Ensure root directory is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import ARTIFACTS_DIR, PROJECT_ROOT
from src.experiments.common import generate_session, btc_sessions, prepare_data
from src.experiments.exp_a_predictive_power import run_experiment_a
from src.experiments.exp_b_signal_survival import run_experiment_b
from src.experiments.exp_c_ablation_attribution import run_experiment_c
from src.experiments.exp_d_latency_vs_profit import run_experiment_d
from src.utils.visualization import (
    plot_model_comparison, plot_signal_decay, plot_ablation_progression, plot_latency_vs_profitability
)


def aggregate(df: pd.DataFrame, keys, value_cols) -> pd.DataFrame:
    """Mean and std across seeds for each group, preserving first-seen group order."""
    grouped = df.groupby(keys, sort=False)[value_cols]
    mean = grouped.mean()
    std = grouped.std(ddof=1).add_suffix("_std")
    out = pd.concat([mean, std], axis=1).reset_index()
    out.insert(len(keys) if isinstance(keys, list) else 1, "n_seeds", grouped.size().values)
    return out


def main():
    parser = argparse.ArgumentParser(description="Master Research Pipeline Orchestrator")
    parser.add_argument("--source", choices=["synthetic", "btc"], default="synthetic")
    parser.add_argument("--segments", type=int, default=4, help="BTC only: number of contiguous segments")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Generator seeds (one session each)")
    parser.add_argument("--events", type=int, default=60000, help="Number of LOB events per session")
    parser.add_argument("--epochs", type=int, default=8, help="Max epochs for deep nets (early stopping on val loss)")
    parser.add_argument("--horizon", type=int, default=20, help="Target horizon k in events")
    parser.add_argument("--run-cpp", action="store_true", help="Also compile and run the C++ latency benchmark")
    parser.add_argument("--fresh", action="store_true", help="Ignore per-session checkpoints and recompute")
    args = parser.parse_args()

    start_time = time.time()
    out_dir = ARTIFACTS_DIR if args.source == "synthetic" else ARTIFACTS_DIR / args.source
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.source == "synthetic":
        print(f"Configuration: synthetic seeds={args.seeds} | events={args.events} | epochs={args.epochs} | horizon={args.horizon}")
        session_iter = (generate_session(seed, args.events) for seed in args.seeds)
    else:
        print(f"Configuration: BTC segments={args.segments} | epochs={args.epochs} | horizon={args.horizon}")
        session_iter = iter(btc_sessions(args.segments))

    rows_a, rows_b, rows_c, rows_imp, rows_d, rows_lat, rows_inf = [], [], [], [], [], [], []
    latencies_ms = None
    # Each finished session is checkpointed, so an interrupted run resumes where it stopped
    ckpt_dir = out_dir / "sessions"
    ckpt_dir.mkdir(exist_ok=True)
    for session in session_iter:
        latencies_ms = session.exec_params["latencies_ms"]
        ckpt = ckpt_dir / f"{session.name}_k{args.horizon}_e{args.epochs}.pkl"
        if ckpt.exists() and not args.fresh:
            print(f"[+] {session.name}: loading checkpoint {ckpt.name}")
            res = pd.read_pickle(ckpt)
        else:
            print(f"\n################ {session.name} (seed {session.seed}, {len(session.lob_40):,} rows) ################")
            data = prepare_data(session)
            df_a, outputs = run_experiment_a(data, horizon_k=args.horizon, epochs=args.epochs, save_artifacts=False)
            df_b = run_experiment_b(data, save_artifacts=False)
            df_c, df_imp = run_experiment_c(data, horizon_k=args.horizon, save_artifacts=False)
            df_d, df_lat, inference = run_experiment_d(data, outputs, save_artifacts=False)
            res = {"a": df_a, "b": df_b, "c": df_c, "imp": df_imp, "d": df_d, "lat": df_lat, "inf": inference}
            pd.to_pickle(res, ckpt)
        rows_a.append(res["a"]); rows_b.append(res["b"]); rows_c.append(res["c"]); rows_imp.append(res["imp"])
        rows_d.append(res["d"]); rows_lat.append(res["lat"]); rows_inf.append(res["inf"])

    a = pd.concat(rows_a); b = pd.concat(rows_b); c = pd.concat(rows_c); imp = pd.concat(rows_imp)
    d = pd.concat(rows_d); lat = pd.concat(rows_lat); inf = pd.DataFrame(rows_inf)
    for name, df in [("experiment_a", a), ("experiment_b", b), ("experiment_c_ablation", c),
                     ("experiment_c_importance", imp), ("experiment_d_strategies", d),
                     ("experiment_d_latency", lat), ("experiment_d_inference_latency", inf)]:
        df.to_csv(out_dir / f"{name}_per_seed.csv", index=False)

    metric_cols = ["accuracy", "macro_f1", "weighted_f1", "f1_flat", "f1_up", "f1_down", "cross_entropy"]
    agg_a = aggregate(a, ["tier", "model"], metric_cols)
    agg_b = aggregate(b, "horizon_k", metric_cols + ["momentum_macro_f1", "test_frac_flat"])
    agg_c = aggregate(c, ["model_tier", "num_features"], metric_cols)
    agg_imp = aggregate(imp, "feature", ["gain_share_pct"]).sort_values("gain_share_pct", ascending=False)
    agg_d = aggregate(d, ["model", "strategy"], ["net_pnl", "pnl_per_trade", "num_trades", "sharpe_ratio",
                                                  "max_drawdown_pct", "total_fees", "slippage"])
    lat_cols = [col for col in lat.columns if col not in ("seed", "latency_ms")]
    agg_lat = aggregate(lat, "latency_ms", lat_cols)
    agg_inf = inf[["lightgbm_python_ms", "deeplob_ms"]].agg(["median", "mean", "std"])

    agg_a.to_csv(out_dir / "experiment_a_results.csv", index=False)
    agg_b.to_csv(out_dir / "experiment_b_signal_decay.csv", index=False)
    agg_c.to_csv(out_dir / "experiment_c_ablation_results.csv", index=False)
    agg_imp.to_csv(out_dir / "experiment_c_feature_importance.csv", index=False)
    agg_d.to_csv(out_dir / "experiment_d_execution_strategies.csv", index=False)
    agg_lat.to_csv(out_dir / "experiment_d_latency_sharpe.csv", index=False)
    agg_inf.to_csv(out_dir / "experiment_d_inference_latency.csv")

    plot_model_comparison(agg_a, out_dir / "experiment_a_model_comparison.png")
    plot_signal_decay(agg_b["horizon_k"].tolist(), agg_b["macro_f1"].tolist(), out_dir / "experiment_b_signal_decay.png",
                      f1_std=agg_b["macro_f1_std"].tolist(), baseline_f1=agg_b["momentum_macro_f1"].tolist())
    plot_ablation_progression(agg_c, out_dir / "experiment_c_ablation_progression.png")
    plot_latency_vs_profitability(
        latencies_ms,
        {m: agg_lat[f"pnl_{m}"].tolist() for m in ("LightGBM", "DeepLOB")},
        out_dir / "experiment_d_latency_vs_profit.png",
        stds={m: agg_lat[f"pnl_{m}_std"].tolist() for m in ("LightGBM", "DeepLOB")},
        ylabel="Net PnL ($), mean ± std across seeds",
    )

    pd.set_option("display.width", 200)
    print("\n================ AGGREGATED RESULTS (mean across seeds) ================")
    print(agg_a[["model", "accuracy", "macro_f1", "macro_f1_std", "cross_entropy"]].to_string(index=False))
    print(agg_b[["horizon_k", "macro_f1", "macro_f1_std", "momentum_macro_f1", "accuracy", "test_frac_flat"]].to_string(index=False))
    print(agg_c[["model_tier", "num_features", "macro_f1", "macro_f1_std"]].to_string(index=False))
    print(agg_imp.head(10).to_string(index=False))
    print(agg_d[["model", "strategy", "net_pnl", "net_pnl_std", "num_trades", "pnl_per_trade", "sharpe_ratio", "sharpe_ratio_std"]].to_string(index=False))
    print(agg_lat[["latency_ms", "pnl_LightGBM", "pnl_LightGBM_std", "pnl_DeepLOB", "pnl_DeepLOB_std",
                   "sharpe_LightGBM", "sharpe_DeepLOB"]].to_string(index=False))
    print(agg_inf.to_string())

    if args.run_cpp:
        print("\n>>> C++ LATENCY BENCHMARK")
        exe_path = PROJECT_ROOT / "cpp" / "lob_benchmark.exe"
        if not exe_path.exists():
            subprocess.run([
                "g++", "-std=c++20", "-O3", "-march=native", "-ffast-math",
                "-I", str(PROJECT_ROOT / "cpp" / "include"),
                str(PROJECT_ROOT / "cpp" / "src" / "main_benchmark.cpp"),
                "-o", str(exe_path)
            ], check=True)
        res = subprocess.run([str(exe_path)], capture_output=True, text=True, cwd=PROJECT_ROOT)
        print(res.stdout)

    print(f"\nAll experiments completed in {time.time() - start_time:.1f} s. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
