"""Master Experiment Orchestrator and Artifact Generator.

Executes the full research agenda:
1. Data Generation & Ingestion
2. Experiment A: Model Hierarchy Validation (Tier 0 to Tier 5)
3. Experiment B: Horizon of Signal Survival (Alpha Decay Curve)
4. Experiment C: Ablation & Feature Attribution (SHAP / GBDT)
5. Experiment D: High-Fidelity Execution Simulation & Latency Profitability
6. C++ Low-Latency Benchmark Execution
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
import numpy as np

from src.config import ARTIFACTS_DIR, DATA_DIR, PROJECT_ROOT
from src.data.synthetic_generator import SyntheticLOBGenerator
from src.experiments.exp_a_predictive_power import run_experiment_a
from src.experiments.exp_b_signal_survival import run_experiment_b
from src.experiments.exp_c_ablation_attribution import run_experiment_c
from src.experiments.exp_d_latency_vs_profit import run_experiment_d


def main():
    parser = argparse.ArgumentParser(description="Master Research Pipeline Orchestrator")
    parser.add_argument("--events", type=int, default=30000, help="Number of LOB events to simulate")
    parser.add_argument("--epochs", type=int, default=8, help="Epochs for deep neural nets (DeepLOB, TLOB)")
    parser.add_argument("--horizon", type=int, default=20, help="Target horizon k in events")
    parser.add_argument("--skip_cpp", action="store_true", help="Skip C++ benchmark execution")
    args = parser.parse_args()

    start_time = time.time()
    print("================================================================================")
    print("      HIGH-FREQUENCY LIMIT ORDER BOOK ALPHA & EXECUTION RESEARCH SUITE          ")
    print("================================================================================")
    print(f"Configurations: Events={args.events} | Epochs={args.epochs} | Horizon={args.horizon}")

    # 1. Dataset Generation
    print("\n>>> STEP 1: GENERATING HIGH-FREQUENCY LIMIT ORDER BOOK DATASET")
    gen = SyntheticLOBGenerator(base_price=100.0, tick_size=0.01, regime="large_tick", seed=42)
    lob_40, labels_dict, trades_df = gen.generate_session(n_events=args.events)
    target_labels = labels_dict[args.horizon]
    print(f"[+] Successfully prepared {len(lob_40)} snapshots and {len(trades_df)} trade events.")

    # 2. Experiment A
    print("\n>>> STEP 2: RUNNING EXPERIMENT A (MODEL PROGRESSION HIERARCHY)")
    df_exp_a = run_experiment_a(
        lob_40=lob_40,
        labels=target_labels,
        trades_df=trades_df,
        horizon_k=args.horizon,
        epochs=args.epochs,
        save_artifacts=True
    )

    # 3. Experiment B
    print("\n>>> STEP 3: RUNNING EXPERIMENT B (THE HORIZON OF SIGNAL SURVIVAL)")
    df_exp_b = run_experiment_b(
        lob_40=lob_40,
        labels_dict=labels_dict,
        trades_df=trades_df,
        horizons=[10, 20, 50, 100],
        save_artifacts=True
    )

    # 4. Experiment C
    print("\n>>> STEP 4: RUNNING EXPERIMENT C (FEATURE AND DEPTH ABLATION)")
    df_exp_c, df_importance = run_experiment_c(
        lob_40=lob_40,
        labels=target_labels,
        trades_df=trades_df,
        horizon_k=args.horizon,
        save_artifacts=True
    )

    # 5. Experiment D
    print("\n>>> STEP 5: RUNNING EXPERIMENT D (HIGH-FIDELITY EXECUTION & LATENCY ANALYSIS)")
    df_strat, df_latency = run_experiment_d(
        lob_40=lob_40,
        labels=target_labels,
        trades_df=trades_df,
        horizon_k=args.horizon,
        save_artifacts=True
    )

    # 6. C++ Benchmark
    if not args.skip_cpp:
        print("\n>>> STEP 6: EXECUTING LOW-LATENCY C++ SIMD BENCHMARK ENGINE")
        exe_path = PROJECT_ROOT / "cpp" / "lob_benchmark.exe"
        if not exe_path.exists():
            print("[+] Compiling C++ benchmark engine with g++...")
            compile_cmd = [
                "g++", "-std=c++20", "-O3", "-march=native", "-ffast-math",
                "-I", str(PROJECT_ROOT / "cpp" / "include"),
                str(PROJECT_ROOT / "cpp" / "src" / "main_benchmark.cpp"),
                "-o", str(exe_path)
            ]
            subprocess.run(compile_cmd, check=True)

        print("[+] Running compiled native executable...")
        res = subprocess.run([str(exe_path)], capture_output=True, text=True)
        print(res.stdout)

    elapsed = time.time() - start_time
    print("\n================================================================================")
    print(f"       ALL EXPERIMENTS COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS          ")
    print(f"       Artifacts and plots stored in: {ARTIFACTS_DIR}                          ")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
