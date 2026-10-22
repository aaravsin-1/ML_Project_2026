"""Publication-Quality Visualizations for LOB Alpha Research."""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_plotting_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["figure.dpi"] = 300


def plot_model_comparison(df_results: pd.DataFrame, save_path: Path):
    """Bar plot comparing model tiers across Macro-F1 and Accuracy."""
    set_plotting_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(df_results))
    width = 0.35

    rects1 = ax.bar(x - width/2, df_results["macro_f1"], width, label="Macro F1-Score", color="#2b5c8f")
    rects2 = ax.bar(x + width/2, df_results["accuracy"], width, label="Accuracy", color="#d95f02")

    ax.set_ylabel("Score")
    ax.set_title("Experiment A: Model Hierarchy Performance Progression (Tier 0 to Tier 5)")
    ax.set_xticks(x)
    ax.set_xticklabels(df_results["model"], rotation=25, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)

    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_signal_decay(horizons: List[int], f1_scores: List[float], save_path: Path):
    """Line plot showing signal decay curve across prediction horizons."""
    set_plotting_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(horizons, f1_scores, marker="o", linewidth=2.5, markersize=8, color="#c0392b", label="Optimal Model (Macro-F1)")
    ax.axhline(y=1.0/3.0, color="#7f8c8d", linestyle="--", label="Random Baseline (1/3)")

    ax.set_title("Experiment B: The Horizon of Signal Survival (Alpha Decay Curve)")
    ax.set_xlabel("Prediction Horizon (Events / Ticks)")
    ax.set_ylabel("Out-of-Sample Macro F1-Score")
    ax.set_xticks(horizons)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    for h, f1 in zip(horizons, f1_scores):
        ax.annotate(f"{f1:.3f}", (h, f1), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_ablation_progression(ablation_results: pd.DataFrame, save_path: Path):
    """Bar plot showing predictive lift across restricted feature sets."""
    set_plotting_style()
    fig, ax = plt.subplots(figsize=(9, 4.5))

    colors = ["#95a5a6", "#3498db", "#9b59b6", "#e67e22", "#27ae60"]
    bars = ax.bar(ablation_results["model_tier"], ablation_results["macro_f1"], color=colors, width=0.55)

    ax.set_ylabel("Macro F1-Score")
    ax.set_title("Experiment C: Feature and Depth Ablation Study (Information Isolation)")
    ax.set_ylim(0, max(ablation_results["macro_f1"]) * 1.25)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_execution_equity_curves(equity_dict: Dict[str, np.ndarray], save_path: Path):
    """Equity curve trajectories comparing Academic Mid-Price vs Realistic Execution."""
    set_plotting_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    styles = {
        "Academic Mid-Price (Fantasy)": ("#27ae60", "--", 1.8),
        "Realistic Aggressive Taker (Friction)": ("#c0392b", "-", 2.0),
        "Dynamic Passive Maker (VPIN Defense)": ("#2980b9", "-", 2.0),
    }

    for name, equity in equity_dict.items():
        color, lstyle, lwidth = styles.get(name, ("#34495e", "-", 1.5))
        ax.plot(equity, label=name, color=color, linestyle=lstyle, linewidth=lwidth)

    ax.set_title("Experiment D: Economic Reality - Realized Equity Under Microstructure Friction")
    ax.set_xlabel("Out-of-Sample Trading Events")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_latency_vs_profitability(
    latencies_ms: List[float],
    sharpe_ratios: Dict[str, List[float]],
    save_path: Path
):
    """Plots Sharpe Ratio degradation as latency increases (Speed vs Mathematical Accuracy)."""
    set_plotting_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    ax.plot(latencies_ms, sharpe_ratios["LightGBM (C++ Engine)"], marker="s", linewidth=2.2, color="#2980b9", label="LightGBM (C++ Engine)")
    ax.plot(latencies_ms, sharpe_ratios["DeepLOB (GPU Neural Net)"], marker="o", linewidth=2.2, color="#e67e22", label="DeepLOB (Deep CNN-LSTM)")

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xscale("log")
    ax.set_title("Economic Viability: Latency Decay vs Net Sharpe Ratio")
    ax.set_xlabel("System Latency (milliseconds, log-scale)")
    ax.set_ylabel("Annualized Net Sharpe Ratio")
    ax.legend()
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
