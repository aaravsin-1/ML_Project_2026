"""Figures for the real-data results (reads only saved CSVs in the artifacts folder).

    python scripts/make_real_data_figures.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR
from src.utils.visualization import set_plotting_style

FIG = ARTIFACTS_DIR / "figures_real"


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print(f"[+] {FIG / name}")


def btc_models():
    a = pd.read_csv(ARTIFACTS_DIR / "btc" / "experiment_a_results.csv").sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = ["#c0392b" if "Na" in m else ("#e67e22" if m in ("LSTM", "DeepLOB", "TLOB") else "#2b5c8f") for m in a.model]
    ax.barh(a.model, a.macro_f1, xerr=a.macro_f1_std, color=colors, capsize=3)
    ax.axvline(1 / 3, color="grey", ls="--", lw=1, label="random guessing (1/3)")
    ax.set_xlabel("Test macro-F1 (mean ± std over 4 BTC segments)")
    ax.set_title("Real Coinbase BTC, k = 20 s: which model predicts direction best?")
    ax.legend(loc="lower right")
    save(fig, "btc_model_comparison.png")


def btc_profit():
    d = pd.read_csv(ARTIFACTS_DIR / "btc" / "experiment_d_execution_strategies.csv")
    d = d[d.model == "LightGBM"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    colors = ["#27ae60" if v > 0 else "#c0392b" for v in d.net_pnl]
    ax.bar(d.strategy, d.net_pnl, yerr=d.net_pnl_std, color=colors, capsize=4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Net P&L per ~10.7 h test window ($)")
    ax.set_title("LightGBM on BTC: the edge exists, fees remove it")
    for i, v in enumerate(d.net_pnl):
        label = f"+${v:,.0f}" if v > 0 else f"−${-v:,.0f}"
        ax.annotate(label, (i + 0.06, v / 2), ha="left", va="center", fontsize=9, fontweight="bold")
    save(fig, "btc_fantasy_vs_real_pnl.png")


def tradability():
    p = ARTIFACTS_DIR / "btc" / "tradability" / "tradability_summary.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for (label, k), g in s.groupby(["label", "horizon_k"], sort=False):
        g = g.sort_values("coverage", ascending=False)
        ax.plot(100 * g.coverage, g.mean_gross_bps_mean, marker="o", label=f"{label}, hold {k} s")
    ax.axhline(8, color="#c0392b", ls="--", lw=1.2, label="break-even at 4 bp taker fee (8 bp round trip)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Share of rows traded (%), most confident first  →  more selective")
    ax.set_ylabel("Mean gross edge per round trip (bp)")
    ax.set_title("BTC: being more selective raises the edge, but not to the fee level")
    ax.legend(fontsize=8)
    save(fig, "btc_tradability.png")


def fi2010():
    p = ARTIFACTS_DIR / "fi2010" / "fi2010_setup2_summary.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    models = ["Naive Momentum (lag k)", "Logistic Regression", "LightGBM (144 feats)",
              "Logistic Regression + history", "LightGBM (144 feats + history)", "DeepLOB (ours)", "DeepLOB (published)"]
    s = s[s.model.isin(models)]
    ks = sorted(s.horizon_k.unique())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        g = s[s.model == m].set_index("horizon_k").reindex(ks)
        ax.bar(np.arange(len(ks)) + i * w, g.weighted_f1_mean, w, yerr=g.weighted_f1_std.fillna(0), capsize=2, label=m)
    ax.set_xticks(np.arange(len(ks)) + w * (len(models) - 1) / 2)
    ax.set_xticklabels([f"k = {k}" for k in ks])
    ax.set_ylabel("Weighted F1 (paper's metric)")
    ax.set_ylim(0, 1)
    ax.set_title("FI-2010 Setup 2: with the same price history, simple models catch up with DeepLOB")
    ax.legend(fontsize=7, ncol=4, loc="upper center")
    save(fig, "fi2010_comparison.png")


def cross_asset():
    p = ARTIFACTS_DIR / "btc" / "cross_asset" / "cross_asset_summary.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar([f"k = {k} s" for k in s.horizon_k], s.delta_mean, yerr=s.delta_std, color="#8e44ad", capsize=4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Change in BTC test macro-F1 from adding ETH")
    ax.set_title("Does Ethereum's order book help predict Bitcoin?")
    for i, (d, n, t) in enumerate(zip(s.delta_mean, s.segments_improved, s.segments)):
        ax.annotate(f"{d:+.3f}\n({n}/{t} segments)", (i, d), ha="center", va="bottom" if d > 0 else "top", fontsize=8)
    save(fig, "btc_eth_cross_asset.png")


if __name__ == "__main__":
    set_plotting_style()
    for fn in (btc_models, btc_profit, tradability, fi2010, cross_asset):
        fn()
