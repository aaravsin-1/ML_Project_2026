"""Generates publication-grade figures for the Indian NSE Limit Order Book experiments.

Outputs both PNG (300 dpi) and PDF vector figures in `artifacts/nse/figures/`:
  1. nse_eda_tick_spectrum.png/.pdf       — Monotone tick-size vs L1-imbalance predictability across 8 NSE instruments
  2. nse_model_ladder.png/.pdf            — Tier 0-5 model ladder on RELIANCE, NIFTY_FUT, IDEA (20 s, 4 segments)
  3. nse_signal_decay.png/.pdf            — Signal decay across 5-100 s for all 8 NSE instruments (large vs small tick)
  4. nse_ablation_shap.png/.pdf           — Feature ablation (including NSE order counts & full book) + TreeSHAP
  5. nse_edge_decomposition.png/.pdf      — Round-trip edge waterfall & break-even hit rate across Indian fee tiers
  6. nse_cross_asset_microprice.png/.pdf  — Cross-asset lead-lag (NIFTY_FUT -> equities) + Micro-price rule vs LightGBM

    python scripts/make_nse_figures.py
"""

from __future__ import annotations

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

NSE_DIR = ARTIFACTS_DIR / "nse"
FIG_DIR = NSE_DIR / "figures"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_eda_tick_spectrum() -> None:
    p = NSE_DIR / "nse_dataset_summary.csv"
    if not p.exists():
        return
    df = pd.read_csv(p).sort_values("tick_bp", ascending=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    colors = ["#C8453C" if t < 0.5 else ("#2F6FA8" if t < 1.5 else "#0E7C86") for t in df["tick_bp"]]
    y = np.arange(len(df))
    ax1.barh(y, df["corr_imb1_20s"], color=colors, edgecolor="none")
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{r.symbol} ({r.tick_bp:.2f} bp)" for r in df.itertuples()])
    ax1.set_xlabel("Corr(Level-1 Queue Imbalance, Next 20 s Mid Return)")
    ax1.set_title("Monotone Tick-Size Law Across 64 NSE Trading Days")
    ax1.grid(axis="x", alpha=0.3, linestyle="--")
    for i, v in enumerate(df["corr_imb1_20s"]):
        ax1.text(v + 0.004, i, f"+{v:.3f}", va="center", fontsize=8.5)
    ax1.set_xlim(0, df["corr_imb1_20s"].max() * 1.18)

    ax2.scatter(df["tick_bp"], df["corr_imb1_20s"], s=80, c=colors, zorder=4)
    offsets = {"RELIANCE": (6, -10), "HDFCBANK": (6, 2), "ICICIBANK": (6, 4)}
    for r in df.itertuples():
        xy = offsets.get(r.symbol, (6, 3))
        ax2.annotate(r.symbol, (r.tick_bp, r.corr_imb1_20s), textcoords="offset points", xytext=xy, fontsize=8.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("Tick Size as Share of Price (bp, log scale)")
    ax2.set_ylabel("Corr(L1 Imbalance, Next 20 s Return)")
    ax2.set_title("Larger Relative Tick = Deeper Queues = Stronger Signal")
    ax2.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    _save(fig, "nse_eda_tick_spectrum")


def plot_model_ladder() -> None:
    syms = ["RELIANCE", "NIFTY_FUT", "IDEA"]
    avail = [s for s in syms if (NSE_DIR / "models" / s / "model_ladder_summary.csv").exists()]
    if not avail:
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(4.5 * len(avail), 4.4), sharey=False)
    if len(avail) == 1:
        axes = [axes]

    for ax, sym in zip(axes, avail):
        df_all = pd.read_csv(NSE_DIR / "models" / sym / "model_ladder_summary.csv")
        flat_f1 = float(df_all.loc[df_all["model"] == "Naive Flat", "macro_f1"].iloc[0])
        df = df_all[df_all["model"] != "Naive Flat"].sort_values("macro_f1", ascending=True)
        colors = []
        for m in df["model"]:
            if "LightGBM" in m:
                colors.append("#0E7C86")
            elif m in ("Random Forest", "Logistic Regression"):
                colors.append("#2F6FA8")
            elif m == "Naive Momentum":
                colors.append("#6E7B8B")
            else:
                colors.append("#C8453C")
        y = np.arange(len(df))
        ax.barh(y, df["macro_f1"], xerr=df["macro_f1_std"], color=colors, capsize=3, error_kw={"lw": 1})
        ax.axvline(flat_f1, color="#555", linestyle=":", lw=1, label=f"Naive Flat = {flat_f1:.3f}")
        ax.set_yticks(y)
        ax.set_yticklabels(df["model"])
        ax.set_xlabel("Test Macro-F1 (mean ± std, 4 segments)")
        ax.set_title(f"{sym} (k = 20 s; Flat={flat_f1:.3f})")
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        for i, (v, sd) in enumerate(zip(df["macro_f1"], df["macro_f1_std"])):
            ax.text(v + sd + 0.008, i, f"{v:.3f}", va="center", fontsize=8.5)
        ax.set_xlim(0.14, max(0.62, float((df["macro_f1"] + df["macro_f1_std"]).max()) + 0.065))

    fig.suptitle("Indian NSE Tier 0–5 Model Ladder (64 Trading Days, 4 Contiguous Segments)", y=1.02, fontweight="bold")
    fig.tight_layout()
    _save(fig, "nse_model_ladder")


def plot_signal_decay() -> None:
    p = NSE_DIR / "decay_ablation" / "nse_decay_summary.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    style = {
        "IDEA": ("#0E7C86", "-", "o"),
        "YESBANK": ("#1B9E77", "-", "s"),
        "ITC": ("#2F6FA8", "-", "^"),
        "RELIANCE": ("#D99A2B", "--", "o"),
        "HDFCBANK": ("#E67E22", "--", "s"),
        "ICICIBANK": ("#9B59B6", "--", "D"),
        "TCS": ("#C8453C", ":", "v"),
        "NIFTY_FUT": ("#444444", ":", "x"),
    }
    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values("horizon_s")
        c, ls, mk = style.get(sym, ("#333", "-", "o"))
        ax1.plot(g["horizon_s"], g["lgbm_core_f1"], label=sym, color=c, linestyle=ls, marker=mk, lw=1.8)
        ax2.plot(g["horizon_s"], g["lgbm_minus_mom"], label=sym, color=c, linestyle=ls, marker=mk, lw=1.8)

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 50, 100])
        ax.set_xticklabels(["5 s", "10 s", "20 s", "50 s", "100 s"])
        ax.set_xlabel("Prediction Horizon")
        ax.grid(True, alpha=0.3, linestyle="--")

    ax1.axhline(1 / 3, color="#666", linestyle="--", lw=1)
    ax1.set_ylabel("LightGBM Test Macro-F1")
    ax1.set_title("Absolute Predictability Across Horizons")
    ax1.legend(ncol=2, loc="upper right", frameon=True)

    ax2.axhline(0.0, color="#666", linestyle="--", lw=1)
    ax2.set_ylabel("LightGBM minus Naive Momentum (Macro-F1)")
    ax2.set_title("Genuine Lift Over Naive Momentum")
    ax2.legend(ncol=2, loc="upper right", frameon=True)

    fig.tight_layout()
    _save(fig, "nse_signal_decay")


def plot_ablation_shap() -> None:
    p_abl = NSE_DIR / "decay_ablation" / "nse_ablation_summary.csv"
    p_shap = NSE_DIR / "models" / "RELIANCE" / "shap_groups.csv"
    if not p_abl.exists():
        return
    df_a = pd.read_csv(p_abl)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    for sym in ("IDEA", "ITC", "RELIANCE", "TCS", "NIFTY_FUT"):
        sub = df_a[df_a["symbol"] == sym]
        if len(sub) == 0:
            continue
        ax1.plot(range(len(sub)), sub["macro_f1"], marker="o", lw=1.8, label=sym)
    labels = ["A: Spread\n(1)", "B: L1 snap\n(4)", "C: +L1 lags\n(13)", "D: +L2-5/VPIN\n(25)", "E: +NSE orders\n(30)"]
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("LightGBM Test Macro-F1 (k = 20 s)")
    ax1.set_title("5-Stage Feature Ablation (Including NSE Order Counts)")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend()

    if p_shap.exists():
        df_s = pd.read_csv(p_shap).sort_values("shap_share", ascending=True)
        y = np.arange(len(df_s))
        h = 0.36
        ax2.barh(y + h / 2, df_s["shap_share"] * 100, height=h, label="TreeSHAP", color="#0E7C86")
        ax2.barh(y - h / 2, df_s["gain_share"] * 100, height=h, label="Split Gain", color="#D99A2B")
        ax2.set_yticks(y)
        ax2.set_yticklabels(df_s["group"])
        ax2.set_xlabel("Share of Importance (%)")
        ax2.set_title("RELIANCE LightGBM Feature Attribution")
        ax2.grid(axis="x", alpha=0.3, linestyle="--")
        ax2.legend()

    fig.tight_layout()
    _save(fig, "nse_ablation_shap")


def plot_edge_decomposition() -> None:
    p = NSE_DIR / "execution" / "nse_edge_decomposition.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    top2 = df[df["coverage"] == 0.02].copy()
    if len(top2) == 0:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # Left: waterfall for RELIANCE top 2%
    rel = top2[top2["symbol"] == "RELIANCE"]
    if len(rel) == 0:
        rel = top2.iloc[:1]
    r = rel.iloc[0]
    fee_rt = 4.0  # 2 bp/side intraday equity statutory
    net_val = r["gross_bps"] - fee_rt
    steps = ["Decision\nmid edge", "1 s delay", "Spread &\nbook walk", "Gross\n(pre-fee)", "NSE statutory\n(2×2 bp)", "Net per\nround trip"]
    vals = [r["edge_decision_mid_bps"], -r["delay_cost_bps"], -r["crossing_cost_bps"], r["gross_bps"], -fee_rt, net_val]
    cols = ["#0E7C86", "#D99A2B", "#C8453C", "#2F6FA8", "#C8453C", "#1B2633"]
    bars = ax1.bar(steps, vals, color=cols)
    ax1.axhline(0, color="#222", lw=0.9)
    ax1.set_ylabel("bp per round trip")
    ax1.set_title(f"{r['symbol']} Top 2% Confident Signals (20 s Hold)")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.set_ylim(min(vals) - 0.8, max(vals) + 0.6)
    for b, v in zip(bars, vals):
        va = "bottom" if v >= 0 else "top"
        off = 0.14 if v >= 0 else -0.14
        ax1.text(b.get_x() + b.get_width() / 2, v + off, f"{v:+.2f}", ha="center", va=va, fontsize=8.5, fontweight="bold")

    # Right: Decision edge vs Gross vs Required Statutory Fee across all 6 symbols (Top 2%)
    x = np.arange(len(top2))
    w = 0.28
    ax2.bar(x - w, top2["edge_decision_mid_bps"], width=w, label="Edge at decision mid", color="#0E7C86")
    ax2.bar(x, top2["gross_bps"], width=w, label="Gross (after 1 s + crossing)", color="#2F6FA8")
    req_fee = [5.6 if s == "NIFTY_FUT" else 4.0 for s in top2["symbol"]]
    ax2.bar(x + w, req_fee, width=w, label="NSE statutory fee (RT)", color="#C8453C", alpha=0.8)
    ax2.axhline(0, color="#222", lw=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(top2["symbol"], rotation=20)
    ax2.set_ylabel("bp per round trip")
    ax2.set_title("Top 2% Signals: Gross Edge vs NSE Statutory Fees")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.legend()

    fig.tight_layout()
    _save(fig, "nse_edge_decomposition")


def plot_cross_asset_microprice() -> None:
    p_ca = NSE_DIR / "cross_asset" / "nse_cross_asset_summary.csv"
    p_mp = NSE_DIR / "microprice" / "nse_microprice_summary.csv"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    if p_ca.exists():
        ca = pd.read_csv(p_ca)
        for pair, g in ca.groupby("pair", sort=False):
            g = g.sort_values("horizon_s")
            ax1.plot(g["horizon_s"], g["delta_f1"], marker="o", lw=1.8, label=pair)
        ax1.axhline(0, color="#555", linestyle="--", lw=1)
        ax1.set_xscale("log")
        ax1.set_xticks([5, 10, 20, 50])
        ax1.set_xticklabels(["5 s", "10 s", "20 s", "50 s"])
        ax1.set_xlabel("Horizon")
        ax1.set_ylabel("Δ Macro-F1 (Adding Leader LOB)")
        ax1.set_title("Cross-Asset Lead-Lag on NSE")
        ax1.grid(True, alpha=0.3, linestyle="--")
        ax1.legend(fontsize=8)

    if p_mp.exists():
        mp = pd.read_csv(p_mp)
        m10 = mp[mp["horizon_s"] == 10].pivot(index="symbol", columns="model", values="ic_pearson")
        order = [s for s in ("IDEA", "YESBANK", "ITC", "ICICIBANK", "HDFCBANK", "RELIANCE", "TCS", "NIFTY_FUT") if s in m10.index]
        m10 = m10.loc[order]
        x = np.arange(len(order))
        w = 0.25
        ax2.bar(x - w, m10["microprice_rule"], width=w, label="Micro-price rule (1 param)", color="#0E7C86")
        ax2.bar(x, m10["lgbm_mid"], width=w, label="LightGBM (mid target)", color="#2F6FA8")
        ax2.bar(x + w, m10["lgbm_micro_to_mid"], width=w, label="LightGBM (micro target)", color="#D99A2B")
        ax2.axhline(0, color="#333", lw=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(order, rotation=25)
        ax2.set_ylabel("Test Pearson IC with 10 s Mid Change")
        ax2.set_title("Micro-Price Rule vs LightGBM (k = 10 s, 64 Days)")
        ax2.grid(axis="y", alpha=0.3, linestyle="--")
        ax2.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, "nse_cross_asset_microprice")


def main() -> None:
    plot_eda_tick_spectrum()
    plot_model_ladder()
    plot_signal_decay()
    plot_ablation_shap()
    plot_edge_decomposition()
    plot_cross_asset_microprice()
    print(f"[+] Saved NSE figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
