"""Generates Okabe-Ito serif vector PDF figures for Chapter 8 (Indian NSE Market) of report/report.tex.

Matches the exact typography, font sizes, palette and column widths of report/scripts/make_figures.py.
Run from the repository root:  python report/scripts/make_nse_report_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NSE = ROOT / "artifacts" / "nse"
F = ROOT / "report" / "figures"
F.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 7.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.2,
})

C = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
    "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#7f7f7f", "black": "#222222",
}
MODEL_COL = {
    "Naive Momentum": C["grey"],
    "Logistic Regression": C["sky"],
    "Random Forest": C["green"],
    "LightGBM": C["blue"],
    "LightGBM (+NSE)": "#1B5E8C",
    "LSTM": C["purple"],
    "DeepLOB5": C["red"],
    "TLOB": C["orange"],
}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(F / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def fig_nse_tick_spectrum() -> None:
    df = pd.read_csv(NSE / "nse_dataset_summary.csv").sort_values("tick_bp", ascending=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.55))

    cols = [C["red"] if t < 0.5 else (C["sky"] if t < 1.5 else C["blue"]) for t in df["tick_bp"]]
    y = np.arange(len(df))
    ax1.barh(y, df["corr_imb1_20s"], color=cols, height=0.68)
    ax1.set_yticks(y, [f"{r.symbol} ({r.tick_bp:.2f} bp)" for r in df.itertuples()], fontsize=7.5)
    ax1.set_xlabel("Corr(L1 imbalance, next 20 s move)")
    ax1.set_title("NSE 64-day tick-size spectrum")
    for i, v in enumerate(df["corr_imb1_20s"]):
        ax1.text(v + 0.004, i, f"+{v:.3f}", va="center", fontsize=7.2)
    ax1.set_xlim(0, 0.235)

    ax2.scatter(df["tick_bp"], df["corr_imb1_20s"], s=42, c=cols, zorder=4)
    offsets = {"RELIANCE": (5, -8), "HDFCBANK": (5, 1), "ICICIBANK": (5, 4), "IDEA": (-28, -2), "YESBANK": (-36, 4)}
    for r in df.itertuples():
        xy = offsets.get(r.symbol, (5, 2))
        ax2.annotate(r.symbol, (r.tick_bp, r.corr_imb1_20s), textcoords="offset points", xytext=xy, fontsize=7.2)
    ax2.set_xscale("log")
    ax2.set_xlabel("Relative tick size (bp, log scale)")
    ax2.set_ylabel("Corr(L1 imb, 20 s move)")
    ax2.set_title("Larger tick = deeper queues = stronger signal")
    ax2.set_ylim(0, 0.21)

    fig.tight_layout(w_pad=1.6)
    save(fig, "nse_eda_tick_spectrum")


def fig_nse_model_ladder() -> None:
    syms = ["RELIANCE", "NIFTY_FUT", "IDEA"]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.55))
    for ax, sym in zip(axes, syms):
        df_all = pd.read_csv(NSE / "models" / sym / "model_ladder_summary.csv")
        flat_f1 = float(df_all.loc[df_all["model"] == "Naive Flat", "macro_f1"].iloc[0])
        df = df_all[df_all["model"] != "Naive Flat"].sort_values("macro_f1", ascending=True)
        y = np.arange(len(df))
        cols = [MODEL_COL.get(m, C["grey"]) for m in df["model"]]
        ax.barh(y, df["macro_f1"], xerr=df["macro_f1_std"], color=cols, height=0.66, capsize=2, error_kw={"lw": 0.9})
        ax.axvline(flat_f1, color=C["grey"], ls=":", lw=1.0)
        short = {
            "Logistic Regression": "Logistic",
            "Naive Momentum": "Momentum",
            "Random Forest": "Random Forest",
            "LightGBM (+NSE)": "LGBM (+NSE)",
        }
        ax.set_yticks(y, [short.get(m, m) for m in df["model"]], fontsize=7.2)
        ax.set_xlabel("Macro-F1 (mean ± std)")
        ax.set_title(f"{sym} (20 s; Flat={flat_f1:.3f})", fontsize=8.4)
        for i, (v, sd) in enumerate(zip(df["macro_f1"], df["macro_f1_std"])):
            ax.text(v + sd + 0.008, i, f"{v:.3f}", va="center", fontsize=6.8)
        ax.set_xlim(0.14, 0.59)
    fig.tight_layout(w_pad=1.0)
    save(fig, "nse_model_ladder")


def fig_nse_signal_decay() -> None:
    df = pd.read_csv(NSE / "decay_ablation" / "nse_decay_summary.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.65))
    style = {
        "IDEA": (C["blue"], "-", "o"),
        "YESBANK": (C["green"], "-", "s"),
        "ITC": (C["sky"], "-", "^"),
        "RELIANCE": (C["orange"], "--", "o"),
        "HDFCBANK": (C["purple"], "--", "s"),
        "ICICIBANK": ("#8C564B", "--", "D"),
        "TCS": (C["red"], ":", "v"),
        "NIFTY_FUT": (C["black"], ":", "x"),
    }
    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values("horizon_s")
        c, ls, mk = style[sym]
        ax1.plot(g["horizon_s"], g["lgbm_core_f1"], label=sym, color=c, ls=ls, marker=mk, lw=1.35, ms=3.5)
        ax2.plot(g["horizon_s"], g["lgbm_minus_mom"], label=sym, color=c, ls=ls, marker=mk, lw=1.35, ms=3.5)

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 50, 100], ["5 s", "10 s", "20 s", "50 s", "100 s"])
        ax.set_xlabel("Prediction horizon")
    ax1.axhline(1 / 3, color=C["grey"], ls="--", lw=0.8)
    ax1.set_ylabel("LightGBM test macro-F1")
    ax1.set_title("NSE skill across horizons (64 days)")
    ax1.legend(ncol=2, fontsize=6.5, frameon=True, loc="upper right")

    ax2.axhline(0.0, color=C["grey"], ls="--", lw=0.8)
    ax2.set_ylabel("LightGBM minus momentum")
    ax2.set_title("Genuine lift over naive momentum")
    ax2.legend(ncol=2, fontsize=6.5, frameon=True, loc="lower right")

    fig.tight_layout(w_pad=1.5)
    save(fig, "nse_signal_decay")


def fig_nse_ablation_shap() -> None:
    df_a = pd.read_csv(NSE / "decay_ablation" / "nse_ablation_summary.csv")
    df_s = pd.read_csv(NSE / "models" / "RELIANCE" / "shap_groups.csv").sort_values("shap_share", ascending=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.55), gridspec_kw={"width_ratios": [1.15, 1.0]})

    cols = {"IDEA": C["blue"], "ITC": C["green"], "RELIANCE": C["orange"], "TCS": C["red"], "NIFTY_FUT": C["black"]}
    for sym in ("IDEA", "ITC", "RELIANCE", "TCS", "NIFTY_FUT"):
        sub = df_a[df_a["symbol"] == sym]
        ax1.plot(range(len(sub)), sub["macro_f1"], marker="o", lw=1.4, ms=3.8, label=sym, color=cols[sym])
    ax1.set_xticks(range(5), ["A: Sprd\n(1)", "B: L1\n(4)", "C: +Lags\n(13)", "D: +L2-5\n(25)", "E: +NSE\n(30)"], fontsize=7.2)
    ax1.set_ylabel("LightGBM macro-F1 (20 s)")
    ax1.set_title("5-stage feature ablation")
    ax1.legend(fontsize=6.8, loc="lower right", frameon=True)

    y = np.arange(len(df_s))
    h = 0.36
    short_g = {
        "Deeper book (L2-5, slopes, PCA)": "Deeper book\n(L2-5, PCA)",
        "NSE order counts & full book": "NSE order cnts\n& full book",
    }
    ax2.barh(y + h / 2, df_s["shap_share"] * 100, height=h, label="TreeSHAP", color=C["blue"])
    ax2.barh(y - h / 2, df_s["gain_share"] * 100, height=h, label="Split gain", color=C["orange"])
    ax2.set_yticks(y, [short_g.get(g, g) for g in df_s["group"]], fontsize=7.3)
    ax2.set_xlabel("Share of importance (%)")
    ax2.set_title("RELIANCE attribution")
    ax2.legend(fontsize=7.0, loc="lower right", frameon=True)

    fig.tight_layout(w_pad=1.4)
    save(fig, "nse_ablation_shap")


def fig_nse_edge_decomposition() -> None:
    df = pd.read_csv(NSE / "execution" / "nse_edge_decomposition.csv")
    top2 = df[df["coverage"] == 0.02].copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.65))

    r = top2[top2["symbol"] == "RELIANCE"].iloc[0]
    fee_rt = 4.0
    net_v = r["gross_bps"] - fee_rt
    steps = ["Decision\nmid", "1 s\ndelay", "Spread &\nbook walk", "Gross\n(pre-fee)", "NSE fee\n(2x2 bp)", "Net\nRT"]
    vals = [r["edge_decision_mid_bps"], -r["delay_cost_bps"], -r["crossing_cost_bps"], r["gross_bps"], -fee_rt, net_v]
    cols = [C["blue"], C["orange"], C["red"], C["sky"], C["red"], C["black"]]
    bars = ax1.bar(steps, vals, color=cols, width=0.65)
    ax1.axhline(0, color=C["black"], lw=0.7)
    ax1.set_ylabel("bp per round trip")
    ax1.set_title("RELIANCE top 2% signals (20 s)")
    ax1.tick_params(axis="x", labelsize=7.0)
    ax1.set_ylim(-6.5, 1.45)
    for b, v in zip(bars, vals):
        va = "bottom" if v >= 0 else "top"
        off = 0.16 if v >= 0 else -0.16
        ax1.text(b.get_x() + b.get_width() / 2, v + off, f"{v:+.2f}", ha="center", va=va, fontsize=7.0, fontweight="bold")

    x = np.arange(len(top2))
    w = 0.27
    ax2.bar(x - w, top2["edge_decision_mid_bps"], width=w, label="Decision mid edge", color=C["blue"])
    ax2.bar(x, top2["gross_bps"], width=w, label="Gross (after crossing)", color=C["sky"])
    req = [5.6 if s == "NIFTY_FUT" else 4.0 for s in top2["symbol"]]
    ax2.bar(x + w, req, width=w, label="NSE statutory fee (RT)", color=C["red"])
    ax2.axhline(0, color=C["black"], lw=0.7)
    ax2.set_xticks(x, top2["symbol"], rotation=22, ha="right", fontsize=7.3)
    ax2.set_ylabel("bp per round trip")
    ax2.set_title("Top 2%: crossing kills edge before fees")
    ax2.legend(fontsize=6.8, loc="lower left", frameon=True)

    fig.tight_layout(w_pad=1.4)
    save(fig, "nse_edge_decomposition")


def fig_nse_cross_micro() -> None:
    ca = pd.read_csv(NSE / "cross_asset" / "nse_cross_asset_summary.csv")
    mp = pd.read_csv(NSE / "microprice" / "nse_microprice_summary.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.65))

    p_cols = {
        "ICICIBANK->HDFCBANK": C["red"],
        "HDFCBANK->ICICIBANK": C["purple"],
        "NIFTY_FUT->RELIANCE": C["blue"],
        "NIFTY_FUT->ICICIBANK": C["green"],
        "NIFTY_FUT->HDFCBANK": C["orange"],
    }
    for pair, g in ca.groupby("pair", sort=False):
        g = g.sort_values("horizon_s")
        ax1.plot(g["horizon_s"], g["delta_f1"], marker="o", lw=1.4, ms=3.5, label=pair.replace("->", "→"), color=p_cols.get(pair, C["grey"]))
    ax1.axhline(0, color=C["grey"], ls="--", lw=0.8)
    ax1.set_xscale("log")
    ax1.set_xticks([5, 10, 20, 50], ["5 s", "10 s", "20 s", "50 s"])
    ax1.set_ylim(-0.0085, 0.0305)
    ax1.set_xlabel("Horizon")
    ax1.set_ylabel("Δ macro-F1 (adding leader)")
    ax1.set_title("NSE cross-asset lead-lag")
    ax1.legend(fontsize=6.1, ncol=2, loc="upper right", frameon=True, columnspacing=0.8, handlelength=1.4)

    m10 = mp[mp["horizon_s"] == 10].pivot(index="symbol", columns="model", values="ic_pearson")
    order = ["IDEA", "YESBANK", "ITC", "ICICIBANK", "HDFCBANK", "RELIANCE", "TCS", "NIFTY_FUT"]
    m10 = m10.loc[order]
    x = np.arange(len(order))
    w = 0.27
    ax2.bar(x - w, m10["microprice_rule"], width=w, label="Micro-price rule (1p)", color=C["purple"])
    ax2.bar(x, m10["lgbm_mid"], width=w, label="LightGBM (mid)", color=C["blue"])
    ax2.bar(x + w, m10["lgbm_micro_to_mid"], width=w, label="LightGBM (micro)", color=C["orange"])
    ax2.axhline(0, color=C["black"], lw=0.7)
    ax2.set_xticks(x, order, rotation=28, ha="right", fontsize=7.0)
    ax2.set_ylabel("Test Pearson IC (10 s)")
    ax2.set_title("Micro-price rule vs LightGBM (64 d)")
    ax2.legend(fontsize=6.6, loc="upper left", frameon=True)

    fig.tight_layout(w_pad=1.4)
    save(fig, "nse_cross_asset_microprice")


def fig_nse_universe_25() -> None:
    p_sum = NSE / "universe_25" / "nse_universe_25_summary.csv"
    p_cor = NSE / "universe_25" / "nse_universe_25_correlations.csv"
    if not p_sum.exists():
        return
    df = pd.read_csv(p_sum)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.45, 2.75))

    # Left: 25-instrument Tick-Size Predictability Curve (5 s OvR ROC-AUC vs log tick_bp)
    eq = df[df["asset"] == "EQUITY"]
    fu = df[df["asset"] == "DERIVATIVE"]
    ax1.scatter(eq["tick_bp"], eq["auc_ovr_5s"], color=C["blue"], s=28, alpha=0.88, label="NSE Equities (23)", zorder=3)
    ax1.scatter(fu["tick_bp"], fu["auc_ovr_5s"], color=C["red"], marker="D", s=34, label="Index Futures (2)", zorder=4)

    lx = np.log10(df["tick_bp"].to_numpy())
    yv = df["auc_ovr_5s"].to_numpy()
    coef = np.polyfit(lx, yv, 1)
    xs = np.logspace(np.min(lx) - 0.08, np.max(lx) + 0.08, 100)
    ax1.plot(xs, coef[0] * np.log10(xs) + coef[1], color=C["grey"], ls="--", lw=1.0, zorder=2)

    if p_cor.exists():
        cor = pd.read_csv(p_cor).set_index("metric")
        rho = float(cor.loc["auc_ovr_5s", "spearman_rho_vs_tick"])
        rp = float(cor.loc["auc_ovr_5s", "pearson_r_vs_log_tick"])
        ax1.text(0.04, 0.94, f"N = {len(df)} (36M rows)\nSpearman ρ = {rho:+.2f}\nLog-linear r = {rp:+.2f}",
                 transform=ax1.transAxes, va="top", fontsize=6.7,
                 bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#CCCCCC", "alpha": 0.9})

    label_offs = {
        "IDEA": (-24, -9),
        "YESBANK": (-36, 4),
        "SUZLON": (6, -8),
        "ITC": (6, 3),
        "RELIANCE": (6, -7),
        "TCS": (6, -7),
        "TITAN": (-28, 5),
        "BANKNIFTY_FUT": (7, 4),
        "NIFTY_FUT": (7, -8),
    }
    for r in df.itertuples():
        if r.symbol in label_offs:
            ax1.annotate(r.symbol, (r.tick_bp, r.auc_ovr_5s), textcoords="offset points",
                         xytext=label_offs[r.symbol], fontsize=6.2)

    ax1.set_xscale("log")
    ax1.set_xlabel("Relative tick size (bp, log scale)")
    ax1.set_ylabel("5 s LightGBM ROC-AUC (test)")
    ax1.set_title("Tick-size law across 25 NSE instruments")
    ax1.legend(fontsize=6.4, loc="lower right", frameon=True)

    # Right: Market-Wide Taker Viability Boundary (Decision-mid edge vs Crossing cost)
    xmax = float(df["crossing_bps"].max()) * 1.10
    xx = np.linspace(0, xmax, 100)
    ax2.fill_between(xx, -1.0, xx, color="#F9E5E2", alpha=0.55, zorder=1)
    ax2.plot(xx, xx, color=C["black"], ls="--", lw=1.0, label="Break-even (0 bp fee)", zorder=2)
    ax2.scatter(eq["crossing_bps"], eq["edge_mid_bps"], color=C["blue"], s=28, alpha=0.88, zorder=3)
    ax2.scatter(fu["crossing_bps"], fu["edge_mid_bps"], color=C["red"], marker="D", s=34, zorder=4)

    n_neg = int((df["gross_bps"] < 0).sum())
    ax2.text(0.04, 0.94, f"{n_neg}/{len(df)} negative before fees\n(crossing > mid edge)",
             transform=ax2.transAxes, va="top", fontsize=6.7,
             bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#CCCCCC", "alpha": 0.92})

    viab_offs = {
        "IDEA": (-28, 5, False),
        "YESBANK": (-44, -9, False),
        "BANKNIFTY_FUT": (14, 12, True),
        "NIFTY_FUT": (22, 2, True),
        "RELIANCE": (8, -15, True),
        "ITC": (14, -12, True),
    }
    for r in df.itertuples():
        if r.symbol in viab_offs:
            dx, dy, use_arr = viab_offs[r.symbol]
            ap = {"arrowstyle": "-", "color": "#888888", "lw": 0.6} if use_arr else None
            ax2.annotate(r.symbol, (r.crossing_bps, r.edge_mid_bps), textcoords="offset points",
                         xytext=(dx, dy), fontsize=6.2, arrowprops=ap)

    ax2.set_xlim(0, xmax)
    ax2.set_ylim(-0.2, max(float(df["edge_mid_bps"].max()) * 1.28, 4.2))
    ax2.set_xlabel("Observed spread & book-walk cost (bp, RT)")
    ax2.set_ylabel("Top-signal mid edge (bp, 20 s)")
    ax2.set_title("All 25 fail spread crossing pre-fee")
    ax2.legend(fontsize=6.4, loc="lower right", frameon=True)

    fig.tight_layout(w_pad=1.4)
    save(fig, "nse_universe_25")


def main() -> None:
    fig_nse_tick_spectrum()
    fig_nse_model_ladder()
    fig_nse_signal_decay()
    fig_nse_ablation_shap()
    fig_nse_edge_decomposition()
    fig_nse_cross_micro()
    fig_nse_universe_25()


if __name__ == "__main__":
    main()
