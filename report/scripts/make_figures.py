"""Builds every figure of the LaTeX report from the CSVs in artifacts/ and report/data/.

Run from the repository root:  python report/scripts/make_figures.py
Figures are written as vector PDFs to report/figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "artifacts"
D = ROOT / "report" / "data"
F = ROOT / "report" / "figures"
F.mkdir(parents=True, exist_ok=True)
RAW = ROOT / "data" / "raw"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linestyle": ":", "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "lines.linewidth": 1.6, "lines.markersize": 4.5,
})
# Okabe-Ito palette: colour-blind safe and still separable when printed in grey
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#7f7f7f", "black": "#222222", "yellow": "#F0E442"}
MODEL_COL = {"Naive Momentum": C["grey"], "Logistic Regression": C["sky"], "Random Forest": C["green"],
             "LightGBM": C["blue"], "LSTM": C["purple"], "DeepLOB": C["red"], "TLOB": C["orange"]}


def save(fig, name):
    fig.savefig(F / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ data exploration (raw BTC book)
def btc_book_figures():
    d = np.load(RAW / "crypto" / "BTC_1sec.L10.npz")
    lob = d["lob_40"].astype(np.float64)
    ts = d["timestamps_ms"]
    ask_p, ask_q, bid_p, bid_q = lob[:, 0::4], lob[:, 1::4], lob[:, 2::4], lob[:, 3::4]
    mid = (ask_p[:, 0] + bid_p[:, 0]) / 2

    # Figure: one snapshot as a depth ladder + one hour of mid price
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6), gridspec_kw={"width_ratios": [1, 1.5]})
    i = 200_000
    levels = np.arange(1, 11)
    ax[0].barh(-levels, bid_q[i], color=C["green"], label="bid size (BTC)")
    ax[0].barh(levels, ask_q[i], color=C["red"], label="ask size (BTC)")
    ticks = [-10, -5, -1, 1, 5, 10]
    ax[0].set_yticks(ticks, [f"bid L{-t}  {bid_p[i, -t - 1]:,.2f}" if t < 0 else f"ask L{t}  {ask_p[i, t - 1]:,.2f}" for t in ticks],
                     fontsize=6.5)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("Resting size (BTC)")
    ax[0].set_title("One snapshot: 10 levels per side")
    ax[0].legend(loc="lower right", fontsize=7)
    sl = slice(i, i + 3600)
    t_min = (ts[sl] - ts[i]) / 60000
    ax[1].plot(t_min, mid[sl], color=C["blue"], lw=0.9, label="mid price")
    ax[1].plot(t_min, ask_p[sl, 9], color=C["red"], lw=0.5, alpha=0.6, label="10th ask level")
    ax[1].plot(t_min, bid_p[sl, 9], color=C["green"], lw=0.5, alpha=0.6, label="10th bid level")
    ax[1].set_xlabel("Minutes")
    ax[1].set_ylabel("USD")
    ax[1].set_title("One hour of the BTC-USD book (1 s snapshots)")
    ax[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    save(fig, "btc_book_snapshot")

    # Figure: does the book carry information? future move vs queue imbalance / multi-level imbalance
    k = 20
    fut = (mid[k:] - mid[:-k]) / mid[:-k] * 1e4
    imb1 = ((bid_q[:, 0] - ask_q[:, 0]) / (bid_q[:, 0] + ask_q[:, 0]))[:-k]
    imb10 = ((bid_q.sum(1) - ask_q.sum(1)) / (bid_q.sum(1) + ask_q.sum(1)))[:-k]
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.3))
    for a, x, name, col in [(ax[0], imb1, "Level-1 imbalance", C["blue"]), (ax[1], imb10, "Levels 1-10 imbalance", C["orange"])]:
        q = pd.qcut(x, 10, labels=False, duplicates="drop")
        g = pd.DataFrame({"q": q, "y": fut, "x": x}).groupby("q").agg(x=("x", "mean"), y=("y", "mean"),
                                                                       se=("y", lambda v: v.std() / np.sqrt(len(v))))
        a.errorbar(g["x"], g["y"], yerr=1.96 * g["se"], color=col, marker="o", capsize=2)
        a.axhline(0, color="k", lw=0.6)
        a.set_xlabel(f"{name} (decile mean)")
        a.set_ylabel("Next 20 s mid change (bp)")
        a.set_title(name)
    r = (mid[1:] - mid[:-1]) / mid[:-1] * 1e4
    ax[2].hist(np.clip(fut, -15, 15), bins=np.arange(-15.25, 15.5, 0.5), color=C["grey"])
    ax[2].set_yscale("log")
    ax[2].set_xlabel("20 s mid change (bp, clipped at ±15)")
    ax[2].set_ylabel("Rows (log scale)")
    ax[2].set_title("Heavy-tailed, mostly tiny moves")
    fig.tight_layout()
    save(fig, "btc_imbalance_signal")
    return {"rows": len(lob), "zero_1s_moves": float(np.mean(r == 0)), "abs20_mean": float(np.mean(np.abs(fut))),
            "spread_one_tick": float(np.mean(np.isclose(ask_p[:, 0] - bid_p[:, 0], 0.01, atol=1e-6))),
            "corr_imb1": float(np.corrcoef(imb1, fut)[0, 1]), "corr_imb10": float(np.corrcoef(imb10, fut)[0, 1])}


def lobster_imbalance_figure():
    out = {}
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.4))
    for a, tic, col in [(ax[0], "INTC", C["blue"]), (ax[1], "AMZN", C["orange"])]:
        msg = pd.read_csv(RAW / "lobster" / f"{tic}_2012-06-21_34200000_57600000_message_10.csv", header=None, usecols=[0])
        ob = pd.read_csv(RAW / "lobster" / f"{tic}_2012-06-21_34200000_57600000_orderbook_10.csv", header=None, usecols=[0, 1, 2, 3])
        t = msg[0].to_numpy()
        grid = np.arange(34200 + 60, 57600 - 60, 0.1)
        ix = np.searchsorted(t, grid, side="right") - 1
        ap, aq, bp, bq = (ob[c].to_numpy()[ix].astype(float) for c in range(4))
        mid = (ap + bp) / 2
        h = 50  # 5 s on the 100 ms clock
        fut = (mid[h:] - mid[:-h]) / mid[:-h] * 1e4
        imb = ((bq - aq) / (bq + aq))[:-h]
        q = pd.qcut(imb, 10, labels=False, duplicates="drop")
        g = pd.DataFrame({"q": q, "y": fut, "x": imb}).groupby("q").agg(x=("x", "mean"), y=("y", "mean"),
                                                                        se=("y", lambda v: v.std() / np.sqrt(len(v))))
        a.errorbar(g["x"], g["y"], yerr=1.96 * g["se"], color=col, marker="o", capsize=2)
        a.axhline(0, color="k", lw=0.6)
        tick_bp = 0.01 / np.median(mid) * 1e4 * 1e4
        a.set_title(f"{tic} ({'large' if tic == 'INTC' else 'small'} tick, tick = {tick_bp:.2f} bp)")
        a.set_xlabel("Level-1 queue imbalance (decile mean)")
        a.set_ylabel("Next 5 s mid change (bp)")
        out[tic] = float(np.corrcoef(imb, fut)[0, 1])
    fig.tight_layout()
    save(fig, "lobster_imbalance_signal")
    return out


# ------------------------------------------------------------------ prediction results
def model_ladder():
    btc = pd.read_csv(A / "btc" / "experiment_a_results.csv")
    syn = pd.read_csv(A / "experiment_a_results.csv")
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for a, df, title, rnd in [(ax[0], syn, "Synthetic book (5 seeds, k = 20 events)", None),
                              (ax[1], btc, "Coinbase BTC-USD (4 segments, 20 s)", 1 / 3)]:
        df = df[df.model != "Naïve Flat"].copy()
        df["model"] = df["model"].str.replace("Naïve", "Naive")
        df = df.sort_values("macro_f1")
        cols = [MODEL_COL.get(m, C["grey"]) for m in df.model]
        pos = np.arange(len(df))
        a.barh(pos, df.macro_f1, xerr=df.macro_f1_std, color=cols, capsize=2, error_kw={"lw": 0.8})
        a.set_yticks(pos, df.model)
        for y, v in enumerate(df.macro_f1):
            a.text(v + df.macro_f1_std.iloc[y] + 0.006, y, f"{v:.3f}", va="center", fontsize=7)
        if rnd:
            a.axvline(rnd, color="k", ls="--", lw=0.8)
            a.text(rnd + 0.005, -0.6, "random = 1/3", fontsize=7)
        a.set_xlim(0.3, 0.66)
        a.set_xlabel("Test macro-F1 (mean ± std)")
        a.set_title(title)
    fig.tight_layout()
    save(fig, "model_ladder")


def roc_and_calibration():
    roc = pd.read_csv(D / "btc_roc_curves.csv")
    rel = pd.read_csv(D / "btc_reliability.csv")
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.4))
    for a, cls in zip(ax[:2], ["UP", "DOWN"]):
        for m in ["Logistic Regression", "Random Forest", "LightGBM"]:
            s = roc[(roc["class"] == cls) & (roc.model == m)]
            grid = np.linspace(0, 1, 101)
            tprs = [np.interp(grid, g.fpr, g.tpr) for _, g in s.groupby("segment")]
            a.plot(grid, np.mean(tprs, 0), color=MODEL_COL[m], label=m)
        a.plot([0, 1], [0, 1], color="k", ls="--", lw=0.7)
        a.set_xlabel("False positive rate")
        a.set_ylabel("True positive rate")
        a.set_title(f"ROC, {cls} vs rest (20 s)")
        a.set_aspect("equal")
    ax[0].legend(fontsize=6.5, loc="lower right")
    for m in ["Logistic Regression", "Random Forest", "LightGBM"]:
        s = rel[rel.model == m].groupby("bin").apply(lambda g: pd.Series({
            "c": np.average(g.mean_conf, weights=g.n), "a": np.average(g.accuracy, weights=g.n)}))
        ax[2].plot(s.c, s.a, marker="o", color=MODEL_COL[m], label=m)
    ax[2].plot([0.3, 0.9], [0.3, 0.9], color="k", ls="--", lw=0.7)
    ax[2].set_xlabel("Predicted probability of chosen class")
    ax[2].set_ylabel("Observed accuracy")
    ax[2].set_title("Reliability (calibration)")
    fig.tight_layout()
    save(fig, "btc_roc_calibration")


def confusion():
    cm = pd.read_csv(A / "btc" / "attribution" / "confusion_matrices.csv")
    models = ["Naive Momentum", "Logistic Regression", "LightGBM", "DeepLOB"]
    fig, ax = plt.subplots(1, 4, figsize=(6.8, 1.95))
    for a, m in zip(ax, models):
        M = cm[cm.model == m].set_index("true").loc[["FLAT", "UP", "DOWN"], ["pred_FLAT", "pred_UP", "pred_DOWN"]].to_numpy()
        a.imshow(M, cmap="Blues", vmin=0, vmax=0.6)
        for i in range(3):
            for j in range(3):
                a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                       color="white" if M[i, j] > 0.42 else "black")
        a.set_xticks(range(3), ["F", "U", "D"])
        a.set_yticks(range(3), ["F", "U", "D"])
        a.set_title(m, fontsize=8)
        a.grid(False)
        a.set_xlabel("predicted", fontsize=7)
    ax[0].set_ylabel("true", fontsize=7)
    fig.tight_layout()
    save(fig, "btc_confusion")


def decay():
    b = pd.read_csv(A / "btc" / "experiment_b_signal_decay.csv")
    lob = pd.read_csv(A / "lobster" / "lobster_decay.csv")
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    ax[0].errorbar(b.horizon_k, b.macro_f1, yerr=b.macro_f1_std, marker="o", color=C["blue"], capsize=2, label="LightGBM")
    ax[0].plot(b.horizon_k, b.momentum_macro_f1, marker="s", color=C["grey"], label="Naive momentum")
    ax[0].axhline(1 / 3, color="k", ls="--", lw=0.7)
    ax[0].set_xscale("log")
    ax[0].set_xticks([10, 20, 50, 100], ["10 s", "20 s", "50 s", "100 s"])
    ax[0].set_xlabel("Prediction horizon")
    ax[0].set_ylabel("Test macro-F1")
    ax[0].set_title("BTC: skill decays within ~100 s")
    ax[0].legend()
    order = ["100ms", "500ms", "1s", "5s", "10s", "30s"]
    secs = [0.1, 0.5, 1, 5, 10, 30]
    sty = {"INTC": (C["blue"], "-", "o"), "MSFT": (C["sky"], "-", "s"), "AAPL": (C["orange"], "--", "^"),
           "AMZN": (C["red"], "--", "v"), "GOOG": (C["purple"], "--", "D")}
    for tic, (col, ls, mk) in sty.items():
        s = lob[lob.ticker == tic].set_index("horizon").loc[order]
        ax[1].plot(secs, s.macro_f1 - s.momentum_macro_f1, color=col, ls=ls, marker=mk,
                   label=f"{tic} ({'large' if tic in ('INTC', 'MSFT') else 'small'})")
    ax[1].axhline(0, color="k", lw=0.7)
    ax[1].set_xscale("log")
    ax[1].set_xticks(secs, ["0.1", "0.5", "1", "5", "10", "30"])
    ax[1].set_xlabel("Prediction horizon (s)")
    ax[1].set_ylabel("Best model minus momentum (macro-F1)")
    ax[1].set_title("Nasdaq (LOBSTER): skill peaks at 1-5 s")
    ax[1].legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    save(fig, "signal_decay")


def fi2010():
    s = pd.read_csv(A / "fi2010" / "fi2010_setup2_summary.csv")
    models = [("Logistic Regression + history", "Logistic + history", C["sky"]),
              ("LightGBM (144 feats + history)", "LightGBM + history", C["blue"]),
              ("DeepLOB (ours)", "DeepLOB (ours)", C["red"]),
              ("DeepLOB (published)", "DeepLOB (published)", C["grey"])]
    ks = [10, 20, 50, 100]
    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    w = 0.2
    for j, (m, lab, col) in enumerate(models):
        vals = [s[(s.horizon_k == k) & (s.model == m)].accuracy_mean.squeeze() if len(s[(s.horizon_k == k) & (s.model == m)]) else np.nan for k in ks]
        errs = [s[(s.horizon_k == k) & (s.model == m)].accuracy_std.squeeze() if len(s[(s.horizon_k == k) & (s.model == m)]) else 0 for k in ks]
        x = np.arange(4) + (j - 1.5) * w
        ax.bar(x, np.array(vals) * 100, w, yerr=np.array(errs, float) * 100, color=col, label=lab, capsize=1.5,
               hatch="//" if "published" in lab else None, edgecolor="white" if "published" not in lab else "k", lw=0.3)
    ax.set_xticks(range(4), [f"k = {k}" for k in ks])
    ax.set_ylim(40, 92)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("FI-2010 Setup 2: simple models win short horizons, DeepLOB long ones")
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    save(fig, "fi2010")


def ablation_shap():
    ab = pd.read_csv(A / "btc" / "experiment_c_ablation_results.csv")
    sh = pd.read_csv(A / "btc" / "attribution" / "shap_lightgbm_groups.csv")
    sh = sh[sh.group != "Other"]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    labels = ["A: spread\n(1)", "B: L1 snap.\n(4)", "C: +lags\n(13)", "D: +L2-3\n(16)", "E: all\n(27)"]
    ax[0].errorbar(range(5), ab.macro_f1, yerr=ab.macro_f1_std, marker="o", color=C["blue"], capsize=2)
    for i, v in enumerate(ab.macro_f1):
        ax[0].text(i, v + 0.025, f"{v:.3f}", ha="center", fontsize=7)
    ax[0].axhline(1 / 3, color="k", ls="--", lw=0.7)
    ax[0].set_xticks(range(5), labels, fontsize=7)
    ax[0].set_ylim(0.2, 0.53)
    ax[0].set_ylabel("Test macro-F1")
    ax[0].set_title("Feature ablation (LightGBM, BTC 20 s)")
    y = np.arange(len(sh))
    ax[1].barh(y + 0.2, sh.shap_share_pct, 0.4, color=C["blue"], label="TreeSHAP")
    ax[1].barh(y - 0.2, sh.gain_share_pct, 0.4, color=C["orange"], label="Split gain")
    ax[1].set_yticks(y, sh.group)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("Share of importance (%)")
    ax[1].set_title("What LightGBM uses")
    ax[1].legend()
    fig.tight_layout()
    save(fig, "ablation_shap")


def deeplob_attr():
    d = pd.read_csv(A / "btc" / "attribution" / "deeplob_attribution.csv")
    lvl = d.groupby(["segment", "level"]).share_total.sum().groupby("level").mean()
    kind = d.assign(k=d.kind.str.split("_").str[1]).groupby(["segment", "k"]).share_total.sum().groupby("k").mean()
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.2), gridspec_kw={"width_ratios": [2, 1]})
    ax[0].bar(lvl.index, lvl.values * 100, color=C["red"])
    ax[0].set_xticks(range(1, 11))
    ax[0].set_xlabel("Book level")
    ax[0].set_ylabel("Share of attribution (%)")
    ax[0].set_title("DeepLOB: level 1 counts twice any other level")
    ax[1].bar(["prices", "sizes"], [kind.get("price", 0) * 100, kind.get("size", 0) * 100], color=[C["grey"], C["red"]])
    ax[1].set_title("Prices vs sizes*")
    fig.tight_layout()
    save(fig, "deeplob_attribution")


def microprice():
    b = pd.read_csv(A / "microprice" / "microprice_beta_summary.csv")
    s = pd.read_csv(A / "microprice" / "microprice_summary.csv")
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    sec = {"500ms": 0.5, "1s": 1, "5s": 5, "10s": 10, "20s": 20, "30s": 30, "50s": 50}
    for ds, col, mk in [("LOBSTER-large", C["blue"], "o"), ("LOBSTER-small", C["orange"], "s"), ("BTC", C["green"], "^")]:
        g = b[b.dataset == ds]
        ax[0].plot([sec[h] for h in g.horizon], g.beta_train, marker=mk, color=col, label=ds.replace("LOBSTER-", "Nasdaq ").replace("large", "large-tick").replace("small", "small-tick"))
    ax[0].axhline(1, color="k", ls="--", lw=0.7)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Horizon (s)")
    ax[0].set_ylabel(r"$\beta$ (1 = gap fully closed)")
    ax[0].set_title("Does the mid move to the micro-price?")
    ax[0].legend(fontsize=7)
    cases = [("LOBSTER-large", "5s", "Large tick 5 s"), ("LOBSTER-large", "10s", "Large tick 10 s"),
             ("LOBSTER-small", "1s", "Small tick 1 s"), ("BTC", "10s", "BTC 10 s")]
    mods = [("Micro-price rule (beta x gap)", "mid", "micro-price rule", C["purple"]),
            ("LightGBM", "mid", "LightGBM (mid target)", C["blue"]),
            ("LightGBM", "micro", "LightGBM (micro target)", C["sky"]),
            ("LightGBM, shuffled-target control", "mid", "shuffled control", C["grey"])]
    w = 0.2
    for j, (m, tgt, lab, col) in enumerate(mods):
        vals = []
        for ds, h, _ in cases:
            r = s[(s.dataset == ds) & (s.horizon == h) & (s.model == m) & (s.train_target == tgt)]
            vals.append(r.ic_pearson.mean() if len(r) else np.nan)
        ax[1].bar(np.arange(4) + (j - 1.5) * w, vals, w, color=col, label=lab)
    ax[1].set_xticks(range(4), [c[2] for c in cases], fontsize=7)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_ylabel("IC with future mid change")
    ax[1].set_title("One-parameter rule vs 40-feature LightGBM")
    ax[1].set_ylim(-0.02, 0.40)
    ax[1].legend(fontsize=6.3, ncol=2, loc="upper center")
    fig.tight_layout()
    save(fig, "microprice")


def optiver():
    r = pd.read_csv(A / "optiver_rvp" / "rvp_summary.csv")
    labels = ["Naive\n(current RV)", "LightGBM", "+ NN feats\npast-only", "+ NN feats\nuses future", "LightGBM\nshuffled K-fold"]
    cols = [C["grey"], C["blue"], C["green"], C["red"], C["orange"]]
    fig, ax = plt.subplots(figsize=(5.2, 2.3))
    ax.bar(labels, r["mean"], yerr=r["std"], color=cols, capsize=2)
    for i, v in enumerate(r["mean"]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_ylim(0.15, 0.38)
    ax.set_ylabel("RMSPE (lower is better)")
    ax.set_title("Volatility regression: leakage makes models look better")
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    save(fig, "optiver_leakage")


# ------------------------------------------------------------------ prediction -> profit
def hit_rate_coverage():
    h = pd.read_csv(D / "btc_hit_rate_vs_coverage.csv")
    e = pd.read_csv(A / "btc" / "tradability" / "edge_decomposition.csv")
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for m in ["Logistic Regression", "Random Forest", "LightGBM"]:
        g = h[h.model == m].groupby("coverage").hit_rate_on_moved.agg(["mean", "std"]).sort_index(ascending=False)
        ax[0].errorbar(g.index * 100, g["mean"] * 100, yerr=g["std"] * 100, marker="o", capsize=2, color=MODEL_COL[m], label=m)
    ax[0].set_xscale("log")
    ax[0].set_xticks([2, 5, 10, 20, 50, 100], ["2", "5", "10", "20", "50", "100"])
    ax[0].invert_xaxis()
    ax[0].set_xlabel("Coverage: share of rows traded (%, most confident first)")
    ax[0].set_ylabel("Hit rate on UP/DOWN rows (%)")
    ax[0].set_title("Confidence filtering raises accuracy")
    ax[0].legend(fontsize=7)
    # break-even curve
    m_move = 4.3
    cost = np.linspace(0, 10, 200)
    ax[1].plot(cost[cost <= m_move], (0.5 + cost / (2 * m_move))[cost <= m_move] * 100, color=C["black"], label=r"$p^*=\frac{1}{2}+\frac{C}{2m}$, $m$ = 4.3 bp")
    d = e[e.signal == "direction"]
    for _, r in d.iterrows():
        ax[1].scatter(r.crossing_cost_bps + r.delay_cost_bps, r.hit_rate_at_decision_mid * 100, color=C["blue"], zorder=3)
        if r.coverage in (1.0, 0.02):
            ax[1].annotate("all signals" if r.coverage == 1.0 else "top 2%", (r.crossing_cost_bps + r.delay_cost_bps, r.hit_rate_at_decision_mid * 100),
                           textcoords="offset points", xytext=(5, -6 if r.coverage == 1.0 else 2), fontsize=6.5)
    for c, lab in [(1.75, "no fee"), (3.75, "1 bp fee"), (9.75, "4 bp fee")]:
        ax[1].axvline(c, color=C["red"], ls=":", lw=0.8)
        ax[1].text(c + 0.12, 80, lab, rotation=90, fontsize=6.5, color=C["red"])
    ax[1].axhline(100, color=C["grey"], lw=0.6)
    ax[1].text(5.2, 101.2, "above 100%: impossible", fontsize=6.5, color=C["grey"])
    ax[1].set_xlabel("Round-trip cost C (bp)")
    ax[1].set_ylabel("Hit rate needed / achieved (%)")
    ax[1].set_title("Break-even accuracy vs trading cost")
    ax[1].set_ylim(48, 106)
    ax[1].legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    save(fig, "hit_rate_breakeven")


def waterfall():
    e = pd.read_csv(A / "btc" / "tradability" / "edge_decomposition.csv")
    r = e[(e.signal == "direction") & (e.coverage == 0.02)].iloc[0]
    steps = [("Edge at\ndecision mid", r.edge_at_decision_mid_bps), ("React 1 s\nlater", -r.delay_cost_bps),
             ("Cross spread,\nwalk book", -r.crossing_cost_bps), ("Taker fees\n(2 x 4 bp)", -8.0)]
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    cum = 0
    for i, (lab, v) in enumerate(steps):
        ax.bar(i, v, bottom=cum if v < 0 else 0, color=C["blue"] if v > 0 else C["red"])
        ax.text(i, (cum + v if v < 0 else v) + (0.25 if v > 0 else -0.55), f"{v:+.2f}", ha="center", fontsize=7.5)
        cum += v
    ax.bar(4, cum, color=C["black"])
    ax.text(4, cum - 0.55, f"{cum:+.2f}", ha="center", fontsize=7.5)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(range(5), [s[0] for s in steps] + ["Net per\nround trip"], fontsize=7.5)
    ax.set_ylabel("bp per round trip")
    ax.set_ylim(-9, 3.2)
    ax.set_title("BTC, most confident 2% of LightGBM signals (hit rate 72%)")
    fig.tight_layout()
    save(fig, "edge_waterfall")


def pnl_regimes():
    s = pd.read_csv(A / "btc" / "experiment_d_execution_strategies.csv")
    sy = pd.read_csv(A / "experiment_d_execution_strategies.csv")
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for a, df, title, scale, unit in [(ax[0], sy, "Synthetic (1.5 bp taker fee, 1 ms)", 1, "$"),
                                      (ax[1], s, "BTC (4 bp taker fee, 1 s)", 1000, "k$")]:
        strat = ["Academic Mid-Price (Fantasy)", "Aggressive Taker", "Passive Maker (VPIN gate)"]
        x = np.arange(3)
        for j, (m, col) in enumerate([("LightGBM", C["blue"]), ("DeepLOB", C["red"])]):
            g = df[df.model == m].set_index("strategy").loc[strat]
            a.bar(x + (j - 0.5) * 0.38, g.net_pnl / scale, 0.38, yerr=g.net_pnl_std / scale, color=col, label=m, capsize=2)
        a.axhline(0, color="k", lw=0.7)
        a.set_xticks(x, ["Trade at mid,\nno costs", "Aggressive\ntaker", "Passive maker\n(VPIN gate)"], fontsize=7.5)
        a.set_ylabel(f"Net P&L per test window ({unit})")
        a.set_title(title)
    ax[0].legend()
    fig.tight_layout()
    save(fig, "pnl_regimes")


def latency_pnl():
    sy = pd.read_csv(A / "experiment_d_latency_sharpe.csv")
    bt = pd.read_csv(A / "btc" / "experiment_d_latency_sharpe.csv")
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for m, col, mk in [("LightGBM", C["blue"], "o"), ("DeepLOB", C["red"], "s")]:
        ax[0].plot(sy.latency_ms, sy[f"pnl_{m}"], color=col, marker=mk, label=m)
        ax[0].fill_between(sy.latency_ms, sy[f"pnl_{m}"] - sy[f"pnl_{m}_std"] / np.sqrt(5),
                           sy[f"pnl_{m}"] + sy[f"pnl_{m}_std"] / np.sqrt(5), color=col, alpha=0.12)
        ax[1].plot(bt.latency_ms / 1000, bt[f"pnl_{m}"] / 1000, color=col, marker=mk, label=m)
    ax[0].axvline(2.4, color=C["grey"], ls=":", lw=0.8)
    ax[0].text(2.8, -200, "DeepLOB GPU\ninference 2.4 ms", fontsize=6.5, color=C["grey"])
    ax[0].axhline(0, color="k", lw=0.7)
    ax[0].set_xscale("symlog", linthresh=1)
    ax[0].set_xticks([0, 1, 10, 100], ["0", "1", "10", "100"])
    ax[0].set_xlabel("Added latency (ms)")
    ax[0].set_ylabel("Net P&L per test window ($)")
    ax[0].set_title("Synthetic: profit falls ~\\$4 per ms")
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("Added latency (s)")
    ax[1].set_ylabel("Net P&L (k$)")
    ax[1].set_title("BTC taker: costs dominate latency")
    ax[1].axhline(0, color="k", lw=0.7)
    ax[1].set_xscale("symlog", linthresh=1)
    ax[1].set_xticks([0, 1, 2, 5, 10, 30], ["0", "1", "2", "5", "10", "30"])
    fig.tight_layout()
    save(fig, "latency_pnl")


def speed_accuracy():
    ex = pd.read_csv(D / "btc_extra_metrics_per_segment.csv")
    ex = ex[(ex.horizon_s == 20) & ex.single_row_p50_us.notna()].groupby("model")[["macro_f1", "single_row_p50_us"]].mean()
    a_res = pd.read_csv(A / "btc" / "experiment_a_results.csv").set_index("model").macro_f1
    cpp = pd.read_csv(A / "cpp_trained_model_benchmark.csv").iloc[0]
    btc_dl = pd.read_csv(A / "btc" / "experiment_d_inference_latency.csv", index_col=0).loc["median", "deeplob_ms"]
    pts = [("Logistic regression\n(Python)", ex.loc["Logistic Regression", "single_row_p50_us"], a_res["Logistic Regression"], C["sky"]),
           ("Random Forest\n(Python)", ex.loc["Random Forest", "single_row_p50_us"], a_res["Random Forest"], C["green"]),
           ("LightGBM\n(Python booster)", ex.loc["LightGBM", "single_row_p50_us"], a_res["LightGBM"], C["blue"]),
           ("LightGBM compiled\nto C++", cpp.cpp_p50_ns / 1000, a_res["LightGBM"], C["black"]),
           ("DeepLOB\n(GPU, batch 1)", btc_dl * 1000, a_res["DeepLOB"], C["red"])]
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    for lab, x, y, col in pts:
        ax.scatter(x, y, s=40, color=col, zorder=3)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, -4) if "C++" not in lab else (-10, -22), fontsize=6.8)
    ax.set_xscale("log")
    ax.set_xlim(3, 3e5)
    ax.set_ylim(0.395, 0.485)
    ax.axhline(a_res["Naïve Momentum"], color=C["grey"], ls="--", lw=0.8)
    ax.text(4, a_res["Naïve Momentum"] + 0.003, "naive momentum", fontsize=6.5, color=C["grey"])
    ax.set_xlabel(r"Single-prediction latency, median ($\mu$s, log scale)")
    ax.set_ylabel("BTC macro-F1 (20 s)")
    ax.set_title("Accuracy vs inference speed on BTC")
    fig.tight_layout()
    save(fig, "speed_vs_accuracy")


def latency_budget():
    """Where each model's synthetic P&L line crosses zero: accuracy buys latency budget."""
    sy = pd.read_csv(A / "experiment_d_latency_sharpe.csv")
    out = {}
    for m in ["LightGBM", "DeepLOB"]:
        x, y = sy.latency_ms.to_numpy(), sy[f"pnl_{m}"].to_numpy()
        sl = np.polyfit(x[1:], y[1:], 1)
        out[m] = {"slope_per_ms": sl[0], "zero_ms": -sl[1] / sl[0]}
    return out


if __name__ == "__main__":
    info = btc_book_figures()
    print(info)
    print(lobster_imbalance_figure())
    model_ladder(); roc_and_calibration(); confusion(); decay(); fi2010(); ablation_shap(); deeplob_attr()
    microprice(); optiver(); hit_rate_coverage(); waterfall(); pnl_regimes(); latency_pnl(); speed_accuracy()
    print(latency_budget())
