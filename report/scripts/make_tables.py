"""Writes the LaTeX tables that come from report/data/btc_extra_metrics_per_segment.csv.

Run from the repository root after btc_extra_metrics.py:  python report/scripts/make_tables.py
"""
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[1] / "data"
df = pd.read_csv(D / "btc_extra_metrics_per_segment.csv")
ORDER = ["Naive Momentum", "Logistic Regression", "Random Forest", "LightGBM"]


def ms(g, col, nd=3):
    return f"{g[col].mean():.{nd}f} $\\pm$ {g[col].std():.{nd}f}"


rows = []
for m in ORDER:
    g = df[(df.horizon_s == 20) & (df.model == m)]
    name = f"\\textbf{{{m}}}" if m == "LightGBM" else m
    rows.append(" & ".join([name, f"{g.accuracy.mean():.3f}", f"{g.balanced_accuracy.mean():.3f}", ms(g, "macro_f1"),
                            f"{g.mcc.mean():.3f}", f"{g.cohen_kappa.mean():.3f}", ms(g, "roc_auc_macro_ovr"),
                            f"{g.auc_up_vs_down_moved.mean():.3f}", f"{g.log_loss.mean():.3f}", f"{g.brier.mean():.3f}"]) + r" \\")
table = r"""\begin{table}[H]
\centering\scriptsize
\setlength{\tabcolsep}{3.5pt}
\caption{BTC-USD, 20\,s horizon: threshold and probability metrics for the tabular models (mean over 4 segments, $\pm$ std). AUC (U/D) is UP vs DOWN on rows that moved.}
\label{tab:btc_auc}
\begin{tabular}{@{}lccccccccc@{}}
\toprule
\textbf{Model} & \textbf{Acc.} & \textbf{Bal. acc.} & \textbf{Macro-F1} & \textbf{MCC} & \textbf{$\kappa$} & \textbf{ROC-AUC (OvR)} & \textbf{AUC (U/D)} & \textbf{Log-loss} & \textbf{Brier} \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
Uniform guess & 0.333 & 0.333 & 0.333 & 0 & 0 & 0.500 & 0.500 & 1.099 & 0.667 \\
\bottomrule
\end{tabular}
\end{table}
"""

# per-horizon AUC table
hrows = []
for k in [10, 20, 50]:
    cells = [f"{k}\\,s"]
    for m in ["Logistic Regression", "Random Forest", "LightGBM"]:
        g = df[(df.horizon_s == k) & (df.model == m)]
        cells += [f"{g.macro_f1.mean():.3f}", f"{g.roc_auc_macro_ovr.mean():.3f}"]
    hrows.append(" & ".join(cells) + r" \\")
htable = r"""\begin{table}[H]
\centering\small
\caption{Macro-F1 and one-vs-rest ROC-AUC by horizon (BTC, mean of 4 segments)}
\label{tab:btc_auc_h}
\begin{tabular}{@{}lcccccc@{}}
\toprule
 & \multicolumn{2}{c}{\textbf{Logistic regression}} & \multicolumn{2}{c}{\textbf{Random Forest}} & \multicolumn{2}{c}{\textbf{LightGBM}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
\textbf{Horizon} & Macro-F1 & AUC & Macro-F1 & AUC & Macro-F1 & AUC \\
\midrule
""" + "\n".join(hrows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(D / "table_btc_auc.tex").write_text(table)
(D / "table_btc_auc_h.tex").write_text(htable)

# latency rows
lat = []
names = {"Logistic Regression": "Logistic regression (Python)", "Random Forest": "Random Forest, 100 trees (Python)",
         "LightGBM": "LightGBM booster (Python, 1 thread)"}
for m in ["Logistic Regression", "Random Forest", "LightGBM"]:
    g = df[(df.horizon_s == 20) & (df.model == m)]

    def fmt(us):
        return f"{us / 1000:.2f}\\,ms" if us >= 1000 else f"{us:.0f}\\,$\\mu$s"
    lat.append(f"{names[m]} & {fmt(g.single_row_p50_us.mean())} & {fmt(g.single_row_p99_us.mean())} & "
               f"Batch: {g.batch_us_per_row.mean():.2f}\\,$\\mu$s/row; fit {g.fit_seconds.mean():.0f}\\,s \\\\")
(D / "latency_rows.tex").write_text("\n".join(lat) + "\n")
print(table, htable, "\n".join(lat), sep="\n")
print(df[df.horizon_s == 20].groupby("model")[["auc_FLAT", "auc_UP", "auc_DOWN", "weighted_f1", "macro_precision",
                                                 "macro_recall", "fit_seconds", "n_train", "n_test", "n_trees"]].mean().round(3))
