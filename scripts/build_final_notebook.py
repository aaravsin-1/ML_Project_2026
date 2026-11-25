"""Builds (and optionally executes) the final submission notebook, submission/Final_Project_Notebook.ipynb.

The notebook documents every experiment across Synthetic, Coinbase BTC/ETH, Nasdaq LOBSTER,
FI-2010, Optiver Kaggle, C++ inference benchmarks, and the 64-day 25-instrument Indian NSE
5-level order book study. By default it shows the saved results from the repository's
artifacts/ folder and runs a complete live ML pipeline demo (data generation, preprocessing,
train-only Block PCA & feature engineering, model training/evaluation, confusion matrix, PyTorch
deep architecture forward-pass check, purged CV + shuffled-label control, and execution
simulation), so "Run all" finishes in ~1-2 minutes on Colab or a laptop. Each full experiment
can be recomputed from the RERUN dictionary in Section 1.

    python scripts/build_final_notebook.py            # build and execute (embeds the outputs)
    python scripts/build_final_notebook.py --no-run   # build only
"""

import argparse
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submission" / "Final_Project_Notebook.ipynb"
COLAB_URL = "https://colab.research.google.com/github/aaravsin-1/ML_Project_2026/blob/master/submission/Final_Project_Notebook.ipynb"

CELLS = []


def md(text):
    CELLS.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    CELLS.append(nbf.v4.new_code_cell(text.strip("\n")))


# =====================================================================================  title
md(f"""
# Predicting Limit Order Books, and Whether the Prediction Pays

**UML501 — Machine Learning · Final Project Notebook.**  
**Authors:** Aarav Singhal (`1024030539`) & Sharan Sharma (`1024030511`) · **Guided by:** Dr. Sanju Kumari (CSED, Thapar Institute of Engineering & Technology). November 2026.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

This notebook walks through every stage of our Machine Learning pipeline—data ingestion, preprocessing, anti-leakage chronological splitting, causal feature engineering (including train-only Block PCA), classical tabular models, deep neural networks, statistical evaluation, explainability, C++ tree compilation, and real-world cost/latency evaluation across six datasets (~43M snapshots total, anchored by our 36.0M-row Indian NSE 5-level study).

- Full LaTeX report: [LOB_ML_Project_Report.pdf](../report/LOB_ML_Project_Report.pdf)
- Presentation slides: [LOB_ML_Project_Presentation.pptx](../report/LOB_ML_Project_Presentation.pptx)
- Summary report & repository docs: [PROJECT_REPORT.md](PROJECT_REPORT.md) · [README](../README.md)

**How to run this notebook.** Choose **Runtime → Run all**. It runs in ~1–2 minutes on CPU with no credentials needed:
- Section 1 detects Colab or the local repository root and defines plotting/table helpers.
- Section 3 runs a complete **live end-to-end ML pipeline**: synthetic order-book generation, cleaning & label calibration, causal feature extraction + train-only Block PCA, training & evaluating the tabular model ladder (with confusion matrix, ROC-AUC, and feature-importance plots), instantiating & verifying all PyTorch deep architectures (`LSTM`, `DeepLOB`, `DeepLOB5`, `TLOB`), running purged cross-validation + a shuffled-label control, and simulating realistic execution.
- Sections 4–14 load verified results from `artifacts/` (since full multi-day GPU runs across BTC and 25 Indian NSE stocks take hours). Toggle any flag in `RERUN` (Section 1) to recompute from source.

**Contents**
1. Setup
2. Problem Formulation, Datasets, Preprocessing & Method
3. Live End-to-End Pipeline: Preprocessing, Feature Engineering, Training, Deep Nets & Testing
4. Synthetic Experiments A–D (5 sessions)
5. Real Data: Coinbase BTC-USD (Experiments A–D)
6. From Prediction to Profit on BTC (Experiments E, Edge Decomposition, G)
7. Cross-Asset Lead-Lag: Ethereum → Bitcoin (Experiment F)
8. Market Making (H) and Nasdaq Stocks by Tick Size (LOBSTER, Experiment I)
9. Statistical Significance & Explainable AI (SHAP, CNN Saliency, Experiment J)
10. Micro-Price as Predictor and Regression Target (Experiment K)
11. FI-2010 Academic Benchmark: History vs Architecture
12. Optiver Kaggle Datasets: Leakage Case Study & Closing Auction Regression
13. Inference Latency: Python-to-C++ Tree Compiler
14. Indian NSE 5-Level Order-Book Study (64 Trading Days, 25 Instruments, 36.0M Snapshots)
15. Discussion: Balancing Accuracy, Latency and Cost
16. Limitations, Reproducibility & Conclusion
""")

# =====================================================================================  1 setup
md("""
## 1. Setup

Clones the repository on Colab (locally, it detects the repository root), installs any missing dependencies, and configures table/figure display helpers.
""")

code(r'''
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO = "aaravsin-1/ML_Project_2026"

if IN_COLAB:
    root = Path("/content/ML_Project_2026")
    if not root.exists():
        try:
            from google.colab import userdata
            token = userdata.get("GITHUB_TOKEN")      # only needed if the repository is private
        except Exception:
            token = None
        url = f"https://{token}@github.com/{REPO}.git" if token else f"https://github.com/{REPO}.git"
        subprocess.run(["git", "clone", "--depth", "1", url, str(root)], check=True)
    os.chdir(root)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm>=4.0", "kaggle"], check=False)
else:
    root = Path.cwd()
    while not (root / "src" / "experiments").exists() and root != root.parent:
        root = root.parent
    os.chdir(root)
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
print("Repository:", root)
print("Code at commit:", subprocess.run(["git", "log", "-1", "--format=%h %ci"], capture_output=True, text=True).stdout.strip())
''')

code(r'''
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 32)
ART = root / "artifacts"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"


def load(path):
    return pd.read_csv(ART / path)


def show(df, digits=3):
    """Displays a table with its first column as the row label."""
    display(df.set_index(df.columns[0]).round(digits))


def figure(path, width=900):
    display(Image(filename=str(ART / path), width=width))
''')

md("""
**Rerunning experiments (optional).** Every switch below defaults to `False` so the notebook renders saved experiment artifacts rapidly. Switch any entry to `True` to recompute that pipeline stage from raw/cached data:
- **Checkpoints:** experiments resume from saved checkpoints where present.
- **Public datasets (`btc`, `eth`, `fi2010`, `optiver_*`):** fetched via Kaggle credentials (`KAGGLE_USERNAME`, `KAGGLE_KEY` in Colab secrets or local `~/.kaggle/kaggle.json`).
- **Indian NSE dataset (`nse_*`):** reads a local Parquet/NPZ NSE tick store configured via `MARKET_TICKS` environment variable or cached `data/processed/nse/*.npz`.
""")

code(r'''
RERUN = {
    "validation": False,          # leakage / robustness report, synthetic       CPU, ~5 min
    "synthetic": False,           # Experiments A-D on 5 synthetic sessions      GPU recommended, ~10 min
    "btc": False,                 # Experiments A-D on 4 BTC segments            GPU, ~3 h
    "btc_tradability": False,     # Experiment E                                 CPU, ~20 min
    "btc_edge": False,            # edge decomposition (needs Experiment G)      CPU, ~2 min
    "btc_eth": False,             # Experiment F                                 CPU, ~20 min
    "btc_regimes": False,         # Experiment G                                 CPU, ~10 min
    "btc_market_making": False,   # Experiment H                                 CPU, ~5 min
    "lobster": False,             # Experiment I (needs LOBSTER files)           CPU, ~15 min
    "attribution": False,         # Experiment J; DeepLOB part needs a GPU       GPU, ~1.5 h
    "microprice": False,          # Experiment K (BTC; LOBSTER if present)       CPU, ~1 h
    "fi2010": False,              # FI-2010 Setup 2, 3 seeds                     GPU, ~2 h
    "fi2010_diagnose": False,     # FI-2010 LightGBM diagnostic                  CPU, ~10 min
    "optiver_rvp": False,         # Optiver Realized Volatility                  CPU, ~1 h
    "optiver_close": False,       # Optiver Trading at the Close                 CPU, ~1 h
    "cpp_trained": False,         # trained LightGBM compiled to C++             CPU, ~5 min
    "nse_models": False,          # NSE 5-level Exp A & J (LGBM vs LSTM/DL5/TLOB) GPU, ~1.5 h
    "nse_decay_ablation": False,  # NSE Exp B & C (horizon decay + 5-stage ablat) CPU, ~25 min
    "nse_execution": False,       # NSE Exp D, E, G & edge decomposition          CPU, ~15 min
    "nse_cross_micro": False,     # NSE Exp F (bank lead-lag) & K (micro-price)   CPU, ~25 min
    "nse_universe_25": False,     # NSE 25-instrument 36M-row tick-spectrum study CPU, ~35 min
}

PY = f"{sys.executable} -u -W ignore"
STEPS = {
    "validation": (f"{PY} scripts/verify_robustness_and_leakage.py --seed 42 --events 60000", []),
    "synthetic": (f"{PY} scripts/run_all_experiments.py --source synthetic --seeds 0 1 2 3 4 --epochs 10", []),
    "btc": (f"{PY} scripts/run_all_experiments.py --source btc --segments 4 --epochs 8", ["btc"]),
    "btc_tradability": (f"{PY} -m src.experiments.exp_e_tradability", ["btc"]),
    "btc_edge": (f"{PY} scripts/decompose_btc_edge.py", ["btc"]),
    "btc_eth": (f"{PY} -m src.experiments.exp_f_cross_asset", ["btc", "eth"]),
    "btc_regimes": (f"{PY} -m src.experiments.exp_g_regimes", ["btc"]),
    "btc_market_making": (f"{PY} -m src.experiments.exp_h_market_making", ["btc"]),
    "lobster": (f"{PY} -m src.experiments.exp_i_lobster", []),
    "attribution": (f"{PY} -m src.experiments.exp_j_attribution" + (" --deeplob" if IN_COLAB else ""), ["btc"]),
    "microprice": (f"{PY} -m src.experiments.exp_k_microprice && {PY} scripts/check_microprice_results.py", ["btc"]),
    "fi2010": (f"{PY} -m src.experiments.fi2010_benchmark --epochs 15 --seeds 0 1 2", ["fi2010"]),
    "fi2010_diagnose": (f"{PY} scripts/diagnose_fi2010_lightgbm.py", ["fi2010"]),
    "optiver_rvp": (f"{PY} -m src.experiments.optiver_rvp", ["optiver_rvp"]),
    "optiver_close": (f"{PY} -m src.experiments.optiver_close", ["optiver_close"]),
    "cpp_trained": (f"{PY} scripts/benchmark_trained_model.py", ["btc"]),
    "nse_models": (f"{PY} -m src.experiments.nse_exp_a_j_models --epochs 10", []),
    "nse_decay_ablation": (f"{PY} -m src.experiments.nse_exp_b_c_decay_ablation", []),
    "nse_execution": (f"{PY} -m src.experiments.nse_exp_d_e_g_execution", []),
    "nse_cross_micro": (f"{PY} -m src.experiments.nse_exp_f_k_cross_micro", []),
    "nse_universe_25": (f"{PY} -m src.experiments.nse_exp_broad_universe", []),
}


def rerun(name):
    """Reruns one experiment if it is switched on in RERUN; otherwise says the saved results are shown."""
    if not RERUN.get(name):
        print(f"Showing saved results. (Set RERUN['{name}'] = True in section 1 to recompute.)")
        return
    cmd, data = STEPS[name]
    if data and IN_COLAB:
        from google.colab import userdata
        for key in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
            try:
                os.environ[key] = userdata.get(key)
            except Exception:
                pass
    for d in data:
        if subprocess.run([sys.executable, "scripts/fetch_datasets.py", d]).returncode != 0:
            print(f"[!] Could not download {d}; showing saved results instead.")
            return
    print(f"[>] {cmd}\n", flush=True)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    print(f"\n[{'+' if proc.wait() == 0 else '!'}] {name} finished with exit code {proc.returncode}")
''')

# =====================================================================================  2 question
md(r"""
## 2. Problem Formulation, Datasets, Preprocessing & Method

### 2.1 Supervised Formulation and Real-World Cost Decomposition

We formulate short-horizon Limit Order Book (LOB) forecasting as a **3-class supervised classification problem** (`FLAT = 0`, `UP = 1`, `DOWN = 2`), plus continuous **mid-price change regression** in Section 10 and Section 14.5.

To connect ML classification accuracy to real-world execution viability, for one round trip (enter, hold for horizon $h$, exit), net edge splits into four parts:

$$\text{net edge} = \underbrace{E[s\,\Delta m_{t\to t+h}]}_{\text{accuracy}} - \underbrace{\big(E[s\,\Delta m_{t\to t+h}] - E[s\,\Delta m_{t+L\to t+L+h}]\big)}_{\text{latency } L} - \underbrace{c_{\text{cross}}}_{\text{spread, depth}} - \underbrace{2f}_{\text{fees}}$$

Here $s=\pm1$ is the predicted direction and $\Delta m$ the change in mid-price. If moves have average size $m$ and accuracy is $p$, gross accuracy edge is $\approx (2p-1)m$, so clearing round-trip cost $C$ requires hit rate:

$$p^* = \tfrac12 + \frac{C}{2m}.$$

When $C > m$, $p^* > 1$ and no classifier alone can turn standalone round-trip trading profitable.

### 2.2 Datasets (~43M Total Snapshots)

| Dataset | Description | Role |
|:---|:---|:---|
| **Indian NSE (Local 5-level)** | **25 instruments across 10 equity sectors + NIFTY_FUT & BANKNIFTY_FUT**, 64 trading days × 22,500 1 s rows/day (**36.0M snapshots** total) | **Largest real-market empirical study**: tick-size predictability law, covariate shift, `DeepLOB5`, ablation & Indian transaction-cost reality |
| **Coinbase BTC-USD & ETH-USD** | 15-level book resampled to 1 s (~1.03M snapshots each, ~12 days Apr 2021, 4 segments) + trade tape | Cross-asset lead-lag, market making, confusion/calibration & C++ compiler benchmark |
| **Nasdaq LOBSTER** | Full order-event books for `INTC`, `MSFT`, `AAPL`, `AMZN`, `GOOG` (100 ms grid) | Large-tick vs small-tick micro-price & queue-position case study |
| **FI-2010** | Standard 5-stock academic LOB benchmark (10 days, 144 features + 40 raw columns) | Faithful replication & history-vs-architecture ablation against Zhang et al. (2019) |
| **Optiver Kaggle (RVP & Close)** | Kaggle 2021 volatility & 2023 closing auction datasets | Time-series leakage case study & closing auction predictability check |
| **Synthetic Ground-Truth** | Planted-signal 10-level discrete-event LOB simulator (`5 × 60,000` events) | Controlled pipeline validation & millisecond latency sweep |

**Tick size** (minimum price increment relative to share price, measured in basis points where `1 bp = 0.01%`) governs queue depth and predictability:
- **Ultra-large / large tick** (`IDEA` `7.41 bp`, `YESBANK` `4.52 bp`, `ITC` `1.88 bp`, Nasdaq `INTC`/`MSFT` `3.3–3.7 bp`): spreads lock at 1 tick and deep queues form, yielding up to **0.841 ROC-AUC (5 s)**.
- **Medium / small / tiny tick** (`RELIANCE` `0.80 bp`, `TCS` `0.43 bp`, `NIFTY_FUT` `0.043 bp`, `BANKNIFTY_FUT` `0.035 bp`, `BTC` `0.0017 bp`): thin Level-1 queues and rapid price jumping.

### 2.3 Preprocessing & Anti-Leakage ML Pipeline

1. **Canonical Cleaning & Session Alignment:**
   - Resample irregular exchange streams onto a uniform 1 s clock (`100 ms` on LOBSTER) via causal last-observation-carried-forward (`searchsorted`).
   - Drop rows with missing top-of-book quotes, non-positive prices/sizes, or crossed books (`Ask_1 <= Bid_1`); reorder columns canonically (`[ap1, aq1, bp1, bq1, ...]`).
   - On NSE, clip strictly to regular continuous trading (`09:15:00–15:29:59 IST`), exclude special 1-hour Muhurat sessions, and reconstruct signed trades from positive cumulative volume changes using the Lee–Ready quote/tick rule.
2. **Whole-Day Isolation & Purged Chronological Splitting:**
   - **Never use random K-Fold CV** on autocorrelated time-series rows.
   - Divide continuous streams chronologically into **70% train / 15% validation / 15% test** with a **purged gap of `k` rows** at each boundary; on NSE, split by **whole trading days (`11 / 2 / 3`)** and reset `OFI`, lags, `VPIN` volume buckets, smoothed forward labels, and `100`-step sequence windows at every overnight boundary.
3. **Train-Only Label Calibration & Transforms:**
   - Smooth mid-price returns with a 5-step trailing window (within day on NSE) and calibrate symmetric threshold `±theta` **on training rows only** (`~1/3 FLAT` target on balanced datasets).
   - **Block PCA on MLOFI:** Fit 3-component mean-centred PCA (`ofi_pca1..3`) on training-split Multi-Level Order Flow Imbalance only (`MLOFI`). On 10-level BTC this reduces levels `6–10` into 3 factors; on 5-level NSE it augments the 5 raw levels with 3 orthogonal rotated directions so axis-aligned trees can split along combined multi-level flow (`80.5%` explained variance on RELIANCE).
   - Fit `StandardScaler` (for Logistic Regression), `DecimalPrecisionNormalizer` (for `LSTM` / `DeepLOB` / `DeepLOB5`), and `VPIN` bucket volume **strictly on training rows**.
4. **Balanced Model Ladder (6 Tiers, Uniform Class Weighting `w_c = N / (3 * N_c)`):**

| Tier | Model | Inputs & Key Idea |
|:---:|:---|:---|
| 0 | Naive Flat & Naive Momentum (`lag k`) | Majority-class & persistence floor |
| 1 | Logistic Regression (`L2`, `StandardScaler`) | Linear baseline testing linear separability |
| 2 | Random Forest (`100` trees, `max_depth=10–12`) | Parallel bagging over bootstrap row/feature subsets |
| 3 | **LightGBM** (`num_leaves=63`, `lr=0.05`, early stopping) | Histogram gradient boosting on 25–27 causal features (+ C++ compiler) |
| 4 | **2-Layer LSTM** | Gated recurrent network on 100-step normalised raw book sequences |
| 5 | **DeepLOB / DeepLOB5 & TLOB** | Spatio-temporal CNN (`1×2`, `1×2`, `1×5/10`) + multi-scale Inception + LSTM; and Dual-Attention Transformer (`TLOB`) with adaptive Bilinear Normalisation (`BiN`) |

5. **Honest Evaluation & Validation:** Report **Macro-F1**, **One-vs-Rest & UP-vs-DOWN ROC-AUC**, **Log-Loss / Brier Calibration**, **5-minute Moving-Block Bootstrap 95% CIs**, **Shuffled-Label Negative Controls**, and **43 automated pytest unit tests**.
""")

# =====================================================================================  3 live demo
md("""
## 3. Live End-to-End Pipeline: Preprocessing, Feature Engineering, Training, Deep Nets & Testing

This section runs live on CPU in ~1 minute so every stage of the ML codebase can be inspected end-to-end:
1. Generate order-book stream, verify data hygiene, create train-only `±theta` 3-class labels, and check chronological purged splits.
2. Run feature extraction, inspect train-only Block PCA variance & zero-NaN guarantees, and verify scaler isolation.
3. Train & test all classical ML models with **Macro-F1**, **ROC-AUC (OvR)**, **Log-Loss**, **MCC**, a **Confusion Matrix**, and **LightGBM Feature Importances**.
4. Instantiate all PyTorch deep neural architectures (`LSTMLOBNet`, `DeepLOB`, `DeepLOB5`, `TLOB`) and verify tensor shapes + forward passes.
5. Run 5-fold **Purged Cross-Validation with Embargo** and the **Shuffled-Label Leakage Check**.
6. Feed test predictions into the **Execution Simulator** (Fantasy mid vs Realistic Aggressive Taker vs VPIN-gated Passive Maker).

### 3.1 Generate Data, Verify Cleaning & Calibrate Chronological Splits
""")

code(r'''
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, log_loss, confusion_matrix
from src.experiments.common import generate_session, prepare_data
from src.models.baselines import NaiveMomentumBaseline, NaiveFlatBaseline, LogisticRegressionBaseline
from src.models.tabular_trees import RandomForestLOB, LightGBMLOB

SEED, k = 42, 20
np.random.seed(SEED)
session = generate_session(SEED, n_events=60000)
y_all = session.labels_dict[k]
data = prepare_data(session)                 # fits normaliser, VPIN bucket & Block PCA on training rows ONLY
idx = data.splits(k)                         # chronological 70/15/15 split with a purge gap of k rows

# Preprocessing sanity checks
lob = session.lob_40
n_crossed = int(np.sum(lob[:, 0] <= lob[:, 2]))
n_nan_lob = int(np.sum(~np.isfinite(lob)))
n_nan_tab = int(np.sum(~np.isfinite(data.tab_feat)))
gap_tr_va = int(idx["val"][0] - idx["train"][-1] - 1)
gap_va_te = int(idx["test"][0] - idx["val"][-1] - 1)

print(f"Snapshots: {len(lob):,} over {session.timestamps_ms[-1] / 1000:.0f} s | Crossed books: {n_crossed} | NaNs (LOB/Tab): {n_nan_lob}/{n_nan_tab}")
print(f"Purged gaps >= k={k}: train->val gap = {gap_tr_va} rows, val->test gap = {gap_va_te} rows")
print(f"Split sizes -> Train: {len(idx['train']):,} | Val: {len(idx['val']):,} | Test: {len(idx['test']):,}")
print("Test label shares (FLAT=0 / UP=1 / DOWN=2):", np.round(np.bincount(y_all[idx["test"]], minlength=3) / len(idx["test"]), 3))
''')

md("""
### 3.2 Inspect Engineered Features & Train-Only Block PCA

Notice below how **Block PCA** fits strictly on `train` rows (`[0 : train_end]`) of the Multi-Level Order Flow (`MLOFI`) block and extracts 3 orthogonal flow factors (`ofi_pca1..3`), explaining most of the cross-level flow variance without mixing units across heterogeneous columns or leaking future statistics.
""")

code(r'''
X = data.tab_feat
X_tr, y_tr = X[idx["train"]], y_all[idx["train"]]
X_va, y_va = X[idx["val"]], y_all[idx["val"]]
X_te, y_te = X[idx["test"]], y_all[idx["test"]]

pca_var = data.pipeline.pca_model.explained_variance_ratio_
print(f"Engineered {len(data.feat_names)} causal features:")
print(" ", ", ".join(data.feat_names))
print(f"Train-only MLOFI Block PCA explained variance ratios: {np.round(pca_var, 3)} (Cumulative: {pca_var.sum():.1%})")
print(f"Deep tensor shape per snapshot: {data.deep_feat.shape} (Decimal-precision normalised on train only)")
''')

md("""
### 3.3 Train & Test the Classical Model Ladder (Metrics, Confusion Matrix & Feature Importance)

Every learned model receives balanced class weights ($w_c = N / (3 N_c)$). Below we compare Accuracy, **Macro-F1**, **One-vs-Rest ROC-AUC**, **Log-Loss**, and **MCC** on identical held-out test timestamps, alongside LightGBM's confusion matrix and gain-based feature importance.
""")

code(r'''
rows = []
def record(name, pred, proba=None):
    auc = roc_auc_score(y_te, proba, multi_class="ovr") if proba is not None else np.nan
    ll = log_loss(y_te, proba, labels=[0, 1, 2]) if proba is not None else np.nan
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_te, pred),
        "Macro-F1": f1_score(y_te, pred, average="macro"),
        "ROC-AUC (OvR)": auc,
        "Log-Loss": ll,
        "MCC": matthews_corrcoef(y_te, pred),
    })

nf = NaiveFlatBaseline()
nm = NaiveMomentumBaseline(k)
record("Naive flat", nf.predict(X_te), nf.predict_proba(X_te))
record("Naive momentum (lag k)", nm.predict(y_all, idx["test"]), nm.predict_proba(y_all, idx["test"]))

lr = LogisticRegressionBaseline(C=1.0); lr.fit(X_tr, y_tr)
record("Logistic regression", lr.predict(X_te), lr.predict_proba(X_te))

rf = RandomForestLOB(n_estimators=100, max_depth=10, random_state=SEED); rf.fit(X_tr, y_tr)
record("Random Forest", rf.predict(X_te), rf.predict_proba(X_te))

lgb = LightGBMLOB(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=200, random_state=SEED)
lgb.fit(X_tr, y_tr, X_va, y_va, early_stopping_rounds=15)
pred_lgb = lgb.predict(X_te)
prob_lgb = lgb.predict_proba(X_te)
record("LightGBM", pred_lgb, prob_lgb)
show(pd.DataFrame(rows))

# Plot Confusion Matrix & Top-10 Feature Importances
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
cm = confusion_matrix(y_te, pred_lgb, labels=[0, 1, 2], normalize="true")
im = axes[0].imshow(cm, cmap="Blues", vmin=0, vmax=0.8)
labels_txt = ["FLAT", "UP", "DOWN"]
axes[0].set_xticks([0, 1, 2]); axes[0].set_yticks([0, 1, 2])
axes[0].set_xticklabels(labels_txt); axes[0].set_yticklabels(labels_txt)
axes[0].set_xlabel("Predicted class"); axes[0].set_ylabel("True class")
axes[0].set_title("LightGBM Row-Normalised Confusion Matrix (Test)")
for i in range(3):
    for j in range(3):
        axes[0].text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                     color="white" if cm[i, j] > 0.45 else "black", fontweight="bold")

imp = pd.Series(lgb.feature_importances_, index=data.feat_names).sort_values(ascending=True).tail(10)
axes[1].barh(imp.index, imp.values / imp.values.sum() * 100.0, color=BLUE)
axes[1].set_xlabel("Relative Gain Importance (%)")
axes[1].set_title("Top-10 Causal Features in LightGBM (Synthetic Live Demo)")
plt.tight_layout(); plt.show()
''')

md("""
Notice two clean ML takeaways right in the live demo:
- `Naive flat` achieves ~45%+ plain accuracy simply by predicting `0` every time, exposing why **Macro-F1** and **ROC-AUC** are essential on imbalanced classes.
- `LightGBM` beats `Logistic regression` on Macro-F1 and Log-Loss because order-book queue thresholds are non-linear.

### 3.4 Deep Sequence Architectures & Live Tensor Verification (`PyTorch`)

Full multi-day training of `LSTM`, `DeepLOB`, `DeepLOB5`, and `TLOB` runs on GPU (`RERUN["synthetic"]`, `RERUN["btc"]`, `RERUN["nse_models"]`). Here we instantiate all four architectures from `src.models`, report exact parameter counts, and verify end-to-end forward passes on real `(batch, 1, 100, cols)` windows from `data.deep_feat`.
""")

code(r'''
import torch
from src.models.recurrent import LSTMLOBNet
from src.models.deeplob import DeepLOB
from src.models.deeplob5 import DeepLOB5
from src.models.tlob import TLOB

# Build 4 real 100-step sequence windows from test indices
win_10 = np.stack([data.deep_feat[t - 99:t + 1] for t in idx["test"][:4]], axis=0)  # (4, 100, 40)
x10 = torch.tensor(win_10, dtype=torch.float32).unsqueeze(1)                       # (4, 1, 100, 40)
x5 = x10[:, :, :, :20]                                                             # (4, 1, 100, 20) for 5-level NSE

deep_specs = [
    ("LSTMLOBNet (10-level BTC / Synth)", LSTMLOBNet(num_features=40, hidden_dim=64, num_layers=2), x10),
    ("DeepLOB (10-level BTC / FI-2010)", DeepLOB(num_classes=3, lstm_hidden=64), x10),
    ("DeepLOB5 (Native 5-level NSE)", DeepLOB5(num_classes=3, lstm_hidden=64), x5),
    ("TLOB Transformer + BiN (10-level)", TLOB(num_features=40, seq_len=100, d_model=64, num_heads=4, num_layers=2), x10),
    ("TLOB Transformer + BiN (5-level NSE)", TLOB(num_features=20, seq_len=100, d_model=64, num_heads=4, num_layers=2), x5),
]
arch_rows = []
with torch.no_grad():
    for name, net, inp in deep_specs:
        net.eval()
        out = net(inp)
        n_params = sum(p.numel() for p in net.parameters())
        arch_rows.append({"Architecture": name, "Input Tensor": str(tuple(inp.shape)),
                          "Output Logits": str(tuple(out.shape)), "Parameters": f"{n_params:,}"})
show(pd.DataFrame(arch_rows))
''')

md("""
### 3.5 Anti-Leakage Checks: Purged Cross-Validation & Shuffled-Target Control
- **Purged CV with embargo** checks that skill holds consistently across chronological folds.
- **Shuffled-label permutation check** breaks the feature–target link; a leak-free pipeline must drop straight to chance (`Macro-F1 ≈ 0.33`, `MCC ≈ 0.00`).
""")

code(r'''
from src.utils.validation_checks import OverfittingDiagnostics

cv = OverfittingDiagnostics.run_purged_cv_evaluation(LightGBMLOB, X, y_all, n_splits=5, horizon_k=k, embargo_pct=0.01)
print(f"Purged CV macro-F1: {cv['test_macro_f1'].mean():.3f} ± {cv['test_macro_f1'].std():.3f} over {len(cv)} folds")
perm = OverfittingDiagnostics.run_permutation_reality_check(LightGBMLOB, X_tr, y_tr, X_te, y_te, n_permutations=3)
print(f"Shuffled-target macro-F1: {perm['permuted_macro_f1']:.3f} (chance ≈ 0.33), MCC {perm['permuted_mcc']:+.3f}",
      "-> FAIL" if perm["leakage_detected"] else "-> PASS: no skill without real labels")
''')

md("""
### 3.6 Simulate Execution on LightGBM's Signals
Three execution regimes on held-out test timestamps (1 ms latency): **Fantasy mid** (unrealistic upper bound), **Aggressive taker** (pays spread + 1.5 bp fee), and **Passive maker with VPIN gate**.
""")

code(r'''
from src.execution.simulator import ExecutionSimulator
from src.execution.strategies import MicrostructureStrategyRunner
from src.execution.metrics import PerformanceMetrics

t_idx = idx["test"]
lob_te, ts_te = session.lob_40[t_idx], session.timestamps_ms[t_idx]
vpin = X[:, data.feat_names.index("vpin")]
vpin_cut = float(np.quantile(vpin[:data.train_end], 0.90))
sim = ExecutionSimulator(initial_cash=1_000_000.0, maker_fee_bps=-0.5, taker_fee_bps=1.5, lot_size=100.0,
                         latency_ms=1.0, timestamps_ms=ts_te)
runner = MicrostructureStrategyRunner(sim)
runs = {"Fantasy (mid, no costs)": runner.run_academic_midprice(pred_lgb, (lob_te[:, 0] + lob_te[:, 2]) / 2),
        "Aggressive taker": runner.run_realistic_aggressive(pred_lgb, lob_te),
        "Passive maker (VPIN gate)": runner.run_dynamic_passive_vpin(pred_lgb, lob_te, vpin[t_idx], vpin_cut)}
out = []
for name, res in runs.items():
    perf = PerformanceMetrics.calculate_metrics(res["equity"], 1_000_000.0, timestamps_ms=ts_te, bar_ms=100.0)
    out.append({"Regime": name, "Net P&L ($)": perf["net_pnl"], "Trades": res["num_trades"],
                "Fees ($)": res.get("total_fees", 0.0), "Slippage ($)": res.get("total_slippage", 0.0)})
print(f"Test window: {(ts_te[-1] - ts_te[0]) / 1000:.0f} s of simulated trading")
show(pd.DataFrame(out), 1)
''')

md("""
The same signals that look strongly profitable at mid-price lose most or all profit once spread crossing and transaction costs are included—motivating the empirical tests across BTC, Nasdaq, and 25 Indian NSE stocks below.
""")

# =====================================================================================  4 synthetic
md("""
## 4. Synthetic Experiments A–D (5 sessions)

The full pipeline over 5 independently generated sessions, including all deep sequence architectures trained on GPU.

### 4.0 Automated Leakage and Robustness Report
""")
code(r'''
rerun("validation")
show(load("overfitting_verification_report.csv"))
''')

md("""
### 4.1 Experiment A: Model Comparison (`k = 20` events)
""")
code(r'''
rerun("synthetic")
a = load("experiment_a_results.csv")
show(a[["model", "accuracy", "macro_f1", "macro_f1_std", "cross_entropy"]])
figure("experiment_a_model_comparison.png", 750)
''')
md("""
LightGBM ranks first (`macro-F1 0.584 ± 0.012`), ahead of DeepLOB (`0.559`) and LSTM (`0.521`).

### 4.2 Experiment B: Horizon Signal Decay
""")
code(r'''
show(load("experiment_b_signal_decay.csv")[["horizon_k", "macro_f1", "macro_f1_std", "momentum_macro_f1", "test_frac_flat"]])
''')
md("""
Skill peaks at `k = 20` events and decays beyond it, matching the `15–60`-event informed bursts planted by the simulator.

### 4.3 Experiment C: Feature-Group Ablation
""")
code(r'''
show(load("experiment_c_ablation_results.csv")[["model_tier", "num_features", "macro_f1", "macro_f1_std"]])
''')
md("""
Level-1 features carry the majority of the signal; short lags add `+0.049` and deeper levels add `+0.029`.

### 4.4 Experiment D: Execution & Microsecond-to-Millisecond Latency Sweep
""")
code(r'''
d = load("experiment_d_execution_strategies.csv")
show(d[["model", "strategy", "net_pnl", "net_pnl_std", "num_trades", "total_fees", "slippage"]], 1)
lat = load("experiment_d_latency_sharpe.csv")[["latency_ms", "pnl_LightGBM", "pnl_DeepLOB"]]
show(lat, 0)
inf = load("experiment_d_inference_latency.csv").rename(columns={"Unnamed: 0": "statistic"})
show(inf)
''')
md("""
- **Costs dominate execution:** crossing the spread + fees removes ~75% (LightGBM) and ~94% (DeepLOB) of fantasy mid-price profit.
- **Accuracy matters ~35× more than raw inference speed:** LightGBM's zero-latency signal advantage (`~$350`) dwarfs DeepLOB's `~2.3 ms` extra GPU inference cost (`~$10`).
""")

# =====================================================================================  5 BTC A-D
md("""
## 5. Real Data: Coinbase BTC-USD (Experiments A–D)

Four contiguous 3-day segments (~1.03M 1-second snapshots total), each split chronologically `70/15/15` with purged horizon gaps.
""")
code(r'''
rerun("btc")
b = load("btc/experiment_a_results.csv")
show(b[["model", "accuracy", "macro_f1", "macro_f1_std"]])
figure("figures_real/btc_model_comparison.png", 750)
''')
md("""
LightGBM achieves **`0.467`** (`47.6%` balanced accuracy) vs `0.413` for Naive Momentum, while `DeepLOB` (`0.401`) and `LSTM` (`0.397`) trail momentum. `TLOB` reaches `0.449`.
""")
code(r'''
print("Signal decay (LightGBM vs momentum):")
show(load("btc/experiment_b_signal_decay.csv")[["horizon_k", "macro_f1", "macro_f1_std", "momentum_macro_f1"]])
print("Feature ablation (k = 20 s):")
show(load("btc/experiment_c_ablation_results.csv")[["model_tier", "num_features", "macro_f1", "macro_f1_std"]])
''')
md("""
- **Rapid horizon decay:** skill peaks at `k = 10 s` and vanishes back to baseline by `100 s`.
- **Top-of-book concentration:** Level-1 snapshot reaches `0.436`, lags add `+0.024`, and deeper levels add `< 0.01`.

**Execution:**
""")
code(r'''
show(load("btc/experiment_d_execution_strategies.csv")[["model", "strategy", "net_pnl", "net_pnl_std", "num_trades", "total_fees"]], 0)
show(load("btc/experiment_d_latency_sharpe.csv")[["latency_ms", "pnl_LightGBM", "pnl_DeepLOB"]], 0)
figure("figures_real/btc_fantasy_vs_real_pnl.png", 650)
''')
md("""
Fees (`~$37.6k`) are `4.7×` larger than fantasy gross profit (`~$8k`), so aggressive trading loses roughly `$2.3` per trade regardless of `0 s` vs `30 s` latency.
""")

# =====================================================================================  6 profit
md("""
## 6. From Prediction to Profit on BTC

### 6.1 Experiment E: Confidence Filtering & Cost-Aware Labels
""")
code(r'''
rerun("btc_tradability")
t = load("btc/tradability/tradability_summary.csv")
cols = ["label", "horizon_k", "coverage", "trades_mean", "mean_mid_move_bps_mean", "mean_gross_bps_mean", "hit_rate_gross_mean"]
show(t[cols].rename(columns=lambda c: c.replace("_mean", "")))
figure("figures_real/btc_tradability.png", 700)
''')
md("""
Confidence filtering raises gross pre-fee edge up to `0.70 bp`, well below Coinbase's `8 bp` round-trip taker fee (`4 bp/side`).

### 6.2 Edge Decomposition (Accuracy vs 1 s Delay vs Spread Walk)
""")
code(r'''
rerun("btc_edge")
e = load("btc/tradability/edge_decomposition.csv")
show(e[["signal", "horizon_k", "coverage", "trades", "edge_at_decision_mid_bps", "delay_cost_bps", "crossing_cost_bps",
        "gross_bps", "mean_abs_move_bps", "hit_rate_at_decision_mid", "breakeven_hit_rate_fee0",
        "breakeven_hit_rate_fee1", "breakeven_hit_rate_fee4"]])
figure("figures_real/accuracy_latency_balance.png", 950)
''')
md("""
1. Top-2% confident predictions double gross mid edge (`2.2 bp`, `72%` directional hit rate) vs all signals (`1.0 bp`, `61%`).
2. Higher-confidence predictions suffer almost `3×` larger 1 s delay decay (`0.75 bp`).

### 6.3 Experiment G: Regime Analysis (VPIN, Spread, Volatility Terciles)
""")
code(r'''
rerun("btc_regimes")
g = load("btc/regimes/regime_tradability.csv")
best = g.sort_values("mean_gross_bps", ascending=False).head(6)
show(best[["signal", "coverage", "regime_var", "tercile", "trades", "mean_gross_bps", "gross_ci_low", "gross_ci_high", "segments_gross_positive"]], 2)
skill = load("btc/regimes/regime_skill.csv")
print("Macro-F1 by regime (direction model):")
show(skill[skill.signal == "direction"][["regime_var", "tercile", "mean", "std"]])
''')

# =====================================================================================  7 ETH
md("""
## 7. Cross-Asset Lead-Lag: Ethereum → Bitcoin (Experiment F)
Causal ETH-USD book features aligned onto BTC timestamps yield a consistent `+0.016` Macro-F1 lift at `10 s` across all 4 segments, fading by `50 s`.
""")
code(r'''
rerun("btc_eth")
show(load("btc/cross_asset/cross_asset_summary.csv"))
figure("figures_real/btc_eth_cross_asset.png", 600)
''')

# =====================================================================================  8 MM + LOBSTER
md("""
## 8. Market Making (H) and Nasdaq Stocks by Tick Size (LOBSTER, Experiment I)

### 8.1 Experiment H: Passive Market Making & Toxic Fill Classification
""")
code(r'''
rerun("btc_market_making")
mm = load("btc/market_making/mm_summary.csv")
show(mm.pivot(index="strategy", columns="fill_rule", values="net_pnl_usd")[["pessimistic", "queue", "optimistic"]].reset_index(), 0)
asel = load("btc/market_making/adverse_selection_summary.csv")
show(asel[["fill_rule", "side", "fill_rate_test", "mean_markout_bp_test", "toxic_auc_test"]])
''')

md("""
### 8.2 Experiment I: Large-Tick vs Small-Tick Nasdaq Stocks (`100 ms` to `30 s`)
""")
code(r'''
rerun("lobster")
dec = load("lobster/lobster_decay.csv")
dec["lift_over_momentum"] = dec.macro_f1 - dec.momentum_macro_f1
order = ["100ms", "500ms", "1s", "5s", "10s", "30s"]
pivot = dec.pivot(index="ticker", columns="horizon", values="lift_over_momentum")[order]
show(pivot.reset_index())

fig, ax = plt.subplots(figsize=(8, 4))
for ticker, color in zip(["INTC", "MSFT", "AAPL", "AMZN", "GOOG"], [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]):
    style = "-" if ticker in ("INTC", "MSFT") else "--"
    ax.plot(order, pivot.loc[ticker], style, color=color, lw=2, marker="o", ms=5,
            label=f"{ticker} ({'large' if style == '-' else 'small'} tick)")
ax.axhline(0, color="#0b0b0b", lw=0.8)
ax.set_xlabel("Prediction horizon"); ax.set_ylabel("Macro-F1 minus momentum")
ax.set_title("Large-tick stocks are predictable at 0.5-10 s; small-tick stocks mostly are not")
ax.legend(frameon=False, fontsize=8); plt.tight_layout(); plt.show()
''')
code(r'''
ls = load("lobster/lobster_adverse_selection.csv")
mk = ls.groupby(["ticker", "fill_rule"]).mean_markout_bp_test.mean().unstack()[["pessimistic", "queue", "optimistic"]]
show(mk.reset_index(), 2)
''')

# =====================================================================================  9 J
md("""
## 9. Statistical Significance & Explainable AI (Experiment J)

Because adjacent 1-second rows have overlapping 20-second forward labels, hypothesis tests that assume row independence are invalid. We therefore use **5-minute Moving-Block Bootstrap 95% Confidence Intervals** (1,000 resamples) to test whether LightGBM's lead over every baseline is statistically real.
""")
code(r'''
rerun("attribution")
show(load("btc/attribution/significance_vs_lightgbm.csv")[["other", "macro_f1_diff", "ci_low", "ci_high", "segments_reference_better"]], 4)
''')
md("""
LightGBM significantly outperforms Logistic Regression, LSTM, and DeepLOB (`+0.058`, positive across all 4 segments).

**Explainable AI (`TreeSHAP` + CNN Saliency):** Exact TreeSHAP values for LightGBM and `|Gradient × Input|` attribution for DeepLOB (`2,000` test windows/segment):
""")
code(r'''
show(load("btc/attribution/shap_lightgbm_groups.csv"), 1)

att = load("btc/attribution/deeplob_attribution.csv")
per_seg = att.groupby("segment").apply(lambda g: pd.Series({
    "level 1": g[g.level == 1].share_total.sum(),
    "each of levels 2-10 (avg)": g[g.level > 1].share_total.sum() / 9,
    "last 10 of 100 snapshots": g.share_last_10_steps.sum(),
    "order sizes (vs prices)": g[g.kind.str.endswith("size")].share_total.sum()}), include_groups=False)
print("DeepLOB attribution shares, mean over 4 segments:")
display((per_seg.mean() * 100).round(1).rename("share %").to_frame())
''')
code(r'''
conf = load("btc/attribution/confusion_matrices.csv")
show(conf[conf.model.isin(["LightGBM", "DeepLOB"])])
''')
md("""
Both TreeSHAP and DeepLOB gradient saliency agree: predictive signal sits at **Level 1** and the **most recent 10 seconds**. LightGBM's confusion matrix spreads errors evenly (`UP 50%`, `DOWN 52%`), whereas DeepLOB misclassifies `41%` of `FLAT` rows as `DOWN`.
""")

# =====================================================================================  10 K
md(r"""
## 10. Micro-Price as Predictor and Regression Target (Experiment K)
Continuous regression evaluation (Ridge & LightGBM Regressor, scored by Pearson Information Coefficient `IC`, $R^2$, and directional hit rate) testing weighted micro-price gap $I_1 \times \text{spread}/2$.
""")
code(r'''
rerun("microprice")
beta = load("microprice/microprice_beta_summary.csv")
show(beta[["dataset", "horizon", "beta_train", "beta_test", "gap_abs_mean_bp", "mid_move_abs_mean_bp"]])

fig, ax = plt.subplots(figsize=(7, 3.6))
for name, color, label in (("LOBSTER-large", BLUE, "Large-tick stocks (INTC, MSFT)"), ("LOBSTER-small", ORANGE, "Small-tick stocks")):
    s = beta[beta.dataset == name]
    ax.plot(s.horizon, s.beta_train, color=color, lw=2, marker="o", ms=5, label=label)
ax.axhline(1, color="#52514e", lw=0.8, ls=":")
ax.set_xlabel("Horizon"); ax.set_ylabel("β (1 = gap fully closed)")
ax.set_title("The mid closes the micro-price gap within ~10 s on large-tick stocks")
ax.legend(frameon=False); plt.tight_layout(); plt.show()
''')
code(r'''
m = load("microprice/microprice_summary.csv")
pick = m[m.model.isin(["Micro-price rule (beta x gap)", "Momentum (past k-step return)", "LightGBM", "LightGBM, shuffled-target control"])]
pick = pick.assign(model=pick.model + " (trained on " + pick.train_target + ")")
ic = pick.pivot_table(index="model", columns=["dataset", "horizon"], values="ic_pearson")
ic = ic[[("LOBSTER-large", "5s"), ("LOBSTER-large", "10s"), ("LOBSTER-small", "1s"), ("BTC", "10s")]]
ic.columns = ["Large-tick 5 s", "Large-tick 10 s", "Small-tick 1 s", "BTC 10 s"]
print("Information coefficient with the future mid change, test period:")
show(ic.reset_index())
''')

# =====================================================================================  11 FI-2010
md("""
## 11. FI-2010 Academic Benchmark: History Beats Architecture at Short Horizons
Giving simple Logistic Regression lagged changes from the same 100-step window flips short-horizon accuracy above DeepLOB (`86.1%` vs `83.1%` at `k = 10`). DeepLOB wins clearly at longer horizons (`k = 50, 100`).
""")
code(r'''
rerun("fi2010")
fi = load("fi2010/fi2010_setup2_summary.csv")
acc = fi.pivot(index="model", columns="horizon_k", values="accuracy_mean").round(3)
order = ["Naive Flat", "Logistic Regression", "LightGBM (144 feats)", "Logistic Regression + history",
         "LightGBM (144 feats + history)", "DeepLOB (ours)", "DeepLOB (published)"]
print("Test accuracy by horizon k (mean over 3 seeds):")
show(acc.loc[order].reset_index())
figure("figures_real/fi2010_comparison.png", 750)
''')
code(r'''
rerun("fi2010_diagnose")
show(load("fi2010/diagnose_lightgbm_history.csv"))
''')

# =====================================================================================  12 Optiver
md("""
## 12. Optiver Kaggle Datasets: Leakage Case Study & Closing Auction Regression
""")
code(r'''
rerun("optiver_rvp")
show(load("optiver_rvp/rvp_summary.csv"), 4)
rerun("optiver_close")
show(load("optiver_close/close_holdout_results.csv"), 4)
''')

# =====================================================================================  13 C++
md("""
## 13. Inference Latency: Python-to-C++ Tree Compiler
Compiling the trained 327-tree BTC LightGBM model into standalone C++ (`scripts/lgbm_to_cpp.py`) delivers bit-identical scores (`max_abs_diff = 0`) at **`10.5 µs`** median single-row latency (`230–540×` faster than DeepLOB on GPU).
""")
code(r'''
rerun("cpp_trained")
tm = load("cpp_trained_model_benchmark.csv")
show(tm[["num_trees", "total_leaves", "num_features", "cpp_p50_ns", "cpp_p99_ns", "cpp_max_abs_diff_vs_lightgbm", "cpp_argmax_agreement"]], 1)
show(load("cpp_latency_benchmarks.csv"), 1)
''')

# =====================================================================================  14 NSE
md(r"""
## 14. Indian NSE 5-Level Order-Book Study (64 Trading Days, 25 Instruments, 36.0M Snapshots)

Our largest real-market empirical study covers **64 regular trading days** (`2026-08-07` to `2026-11-09`, `09:15–15:29 IST`, `22,500` 1-second snapshots/day = **1.44M rows per instrument**, **36.0M total rows** across 25 NSE symbols).

### 14.1 Whole-Day Preprocessing, Feed Caveats & Tick-Size Spectrum
- **Strict daily isolation:** Split each 64-day stream into 4 contiguous 16-day segments (`11 train / 2 val / 3 test` whole trading days). All rolling features, lags, OFI diffs, VPIN buckets, smoothed targets, and 100-step deep windows reset cleanly at `09:15` each morning.
- **Native 5-level deep architecture (`DeepLOB5`):** Standard DeepLOB expects 10 price levels (`40` cols). Rather than padding 20 fake zeros, `DeepLOB5` replaces Conv Block 3's `(1, 10)` spatial kernel with a native `(1, 5)` kernel for NSE's 20-column input.
- **Realistic Indian market parameters:** `NIFTY_FUT` lot size is `65` units (`BANKNIFTY_FUT = 30`). Futures statutory execution cost reflects post-Union-Budget-2026 STT (`0.05%` sell-side from Apr 2026 $\rightarrow$ `2.8 bp/side`, `5.6 bp` round-trip); intraday equity costs `2.0 bp/side` (`4.0 bp` RT).
- **Sampled Snapshot Feed Caveat:** The live Kite WebSocket delivers ~1 snapshot/second (`1.49M` raw ticks on `NIFTY_FUT`, unchanged in `41%` of 1 s slots) rather than full order-by-order ITCH events, so sub-second queue transitions are unseen and NIFTY's `1.77 bp` median spread reflects this sampled feed.
""")
code(r'''
nse_eda = load("nse/nse_dataset_summary.csv")
show(nse_eda[["symbol", "asset", "tick_class", "days", "rows_1s", "median_mid_inr", "tick_inr",
              "tick_bp", "one_tick_pct", "median_spread_bp", "median_bq1", "corr_imb1_20s"]], 3)
figure("nse/figures/nse_eda_tick_spectrum.png", 900)
''')

md(r"""
### 14.2 Scaling Across 25 Indian NSE Instruments (36.0M Rows): The Tick-Size Predictability Law

Testing `LightGBM` uniformly across **23 liquid NSE equities spanning 10 sectors + `NIFTY_FUT` & `BANKNIFTY_FUT`** uncovers a clean empirical law:
- **Log-linear predictability law:** $\log_{10}(\text{tick\_bp})$ correlates strongly with 1-tick spread lock-in ($r = +0.86$), Level-1 imbalance correlation ($r = +0.88$), and **5-second `LightGBM` OvR ROC-AUC** ($r = +0.80$, Spearman $\rho = +0.77, p < 10^{-5}$), rising from `0.724` (`NIFTY_FUT`) to `0.841` (`YESBANK`).
- **The ₹250 NSE tick-band natural experiment:** At nearly identical share prices, crossing NSE's ₹250 price boundary shifts the minimum tick `5×` (`₹0.01` $\to$ `₹0.05`): **`ONGC` (`₹233`, `0.43 bp` tick) scores `0.755` 5 s AUC vs `ITC` (`₹265`, `1.88 bp` tick) scoring `0.808` 5 s AUC** with `4×` deeper L1 queues.
""")
code(r'''
rerun("nse_universe_25")
u25_corr = load("nse/universe_25/nse_universe_25_correlations.csv")
show(u25_corr, 4)

u25 = load("nse/universe_25/nse_universe_25_summary.csv")
print(f"Loaded all {len(u25)} NSE instruments ({u25['rows'].sum():,} total 1-second snapshots):")
show(u25[["symbol", "sector", "mid_inr", "tick_bp", "spread_1tick_pct", "bq1_med",
          "f1_5s", "auc_ovr_5s", "f1_20s", "auc_ud_20s", "hit_rate_20s", "gross_bps", "net_bps"]], 3)
figure("nse/figures/nse_universe_25.png", 900)
''')

md("""
### 14.3 Model Ladder (`LightGBM` vs `LSTM`, `DeepLOB5`, `TLOB`) & Distribution Shift (`k = 20 s`)

- **Core result:** Across all 3 deep-evaluated benchmarks (`RELIANCE`, `NIFTY_FUT`, `IDEA`), **`LightGBM` on 25 engineered features ranks #1** (`0.445`, `0.443`, `0.484` Macro-F1; statistically tying `TLOB 0.440` on `RELIANCE`, where the 5-minute moving-block bootstrap 95% CI `[-0.0001, +0.0200]` crosses zero).
- **Why static deep models crashed on `NIFTY_FUT` Segment 3 (Covariate Shift!):** Even after validation-only collapse retry recovered validation Macro-F1 (`0.322`), `NIFTY` rallied **`+10.7%`** from training days (`~₹22,540`) to test days (`₹24,949`). Static decimal-scaled `LSTM` (`0.184`) and `DeepLOB5` (`0.197`) shifted out-of-distribution on test (`ROC-AUC ≈ 0.50`), whereas per-window **Bilinear Normalisation (`BiN`)** in `TLOB` (`0.439`) and **scale-free ratios** in `LightGBM` (`0.432`) suffered zero degradation.
""")
code(r'''
rerun("nse_models")
for sym in ["RELIANCE", "NIFTY_FUT", "IDEA"]:
    print(f"=== {sym} (k = 20 s, 4 segments) ===")
    ml = load(f"nse/models/{sym}/model_ladder_summary.csv")
    show(ml[["model", "accuracy", "macro_f1", "macro_f1_std", "roc_auc_ovr", "roc_auc_ud", "log_loss", "mcc"]], 3)
    sig = load(f"nse/models/{sym}/significance_vs_lightgbm.csv")
    show(sig[["baseline", "lgbm_minus_baseline", "ci_lo", "ci_hi", "segments_lgbm_better"]], 4)
figure("nse/figures/nse_model_ladder.png", 900)
''')

md("""
### 14.4 Horizon Decay, 5-Stage Ablation (Overfitting Extra Features), TreeSHAP & DeepLOB5 Saliency

- **5-Stage Ablation (`A -> E`):** Stages `A -> D` (`1` to `25` causal features) improve Macro-F1 consistently (`L2–5` deeper book + PCA adds `+0.029` on thin-queue `NIFTY_FUT` vs `+0.002` on thick `IDEA`). Adding 5 noisy NSE full-book / order-count columns in **Stage E** (`30` features) **lowers test Macro-F1 on 6 of 8 symbols** (`-0.023` on `ITC`, `-0.020` on `RELIANCE`), even though `TreeSHAP` allocates them `~25%` of split attribution—a textbook demonstration of trees overfitting noisy high-cardinality training drift.
- **Explainability convergence:** `DeepLOB5` gradient saliency places **`63.3%` (`NIFTY_FUT`), `64.3%` (`RELIANCE`), and `70.0%` (`IDEA`)** of its attribution weight on the **last 10 seconds** and concentrates on Level 1.
""")
code(r'''
rerun("nse_decay_ablation")
dec_nse = load("nse/decay_ablation/nse_decay_summary.csv")
piv_f1 = dec_nse.pivot(index="symbol", columns="horizon_s", values="lgbm_core_f1")
piv_mom = dec_nse.pivot(index="symbol", columns="horizon_s", values="momentum_f1")
print("LightGBM (25 core features) Macro-F1 across 8 core NSE symbols & 5-100 s horizons:")
show(piv_f1.reset_index(), 3)

abl = load("nse/decay_ablation/nse_ablation_summary.csv")
print("5-Stage Feature Ablation (k = 20 s Macro-F1):")
show(abl.pivot(index="symbol", columns="feature_set", values="macro_f1").reset_index(), 3)

print("TreeSHAP Feature Group Shares & DeepLOB5 Attribution (RELIANCE / NIFTY_FUT / IDEA):")
for sym in ["RELIANCE", "NIFTY_FUT", "IDEA"]:
    sg = load(f"nse/models/{sym}/shap_groups.csv").assign(symbol=sym)
    dl = load(f"nse/models/{sym}/deeplob5_attribution.csv")[["level_1_share", "recent_10s_share", "recent_20s_share"]].mean()
    print(f"  {sym:10s} | DeepLOB5 L1={dl['level_1_share']*100:.1f}%, Last10s={dl['recent_10s_share']*100:.1f}% | Top SHAP group: {sg.iloc[0]['group']} ({sg.iloc[0]['shap_share']*100:.1f}%)")

figure("nse/figures/nse_signal_decay.png", 900)
figure("nse/figures/nse_ablation_shap.png", 900)
''')

md("""
### 14.5 Cross-Asset Bank Lead-Lag & Continuous Micro-Price Regression on NSE

- **Bank peer lead-lag (`ICICIBANK -> HDFCBANK`):** Adding contemporaneous `ICICIBANK` features lifts `HDFCBANK` Macro-F1 by **`+0.019`** at `5–10 s` across `4/4` segments (comparable in scale to `ETH -> BTC` `+0.016`).
- **Continuous micro-price regression (`Ridge` & `LightGBM`):** Micro-price mean reversion $\\beta_{\\text{test}}$ scales systematically with tick size (`IDEA 0.56` at 50 s vs `NIFTY_FUT 0.02`), while `ICICIBANK`'s 50 s test slope flips sign across individual 16-day segments (unstable rather than zero).
""")
code(r'''
rerun("nse_cross_micro")
ca = load("nse/cross_asset/nse_cross_asset_summary.csv")
show(ca, 4)
mp_beta = load("nse/microprice/nse_microprice_beta_summary.csv")
show(mp_beta.pivot(index=["symbol", "tick_class"], columns="horizon_s", values="beta_test").reset_index(), 3)
figure("nse/figures/nse_cross_asset_microprice.png", 900)
''')

md(r"""
### 14.6 Execution, Edge Decomposition & Post-Budget-2026 Indian Cost Reality

- **0 of 25 NSE instruments are profitable pre-fee as aggressive takers:** Top-2% confidence predictions achieve **`72.9%` (`NIFTY_FUT`)** and **`73.4%` (`BANKNIFTY_FUT`)** directional hit rate (`+0.89` to `+1.21 bp` gross mid move), yet spread crossing (`1.97–2.00 bp` on futures, up to `9.01 bp` on `IDEA`) turns gross edge negative (`-0.97 bp` at best) *before* paying `4.0–5.6 bp` round-trip statutory taxes/fees.
- **Passive Maker (`NIFTY_FUT`, `lot = 65`):** VPIN-gated passive quoting is positive *before fees* in `4/4` segments (`+₹2.58L / window`, `+₹160.5 / fill`), illustrating both the value of queue-position capture and how statutory transaction charges (`2.8 bp/side` $\approx ₹426$/fill) still decide net feasibility.
""")
code(r'''
rerun("nse_execution")
ed_nse = load("nse/execution/nse_edge_decomposition.csv")
show(ed_nse, 3)
ex_nse = load("nse/execution/nse_exp_d_summary.csv")
show(ex_nse[["symbol", "strategy", "net_pnl_inr", "num_trades", "pnl_per_trade_inr", "total_fees_inr", "slippage_inr"]], 1)
figure("nse/figures/nse_edge_decomposition.png", 900)
''')

# =====================================================================================  15 discussion
md("""
## 15. Discussion: Balancing Accuracy, Latency and Cost

| Step | Observed Time / Scale |
|:---|:---|
| Coinbase BTC / Indian NSE clock | `1 s` snapshot resolution |
| Nasdaq LOBSTER clock | `100 ms` resolution |
| DeepLOB / DeepLOB5, single sample (GPU) | `2.4–7.1 ms` |
| LightGBM, single sample (Python) | `128–190 µs` |
| **Trained LightGBM compiled to C++** | **`10.5 µs` median (`0` score difference)** |
| Closed-form micro-price rule | `< 1 µs` (a few arithmetic ops) |

**Core Machine Learning Takeaways:**
1. **Honest evaluation hygiene changes headline conclusions:** Random K-fold (`57.0%` vs `35.2%` chronological on Optiver RVP), overnight lookahead, and improper independence tests inflate apparent performance; chronological purged/day-isolated splits and shuffled-label controls keep evaluation honest.
2. **Input representation beats parameter count at short horizons:** Across Synthetic, BTC, FI-2010 (`k=10, 20`), and Indian NSE benchmarks, well-engineered causal features + `LightGBM` match or outperform `200k`-parameter deep networks (`LSTM`, `DeepLOB`, `DeepLOB5`), while running `230–540×` faster once compiled to C++.
3. **Feature selection & distribution shift matter in production:** Adding 5 noisy NSE full-book columns caused overfitting on `6/8` stocks, and a `+10.7%` NIFTY price rally broke static neural scaling while scale-free ratios (`LightGBM`) and adaptive per-window Bilinear Normalisation (`TLOB`) held steady.
4. **Market structure governs predictability, while friction governs deployment:** Across 25 Indian instruments (`36.0M` rows), coarser relative tick sizes yield up to `0.84` ROC-AUC (`r = +0.80`), yet `0/25` instruments beat round-trip spread + statutory cost as standalone takers—making directional ML signals best suited for **execution timing & passive adverse-selection gating** rather than standalone high-frequency taker trading.
""")

# =====================================================================================  16 conclusion
md(r"""
## 16. Limitations, Reproducibility & Conclusion

**Limitations**
- **Sampled 1 Hz feeds:** Coinbase BTC and Kite NSE provide `~1 s` snapshot grids (`~41%` unchanged seconds on `NIFTY_FUT`), masking sub-second queue arrivals and inflating passive simulation bounds.
- **Unobserved queue position:** Passive maker simulations report pessimistic (trade-through), queue-proxy, and optimistic bounds rather than true exchange queue priority.
- **Zero permanent market impact:** Simulated orders do not deplete subsequent snapshots.
- **Narrow NIFTY spread caveat:** NIFTY's `1.77 bp` (`~₹4`) median spread on the sampled Kite snapshot feed is wider than raw tick-by-tick exchange prints, reinforcing rather than altering the negative taker conclusion.

**Reproducibility**
- Automated verification suite: `python -m pytest tests -q` (**43 passing tests** covering causality, whole-day boundary isolation, `DeepLOB5`, C++ compilation, and execution accounting).
- Rebuild report PDF: `report/build.sh` $\rightarrow$ [LOB_ML_Project_Report.pdf](../report/LOB_ML_Project_Report.pdf).
- Rebuild presentation slides: `cd report/slides && node build_deck.js` $\rightarrow$ [LOB_ML_Project_Presentation.pptx](../report/LOB_ML_Project_Presentation.pptx).
- Rebuild this executed notebook: `python scripts/build_final_notebook.py`.
""")


def build(run: bool) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "colab": {"provenance": [], "toc_visible": True}}
    if run:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=1800, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT / "submission")}}).execute()
    OUT.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"[+] {OUT} ({len(CELLS)} cells{', executed' if run else ''})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-run", action="store_true", help="build without executing")
    build(run=not parser.parse_args().no_run)
