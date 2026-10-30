# High-Frequency Limit Order Book Alpha & Microstructure Execution Engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aaravsin-1/ML_Project_2026/blob/master/notebooks/colab_run_all.ipynb)
[![Tests](https://img.shields.io/badge/pytest-24%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](setup.py)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%20CUDA-orange.svg)](src/models/)
[![C++20](https://img.shields.io/badge/C%2B%2B-20%20SIMD-darkblue.svg)](cpp/)

A research framework for short-horizon price prediction from Limit Order Book (LOB) data and for testing whether predictions survive realistic execution. It covers microstructure features (OFI, multi-level OFI, VPIN, micro-price), a model hierarchy from naive baselines to DeepLOB and TLOB, an execution simulator with spread crossing, book walking, fees, latency and passive fills, and a C++20 inference-latency benchmark.

> **Important: all results below are on synthetic data.** The [generator](src/data/synthetic_generator.py) plants a hidden "informed trader" state that persists for 15–60 events and skews order flow. Any predictability the models find is the predictability built into that generator. The results show the pipeline working end to end and let you compare methods under controlled conditions; they say nothing about real markets. Real-data experiments (Coinbase BTC, FI-2010, two Optiver Kaggle competitions) are implemented and tested; see [Real-Data Experiments](#real-data-experiments-in-progress). Their results will be added once the runs finish.

**Headline findings (5 independent sessions, mean ± std across seeds):**
- LightGBM is the strongest model (macro-F1 **0.584 ± 0.012** at k = 20). It beats DeepLOB in 5/5 seeds (+0.025 on average) and logistic regression in 5/5 (+0.021). Its edge over Random Forest (+0.006) is within seed noise.
- An honest naive baseline (repeat the last *realized* k-event move) already reaches **0.519 ± 0.013**, so the learned models add about 0.04–0.07 macro-F1 on top of it.
- Most of the signal is in levels 1–3 of the book. Cumulative L3 depth imbalance alone accounts for about half of LightGBM's split gain.
- With an aggressive taker, LightGBM's signals net more than DeepLOB's in 4/5 seeds. **The gap is already there at zero latency, so it comes from signal quality, not speed.** Both models lose roughly the same ~$390–400 when latency rises from 0 to 100 ms. At their measured inference latencies (0.14 ms vs 2.4 ms), the latency difference is worth only about $10.

---

## Table of Contents
1. [Core Architecture](#core-architecture)
2. [Mathematical Foundations & Microstructure Signals](#mathematical-foundations--microstructure-signals)
3. [Model Hierarchy (Tiers 0–5)](#model-hierarchy-tiers-05)
4. [Methodology: Leakage Controls](#methodology-leakage-controls)
5. [Empirical Results](#empirical-results)
6. [Real-Data Experiments (in progress)](#real-data-experiments-in-progress)
7. [C++20 Latency Benchmark](#c20-latency-benchmark)
8. [Repository Structure](#repository-structure)
9. [Running on Google Colab](#running-on-google-colab)
10. [Installation & Local Quickstart](#installation--local-quickstart)
11. [Automated Test Suite](#automated-test-suite)
12. [Known Limitations](#known-limitations)
13. [References](#references)

---

## Core Architecture

```
                 +----------------------------------------------+
                 | Synthetic LOB generator (10 levels, ~10 ms)  |
                 | snapshots + trade tape + timestamps          |
                 +----------------------------------------------+
                                       |
                 +---------------------+----------------------+
                 v                                            v
   +-----------------------------+            +------------------------------+
   | Microstructure features     |            | Raw LOB tensor (40 features) |
   | OFI, MLOFI + PCA, VPIN,     |            | decimal-precision normalized |
   | micro-price, imbalances,    |            +------------------------------+
   | slopes, spread in ticks     |                           |
   +-----------------------------+                           |
                 |                                           |
                 v                                           v
   +-----------------------------+            +------------------------------+
   | Tabular models              |            | Sequence models (100 steps)  |
   | momentum, logistic, RF,     |            | LSTM, DeepLOB, TLOB          |
   | LightGBM                    |            +------------------------------+
   +-----------------------------+                           |
                 +---------------------+---------------------+
                                       | predictions (FLAT / UP / DOWN)
                                       v
                 +----------------------------------------------+
                 | Execution simulator                          |
                 | spread crossing, book walking, taker fees,   |
                 | maker rebates, trade-through passive fills,  |
                 | wall-clock latency, VPIN gating              |
                 +----------------------------------------------+
```

---

## Mathematical Foundations & Microstructure Signals

### 1. Limit Order Book Dynamics
At event time $t$ the top $M$ levels of the book are
$$LOB(t) = \left\{ (p_m^a(t), q_m^a(t), p_m^b(t), q_m^b(t)) \right\}_{m=1}^M$$
with spread $S(t) = p_1^a(t) - p_1^b(t)$ and mid-price $P_{mid}(t) = \tfrac{1}{2}(p_1^a(t) + p_1^b(t))$.

### 2. Order Flow Imbalance (OFI & MLOFI)
Following **Cont, Kukanov, and Stoikov (2014)**, the bid-side demand change is
$$\Delta W(t) = \begin{cases} q_1^b(t), & p_1^b(t) > p_1^b(t-1) \\ q_1^b(t) - q_1^b(t-1), & p_1^b(t) = p_1^b(t-1) \\ -q_1^b(t-1), & p_1^b(t) < p_1^b(t-1) \end{cases}$$
the ask-side supply change $\Delta V(t)$ is defined symmetrically, and $OFI(t) = \Delta W(t) - \Delta V(t)$.
**Multi-Level OFI** applies the same formula at each of the $M = 10$ levels. PCA (fit on the training period only) compresses the 10-level vector into 3 components.

### 3. Volume-Synchronized Probability of Informed Trading (VPIN)
Following **Easley, López de Prado, and O'Hara (2012)**, trades are grouped into buckets of constant volume $V$, and
$$VPIN = \frac{\sum_{\tau=1}^N |V_\tau^B - V_\tau^S|}{N \times V}.$$
VPIN is computed from the generator's trade tape and mapped onto each snapshot using only buckets completed at or before that snapshot's timestamp.

### 4. The Micro-Price Estimator
Following **Stoikov (2018)**, with $I(t) = \frac{q_1^b - q_1^a}{q_1^b + q_1^a}$,
$$P_{micro}(t) = \frac{q_1^b p_1^a + q_1^a p_1^b}{q_1^b + q_1^a} = P_{mid}(t) + \frac{I(t)}{2} S(t).$$
The deviation $P_{micro} - P_{mid}$ is used as a feature.

### 5. Spread Feature
The spread enters as an integer number of ticks. An earlier version used relative spread $S/P_{mid}$. With a fixed one-tick spread this is just a monotone function of the price level, and float32 rounding gave it two distinct values. LightGBM ranked that artifact as its top feature, so it was removed.

### 6. Normalization
- **Decimal precision normalization** for the sequence models: $(P - P_0)/P_0$ and $\log(1 + V/V_0)$, with $P_0, V_0$ taken from the first 100 training snapshots.
- **Causal rolling z-score** and **Bilinear Normalization (BiN)** (Tran et al., 2020) are implemented in [normalizers.py](src/data/normalizers.py). BiN is used inside TLOB.

---

## Model Hierarchy (Tiers 0–5)

| Tier | Model | Purpose |
|:---:|:---|:---|
| **Tier 0** | Naïve Momentum, Naïve Flat | Momentum repeats the label at $t-k$, the latest move actually known at decision time. Flat predicts the majority class. |
| **Tier 1** | Logistic Regression | Linear baseline on standardized features. |
| **Tier 2** | Random Forest | Non-linear baseline. |
| **Tier 3** | LightGBM | Gradient-boosted trees with early stopping on validation. |
| **Tier 4** | LSTM | Recurrent net over 100-snapshot windows. |
| **Tier 5** | DeepLOB, TLOB | CNN + Inception + LSTM, and a dual-attention transformer with BiN. |

Every learned model uses **balanced class weights**, so all of them optimize the same objective and macro-F1 is comparable across tiers.

---

## Methodology: Leakage Controls

- **Train-only fitting.** PCA loadings and normalizer scales are estimated on the training period. All features are causal: row $t$ uses snapshots $\le t$.
- **Purged splits.** The split is 70/15/15 chronological. The last $k$ rows of train and validation are dropped because their labels look into the next split. So are the final $k$ rows of the session, whose labels are undefined. See [purged_split_indices](src/data/lob_dataset.py).
- **Same test rows for every model.** Sequence windows end exactly at the tabular test rows, taking earlier rows as causal context.
- **Labels.** Smoothed mid-price ($5$-event trailing mean) at $t+k$ vs $t$, with a ±1.5 bp threshold for UP/DOWN.
- **Seeds.** Every experiment runs on 5 independently generated sessions (seeds 0–4, 60,000 events each, about 10 minutes of simulated trading). Tables report the mean ± sample std across seeds.

The [verification script](scripts/verify_robustness_and_leakage.py) adds purged 5-fold CV with a 1% embargo, a target-permutation check, the momentum-baseline comparison, and a transfer test to a small-tick regime. Each PASS/FAIL is computed from the value against the stated criterion ([report](artifacts/overfitting_verification_report.csv)). The permutation check only detects label/row misalignment. It cannot prove that features don't look ahead; that is what the train-only fitting and the causality test in [test_features.py](tests/test_features.py) are for.

---

## Empirical Results

Reproduce with `python scripts/run_all_experiments.py` (about 8 minutes on a CUDA GPU). Per-seed rows are in `artifacts/*_per_seed.csv`.

### Experiment A: Model Hierarchy (k = 20)

| Tier | Model | Accuracy | Macro-F1 | Cross-Entropy |
|:---:|:---|:---:|:---:|:---:|
| 0 | Naïve Flat | 0.698 ± 0.013 | 0.274 ± 0.003 | 4.812 |
| 0 | Naïve Momentum (lag k) | 0.645 ± 0.012 | 0.519 ± 0.013 | 0.961 |
| 1 | Logistic Regression | 0.587 ± 0.011 | 0.564 ± 0.008 | 0.812 |
| 2 | Random Forest | 0.609 ± 0.008 | 0.578 ± 0.008 | 0.757 |
| 3 | **LightGBM** | 0.620 ± 0.014 | **0.584 ± 0.012** | **0.743** |
| 4 | LSTM | 0.552 ± 0.049 | 0.521 ± 0.023 | 0.859 |
| 5 | DeepLOB | 0.597 ± 0.025 | 0.559 ± 0.010 | 0.823 |
| 5 | TLOB | 0.535 ± 0.026 | 0.492 ± 0.018 | 0.965 |

Accuracy is below the Naïve Flat rate for every learned model because balanced class weights trade majority-class accuracy for recall on UP/DOWN. Macro-F1 is the comparison metric. On this data, sequence models do not beat the tabular models built on hand-crafted microstructure features. The deep nets also train for at most 10 epochs, so they may be under-trained.

![Experiment A](artifacts/experiment_a_model_comparison.png)

### Experiment B: Signal Decay Across Horizons (LightGBM)

| Horizon k | Macro-F1 | Naïve Momentum Macro-F1 | Accuracy | Test FLAT share |
|:---:|:---:|:---:|:---:|:---:|
| 10 | 0.521 ± 0.011 | 0.438 | 0.715 | 88.3% |
| 20 | **0.581 ± 0.011** | 0.519 | 0.614 | 69.8% |
| 50 | 0.534 ± 0.012 | 0.467 | 0.542 | 38.7% |
| 100 | 0.453 ± 0.013 | 0.413 | 0.508 | 22.1% |

Skill peaks at k = 20. At k = 10 most moves fall under the 1.5 bp threshold (88% FLAT), which makes UP/DOWN hard to recall. Beyond k = 20 the signal decays. The informed-flow bursts the generator plants last 15–60 events, which matches this shape.

![Experiment B](artifacts/experiment_b_signal_decay.png)

### Experiment C: Feature Ablation (LightGBM, k = 20)

| Feature set | # Features | Macro-F1 |
|:---|:---:|:---:|
| A: Spread only | 1 | 0.160 ± 0.102 |
| B: L1 snapshot (spread, L1 imbalance, L1 OFI, micro-price deviation) | 4 | 0.492 ± 0.011 |
| C: B + lags of L1 signals | 13 | 0.541 ± 0.006 |
| D: C + levels 2–3 (L3 imbalance, L2/L3 OFI) | 16 | 0.570 ± 0.008 |
| E: All features (adds L4–10 depth, slopes, MLOFI PCA, VPIN) | 27 | 0.581 ± 0.011 |

The spread is always one tick in this regime, so Model A carries no information. With balanced weights it collapses to a single class, and which class varies by seed (hence the large std). Short history adds +0.049 on top of the L1 snapshot, and levels 2–3 add +0.029. The rest of the book adds +0.011.

**Feature attribution** (share of total LightGBM split gain on the full model, mean over seeds): `imbalance_l3` 49.3%, `vpin` 7.5%, `ofi_pca1` 6.3%, `slope_bid` 5.3%, `slope_ask` 4.9%, `imbalance_l5` 4.1%, `ofi_lag_1` 4.0%, `imbalance_l1` 3.4%. Gain importance is a model-internal measure. It is not a causal attribution, and it splits credit arbitrarily between correlated features.

![Experiment C](artifacts/experiment_c_ablation_progression.png)

### Experiment D: Execution and Latency

The test-set signals of the trained LightGBM and DeepLOB models from Experiment A are executed over each seed's test window (8,980 events, about 89 s of simulated trading). Settings: $1M capital, 100-share lots, ±500 share inventory cap, 1.5 bp taker fee, 0.5 bp maker rebate, and 1 ms latency unless stated. Latency is applied in wall-clock time: an order reaches the book at the first snapshot whose timestamp is at least $t$ + latency.

| Model | Regime | Net PnL ($) | Trades | PnL / trade ($) | Fees ($) | Slippage vs decision mid ($) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| LightGBM | Academic mid-price (fantasy) | 1,858 ± 136 | 658 | 2.86 | 0 | 0 |
| LightGBM | Aggressive taker | **457 ± 257** | 658 | 0.74 | 982 | 422 |
| LightGBM | Passive maker (VPIN gate) | 2 ± 31 | 4 | — | −2 | 0 |
| DeepLOB | Academic mid-price (fantasy) | 1,782 ± 142 | 797 | 2.25 | 0 | 0 |
| DeepLOB | Aggressive taker | 104 ± 160 | 797 | 0.14 | 1,186 | 501 |
| DeepLOB | Passive maker (VPIN gate) | 7 ± 173 | 11 | — | −5 | 0 |

- Crossing the spread and paying fees removes about 75% (LightGBM) and 94% (DeepLOB) of the fantasy PnL. DeepLOB's signals flip more often, so it trades more and pays more friction.
- The **passive maker** joins the touch and counts as filled only when the market trades *through* its price within 50 events (it assumes it sits at the back of the queue). It cancels when the signal flips or VPIN exceeds its 90th training-period percentile. Under this conservative rule almost nothing fills, and what does fill is adversely selected by construction. An earlier version assumed every passive order filled instantly at the touch, which is why it used to beat even the fantasy regime.
- **Sharpe ratios are not reported** in these tables. Annualizing about 90 seconds of returns gives numbers in the hundreds or thousands that mean nothing. The CSVs still contain a Sharpe computed on 100 ms bars for relative comparison.

**Latency sweep** (aggressive taker, added latency, net PnL mean ± std):

| Added latency | LightGBM | DeepLOB |
|:---:|:---:|:---:|
| 0 ms | 508 ± 255 | 155 ± 156 |
| 1 ms | 457 ± 257 | 104 ± 160 |
| 2 ms | 454 ± 258 | 100 ± 160 |
| 10 ms | 427 ± 267 | 71 ± 173 |
| 25 ms | 369 ± 265 | 14 ± 181 |
| 100 ms | 119 ± 278 | −247 ± 210 |

**Measured single-sample inference latency** (median, Python): LightGBM booster **0.14 ms** (CPU), DeepLOB **2.4 ms** (CUDA, batch of 1).

Interpretation: both models lose about the same amount per millisecond of latency (≈$390–400 from 0 to 100 ms). LightGBM's advantage over DeepLOB is already present at zero latency, so it comes from better signals, not faster inference. At the measured latencies, DeepLOB's extra ~2.3 ms costs about $10 (≈$109 at 0.14 ms vs ≈$99 at 2.4 ms), which is small next to the ~$350 signal-quality gap. The drop from 0 to 0.05 ms for both models is a discretization effect: at zero latency the order fills against the very snapshot the decision was made on, while any positive latency means the next event. In seed 0 DeepLOB edges out LightGBM. The seed-to-seed spread in PnL is large compared with the means, so treat the absolute numbers as indicative.

![Experiment D Equity](artifacts/experiment_d_equity_curves.png)
![Experiment D Latency](artifacts/experiment_d_latency_vs_profit.png)

---

## Real-Data Experiments (in progress)

Four real datasets are wired in and tested. Their full runs are heavy, so they are meant to run on [Colab](#running-on-google-colab). Download them with `python scripts/fetch_datasets.py`.

| Dataset | Source | What the experiment does |
|:---|:---|:---|
| **Coinbase BTC-USD**, 15-level book, 1 s snapshots, ~12 days (Apr 2021) | [Kaggle, CC0](https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data) | Reruns Experiments A–D on 4 contiguous ~3-day segments. Labels use a per-horizon threshold calibrated on training rows only (one third FLAT). VPIN comes from the real signed buy/sell volume. Execution uses Coinbase's top-tier fees (0 bp maker, 4 bp taker), 0.1 BTC lots, and a latency sweep in whole seconds (the book is only observed once a second). `python scripts/run_all_experiments.py --source btc` |
| **FI-2010**, Setup 2 | [Kaggle mirror](https://www.kaggle.com/datasets/ulfricirons/fi-2010) of Ntakaris et al. (2018) | Trains on days 1–7 and tests on days 8–10. Compares our DeepLOB, LightGBM, logistic regression and baselines with the published DeepLOB numbers (84.47% accuracy / 83.40% weighted-F1 at k = 10). `python -m src.experiments.fi2010_benchmark` |
| **Optiver Realized Volatility** (Kaggle 2021) | [competition](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction) | Book/trade volatility features. Recovers the shuffled time order with the [1st-place](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/writeups/nyanp-1st-place-solution-nearest-neighbors) tick-size + t-SNE trick, then validates forward in time. Compares the winner's nearest-neighbor features as published (neighbors drawn from all time windows, including future ones) with a past-only version, plus shuffled K-fold CV for contrast. `python -m src.experiments.optiver_rvp` |
| **Optiver Trading at the Close** (Kaggle 2023) | [competition](https://www.kaggle.com/competitions/optiver-trading-at-the-close) | Auction-imbalance features, including the [1st-place](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution) seconds-group and cross-sectional rank features. Uses the winner's split (first 400 days train, last 81 days holdout) and compares a model trained once with one retrained every 12 days. `python -m src.experiments.optiver_close` |

The Optiver data is competition data: accept each competition's rules on kaggle.com once before downloading.

---

## C++20 Latency Benchmark

[main_benchmark.cpp](cpp/src/main_benchmark.cpp) times feature extraction plus tree-ensemble inference plus a simulated order on 100,000 snapshots from a random-walk book.

**Scope.** The ensemble is **randomly generated** (50 trees × 3 classes, depth 6, 10 real features plus 6 zero inputs). It is not the trained LightGBM model, and C++ features are not checked against the Python pipeline. The numbers measure the cost of evaluating a tree ensemble of that size. They are not the latency of the model evaluated above, and they do not include network or market-data parsing ("wire-to-order").

Latest run ([artifacts/cpp_latency_benchmarks.csv](artifacts/cpp_latency_benchmarks.csv), GCC 15.2, `-O3 -march=native`):

| Stage | P50 | P99 |
|:---|:---:|:---:|
| Feature extraction | below timer resolution | 0.2 µs |
| Tree inference | 1.8 µs | 6.1 µs |
| End-to-end loop | 1.9 µs | 6.3 µs (P99.9 15.1 µs) |

Throughput: about 399k snapshots/s, including timing overhead. The [tutorial](cpp/tutorial/) and [optimization_comparison.cpp](cpp/src/optimization_comparison.cpp) compare AoS vs SoA layouts and branchless tree evaluation on the same kind of synthetic ensemble.

---

## Repository Structure

```
ML_Project_2026/
├── README.md
├── cpp/                                    # C++20 latency benchmark & SIMD tutorial
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── synthetic_generator.py          # Simulated LOB, trade tape, timestamps, labels
│   │   ├── lob_dataset.py                  # Purged splits, indexed sequence loaders
│   │   ├── normalizers.py                  # Decimal, causal rolling z, BiN
│   │   ├── purged_cv.py                    # Purged & embargoed CV
│   │   └── fi2010_loader.py / lobster_loader.py / crypto_loader.py
│   ├── microstructure/                     # OFI, VPIN, micro-price, feature pipeline
│   ├── models/                             # Baselines, RF, LightGBM, LSTM, DeepLOB, TLOB, trainer
│   ├── execution/                          # Simulator, strategies, performance metrics
│   ├── experiments/
│   │   ├── common.py                       # Sessions (synthetic seeds / BTC segments), train-only fit, purged splits
│   │   ├── exp_a ... exp_d                 # Experiments A-D
│   │   ├── fi2010_benchmark.py             # FI-2010 Setup 2 vs published DeepLOB
│   │   ├── optiver_rvp.py                  # Optiver Realized Volatility (forward CV, NN features)
│   │   └── optiver_close.py                # Optiver Trading at the Close (date split, retraining)
│   └── utils/                              # Metrics, validation checks, plots
├── scripts/
│   ├── run_all_experiments.py              # Multi-seed runner (aggregated + per-seed artifacts)
│   ├── verify_robustness_and_leakage.py    # Validation report
│   ├── fetch_datasets.py                   # Downloads the real datasets from Kaggle
│   ├── build_and_execute_notebook.py       # Builds and runs the submission notebook
│   └── download_or_generate_data.py
├── notebooks/colab_run_all.ipynb           # One-click Colab runner for everything
├── submission/                             # Executed notebook + submission.csv
├── tests/
└── artifacts/                              # CSVs and figures
```

---

## Running on Google Colab

The heavy runs (BTC, FI-2010, Optiver) need a GPU and several hours, so they are set up to run on Colab: open [notebooks/colab_run_all.ipynb](https://colab.research.google.com/github/aaravsin-1/ML_Project_2026/blob/master/notebooks/colab_run_all.ipynb) and run the cells top to bottom.

1. **Runtime → Change runtime type → T4 GPU.**
2. Add your Kaggle API key as two Colab **Secrets** (🔑 panel): `KAGGLE_USERNAME` and `KAGGLE_KEY`. The values are in the `kaggle.json` from *kaggle.com → Settings → API → Create New Token*.
3. For the Optiver data, accept both competitions' rules on kaggle.com with the same account.
4. If the repo is private, also add a `GITHUB_TOKEN` secret with read access.

Results and checkpoints go to Google Drive (`MyDrive/ML_Project_2026/artifacts`). Every long run checkpoints after each segment or horizon, so after a disconnect you just rerun the setup cells and the experiment cell. The last cell downloads the results as a zip to unzip into `artifacts/` locally.

**Moving a run between machines.** Every long experiment checkpoints its progress (BTC per segment, FI-2010 per horizon, Optiver Close per training stage, Optiver RVP per fold and model). `python scripts/sync_checkpoints.py --loop 600` pushes the artifacts folder (checkpoints and results, never raw data) to the repo's `run-state` branch every 10 minutes. The notebook's "Resume from earlier progress" cell runs `python scripts/sync_checkpoints.py --restore` to pull them in. So if the laptop dies mid-run, Colab continues from the last sync, and the reverse works too. Raw data is always re-downloaded from Kaggle by `scripts/fetch_datasets.py` (Kaggle's rules do not allow redistributing competition data).

Paths can be redirected anywhere with the `LOB_DATA_DIR` and `LOB_ARTIFACTS_DIR` environment variables.

---

## Installation & Local Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# Synthetic experiments, 5 seeds (add --run-cpp to also build and run the C++ benchmark)
python scripts/run_all_experiments.py --seeds 0 1 2 3 4 --events 60000 --epochs 10

# Real data (heavy: GPU recommended)
python scripts/fetch_datasets.py                # needs Kaggle API credentials
python scripts/run_all_experiments.py --source btc --segments 4
python -m src.experiments.fi2010_benchmark
python -m src.experiments.optiver_close
python -m src.experiments.optiver_rvp

# Validation report
python scripts/verify_robustness_and_leakage.py --seed 42 --events 60000

# C++ benchmark on its own
g++ -std=c++20 -O3 -march=native -ffast-math -I cpp/include cpp/src/main_benchmark.cpp -o cpp/lob_benchmark.exe
./cpp/lob_benchmark.exe
```

---

## Automated Test Suite

```bash
python -m pytest tests -v
```

24 tests cover:
- feature formulas (OFI, micro-price, VPIN, BiN);
- the pipeline fitting only on training rows, real VPIN, and the integer tick spread;
- the purge gap;
- the momentum baseline using only realized labels;
- model forward passes;
- spread crossing and book walking;
- trade-through passive fills;
- timestamp-based latency;
- time-bar Sharpe;
- purged CV and the permutation check;
- real-data plumbing without downloads: train-only label thresholds, Coinbase format conversion, causality of the Optiver Close features, and past-only neighbors and forward folds for Optiver RVP.

---

## Known Limitations

- **Synthetic results only, so far.** Conclusions above are about the generator. The real-data experiments are implemented, but their results are not in yet.
- **Short test windows.** About 90 s of trading per seed means PnL is noisy and annualized risk metrics are not meaningful.
- **Simple strategy.** One lot per signal up to a ±500 inventory cap, with no position sizing by confidence and no exit logic.
- **Simple book impact.** The book does not react to our own orders beyond walking the snapshot.
- **Limited tuning.** Deep models get at most 10 epochs and little hyperparameter search.
- **Separate C++ path.** The C++ benchmark runs a random ensemble, not the trained model.

---

## References

1. **Cont, R., Kukanov, A., & Stoikov, S. (2014).** *The Price Impact of Order Book Events.* Journal of Financial Econometrics, 12(1), 47–88.
2. **Easley, D., López de Prado, M. M., & O'Hara, M. (2012).** *Flow Toxicity and Liquidity in a High-Frequency World.* The Review of Financial Studies, 25(5), 1457–1493.
3. **Stoikov, S. (2018).** *The Micro-Price: A High-Frequency Estimator of Future Prices.* Quantitative Finance, 18(12), 1959–1966.
4. **Zhang, Z., Zohren, S., & Roberts, S. (2019).** *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books.* IEEE Transactions on Signal Processing, 67(11), 3001–3012.
5. **López de Prado, M. (2018).** *Advances in Financial Machine Learning.* Wiley.
6. **Tran, D. T., Kanniainen, J., Gabbouj, M., & Iosifidis, A. (2021).** *Data Normalization for Bilinear Structures in High-Frequency Financial Time-Series.* ICPR 2020.
