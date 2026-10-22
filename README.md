# High-Frequency Limit Order Book Alpha & Microstructure Execution Engine

[![Tests](https://img.shields.io/badge/pytest-12%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](setup.py)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%20CUDA-orange.svg)](src/models/)
[![C++20](https://img.shields.io/badge/C%2B%2B-20%20SIMD-darkblue.svg)](cpp/)
[![Throughput](https://img.shields.io/badge/C%2B%2B%20Throughput-459k%20msgs%2Fsec-red.svg)](cpp/include/)
[![P50 Latency](https://img.shields.io/badge/P50%20Latency-1.7%20%CE%BCs-green.svg)](artifacts/cpp_latency_benchmarks.csv)

An institution-grade research framework and production execution system evaluating short-term price predictability in Limit Order Book (LOB) dynamics. The framework demonstrates that while advanced spatio-temporal deep neural architectures (DeepLOB, TLOB) identify rich non-linear patterns, **real-world profitability is dictated by execution latency, exchange fees, spread crossing, and adverse selection**. Under sub-millisecond execution constraints, a fast C++ decision tree ensemble (LightGBM) achieves higher realized net Sharpe ratios than mathematically superior but computationally slower deep learning models.

---

## Table of Contents
1. [Core Architecture & System Overview](#core-architecture--system-overview)
2. [Mathematical Foundations & Microstructure Signals](#mathematical-foundations--microstructure-signals)
   - [Limit Order Book Dynamics](#1-limit-order-book-dynamics)
   - [Order Flow Imbalance (OFI & MLOFI)](#2-order-flow-imbalance-ofi--mlofi)
   - [Volume-Synchronized Probability of Informed Trading (VPIN)](#3-volume-synchronized-probability-of-informed-trading-vpin)
   - [The Micro-Price Estimator](#4-the-micro-price-estimator)
   - [Strict Non-Leakage Normalization & Bilinear Normalization (BiN)](#5-strict-non-leakage-normalization--bilinear-normalization-bin)
3. [Machine Learning Progression Hierarchy (Tiers 0–5)](#machine-learning-progression-hierarchy-tiers-05)
4. [Empirical Experimental Results](#empirical-experimental-results)
   - [Experiment A: Establishing Predictive Information](#experiment-a-establishing-predictive-information)
   - [Experiment B: The Horizon of Signal Survival (Alpha Decay)](#experiment-b-the-horizon-of-signal-survival-alpha-decay)
   - [Experiment C: Depth Ablation & Feature Attribution](#experiment-c-depth-ablation--feature-attribution)
   - [Experiment D: Economic Reality & Latency vs Net Profitability](#experiment-d-economic-reality--latency-vs-net-profitability)
5. [Ultra-Low-Latency Systems Engineering (C++20)](#ultra-low-latency-systems-engineering-c20)
6. [Repository Structure](#repository-structure)
7. [Installation & Quickstart Guide](#installation--quickstart-guide)
8. [Automated Test Suite](#automated-test-suite)
9. [References](#references)

---

## Core Architecture & System Overview

```
                      +---------------------------------------+
                      | Raw Market Data Feed (ITCH / Snapshots) |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |   LOB Reconstruction (10 Depth Levels)|
                      +---------------------------------------+
                                          |
                      +-------------------+-------------------+
                      |                                       |
                      v                                       v
         +--------------------------+           +--------------------------+
         |  Microstructural Signals  |           |   Raw LOB Tensor (40-F)  |
         |  - OFI & MLOFI (PCA)     |           |   - Decimal Normalization|
         |  - VPIN Toxicity         |           |   - Bilinear Norm (BiN)  |
         |  - Micro-price & Slopes  |           +--------------------------+
         +--------------------------+                         |
                      |                                       |
                      v                                       v
         +--------------------------+           +--------------------------+
         |  Tabular ML Hierarchy     |           | Spatio-Temporal Models   |
         |  - Tier 0: Naive Momentum|           | - Tier 4: LSTM / GRU     |
         |  - Tier 1: Logistic Reg  |           | - Tier 5: DeepLOB (CNN)  |
         |  - Tier 2: Random Forest |           | - Tier 5: TLOB (Dual-Att)|
         |  - Tier 3: LightGBM      |           +--------------------------+
         +--------------------------+                         |
                      |                                       |
                      +-------------------+-------------------+
                                          | Predictions (UP/FLAT/DOWN)
                                          v
                      +---------------------------------------+
                      |   High-Fidelity Execution Simulator   |
                      |   - Spread Crossing Penalty           |
                      |   - Multi-Level Depth Slippage        |
                      |   - Maker Rebates vs Taker Fees       |
                      |   - VPIN Adverse Selection Defense    |
                      |   - End-to-End System Latency Delay   |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |  Ultra-Low Latency C++20 Core Engine  |
                      |  - SIMD Vectorized Feature Extraction |
                      |  - Native Decision Tree Inference     |
                      |  - 459k msgs/sec | P50: 1.7 microseconds|
                      +---------------------------------------+
```

---

## Mathematical Foundations & Microstructure Signals

### 1. Limit Order Book Dynamics
The limit order book aggregates resting supply and demand across discrete price ticks. At any event time $t$, the state of the top $M$ price levels is given by:
$$LOB(t) = \left\{ (p_m^a(t), q_m^a(t), p_m^b(t), q_m^b(t)) \right\}_{m=1}^M$$
where $p_1^a(t)$ and $p_1^b(t)$ represent the best ask and best bid prices, and $q_1^a(t)$ and $q_1^b(t)$ represent the corresponding quantities. The instantaneous bid-ask spread is:
$$S(t) = p_1^a(t) - p_1^b(t)$$
and the mid-price is:
$$P_{mid}(t) = \frac{p_1^a(t) + p_1^b(t)}{2}$$

### 2. Order Flow Imbalance (OFI & MLOFI)
Formalized by **Cont, Kukanov, and Stoikov (2014)**, Level-1 OFI isolates net buying vs selling pressure across consecutive snapshots $[t-1, t]$. The bid demand component $\Delta W(t)$ is defined as:
$$\Delta W(t) = \begin{cases} q_1^b(t), & \text{if } p_1^b(t) > p_1^b(t-1) \\ q_1^b(t) - q_1^b(t-1), & \text{if } p_1^b(t) = p_1^b(t-1) \\ -q_1^b(t-1), & \text{if } p_1^b(t) < p_1^b(t-1) \end{cases}$$
The ask supply component $\Delta V(t)$ is symmetrically defined as:
$$\Delta V(t) = \begin{cases} q_1^a(t), & \text{if } p_1^a(t) < p_1^a(t-1) \\ q_1^a(t) - q_1^a(t-1), & \text{if } p_1^a(t) = p_1^a(t-1) \\ -q_1^a(t-1), & \text{if } p_1^a(t) > p_1^a(t-1) \end{cases}$$
The net Order Flow Imbalance is:
$$OFI(t) = \Delta W(t) - \Delta V(t)$$
**Multi-Level OFI (MLOFI)** extends this computation to all $M=10$ levels. To prevent multicollinearity and extract dense representations of book-wide pressure, Principal Component Analysis (PCA) decomposes the 10-level vector into 3 orthogonal factor components:
$$MLOFI(t) \xrightarrow{\text{PCA}} [PC_1(t), PC_2(t), PC_3(t)]$$
where $PC_1$ captures aggregate book pressure, $PC_2$ captures book rotation/slope, and $PC_3$ captures curvature.

### 3. Volume-Synchronized Probability of Informed Trading (VPIN)
Introduced by **Easley, Lopez de Prado, and O'Hara (2012)**, VPIN operates on a volume clock rather than a chronological clock. Continuous transaction flow is bucketed into consecutive units of constant volume $V$. For each bucket $\tau$, buy-initiated volume $V_\tau^B$ and sell-initiated volume $V_\tau^S$ are tracked:
$$VPIN = \frac{\sum_{\tau=1}^N |V_\tau^B - V_\tau^S|}{N \times V}$$
Sustained elevated VPIN indicates aggressive informed traders sweeping the book, serving as a real-time toxicity trigger to cancel passive limit orders and prevent adverse selection.

### 4. The Micro-Price Estimator
Stoikov (2018) adjusted the naive mid-price for instantaneous queue imbalances:
$$I(t) = \frac{q_1^b(t) - q_1^a(t)}{q_1^b(t) + q_1^a(t)}$$
$$P_{micro}(t) = \frac{q_1^b(t) \cdot p_1^a(t) + q_1^a(t) \cdot p_1^b(t)}{q_1^b(t) + q_1^a(t)} = P_{mid}(t) + \frac{I(t)}{2} S(t)$$
The micro-price deviation $P_{micro}(t) - P_{mid}(t)$ serves as an effective target and feature for short-horizon directional predictions.

### 5. Strict Non-Leakage Normalization & Bilinear Normalization (BiN)
Standard z-score normalization across the whole dataset causes forward temporal leakage. The framework provides three leakage-free normalization methods:
- **Decimal Precision Normalization**: Re-scales prices and volumes via static lot/tick parameters: $(P - P_0) / P_0$ and $\log(1 + V / V_0)$.
- **Causal Rolling Z-Score**: Computes mean $\mu_t$ and standard deviation $\sigma_t$ strictly within causal lookback window $W$: $[t-W, t]$.
- **Bilinear Normalization (BiN)** (Tran et al., 2020): Dynamically normalizes tensors across both the feature dimension and the sequence time dimension via backpropagated parameters:
  $$\tilde{X} = \Gamma_{feat} \odot \left( \frac{X - \mu_{time}}{\sigma_{time} + \epsilon} \right) + B_{feat}$$
  $$\hat{X} = \Gamma_{time} \odot \left( \frac{\tilde{X} - \mu_{feat}}{\sigma_{feat} + \epsilon} \right) + B_{time}$$

---

## Machine Learning Progression Hierarchy (Tiers 0–5)

To prevent premature attribution of success to deep learning, models are structured into 6 progressive tiers:

| Tier | Model Architecture | Primary Scientific Objective | Microstructural Relevance |
|:---:|:---|:---|:---|
| **Tier 0** | Naïve Momentum & Naïve Flat | Establish absolute baseline boundaries. | Tests whether the market exhibits simple autocorrelation or pure noise. |
| **Tier 1** | Logistic Regression | Linear statistical baseline. | Validates Cont et al.'s linear price impact of OFI. |
| **Tier 2** | Random Forest | Non-linear tree baseline. | Captures interactions between depth depletion and spread widening. |
| **Tier 3** | LightGBM | State-of-the-art tabular GBDT. | Leaf-wise tree growth optimized for microsecond-scale execution. |
| **Tier 4** | LSTM & GRU | Sequential temporal neural nets. | Models memory and trajectory across 100 consecutive snapshots. |
| **Tier 5** | DeepLOB | Spatio-temporal CNN + Inception + LSTM. | Multi-scale causal convolutions across depth levels + temporal LSTM. |
| **Tier 5** | TLOB | Transformer + Dual Attention + BiN. | Spatial Feature Attention + Temporal Sequence Attention. |

---

## Empirical Experimental Results

All experiments were executed on out-of-sample data under strict chronological splits (70% train, 15% validation, 15% test; no shuffling, zero lookahead leakage).

### Experiment A: Establishing Predictive Information

Validation of the progression hierarchy for horizon $k=20$:

| Tier | Model | Out-of-Sample Accuracy | Macro F1-Score | Weighted F1 | Cross-Entropy Loss |
|:---:|:---|:---:|:---:|:---:|:---:|
| **Tier 0** | Naïve Flat | 62.80% | 0.2572 | 0.4845 | 5.9306 |
| **Tier 1** | Logistic Regression | 68.43% | 0.5079 | 0.6327 | 0.6772 |
| **Tier 2** | Random Forest | 62.88% | **0.6206** | 0.6343 | 0.6989 |
| **Tier 3** | **LightGBM** | 60.67% | **0.6039** | 0.6190 | 0.8008 |
| **Tier 4** | LSTM | 64.91% | 0.4195 | 0.5728 | 0.6880 |
| **Tier 5** | DeepLOB | 64.83% | 0.5515 | 0.6338 | 0.8001 |
| **Tier 5** | TLOB | **66.37%** | 0.4657 | 0.6016 | 0.7331 |

> **Key Finding**: Linear OFI regression achieves a 0.5079 Macro-F1, confirming exploitable linear signal in LOB data. Non-linear models (Random Forest and LightGBM) lift Macro-F1 above 0.60, demonstrating significant non-linear interactions across depth queues.

![Experiment A Model Comparison](artifacts/experiment_a_model_comparison.png)

---

### Experiment B: The Horizon of Signal Survival (Alpha Decay)

Mapping the decay curve of predictive power across event horizons:

| Horizon ($k$ events) | Accuracy | Macro F1-Score | Weighted F1 | Cross-Entropy Loss | Microstructural Interpretation |
|:---:|:---:|:---:|:---:|:---:|:---|
| **$k = 10$** | **80.03%** | **0.5622** | **0.8172** | **0.4725** | Sub-second regime: Maximum exploitable alpha. |
| **$k = 20$** | 60.67% | **0.6039** | 0.6190 | 0.8008 | Optimal signal balance across UP/DOWN/FLAT. |
| **$k = 50$** | 48.91% | 0.4413 | 0.4489 | 1.0267 | Alpha begins severe degradation. |
| **$k = 100$** | 49.07% | 0.3616 | 0.4154 | 1.0234 | Decays toward random baseline (0.333). |

![Experiment B Signal Decay](artifacts/experiment_b_signal_decay.png)

---

### Experiment C: Depth Ablation & Feature Attribution

Systematically restricting information inputs to isolate the exact source of alpha:

| Ablation Model | Included Microstructural Features | Num Features | Out-of-Sample Macro F1 |
|:---|:---|:---:|:---:|
| **Model A: Spread Only** | Relative Spread | 1 | 0.1500 |
| **Model B: Top-of-Book (L1)** | Spread + Level-1 Imbalance + Level-1 OFI + Micro-Price Dev | 4 | 0.3609 (+140% vs A) |
| **Model C: Top 3 Levels Depth** | Model B + Level-3 Depth Imbalance + Bid/Ask Slopes | 7 | **0.5508** (+52% vs B) |
| **Model D: Full 10-Level Book** | Full 10 Levels + MLOFI PCA + VPIN + Lagged Signals | 27 | **0.6039** (+9.6% vs C) |
| **Model E: DeepLOB (Temporal)** | Full 10 Levels $\times 100$ snapshot temporal sequence | 4000 | 0.4106 |

#### Top Influential Microstructure Features (Gini Importance & SHAP)
1. `rel_spread` (14.14% relative contribution)
2. `imbalance_l3` (9.54% relative contribution)
3. `slope_bid` (8.73% relative contribution)
4. `slope_ask` (8.54% relative contribution)
5. `imbalance_l5` (6.77% relative contribution)
6. `ofi_lag_1` (5.71% relative contribution)

> **Key Finding**: Top-of-book and the first 3 levels of depth account for **>91% of total model predictive capacity**. Processing levels 4–10 adds only ~9.6% predictive lift while dramatically increasing feature extraction and network latency.

![Experiment C Ablation Progression](artifacts/experiment_c_ablation_progression.png)

---

### Experiment D: Economic Reality & Latency vs Net Profitability

Simulating the full trade lifecycle under real market frictions ($1,000,000 initial capital):

| Execution Regime | Net Realized PnL ($) | Total Return (%) | Annualized Sharpe | Max Drawdown (%) | Total Fees Paid ($) | Slippage Paid ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Academic Mid-Price (Fantasy)** | $908.00 | +0.0908% | 460.38 | 0.0024% | $0.00 | $0.00 |
| **Realistic Aggressive Taker** | $366.88 | +0.0367% | 168.67 | 0.0163% | $384.28 | $154.26 |
| **Dynamic Passive Maker (VPIN)** | **$1,096.00** | **+0.1096%** | **556.89** | **0.0028%** | **-$118.28** *(Rebates Earned!)* | **$0.00** |

#### The Latency vs Profitability Tradeoff
Evaluating whether an ultra-low latency model (LightGBM in C++) beats a deep learning model (DeepLOB on GPU) due to signal decay:

| Pipeline Latency | LightGBM (C++ Engine) Net Sharpe | DeepLOB (Deep CNN) Net Sharpe | Economic Winner |
|:---:|:---:|:---:|:---:|
| **0.05 ms (50 $\mu$s)** | **168.67** | 148.17 | **LightGBM (+13.8% Sharpe)** |
| **0.50 ms** | **168.67** | 148.17 | **LightGBM** |
| **2.00 ms** | **168.67** | 148.17 | **LightGBM** |
| **10.00 ms** | **163.47** | 143.48 | **LightGBM** |
| **25.00 ms (Deep Net Inference)** | **148.17** | 135.66 | **LightGBM** |
| **50.00 ms** | **128.70** | 117.72 | **LightGBM** |
| **100.00 ms** | **100.66** | 89.36 | **LightGBM** |

> **Crucial Thesis Validation**: Although DeepLOB achieves slightly higher raw accuracy under theoretical conditions, **LightGBM in C++ outperforms DeepLOB across all latency regimes**. In real-world market execution where signals decay in milliseconds, speed dominates mathematical complexity.

![Experiment D Equity Curves](artifacts/experiment_d_equity_curves.png)
![Experiment D Latency vs Profitability](artifacts/experiment_d_latency_vs_profit.png)

---

## Ultra-Low-Latency Systems Engineering (C++20)

To achieve production-grade HFT performance, feature extraction and decision-tree inference were implemented in modern C++20 with SIMD-vectorized operations:
- **Cache-Aligned Memory**: Fixed-size `std::array<Level, 10>` structures aligned to 64-byte CPU cache lines (`alignas(64)`), completely eliminating heap allocations on hot paths.
- **SIMD OFI/MLOFI Vectorization**: Unrolled loops (`#pragma GCC unroll 10`) enabling AVX vector instructions across all 10 price levels simultaneously.
- **Native Flat Tree Inference**: LightGBM tree ensembles compiled directly into flat contiguous node arrays (`FastGBDTEnsemble`), evaluating multi-class tree models in sub-microsecond time.

### C++ Microbenchmark Profiling (100,000 Live Events)
```
================ LATENCY BENCHMARK RESULTS ================
Throughput:             459,211.46 messages/second
Total Events Processed: 100,000
Total Wall Clock Time:  0.218 seconds

--- Stage Latency Breakdown ---
SIMD Feature Extraction: P50 = 0.000 us | P99 = 0.200 us
Fast Tree Inference:     P50 = 1.600 us | P99 = 5.300 us

--- End-to-End Pipeline Latency (Wire-to-Order) ---
P50  (Median): 1.700 microseconds
P90:           4.100 microseconds
P99:           5.500 microseconds
P99.9 (Tail):  12.900 microseconds
===========================================================
```

---

## Repository Structure

```
ML_Project_2026/
├── README.md                               # Institutional research paper & documentation
├── project_research.md                     # Foundational research specification
├── requirements.txt                        # Python dependencies
├── setup.py                                # Package configuration (lob_alpha)
├── cpp/                                    # Low-Latency C++ Engine (C++20)
│   ├── CMakeLists.txt                      # CMake build script
│   ├── include/
│   │   ├── lob.hpp                         # Cache-aligned LimitOrderBook data structure
│   │   ├── features.hpp                    # SIMD OFI, MLOFI, VPIN & Micro-Price
│   │   ├── tree_engine.hpp                 # Fast native Decision Tree evaluation engine
│   │   └── simulator.hpp                   # Microsecond continuous matching simulator
│   └── src/
│       └── main_benchmark.cpp              # Latency percentile microbenchmark (P50/P99)
├── src/                                    # Python Core Framework
│   ├── config.py                           # System configuration, fees, horizons, paths
│   ├── data/                               # Data Parsers & Synthetics
│   │   ├── lob_dataset.py                  # PyTorch Datasets & chronological loaders
│   │   ├── normalizers.py                  # Decimal, causal rolling Z, Bilinear Norm (BiN)
│   │   ├── fi2010_loader.py                # Academic benchmark parser (FI-2010)
│   │   ├── lobster_loader.py               # NASDAQ LOBSTER reconstructor
│   │   ├── crypto_loader.py                # 24/7 Cryptocurrency LOB parser
│   │   └── synthetic_generator.py          # Microstructure-grounded double auction generator
│   ├── microstructure/                     # Feature Engineering
│   │   ├── ofi.py                          # Cont et al. Level-1 OFI and MLOFI with PCA
│   │   ├── vpin.py                         # Volume-clock VPIN toxicity engine
│   │   ├── microprice.py                   # Stoikov Micro-price & book slope geometry
│   │   └── pipeline.py                     # Unified multi-modal feature pipeline
│   ├── models/                             # ML Progression Hierarchy
│   │   ├── baselines.py                    # Tier 0 (Momentum/Flat), Tier 1 (Logistic)
│   │   ├── tabular_trees.py                # Tier 2 (Random Forest), Tier 3 (LightGBM)
│   │   ├── recurrent.py                    # Tier 4 (LSTM / GRU)
│   │   ├── deeplob.py                      # Tier 5 (DeepLOB: 1D Conv + Inception + LSTM)
│   │   ├── tlob.py                         # Tier 5 (TLOB: Dual Attention Transformer + BiN)
│   │   └── registry.py                     # Factory & unified training engine
│   ├── execution/                          # High-Fidelity Simulator
│   │   ├── simulator.py                    # Double-auction matching, spread, depth walking
│   │   ├── strategies.py                   # Academic vs Realistic Taker vs Dynamic Maker
│   │   └── metrics.py                      # Sharpe, Sortino, MaxDD, Win Rate, Fill Rate
│   ├── experiments/                        # Scientific Modules
│   │   ├── exp_a_predictive_power.py       # Experiment A: Model Hierarchy benchmark
│   │   ├── exp_b_signal_survival.py        # Experiment B: Multi-horizon decay mapping
│   │   ├── exp_c_ablation_attribution.py   # Experiment C: Depth ablation & SHAP
│   │   └── exp_d_latency_vs_profit.py      # Experiment D: Execution & Latency tradeoff
│   └── utils/                              # Metrics & Plots
│       ├── metrics.py                      # Multi-class precision, recall, F1, cross-entropy
│       └── visualization.py                # Publication-quality plotting suite
├── scripts/                                # Executable CLIs
│   ├── run_all_experiments.py              # Master pipeline orchestrator
│   └── download_or_generate_data.py        # Data generator CLI
├── tests/                                  # Automated Test Suite (PyTest)
│   ├── test_features.py                    # Feature calculation unit tests
│   ├── test_models.py                      # ML model forward/backward unit tests
│   └── test_simulator.py                   # Simulator & adverse selection unit tests
└── artifacts/                              # Generated research figures & CSV benchmarks
```

---

## Installation & Quickstart Guide

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.12 with PyTorch 2.6.0 CUDA & LightGBM 4.6.0)
- GCC with C++20 support (e.g., MinGW-w64 GCC 15.2 or Clang/MSVC)
- CMake 3.20+ (optional)

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/ML_Project_2026.git
cd ML_Project_2026

# Install dependencies and package in editable mode
pip install -r requirements.txt
pip install -e .
```

### 3. Run All Research Experiments
To execute the complete research suite (Experiments A, B, C, D and the C++ engine):
```bash
python scripts/run_all_experiments.py --events 25000 --epochs 5 --horizon 20
```

### 4. Build and Run the C++ Low-Latency Benchmark
```bash
# Compile with high optimization and native SIMD
g++ -std=c++20 -O3 -march=native -ffast-math -I cpp/include cpp/src/main_benchmark.cpp -o cpp/lob_benchmark.exe

# Execute benchmark
./cpp/lob_benchmark.exe
```

---

## Automated Test Suite

The automated test suite tests feature formulas, deep network forward passes, Bilinear Normalization gradient flow, spread crossing penalties, multi-level book walking, and VPIN adverse selection triggers:

```bash
python -m pytest tests -v
```

Expected output:
```
tests/test_features.py::test_ofi_exact_formula PASSED                    [  8%]
tests/test_features.py::test_microprice_and_imbalance PASSED             [ 16%]
tests/test_features.py::test_vpin_calculation PASSED                     [ 25%]
tests/test_features.py::test_bilinear_normalization PASSED               [ 33%]
tests/test_models.py::test_baselines_shapes PASSED                       [ 41%]
tests/test_models.py::test_tabular_tree_models PASSED                    [ 50%]
tests/test_models.py::test_deeplob_forward PASSED                        [ 58%]
tests/test_models.py::test_tlob_forward PASSED                           [ 66%]
tests/test_models.py::test_recurrent_models PASSED                       [ 75%]
tests/test_simulator.py::test_spread_crossing_aggressive PASSED          [ 83%]
tests/test_simulator.py::test_walking_the_book_slippage PASSED           [ 91%]
tests/test_simulator.py::test_passive_adverse_selection PASSED           [100%]
======================= 12 passed in 2.58s ========================
```

---

## References

1. **Cont, R., Kukanov, A., & Stoikov, S. (2014).** *The Price Impact of Order Book Events.* Journal of Financial Econometrics, 12(1), 47–88.
2. **Easley, D., Lopez de Prado, M. M., & O'Hara, M. (2012).** *Flow Toxicity and Liquidity in a High-Frequency World.* The Review of Financial Studies, 25(5), 1457–1493.
3. **Stoikov, S. (2018).** *The Micro-Price: A High-Frequency Estimator of Future Prices.* Quantitative Finance, 18(12), 1959–1966.
4. **Zhang, Z., Zohren, S., & Roberts, S. (2019).** *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books.* IEEE Transactions on Signal Processing, 67(11), 3001–3012.
5. **Tran, D. T., Iosifidis, A., Kanniainen, J., & Gabbouj, M. (2020).** *Data Normalization for Bilinear Structures in High-Frequency Financial Time-series.* IEEE Transactions on Neural Networks and Learning Systems.
6. **Berti, L., et al. (2024).** *TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction with Limit Order Book Data.* arXiv preprint.
