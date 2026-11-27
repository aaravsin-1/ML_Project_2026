# Literature Survey & Saved Paper References

Saved bibliography and research positioning notes for our Limit Order Book (LOB) ML, microstructure, and cross-market execution work.

---

## Saved Primary Survey Link

- **Gould, M. D., Porter, M. A., Williams, S., McDonald, M., Fenn, D. J., & Howison, S. D. (2013).** *Limit Order Books.* *Quantitative Finance*, 13(11), 1709–1742.
  - **Direct PDF:** https://www.math.ucla.edu/~mason/papers/gould-qf-final.pdf
  - **DOI:** `10.1080/14697688.2013.803148` | **Preprint:** [arXiv:1012.0349](https://arxiv.org/abs/1012.0349)
  - **Why it matters for our paper:** Foundational empirical and theoretical survey of continuous double-auction limit order books, stylized facts (tick-size effects, spread distributions, queue dynamics, order flow persistence), and the gap between statistical LOB models and real market mechanics.

---

## Core LOB Microstructure, Feature Engineering & Architecture References

| Reference | Venue / Year | Link / Identifier | Relevance to Our Project |
| :--- | :--- | :--- | :--- |
| **Easley, López de Prado, & O'Hara (2012)** — *Flow Toxicity and Liquidity in a High-Frequency World* | *Review of Financial Studies*, 25(5) | — | Defines volume-clock VPIN toxicity used in our core feature set and market-making gating experiments. |
| **Cont, Kukanov, & Stoikov (2014)** — *The Price Impact of Order Book Events* | *Journal of Financial Econometrics*, 12(1) | — | Foundational Level-1 Order Flow Imbalance (OFI) definition used across our tabular baselines and causal features. |
| **Stoikov (2018)** — *The Micro-Price: A High-Frequency Estimator of Future Prices* | *Quantitative Finance*, 18(12) | — | Defines micro-price / imbalance-adjusted fair value benchmarked against LightGBM on large-tick regimes. |
| **Zhang, Zohren, & Roberts (2019)** — *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books* | *IEEE Trans. Signal Processing*, 67(11) | — | Canonical CNN-Inception-LSTM LOB baseline benchmarked on FI-2010, BTC, and NSE (`DeepLOB5`). |
| **Tran, Kanniainen, Gabbouj, & Iosifidis (2021)** — *Data Normalization for Bilinear Structures in High-Frequency Financial Time-Series* | *ICPR 2020* | — | Introduces Bilinear Normalization (`BiN`) that protects deep LOB encoders under non-stationary price-level shifts. |
| **Cont, Cucuringu, & Zhang (2023)** — *Cross-Impact of Order Flow Imbalance in Equity Markets* | *Quantitative Finance* (2023) | [arXiv:2112.13213](https://arxiv.org/abs/2112.13213) | Establishes PCA-integrated Multi-Level OFI (`MLOFI`) and lagged cross-asset OFI; our NSE `SSF → Cash` study extends cross-impact to Single-Stock Futures. |
| **Kolm, Turiel, & Westray (2023)** — *Deep Order Flow Imbalance: Extracting Alpha at Multiple Horizons from the Limit Order Book* | *Mathematical Finance* (2023) | `DOI:10.1111/mafi.12413` | Shows stationary OFI features rival deep nets and links Information Richness / tick size to predictability. |
| **Prata, Masi, Berti, et al. (2024)** — *LOB-Based Deep Learning Models for Stock Price Trend Prediction: A Benchmark Study* | *Artificial Intelligence Review*, 57(5) | `LOBCAST` | Benchmarks 15 deep LOB models and highlights severe out-of-distribution drops; our frozen 22-stock NSE test directly evaluates OOD generalization. |
| **Briola, Bartolucci, & Aste (2024/2025)** — *Deep Limit Order Book Forecasting: A Microstructural Guide* | *Quantitative Finance* | [arXiv:2403.09267](https://arxiv.org/abs/2403.09267) (`LOBFrame`) | Demonstrates that standard ML metrics (accuracy/F1) diverge from actionable execution edge across NASDAQ tick regimes. |
| **Berti & Kasneci (2025)** — *TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction with Limit Order Book Data* | Preprint (Feb 2025) | [arXiv:2502.15757](https://arxiv.org/abs/2502.15757) | Introduces the `TLOB` Dual-Attention Transformer + `MLPLOB` and analyzes horizon-labeling bias; `TLOB` forms Branch B of our Flagship Hybrid. |
| **Nagy, Frey, Li, et al. (2025)** — *LOB-Bench: Benchmarking Generative AI for Finance* | Preprint (Feb 2025) | [arXiv:2502.09172](https://arxiv.org/abs/2502.09172) | Evaluates generative LOB models using conditional order-flow and market-impact response tests. |

---

## Positioning Notes for Potential Paper Submission

- **Recommended Framing:** Empirical Market Microstructure, Cross-Market (`SSF → Cash`) Lead-Lag, & Latency-Aware Hybrid Forecasting (*ACM ICAIF*, *Quantitative Finance*, *Journal of Financial Data Science*, *IEEE CIFEr*, or *Expert Systems with Applications*).
- **Working Title Idea:** *When Does Limit Order Book Predictability Pay? Tick-Size Scaling, Single-Stock Futures Lead-Lag, and Latency-Aware Hybrid Forecasting Across 36 Million NSE Snapshots*
- **Core Differentiators:**
  1. Strict zero-leakage frozen evaluation across `25` Indian NSE instruments (`22` untouched out-of-universe, `300/300` test days `>58.8%` directional hit).
  2. Market+sector-residualized Single-Stock Futures (`SSF → Cash`) precedence across `12/12` equities (`10/10` unseen, Wilcoxon `p = 2.44 × 10⁻⁴`).
  3. Unified accuracy–latency–cost break-even equation across `6` datasets (`BTC`, `LOBSTER`, `FI-2010`, `5` NSE tick regimes, `Optiver`, synthetic).
  4. Mechanistic error anatomy (`63.4%` `±α` boundary share, `15:15 IST` closing-stall fix `+31.4 pp`, off-book sweep Bayes ceiling, `1 s` tree flip vs `100 s` `TLOB` attention complementarity) + `10.5 µs` C++ deployment.
