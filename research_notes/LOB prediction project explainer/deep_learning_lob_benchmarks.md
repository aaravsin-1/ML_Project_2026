# Deep Learning for LOB Mid-Price Prediction: Architectures, Benchmarks, Pitfalls, Reported Results

Scope note: where this file gives an exact number, it was read directly from the primary PDF or HTML (DeepLOB arXiv PDF, FI-2010 arXiv PDF, LOBCAST arXiv PDF, Lucchese et al. arXiv PDF, TLOB arXiv HTML, LiT PMC full text, Briola et al. 2025 PMC full text, Lee 2024 arXiv HTML). Anything that comes only from a search-result snippet or abstract is marked that way.

## 1. FI-2010 (Ntakaris et al. 2018): contents, labels, setups, known problems

### Takeaway
FI-2010 is small: 10 days, 5 Finnish stocks, from 2010. It has already been pre-processed, normalized and labelled, and the original order book cannot be reconstructed from it. Each label compares the mean of the next k mid-prices against the current mid-price, using a fixed ±0.002 threshold. The threshold was picked to balance the classes, not for any financial meaning. This smoothing makes the target easier to predict and autocorrelated. Together with the very small, old, less-liquid market, it is the main reason FI-2010 F1 scores (often 80-90%) collapse on modern NASDAQ data (48-61%).

### Cited Findings
- **What it is:** NASDAQ Nordic ITCH data for five stocks (Kesko Oyj, Outokumpu Oyj, Sampo, Rautaruukki, Wärtsilä Oyj), 1-14 June 2010, which is 10 trading days. That is about 4 million events and 10 LOB levels. — [Ntakaris et al., arXiv 1705.03233](https://arxiv.org/abs/1705.03233); dates and company names from [LOBCAST, arXiv 2308.01915](https://arxiv.org/abs/2308.01915)
- **Sampling:** event-based. "each of our representations is a vector that contains information for 10 consecutive events", which gives 394,337 representations (samples). — [Ntakaris et al.](https://arxiv.org/abs/1705.03233)
- **144 features:** the 144-dimensional representation of Kercheval & Zhang (2015). It has three groups: (a) raw 10-level prices and volumes for bid and ask (40 features), (b) features describing the LOB state built from past information (spreads, mid-prices, price differences, means, accumulated differences), and (c) time-sensitive features (derivatives, intensities of trades/orders/cancellations, accelerations). The authors note it lacks order-flow (message sequence) information. — [Ntakaris et al.](https://arxiv.org/abs/1705.03233)
- **Normalizations:** the dataset ships in three versions: z-score, min-max, and decimal-precision. DeepLOB used the z-score version "without any emendation" and found only "subtle difference" with the other two. — [Ntakaris et al.](https://arxiv.org/abs/1705.03233); [DeepLOB, arXiv 1808.03668](https://arxiv.org/abs/1808.03668)
- **Label definition (Eq. 8 of the paper):** l_i = ((1/k) Σ_{j=i+1}^{i+k} m_j − m_i) / m_i, the mean of the next k mid-prices against the current mid-price. Thresholds: l ≥ 0.002 is label 1 (up), −0.00199 to 0.00199 is label 2 (stationary), l ≤ −0.002 is label 3 (down). Horizons are "the next 1, 2, 3, 5, and 10 events in our representations". — [Ntakaris et al.](https://arxiv.org/abs/1705.03233)
- **Horizon naming confusion:** LOBCAST indexes horizons as K = {1, 2, 3, 5, 10}. DeepLOB reports k = 10, 20, 50 (and 100 in Setup 1). TLOB reports h = 10, 20, 50, 100. — [LOBCAST](https://arxiv.org/abs/2308.01915); [DeepLOB](https://arxiv.org/abs/1808.03668); [TLOB](https://arxiv.org/html/2502.15757)
- **Setup 1 vs Setup 2 (as named in DeepLOB, following earlier convention):**
  - Setup 1 is the original FI-2010 protocol. It is a day-based anchored forward cross-validation with 9 folds: train on the first i days, test on day i+1. Metrics are averaged over folds.
  - Setup 2 trains on the first 7 days and tests on the last 3 days. It came from the deep-learning papers, because deep nets need more training data.
  - LOBCAST followed Setup 2, splitting the first 7 days 80/20 into train and validation.
  — [DeepLOB](https://arxiv.org/abs/1808.03668); [Ntakaris et al.](https://arxiv.org/abs/1705.03233); [LOBCAST](https://arxiv.org/abs/2308.01915)
- **Original baselines:** the FI-2010 authors' ridge regression and SLFN baselines reached an average out-of-sample F1 of about 46%. They chose F1 as the main metric because the classes are unbalanced. — [Ntakaris et al.](https://arxiv.org/abs/1705.03233)
- **Class imbalance varies with horizon:**
  - k = 1: U 18%, S 63%, D 19%
  - k = 5: U 32%, S 35%, D 33%
  - k = 10: U 37%, S 25%, D 38%
  - θ "was chosen to obtain a balanced dataset in the number of classes for the horizon k = 5", so "θ is not chosen in accordance with its financial implication".
  - Because one fixed θ is applied across stocks, "stocks with low returns become associated with stable trends".
  — [LOBCAST](https://arxiv.org/abs/2308.01915)
- **Pre-processing and instability:** FI-2010 has "the intrinsic limitation of being already pre-processed (filtered, normalized, and labelled) so that the original LOB cannot be backtracked". Its labelling "is found to be prone to instability". — [LOBCAST](https://arxiv.org/abs/2308.01915)
- **DeepLOB's own critique of the FI-2010 label:** smoothing only the future side makes labels "less consistent", which is "detrimental for designing trading algorithms as signals are not consistent here leading to many redundant trading actions thus incurring larger transaction costs". On LSE data the FI-2010-style labels were "rather stochastic", so DeepLOB switched to l_t = (m+ − m−)/m−, which smooths both past and future. DeepLOB also calls FI-2010 "far too short, downsampled and taken from a less liquid market", and says that with 10 days "the problem of overfitting to backtest data is severe". — [DeepLOB](https://arxiv.org/abs/1808.03668)
- **TLOB's critique:**
  - "A key drawback is that the window length k coincides with the prediction horizon h". A short h gets too little smoothing and a long h too much.
  - θ is often picked to balance classes rather than to reflect trading costs.
  - On FI-2010, scores for most models rise with horizon, which is the opposite of what happens on NASDAQ and BTC data. TLOB calls FI-2010 a lower-complexity market.
  — [TLOB, arXiv 2502.15757](https://arxiv.org/html/2502.15757)
- **Smoothed labels are easy because they embed overlapping information (Lee 2024, AAPL 2022 ITCH data):**
  - For a target comparing future and past 20-observation averages (r20), the paper notes the target "contains future information".
  - On r20, DeepLOB gets 0.659 accuracy against 0.648 for a naive forecast that simply uses the lagged target. The author says DL "does not substantially differ from naive predictions".
  - On a non-smoothed target that uses the current price as base (r1,20), DeepLOB accuracy falls to 0.546, with up/down F1 about 0.54.
  — [Lee, arXiv 2409.14157](https://arxiv.org/html/2409.14157v1)
- **Weighted vs macro F1:**
  - In DeepLOB's FI-2010 and LSE tables, recall equals accuracy exactly in every row (84.47/84.47, 74.85/74.85, 80.51/80.51, LSE 70.17/70.17). Support-weighted recall is mathematically identical to accuracy, which is the signature of weighted averaging.
  - TLOB does not state which averaging it uses. It says only that F1 "is robust to the class imbalance problem".
  - LOBCAST's text does not state the averaging either (checked by searching for "macro" and "weighted").
  — [DeepLOB](https://arxiv.org/abs/1808.03668); [TLOB](https://arxiv.org/html/2502.15757); [LOBCAST](https://arxiv.org/abs/2308.01915)

### Inferences
- With k-step forward averaging, labels at t and t+1 share k−1 of the same future mid-prices. Consecutive labels are therefore heavily autocorrelated. Within the test set, a model that partly reconstructs "where the recent average is heading", or effectively uses the recent past label, gets credit far above a fair coin. Lee's naive-lagged-target result (64.8%, against 65.9% for DeepLOB) is direct evidence of this. A project should always report a "persistence / previous-label" baseline next to DeepLOB and TLOB.
- Because labels overlap, a random (shuffled) train/test split, or a validation slice taken from inside the training days, leaks information. Only contiguous time splits with a gap at least k samples long are clean. This is a standard argument from the label construction (the purging/embargo idea), not a claim made in any one paper above.
- Weighted F1 inflates scores when the dominant class is easy, as with "stationary" at short horizons. For the project, report macro F1, with weighted F1 next to it only for comparison with DeepLOB's published numbers.
- "Data snooping" on FI-2010 has several sources: one public 10-day test set, years of papers tuning to the same 3 test days, and θ chosen with knowledge of the whole dataset's class balance. LOBCAST's finding that claimed results often do not reproduce, and that rankings change, is consistent with this.

### Gaps
- I did not find (in the time available) a standalone paper titled as an "FI-2010 label leakage" critique beyond LOBCAST, DeepLOB, TLOB, Lee (2024) and Briola et al. (2025). Zhang et al. [26] in LOBCAST's text is cited as showing label instability, but I did not open that reference.
- It is not verified whether the FI-2010 z-score normalization statistics were computed only on past or training data, or on the full fold. I did not confirm this from the paper text extracted.
- The exact mapping of DeepLOB's "k = 10/20/50" to FI-2010 label columns (raw events vs 10-event samples) is ambiguous between papers.

## 2. DeepLOB (Zhang, Zohren & Roberts 2019)

### Takeaway
DeepLOB has three stages. A CNN learns micro-price-like features across book levels. An Inception module adds multi-timescale "learned moving averages". An LSTM with 64 units models time. It has about 60k parameters and a forward pass of about 0.25 ms. It reported Setup-2 FI-2010 F1 of 83.40 / 72.82 / 80.35 at k = 10 / 20 / 50, and LSE one-year F1 of about 60-70%. Independent re-runs (LOBCAST, TLOB) get roughly 71% average F1 on FI-2010 and about 59.5% on 2021-2022 NASDAQ data.

### Cited Findings
- **Venue:** IEEE Transactions on Signal Processing, vol. 67, no. 11, 2019. — [arXiv 1808.03668](https://arxiv.org/abs/1808.03668)
- **Input:** the 100 most recent LOB states × 40 raw features (10 levels × {ask price, ask volume, bid price, bid volume}). — [DeepLOB](https://arxiv.org/abs/1808.03668)
- **Convolution blocks:**
  - The first conv is (1×2) with stride (1×2). It pairs each price with its volume at the same level. The stride stops price and volume from sharing parameters.
  - The second (1×2) stride-2 conv combines bid and ask, "actually form[ing] the micro-price", where imbalance I = v_b/(v_a+v_b).
  - A (1×10) conv then integrates all levels, giving a 100×1 feature map.
  - Leaky-ReLU (slope 0.01) throughout, no pooling outside Inception. The authors say pooling caused under-fitting.
  — [DeepLOB](https://arxiv.org/abs/1808.03668)
- **Inception module (Inception@32):** parallel 1×1→3×1, 1×1→5×1, and maxpool 3×1→1×1 branches, concatenated. It is described as learning "different moving averages" over several timescales. — [DeepLOB](https://arxiv.org/abs/1808.03668)
- **LSTM and output:** 64 LSTM units replace the fully connected layers (about 60,000 parameters, against more than 630k for one 64-unit FC layer), followed by a softmax over 3 classes. — [DeepLOB](https://arxiv.org/abs/1808.03668)
- **Training:** Adam (lr 0.01, epsilon 1), batch 32, early stopping after 20 epochs without improvement in validation accuracy, Keras/TensorFlow, single P100. — [DeepLOB](https://arxiv.org/abs/1808.03668)
- **FI-2010 Setup 2 (Table II, in %: Acc / Prec / Rec / F1):**
  - k=10: DeepLOB 84.47 / 84.00 / 84.47 / 83.40; C(TABL) 84.70 / 76.95 / 78.44 / 77.63
  - k=20: DeepLOB 74.85 / 74.06 / 74.85 / 72.82; C(TABL) 73.74 / 67.18 / 66.94 / 66.93
  - k=50: DeepLOB 80.51 / 80.38 / 80.51 / 80.35; C(TABL) 79.87 / 79.05 / 77.04 / 78.44
  - Earlier LSTM baseline F1: 66.33 / 62.37 / 61.43
  — [DeepLOB](https://arxiv.org/abs/1808.03668)
- **FI-2010 Setup 1 (Table I, F1 %):**
  - DeepLOB: 77.66 (k=10), 74.96 (k=50), 76.58 (k=100)
  - C(TABL): 72.84 / 74.32 / 73.52
  - Ridge regression: 41.00 / 42.70 / 41.60
  — [DeepLOB](https://arxiv.org/abs/1808.03668)
- **Inference time (Table III, forward pass):** DeepLOB 0.253 ms with 60k parameters. CNN-I 0.025 ms (768k), LSTM 0.061 ms, C(TABL) 0.229 ms. — [DeepLOB](https://arxiv.org/abs/1808.03668)
  - Contrast: LOBCAST later measured DeepLOB at inference times "in the order of milliseconds, potentially unsuitable for HFT". — [LOBCAST](https://arxiv.org/abs/2308.01915)
- **LSE data:** LLOY, BARC, TSCO, BT, VOD from 3 Jan to 24 Dec 2017, 08:30-16:00. More than 134M samples, about 150k events per day per stock, average inter-event time 0.192 s. Split 6 months train, 3 months validation, 3 months test. Normalization uses the z-score of the previous 5 days, per instrument. — [DeepLOB](https://arxiv.org/abs/1808.03668)
- **LSE results (Table IV, Acc / F1 %):** k=20: 70.17 / 70.15; k=50: 63.93 / 63.49; k=100: 61.52 / 60.65. — [DeepLOB](https://arxiv.org/abs/1808.03668)
- **Transfer to five untrained stocks** (GLEN, HSBC, CNA, BP, ITV): F1 68.48 / 62.84 / 60.77. The authors read this as evidence of "universal patterns". — [DeepLOB](https://arxiv.org/abs/1808.03668)
- **Trading simulation:** uses "mid-prices without transaction costs". The authors concede that "the second assumption is not a reasonable assumption for a standalone strategy". — [DeepLOB](https://arxiv.org/abs/1808.03668)
- **Independent reproduction (LOBCAST):** DeepLOB claimed 78.9 ± 4.4 average F1 on FI-2010. LOBCAST measured 71.4 ± 5.3, with LOB-2021 at 59.5 ± 3.0 and LOB-2022 at 59.5 ± 2.9. — [LOBCAST](https://arxiv.org/abs/2308.01915)
- **TLOB's re-run of DeepLOB on FI-2010 (F1):** 71.1 / 62.4 / 75.4 / 77.6 at h = 10 / 20 / 50 / 100. — [TLOB](https://arxiv.org/html/2502.15757)
- **Official code:** github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books. The repo name is from memory and its contents were not fetched.

### Inferences
- The re-run gap (83.4 claimed against about 71 reproduced at k=10) is about 12 F1 points. That is larger than most reported "improvements" between successive SOTA models.
- Weighted F1 (see section 1) makes DeepLOB's published numbers not directly comparable with macro-F1 numbers in a student project.

### Gaps
- I did not fetch the official DeepLOB repo to confirm the exact sklearn averaging it uses.

## 3. TLOB (Berti & Kasneci 2025)

### Takeaway
TLOB is a BiN plus Transformer model. Each block alternates temporal self-attention (across snapshots) and "spatial" self-attention (across features), with an MLP-mixer-style feed-forward block. The same paper introduces the simpler MLPLOB. Both beat older models on FI-2010 and on 2015 TSLA/INTC and 2023 BTC data. The paper also reports that F1 falls sharply with horizon on real NASDAQ and BTC data, that INTC was harder to predict in 2015 than in 2012, and that with a cost-aware threshold the F1 falls further.

### Cited Findings
- **Title and code:** "TLOB: A Novel Transformer Model with Dual Attention for Stock Price Trend Prediction with Limit Order Book Data", Berti & Kasneci, arXiv 2502.15757 (2025). — [arXiv](https://arxiv.org/html/2502.15757); code at github.com/LeonardoBerti00/TLOB (repo not fetched)
- **Architecture:**
  - Bilinear Normalization first, then sinusoidal positional encoding.
  - Each block has temporal self-attention, feature (spatial) self-attention, and an MLPLOB block replacing the usual FFN.
  - Hyperparameters: Adam, sequence length 128, lr 1e-4, 4 layers, 1 head ("1, 2, and 4 heads showed no performance difference").
  - Size and speed: about 1×10⁷ parameters and 2.24 ms inference. MLPLOB has about 6.3×10⁷ parameters and 4.79 ms.
  - Ablation at h=10 (FI-2010): full 81.55, without spatial attention 79.59, without temporal attention 80.27.
  — [TLOB](https://arxiv.org/html/2502.15757)
- **New label proposal:** decouple the smoothing window k from the horizon h. Compare a forward window average anchored at t+h with a backward window average anchored at t, using threshold θ. FI-2010 keeps its original labels for comparability, and the new labels are used on TSLA/INTC. — [TLOB](https://arxiv.org/html/2502.15757)
- **Datasets:**
  - FI-2010.
  - TSLA and INTC, 2-30 January 2015, about 24M samples, sampled every 500 shares traded. Split 17 days train, 1 day validation, 2 days test. Adding order messages gave about +1.5 F1.
  - BTC: Binance BTCUSDT perpetual, 9-20 January 2023, 3,730,870 rows, 250 ms sampling. h = 10/20/50/100 corresponds to about 1 s / 2.5 s / 5 s / 10 s. Split 9 days train, 1 day validation, 1 day test.
  — [TLOB](https://arxiv.org/html/2502.15757)
- **F1 results (%) at h = 10 / 20 / 50 / 100:**

  | Data | Model | h=10 | h=20 | h=50 | h=100 |
  |---|---|---|---|---|---|
  | FI-2010 | TLOB | 81.55 | 82.68 | 90.03 | 92.81 |
  | FI-2010 | MLPLOB | 81.64 | 84.88 | 91.39 | 92.62 |
  | FI-2010 | BiNCTABL | 81.1 | 71.5 | 87.7 | 92.1 |
  | FI-2010 | DeepLOB | 71.1 | 62.4 | 75.4 | 77.6 |
  | TSLA | TLOB | 60.50 | 49.74 | 43.48 | 39.84 |
  | TSLA | MLPLOB | 60.72 | 50.25 | 38.97 | 32.95 |
  | TSLA | DeepLOB | 36.25 | 36.58 | 35.29 | 34.43 |
  | INTC | TLOB | 80.15 | 72.75 | 62.07 | 50.14 |
  | INTC | DeepLOB | 68.13 | 63.70 | 40.3 | 30.1 |
  | BTC | TLOB | 74.7 | 61.74 | 48.54 | 41.49 |
  | BTC | BiNCTABL | 73.4 | 61.34 | 47.05 | 40.59 |
  | BTC | DeepLOB | 68.07 | 57.87 | 45.13 | 37.43 |

  — [TLOB](https://arxiv.org/html/2502.15757)
- **Headline claims (abstract):** average F1 gains of 3.7 over SOTA on FI-2010, 1.3 on TSLA, 7.7 on INTC, and 1.1 on BTC. — [TLOB](https://arxiv.org/html/2502.15757)
- **Horizon finding:** "As expected, the longer the horizon, the more difficult to forecast" holds on TSLA, INTC and BTC. FI-2010 shows the opposite pattern. — [TLOB](https://arxiv.org/html/2502.15757)
- **Over-time finding:** INTC at h=50 gives TLOB F1 66.87 on a 2012 day and 60.19 on a 2015 day. The authors read this as markets becoming more efficient. It rests on a single test day per year. — [TLOB](https://arxiv.org/html/2502.15757)
- **Cost-aware threshold:** with θ set to the average spread as a % of mid-price, TSLA TLOB F1 is 41.39 / 36.48 / 30.82 at h = 50 / 100 / 200. For INTC this was not feasible, because about 99.99% of labels became "stable". There is no backtest and no profit figures; profitability is named as future work. — [TLOB](https://arxiv.org/html/2502.15757)
- **F1 averaging:** not stated. — [TLOB](https://arxiv.org/html/2502.15757)

### Inferences
- The INTC result is very telling. Once the up/down threshold has to exceed the spread, almost no events qualify. The predictable moves in a large-tick stock are therefore mostly smaller than the cost of crossing the spread.
- MLPLOB (pure MLP-mixer) roughly ties TLOB, which suggests attention itself is not the key ingredient. BiN normalization and careful training may matter as much.

### Gaps
- I did not verify whether TLOB uses macro or weighted F1 from the code.
- Note: the brief describes the 2012 vs 2015 drop as being "on recent NASDAQ data". TLOB's "recent" NASDAQ data is 2015, so it is not post-2020.

## 4. BiN (Tran et al.) and TABL / C(TABL) (Tran et al. 2019)

### Takeaway
TABL treats each input as a features × time matrix. It uses bilinear projections (left and right weight matrices) plus a learned temporal attention mask, so it is tiny and fast. BiN normalizes the input jointly along the time and feature axes with learnable parameters. BiN-CTABL was the most robust model in LOBCAST, and it is a strong baseline that is cheaper than DeepLOB.

### Cited Findings
- **TABL:** "Temporal attention-augmented bilinear network for financial time-series data analysis", Tran, Iosifidis, Kanniainen, Gabbouj, IEEE TNNLS 2019 (cited as [25] in DeepLOB).
  - FI-2010 Setup 2 F1 for C(TABL), the two-hidden-layer version: 77.63 / 66.93 / 78.44 at k = 10 / 20 / 50.
  - For B(TABL), one hidden layer: 69.20 / 62.22 / 73.64.
  - Forward pass 0.229 ms.
  — [DeepLOB Tables II-III](https://arxiv.org/abs/1808.03668)
- **BiN:** "Data Normalization for Bilinear Structures in High-Frequency Financial Time-series", Tran, Kanniainen, Gabbouj, Iosifidis, arXiv 2003.00598 (2020). On LOB data of more than 4M events, "BiN-TABL" beats TABL with other normalization schemes "by a wide margin". A later version, "Bilinear Input Normalization for Neural Networks in Financial Forecasting" (arXiv 2109.00983), tests Nordic and US LOB data. — [arXiv 2003.00598](https://arxiv.org/abs/2003.00598v2); [arXiv 2109.00983](https://arxiv.org/pdf/2109.00983) (abstract/snippet-level only)
- **LOBCAST on BiNCTABL:**
  - It enhances CTABL "by adding an Adaptive Bilinear Normalization layer, enabling joint normalization of the input time series along both temporal and feature dimensions".
  - Best average F1 on FI-2010: 82.6 ± 7.0 (claimed 80.1 ± 6.9), robustness score 99.7.
  - Still best on LOB-2021 (61.2 ± 2.7), but that is a drop of about 19.6% (generalizability score 73.5).
  - "up to 9.2%" better than DLA and "up to 13%" better than CTABL.
  — [LOBCAST](https://arxiv.org/abs/2308.01915)

### Inferences
- If compute is tight, BiN-CTABL is a better "strong DL baseline" than DeepLOB on the reproducibility evidence.

### Gaps
- I did not open the TABL paper or the BiN paper in full, so I have no exact architecture equations or their own reported tables beyond what DeepLOB and LOBCAST reproduce.

## 5. LOBCAST (Prata, Masi, Berti et al. 2024): robustness, generalizability, reproducibility

### Takeaway
LOBCAST re-implemented 15 DL LOB models (2017-2022) in one PyTorch Lightning framework. Many models did not reach their published FI-2010 F1. Rankings changed. About half of hyperparameter runs diverged. On 2021-2022 NASDAQ (LOBSTER) data every model fell to 48-61% F1, and simple models such as MLP and CNN1 lost almost nothing, which closes most of the gap between SOTA and naive DL.

### Cited Findings
- **Paper:** "LOB-based deep learning models for stock price trend prediction: a benchmark study" by Prata, Masi, Berti, Arrigoni, Coletta, Cannistraci, Vyetrenko, Velardi, Bartolini. Artificial Intelligence Review (2024), DOI 10.1007/s10462-024-10715-4. Code at github.com/matteoprata/LOBCAST. — [arXiv 2308.01915](https://arxiv.org/abs/2308.01915); [Springer](https://link.springer.com/article/10.1007/s10462-024-10715-4)
- **Abstract finding:** "all models exhibit a significant performance drop when exposed to new data". — [LOBCAST](https://arxiv.org/abs/2308.01915)
- **Data:**
  - LOB-2021: 1-15 July 2021, 10 trading days.
  - LOB-2022: 1-15 February 2022, a higher-volatility period linked to the Ukraine crisis.
  - Six NASDAQ stocks: SOFI, NFLX, CSCO, WING, SHLS, LSTR.
  - Built like FI-2010: sampled every 10 events, 6-2-2 day split, z-score normalization, same labels.
  - Not released, for copyright reasons.
  — [LOBCAST](https://arxiv.org/abs/2308.01915)
- **Score definitions:** robustness and generalizability scores are 100 − (|A| + S). A is the mean gap between claimed and measured F1, and S is its standard deviation. — [LOBCAST](https://arxiv.org/abs/2308.01915)
- **Selected Table 2 rows (average F1 over horizons and 5 seeds; claim → FI-2010 measured → LOB-2021 → LOB-2022):**

  | Model | Claimed | FI-2010 measured | LOB-2021 | LOB-2022 |
  |---|---|---|---|---|
  | MLP | 51.8 | 48.0 | 55.5 | 53.1 |
  | LSTM | 63.4 | 63.4 | 56.9 | 56.1 |
  | CNN1 | 57.9 | 58.1 | 57.5 | 57.1 |
  | CTABL | 74.3 | 69.6 | 59.7 | 58.1 |
  | DeepLOB | 78.9 | 71.4 | 59.5 | 59.5 |
  | TransLOB | 87.3 | 59.4 | 57.7 | 50.4 |
  | BiNCTABL | 80.1 | 82.6 | 61.2 | 59.2 |
  | DeepLOBAtt | 78.8 | 67.3 | 60.1 | 58.9 |
  | DLA | 78.7 | 73.4 | 57.7 | 56.6 |
  | ATNBoF | 67.1 | 40.9 | 53.1 | 48.0 |
  | AxialLOB | 82.0 | 73.4 | 59.5 | 58.6 |
  | METALOB (ensemble) | n/a | 82.2 | 55.9 | 53.2 |
  | MAJORITY (ensemble) | n/a | 60.0 | 55.5 | 47.9 |

  — [LOBCAST](https://arxiv.org/abs/2308.01915)
- **Key bullet findings:**
  1. There is a considerable gap between claimed and measured performance.
  2. "All models are very sensitive to hyperparameters": they diverged (F1 ≤ 33%) in about half of hyperparameter-search runs.
  3. Rankings change against the claimed ones, but the top 6 on FI-2010 stay the top 6 on LOB-2021/22.
  4. Five of the best six use attention.
  5. Ensembles do not beat the best single model.
  — [LOBCAST](https://arxiv.org/abs/2308.01915)
- **Reproducibility:**
  - Failures were attributed to "the lack of the original code and the missing hyperparameters declaration".
  - TransLOB and ATNBoF had the largest discrepancies.
  - CNN1, CNN2, CNNLSTM, TLONBoF and DLA had seed standard deviations above 5 F1 points.
  - Models were re-implemented when code was missing or TensorFlow-only.
  — [LOBCAST](https://arxiv.org/abs/2308.01915)
- **Generalizability:** NASDAQ performance "ranging 48-61% in F1-score".
  - Proposed causes: higher complexity (about 3× more events than FI-2010 for the same period) and overfitting to FI-2010.
  - CSCO was the most predictable stock, attributed to a dominant stationary class (train balance 18-65-17%).
  - A BiNCTABL confusion matrix on LOBSTER at k=1 shows about 35-37% of up and down moves predicted as the opposite direction.
  — [LOBCAST](https://arxiv.org/abs/2308.01915)
- **Other findings:**
  - None of the top three models needs a 100-step input window: "the most influential and relevant dynamics … occur within a short time frame".
  - A trading simulation on LOB-2021 showed "profitability is far from guaranteed".
  - The authors urge trading simulations and reactive simulators beyond historical backtests.
  — [LOBCAST](https://arxiv.org/abs/2308.01915)

### Inferences
- On new data a plain MLP (55.5) or CNN1 (57.5) is within about 4-6 F1 points of the best SOTA model (61.2). That is exactly the regime where LightGBM on engineered features would be expected to be competitive.

### Gaps
- The details of the LOBCAST trading-simulation results (Section 7 of the SUP) were not extracted.

## 6. Other recent benchmarks and transformer variants (2024-2026): LOBench, LOB-Bench, LOBFrame, LiT

### Takeaway
"LOBench" and "LOB-Bench" are different projects. LOBench (Zhong, Lin, Yang 2025) is a representation-learning benchmark on China A-share data. LOB-Bench (ICML 2025) evaluates generative message-level models. LOBFrame (Briola, Bartolucci, Aste 2025) is the most relevant to evaluation critique. LiT is real (Frontiers in AI, October 2025) and is evaluated only on Binance data, with small margins over DeepLOB.

### Cited Findings
- **LOBench:** "Representation Learning of Limit Order Book: A Comprehensive Study and Benchmarking", Muyao Zhong, Yushi Lin, Peng Yang, arXiv 2505.02139. It uses China A-share data and covers reconstruction, prediction and imputation tasks. Code is at github.com/financial-simulation-lab/LOBench. — [arXiv 2505.02139](https://arxiv.org/abs/2505.02139v1) (search snippet level)
- **LOB-Bench:** "Benchmarking Generative AI for Finance: an Application to Limit Order Book Data", arXiv 2502.09172, ICML 2025, on LOBSTER-format message data. — [arXiv 2502.09172](https://arxiv.org/pdf/2502.09172) (search snippet level)
- **LOBFrame / "Deep limit order book forecasting: a microstructural guide":** Briola, Bartolucci, Aste, Quantitative Finance 25(7):1101-1131 (2025), open access.
  - Data: 15 NASDAQ stocks, LOBSTER, 2017-2019. Model: DeepLOB. Horizons: 10 / 50 / 100 LOB updates.
  - MCC without probability threshold:

    | Group | H10 | H50 | H100 |
    |---|---|---|---|
    | Small-tick | 0.11 | 0.04 | 0.01 |
    | Medium-tick | 0.13 | 0.085 | 0.036 |
    | Large-tick | 0.29 | 0.36 | 0.26 |

  - Large-tick stocks reach accuracy above 0.7 (above 0.9 with confidence thresholds).
  - Proposed metric: pT = CT / (PT + TT − CT), the probability of correctly forecasting complete transactions.
  - Quote: "traditional machine learning metrics fail to adequately assess the quality of forecasts in the Limit Order Book context."
  — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/); [arXiv 2403.09267](https://arxiv.org/abs/2403.09267v1)
- **LiT: limit order book transformer:** Xiao, Ventre, Wang, Li, Huan, Liu, Frontiers in Artificial Intelligence 8:1616485 (13 October 2025).
  - Architecture: structured patches (one book side × a short time window), self-attention, then LSTM. Input is the 64 most recent snapshots.
  - Data: Binance L2 data, top 20 levels per side, September to December 2024, more than 1M snapshots.
  - Horizons: windows of 300-500 ms up to 500-1000 ms. Threshold p = 0.000015 × mid-price.
  - F1 (LiT / DeepLOB / TransLOB):

    | Window | LiT | DeepLOB | TransLOB |
    |---|---|---|---|
    | 300-500 ms | 58.99 | 57.09 | 58.77 |
    | 300-1000 ms | 66.40 | 66.23 | 65.68 |
    | 500-1000 ms | 64.32 | 64.57 | 63.49 |

    DeepLOB is better at 500-1000 ms.
  - F1 averaging is not explicitly stated.
  — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12555381/)
- **Other transformer variants:** TransLOB (Wallbridge 2020, "Transformers for Limit Order Books", FI-2010) and LOBERT (arXiv 2511.12563, a BERT-style model with a mid-price direction head at horizons of 10/50/100 messages). — [arXiv 2003.00130](https://arxiv.org/pdf/2003.00130); [arXiv 2511.12563](https://arxiv.org/pdf/2511.12563) (snippet level)

### Inferences
- Briola et al.'s MCC of 0.01-0.04 at H50-H100 for small-tick stocks means DeepLOB is near chance on exactly the stocks whose prices move most freely. "Predictability" in LOB papers is concentrated in large-tick stocks, where the mid-price rarely changes and queue imbalance is very informative.

### Gaps
- I did not verify LOBench's quantitative results or any 2026-dated LOB benchmark. I found no "LOBench" results paper with prediction leaderboards.

## 7. Crypto LOB prediction results

### Takeaway
On crypto (mostly BTC on Binance), the recent primary sources report 3-class F1 of about 57-75% at sub-second to 1-second horizons, falling to about 37-49% at 5-10 s (TLOB, BTC 2023). The gains of new architectures over DeepLOB are usually 1-2 F1 points.

### Cited Findings
- **TLOB, BTCUSDT perpetual, 250 ms sampling:**
  - At h=10 (about 1 s): TLOB F1 74.7, DeepLOB 68.07.
  - At h=100 (about 10 s): TLOB 41.49, DeepLOB 37.43.
  — [TLOB](https://arxiv.org/html/2502.15757)
- **LiT, Binance late 2024:** F1 about 57-66% across 300-1000 ms windows for LiT, DeepLOB and TransLOB alike. — [LiT](https://pmc.ncbi.nlm.nih.gov/articles/PMC12555381/)
- **"Ascertaining price formation in cryptocurrency markets with Deep Learning" (arXiv 2003.00803):** reportedly "a consistent 78% accuracy on the binary prediction of mid-price movements on the live exchange of BTC-USD". This is snippet level and not verified in full text. — [arXiv 2003.00803](https://arxiv.org/pdf/2003.00803)
- **Theses (snippet level, not verified):** an Aalto master's thesis reports DeepLOB reaches accuracy on BTCUSD similar to its equity results. An NTU 2023 thesis on Binance BTC futures reports DeepLOB beating linear models, and says the original LOB representation can give unstable predictions. — [Aalto](https://aaltodoc.aalto.fi/items/02ce9c54-029e-477c-9adb-3c52ff6a41b1); [NTU](https://tdr.lib.ntu.edu.tw/handle/123456789/88514)

### Inferences
- Crypto is a good student-project venue because free order book data exists. Expect the same horizon decay as on NASDAQ, and use a threshold tied to fees plus spread. Binance taker fees are large relative to 1-10 s mid moves; this is a domain-knowledge claim, not verified here.

### Gaps
- I found no peer-reviewed crypto LOB paper with a cost-inclusive backtest showing robust net profits.

## 8. Statistical predictability is not profitability

### Takeaway
The most careful papers agree that high-frequency mid-price returns are predictable. The predictability is real but short-lived: up to about 2 average price changes (Kolm et al.), or 50-300 book updates, which is milliseconds to about 0.5 s (Lucchese et al.). It is measured on non-tradable mid-to-mid returns. DeepLOB's own trading simulation ignored costs. TLOB's cost-aware threshold makes most labels "stable". Briola et al. (2025) argue that ML metrics do not measure tradability.

### Cited Findings
- **Kolm, Turiel & Westray, "Deep order flow imbalance: Extracting alpha at multiple horizons from the limit order book", Mathematical Finance 33(4):1044-1081 (2023), DOI 10.1111/mafi.12413:**
  - 115 Nasdaq stocks, multi-horizon (term-structure) forecasts.
  - Models trained on order flow (OFI) "generally beat" those trained on raw order book states.
  - The "effective horizon of stock-specific forecasts is about two average price changes".
  — [IDEAS/RePEc](https://ideas.repec.org/a/bla/mathfi/v33y2023i4p1044-1081.html); [SSRN 3900141](https://papers.ssrn.com/abstract=3900141) (abstract/snippet level; the SSRN page returned 403)
- **Lucchese, Pakkanen & Veraart, "The short-term predictability of returns in order book markets: a deep learning perspective" (arXiv 2211.13777; published in International Journal of Forecasting 2024 per the brief, venue not verified here):**
  - LOBSTER data for the same 115-ticker universe as Kolm et al., 2 January 2019 to 31 January 2020. The deep analysis covers a 10-stock subset.
  - Predictability is "not just present, but ubiquitous", and persists "up to 50-300 order book updates ahead … from a few milliseconds to nearly half a second".
  - L2 beats L1. Order-flow and "volume representation" inputs "considerably outperform" the raw level representation.
  - Universal (multi-stock) models show evidence of universal patterns.
  - Caveats: "we focus on mid-to-mid returns, which are not tradable in practice". An order-book clock makes the horizon stochastic. Latency and own market impact matter. The predictability "might not be directly tradable through a standalone strategy" but may help market makers adjust quotes.
  — [arXiv 2211.13777](https://arxiv.org/abs/2211.13777)
- **Briola, Turiel, Marcaccioli, Cauderan, Aste, "Deep Reinforcement Learning for Active High Frequency Trading" (arXiv 2101.07107):**
  - PPO agents trade 1 unit of INTC (LOBSTER). Train February-April 2019, validation May 2019, test June 2019.
  - Costs are included only via the spread (buy at ask, sell at bid). Market impact is assumed zero, and no commissions are modelled.
  - Profitable on average on the test month.
  - Acknowledged limitations: trades a "single unit", and "the impact of our own trades influences in a severe way the final profit of a strategy".
  — [ar5iv](https://ar5iv.labs.arxiv.org/html/2101.07107)
- **DeepLOB:** its profit simulation used "mid-prices without transaction costs". — [DeepLOB](https://arxiv.org/abs/1808.03668)
- **TLOB:** with θ equal to the average spread, F1 drops (TSLA 41.39 / 36.48 / 30.82 at h = 50/100/200). INTC becomes 99.99% "stable". There is no profit analysis. — [TLOB](https://arxiv.org/html/2502.15757)
- **LOBCAST:** profitability on LOB-2021 "is far from guaranteed". — [LOBCAST](https://arxiv.org/abs/2308.01915)
- **Briola, Bartolucci, Aste 2025:** "high predictive power does not necessarily produce actionable trading signals" (paraphrase of the paper). They propose the transaction-level pT metric. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)
- **Lee 2024:** directional prediction from prices alone is about 50% (0.503 for AAPL), but adding volume imbalance gives about 70.5%. "Low-risk profit through directional prediction is limited" because of competition for queue position. — [Lee](https://arxiv.org/html/2409.14157v1)

### Inferences
- For the project, a clean statement is: "mid-price direction is predictable at horizons of a few price changes, mainly from order-flow and queue imbalance. Whether that edge survives spread, fees, latency and queue position is a separate and mostly unanswered question."
- A cost-aware evaluation would label moves relative to the spread, or use ask-to-bid returns.

### Gaps
- I did not obtain Kolm et al.'s exact R² or out-of-sample numbers from the full text, because the SSRN page was blocked.

## 9. Gradient boosting vs deep learning on tabular or engineered LOB features

### Takeaway
I found no head-to-head peer-reviewed LOB benchmark of LightGBM/XGBoost against DeepLOB or TLOB. The indirect evidence favours GBDT being competitive:
- Tree models remain SOTA on general medium-size tabular data.
- Most LOB predictability comes from a few engineered quantities: order-flow imbalance, queue imbalance, micro-price.
- On new data, LOBCAST's simple MLP and CNN are within a few F1 points of SOTA DL.

### Cited Findings
- **Grinsztajn, Oyallon, Varoquaux, "Why do tree-based models still outperform deep learning on tabular data?" (arXiv 2207.08815):**
  - 45 datasets. Tree-based models (XGBoost, RF) "remain state-of-the-art on medium-sized data" (about 10K samples).
  - NN weaknesses: robustness to uninformative features, rotation invariance, and learning irregular functions.
  — [arXiv 2207.08815](https://arxiv.org/abs/2207.08815)
- **Feature evidence:**
  - Order-flow inputs beat raw book inputs ([Kolm et al.](https://ideas.repec.org/a/bla/mathfi/v33y2023i4p1044-1081.html); [Lucchese et al.](https://arxiv.org/abs/2211.13777)).
  - Volume imbalance accounts for most of the directional signal ([Lee 2024](https://arxiv.org/html/2409.14157v1)).
  - DeepLOB's own conv layers are explicitly designed to rebuild micro-price and imbalance ([DeepLOB](https://arxiv.org/abs/1808.03668)).
- **Handcrafted features:** Ntakaris et al. study handcrafted and econometric LOB features for mid-price prediction with DL ("Feature Engineering for Mid-Price Prediction with Deep Learning", IEEE Access, arXiv 1904.05384). The abstract describes "very limited attempts for extracting relevant features based on limit order book data". — [arXiv 1904.05384](https://arxiv.org/pdf/1904.05384) (snippet level)
- **Simple vs SOTA models on new data:** on LOB-2021, MLP gets 55.5 F1, CNN1 57.5, LSTM 56.9, against the best 61.2 (BiNCTABL). — [LOBCAST](https://arxiv.org/abs/2308.01915)

### Inferences
- A LightGBM model on about 20-50 engineered features (multi-level OFI, queue imbalance, spread, recent returns, trade intensity) is a credible competitor to DeepLOB and TLOB on non-FI-2010 data. If it ties, that is a meaningful result, consistent with LOBCAST and the tabular literature, not a failure.
- On FI-2010 specifically, the 144 handcrafted features already exist, so tree models can be applied directly. Compare with macro F1 and the same Setup-2 split.

### Gaps
- No verified paper gives LightGBM/XGBoost against DeepLOB/TLOB on FI-2010 or LOBSTER with the same splits. The Optiver Kaggle competitions (realized volatility, closing auction) are widely reported as GBDT-dominated, but I did not verify that in this session.
