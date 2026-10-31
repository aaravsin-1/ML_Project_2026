# Predicting Limit Order Books, and Whether the Prediction Pays: Accuracy, Latency and Cost

**Machine Learning course project, final report**
**Author:** Aarav Singhal
**Date:** October 2026
**Code:** this repository (`README.md` has the full results; this report is the summary and the argument)

---

## Abstract

Can machine learning predict the next few seconds of price movement from a limit order book (LOB), and can a trading system turn that prediction into profit? Profit depends on three things at once: how **accurate** the prediction is, how **fast** the system acts on it (latency), and what it **costs** to trade.

I built a leakage-controlled pipeline with:
- microstructure features (order flow imbalance, VPIN, micro-price);
- a ladder of models from naive baselines to LightGBM, DeepLOB and a transformer (TLOB);
- an execution simulator that models spread crossing, book walking, fees, latency and passive fills;
- a C++ inference benchmark.

I tested it on five real datasets: Coinbase BTC-USD, Nasdaq event data (LOBSTER), FI-2010, and two Optiver competitions.

**Main results:**
- Direction is predictable, but only slightly. On BTC, LightGBM is significantly better than every baseline (block bootstrap and McNemar tests).
- Simple models match deep learning at short horizons. Deep learning wins only at longer horizons (FI-2010).
- Prediction does not become profit when taking liquidity on BTC. Splitting the round trip shows why: of the 2.2 bp edge of the best signals, reacting one second late costs 0.75 bp, crossing the spread and walking the book costs 1.0 bp, and the fees are 8 bp. A *perfect* direction forecast would earn about 4.3 bp, still below the fees.
- On a synthetic book with millisecond latencies, accuracy matters far more than inference speed. A more accurate model also tolerates more latency before its profit runs out.

The conclusion is a rule for balancing the two:
1. Costs set the bar.
2. Accuracy decides whether you clear it.
3. Latency only matters relative to how fast the signal decays, and better signals decay faster.

---

## 1. Introduction

### 1.1 The question

A limit order book is the list of all resting buy orders (bids) and sell orders (asks) for an asset, at each price. Researchers have reported that deep networks can predict short-term mid-price moves from the book with 80%+ accuracy (DeepLOB, Zhang et al. 2019). This project asks two questions:

1. **Prediction (ML):** How much short-horizon information does the book contain? Which models extract it, and are the gains statistically real?
2. **Exploitation (systems):** Can a trading system act on the prediction fast enough, and cheaply enough, to make money?

### 1.2 Why accuracy alone is not enough

I chose this topic because the final result depends on both halves together. For one round trip (enter, hold for horizon $h$, exit):

$$\text{net edge} \;=\; \underbrace{E[\,s \cdot \Delta m_{t \to t+h}\,]}_{\text{accuracy}} \;-\; \underbrace{\big(E[s\,\Delta m_{t\to t+h}] - E[s\,\Delta m_{t+L \to t+L+h}]\big)}_{\text{latency } L} \;-\; \underbrace{c_{\text{cross}}}_{\text{spread, depth}} \;-\; \underbrace{2f}_{\text{fees}}$$

where $s = \pm 1$ is the predicted direction and $\Delta m$ the mid-price change.
- The **ML model** controls the first term.
- The **system's speed** controls the second.
- The **market and exchange** control the last two.

If moves are roughly symmetric with average size $m$ and the model is right with probability $p$, the first term is about $(2p-1)\,m$. Breaking even after a total round-trip cost $C$ then needs a hit rate of

$$p^* \;=\; \tfrac{1}{2} + \frac{C}{2m}.$$

This formula frames the whole report. Better models raise $p$, faster systems shrink $C$, and when $C > m$, no model can win at all.

---

## 2. Data

| Dataset | What it is | Size used | Role |
|:---|:---|:---|:---|
| **Synthetic** | Simulated large-tick book with a planted "informed trader" signal | 5 sessions × 60,000 events | Controlled tests, latency sweep. Says nothing about real markets |
| **Coinbase BTC-USD** | 15-level book, 1 s snapshots, real trade tape | ~12 days (Apr 2021), 4 segments of ~3 days | Main real-data testbed: models, execution, regimes, market making |
| **LOBSTER** | Every Nasdaq order event, 10 levels, for INTC, MSFT, AAPL, AMZN and GOOG | 1 day (21 June 2012), resampled to 100 ms | Large-tick vs small-tick stocks; horizons from 100 ms to 30 s |
| **FI-2010** | Standard academic LOB benchmark (5 Finnish stocks, 10 days) | Setup 2: train days 1–7, test days 8–10 | Comparison with the published DeepLOB |
| **Optiver Realized Volatility** | Kaggle 2021, 10-minute book/trade buckets | Full training set, 4 forward folds | Leakage case study |
| **Optiver Trading at the Close** | Kaggle 2023, closing auction | 481 days (400 train / 81 holdout) | Regression on auction imbalance |

**Tick size** matters for everything that follows. INTC and MSFT are *large-tick* stocks: the 1-cent tick is 3.3–3.7 bp of the price, the spread sits at one tick 77% of the time, and queues are deep. AAPL, AMZN and GOOG are *small-tick* stocks (0.17–0.45 bp). BTC is extremely small-tick: $0.01 on about $60,000, which is 0.0017 bp.

---

## 3. Method

### 3.1 Prediction task

- **Classification.** Predict whether the (smoothed) mid price $k$ steps ahead is UP, FLAT or DOWN. The UP/DOWN threshold is set on training rows only. On BTC it is calibrated so that one third of training rows are FLAT, which puts random guessing at macro-F1 ≈ 0.333.
- **Regression** (micro-price experiment). Predict the future mid change in basis points (1 bp = 0.01%), scored by the information coefficient (IC, the correlation between forecast and outcome) and out-of-sample R².
- **Metric.** Macro-F1 averages the F1 score of the three classes, so a model cannot score well by always predicting the majority class. Every learned model uses balanced class weights, so all of them optimise the same objective.

### 3.2 Features

All features are causal: row $t$ uses only snapshots at or before $t$.
- **Book snapshot:** spread in ticks; queue imbalance at levels 1, 3, 5 and 10; book slopes.
- **Order Flow Imbalance (OFI):** Cont et al. (2014), the net change in buy minus sell pressure at the best quotes. Also multi-level OFI compressed by PCA, with the loadings fitted on training data.
- **VPIN:** Easley et al. (2012), a volume-clock measure of how one-sided recent trading was ("toxicity").
- **Micro-price deviation:** the imbalance-weighted mid minus the mid, $I \cdot \text{spread}/2$.
- **Lags** of the level-1 signals, and per-level limit, market and cancel order flows (BTC and LOBSTER).

### 3.3 Models (tiers 0–5)

| Tier | Model | Input |
|:---:|:---|:---|
| 0 | Naive momentum (repeat the last realised move), naive flat | none |
| 1 | Logistic regression | 27 tabular features |
| 2 | Random Forest | 27 tabular features |
| 3 | LightGBM (gradient-boosted trees, early stopping on validation) | 27 tabular features |
| 4 | LSTM | windows of 100 raw book snapshots |
| 5 | DeepLOB (CNN + Inception + LSTM), TLOB (dual-attention transformer) | windows of 100 raw book snapshots |

### 3.4 Validation: guarding against leakage

High-frequency data makes it easy to fool yourself, so most of the engineering went here.
- **Chronological, purged splits.** Each split is 70/15/15 in time. The last $k$ rows before each boundary are dropped, because their labels look into the next split.
- **Train-only fitting.** Normalisers, PCA loadings and label thresholds are all fitted on the training period.
- **Same test rows for every model.** Sequence windows end exactly at the tabular test rows.
- **Significance:**
  - a moving-block bootstrap of the macro-F1 difference, using 5-minute blocks because returns are autocorrelated;
  - McNemar's test on the rows each model gets right;
  - TreeSHAP attribution.
- **Negative controls.** A model trained on shuffled targets must score about zero. In Experiment K it averages IC −0.0005 over 37 cases.
- **Tests.** 40 automated tests, including a check that features are causal, the purge gap, the simulator's fills, and an end-to-end test where a planted signal must be found and the shuffled control must not find it.

### 3.5 Execution simulator

The simulator turns predictions into trades against the recorded book:
- **Taker orders** cross the spread and walk the book level by level. Orders larger than the visible depth are extrapolated, filled partially or rejected, by a configurable rule.
- **Fees:** 4 bp taker on BTC (Coinbase's top tier) and 1.5 bp on the synthetic book; Nasdaq-style per-share fees on LOBSTER.
- **Latency** in wall-clock time: an order reaches the book at the first snapshot at or after $t + L$.
- **Passive (maker) orders** use three fill rules that bracket the truth, because queue position is not observed: *pessimistic* (filled only if the price trades through the order), *queue* (filled once traded volume at the order's price exceeds the size ahead of it), and *optimistic* (front of the queue).

### 3.6 C++ inference

A code generator (`scripts/lgbm_to_cpp.py`) compiles the trained BTC LightGBM model (327 trees, 20,601 leaves) into plain C++. A benchmark checks it against LightGBM and times it per prediction. A second benchmark times feature extraction, inference with a randomly generated 50-tree ensemble and a simulated order in one loop. AVX2 versions of the features are in `cpp/include/features_avx2.hpp` and lab 06.

---

## 4. Results: prediction

### 4.1 Which model predicts best? (BTC, 20 s ahead, 4 segments)

| Model | Macro-F1 (mean ± std) |
|:---|:---:|
| Naive momentum | 0.413 ± 0.021 |
| Logistic regression | 0.448 ± 0.016 |
| Random Forest | 0.465 ± 0.017 |
| **LightGBM** | **0.467 ± 0.021** |
| LSTM | 0.397 ± 0.044 |
| DeepLOB | 0.401 ± 0.048 |
| TLOB | 0.449 ± 0.025 |

Random guessing scores 0.333.

**Significance of LightGBM's lead** (pooled over segments; block-bootstrap 95% CI):

| LightGBM vs | Macro-F1 difference | 95% CI | McNemar p |
|:---|:---:|:---:|:---:|
| Naive momentum | +0.059 | 0.050 to 0.068 | < 0.0001 |
| Logistic regression | +0.019 | 0.015 to 0.024 | < 0.0001 |
| Random Forest | +0.004 | 0.001 to 0.007 | 0.015 |
| DeepLOB | +0.067 on average; better on 4/4 segments | *pending: significance test running on Colab* | — |

The signal is real but small: about +0.05 macro-F1 over momentum. LightGBM is significantly better than all the tabular baselines, but only barely better than Random Forest. The deep sequence models do worse than trees on hand-crafted features here. On the synthetic data the ranking is the same (LightGBM 0.584 ± 0.012, DeepLOB 0.559 ± 0.010).

### 4.2 Do deep models ever win? (FI-2010, accuracy %, 3 seeds)

| Horizon k (events) | Logistic + history | LightGBM + history | DeepLOB (ours) | DeepLOB (published) |
|:---:|:---:|:---:|:---:|:---:|
| 10 | **86.1** | 78.0 | 83.1 ± 0.3 | 84.5 |
| 20 | **78.3** | 56.6 | 74.1 ± 0.5 | 74.8 |
| 50 | 66.5 | 64.4 | **78.2 ± 0.2** | 80.5 |
| 100 | 60.2 | 75.7 | **77.5 ± 0.4** | — |

- My DeepLOB reproduces the paper to within 0.7–2.3 points.
- At short horizons, a *linear* model given a window of past book changes beats it. DeepLOB's short-horizon advantage over snapshot features comes from seeing history, not from its architecture.
- DeepLOB wins clearly at longer horizons.
- **LightGBM's poor k = 20 score is a distribution shift, not a bug.** The test days are calmer: FLAT is 62% of test labels against 49% of training labels. A diagnostic script (`scripts/diagnose_fi2010_lightgbm.py`) found validation accuracy of 86–88% for every setting, while test accuracy ranged from 55% to 88%. Validation simply cannot detect the shift here.

### 4.3 How fast does the signal decay?

| Horizon | BTC LightGBM macro-F1 | BTC momentum |
|:---:|:---:|:---:|
| 10 s | 0.540 | 0.463 |
| 20 s | 0.467 | 0.413 |
| 50 s | 0.395 | 0.367 |
| 100 s | 0.350 | 0.358 |

On BTC, skill is highest at the shortest horizon the 1 s data allows, and gone by 100 s.

LOBSTER's 100 ms clock shows the full curve. On the large-tick stocks the best learned model beats momentum from 500 ms to 10 s, peaking at 1–5 s (INTC at 5 s: +0.113 macro-F1). At 100 ms nothing is predictable, because a large-tick mid almost never moves that fast (99% of labels are FLAT). On the small-tick stocks the learned models mostly lose to momentum at every horizon.

**Implication for latency:** the useful information lives at horizons of about 1–10 s and decays after that. A system has to act well inside that window.

### 4.4 What do the models use?

- **Ablation** (BTC, LightGBM): the level-1 snapshot alone reaches 0.436 macro-F1. Lags add +0.024, and everything deeper in the book adds less than +0.01.
- **TreeSHAP** shares: deeper-book shape 39%, level-1 lags 35%, level-1 snapshot 14%, VPIN 7%, levels 2–3 5%. Gain importance under-weights the lags (25%).
- **Cross-asset:** Ethereum's book adds +0.016 macro-F1 at 10 s, improving all 4 segments. The gain fades with horizon, which fits ETH leading BTC by a few seconds.

### 4.5 The micro-price: a cheap, fast predictor (Experiment K)

The spec suggested the micro-price as a better *target*. On the large-tick stocks the mid closes the gap to the micro-price within about 10 s (regression slope β = 0.93).

| IC on future mid change | Large-tick, 5 s | Small-tick, 1 s | BTC, 10 s |
|:---|:---:|:---:|:---:|
| Micro-price rule (β × gap, one parameter) | **0.290** | 0.062 | 0.024 |
| LightGBM (about 40 features) | 0.209 | **0.155** | **0.263** |
| LightGBM trained on micro-price changes | 0.080 | −0.007 | 0.260 |
| Shuffled-target control | −0.003 | −0.003 | 0.002 |

- **As a predictor:** on large-tick stocks, a one-parameter rule that costs a few arithmetic operations matches or beats LightGBM from 5 s on.
- **As a training target:** worse than the mid in 21 of 37 cases (block-bootstrap CIs), and never better.

The predictor result matters for the latency argument (§6).

### 4.6 A leakage case study (Optiver Realized Volatility)

The winning Kaggle solution used nearest-neighbour features drawn from *all* time windows, including future ones.
- Restricted to past windows only, those features keep about **59%** of their gain (RMSPE 0.223 vs 0.214, against 0.235 without them).
- Shuffled K-fold CV makes a plain model look as good as the leaky one (0.214).

Validation design changes the conclusion, which is why §3.4 exists.

---

## 5. Results: from prediction to profit

### 5.1 BTC: the edge exists, costs remove it

| LightGBM on BTC, test windows | Net P&L |
|:---|:---:|
| Trading at the mid, no fees (what many papers report) | **+$8,055 ± 988** |
| Aggressive taker: crosses the spread, 4 bp fee, 1 s latency | **−$36,394 ± 7,271** |
| Passive maker with a VPIN gate, pessimistic fills | −$478 ± 776 |

Fees ($37.6k) are 4.7× the fantasy profit.

### 5.2 Where exactly does the edge go?

I split each taker round trip (`scripts/decompose_btc_edge.py`) into the edge at the decision mid, the cost of reacting one snapshot later, and the cost of crossing the spread and walking the book:

| Signals traded (20 s hold) | Trades | Edge at decision mid | − 1 s delay | − crossing | = gross | Hit rate | Break-even hit rate (0 / 1 / 4 bp fee) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| All | 6,359 | 1.01 bp | 0.27 | 0.73 | 0.01 | 60.7% | 63% / 90% / impossible |
| Top 10% by confidence | 3,776 | 1.97 bp | 0.65 | 0.89 | 0.43 | 68.9% | 68% / 92% / impossible |
| Top 2% by confidence | 1,590 | 2.23 bp | 0.75 | 1.01 | 0.47 | 71.8% | 70% / 94% / impossible |

Hit rate is measured at the decision mid, on rows where the mid moved.

![Accuracy, latency and cost](../artifacts/figures_real/accuracy_latency_balance.png)

Four findings:
1. **Better accuracy doubles the edge.** Filtering to confident signals raises the hit rate from 61% to 72% and the edge from 1.0 to 2.2 bp.
2. **Better signals are more latency-sensitive.** The one-second delay costs 0.27 bp on all signals but 0.75 bp on the top 2%. The most confident signals predict the moves that happen soonest, so waiting hurts them most. The 1 s here is the data's resolution, not a real system latency, but the direction of the effect is the point.
3. **Without fees, confident signals do clear the bar** (72% hit rate against 70% needed). With a 1 bp fee they would need 94%.
4. **At the real 4 bp taker fee, no accuracy is enough.** The average absolute 20 s move on these rows is 4.3 bp, so even a perfect forecast earns less than the 8 bp round-trip fee. In the language of §1.2: $C > m$, so $p^* > 1$.

Splitting trades by regime (VPIN, spread, volatility terciles, 54 cells, Experiment G) finds no regime where the edge survives costs. The best cell earns +1.43 bp gross. The edge is most consistent in *calm* markets, not toxic ones.

### 5.3 Avoiding the spread: market making

Posting passive orders earns the spread instead of paying it, but those orders tend to fill when the price moves against them ("adverse selection").
- **BTC:** naive two-sided quoting loses about $10,400 a day, because there is no spread to earn (0.0017 bp). An adverse-selection model cuts the losses by 86% or more, but breaks even only under optimistic fills.
- **LOBSTER, large-tick stocks:** the outcome depends almost entirely on **queue position**. Measured 30 s after the fill:

| Stock | Fill at the back of the queue | Fill at the front of the queue |
|:---|:---:|:---:|
| INTC | −2.22 bp | +1.24 bp |
| MSFT | −1.78 bp | +0.70 bp |

With about 13,000 shares queued at the best price, a new order is almost always at the back. Getting to the front means posting first when a new price level opens, which is purely a speed race. That is the situation where latency, not prediction, is the edge.

---

## 6. Discussion: balancing accuracy and latency

### 6.1 How much latency costs, and how fast each model is

| Step | Time | Source |
|:---|:---:|:---|
| BTC data interval (fastest possible reaction in this data) | 1 s | data |
| LOBSTER clock used | 100 ms | resampling |
| DeepLOB, one prediction on GPU | 2.4–5.7 ms | Exp. D, synthetic / BTC |
| LightGBM, one prediction from Python | 0.14 ms | Exp. D |
| Trained LightGBM compiled to C++ (327 trees) | 10.5 µs P50, 17.9 µs P99 | C++ benchmark, busy machine, upper bound |
| Features + random 50-tree ensemble + simulated order, C++ loop | 1.9 µs P50, 6.3 µs P99 | C++ benchmark |
| Micro-price rule | a few arithmetic operations | not timed separately |

The C++ model reproduces LightGBM exactly: maximum absolute score difference 0, 100% agreement on the predicted class.

**Latency sweep** (synthetic large-tick book, aggressive taker, events about 10 ms apart, horizon 20 events ≈ 200 ms):

| Added latency | 0 ms | 1 ms | 10 ms | 25 ms | 50 ms | 100 ms |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| LightGBM net P&L | $508 | $457 | $427 | $369 | $290 | $119 |
| DeepLOB net P&L | $155 | $104 | $71 | $14 | −$76 | −$247 |

### 6.2 What the evidence says

**1. Costs set the bar, and no model clears an impossible bar.** On BTC as a taker, the round-trip cost (fees + spread + delay) is larger than the typical move, so the problem is not solvable by better ML. Neither is it solvable by lower latency: going from 1 s to 0 s latency improves the P&L by only about 5% (−$36.4k to −$34.5k). The first step is lowering costs: a lower fee tier, posting passively, or choosing an instrument where the typical move is large relative to the spread (large-tick stocks).

**2. Once the bar can be cleared, accuracy matters most, until latency approaches the signal's horizon.**
- In the synthetic sweep, LightGBM's lead over DeepLOB is about $350 at *zero* latency. It comes from better signals, not faster inference.
- Both models lose about $4 per millisecond of latency, so DeepLOB's extra ~2.3 ms of inference costs only about $10.
- At these speeds, picking the more accurate model is worth 35 times more than picking the faster one.

**3. Accuracy buys latency budget.** A model's profit runs out when the latency cost eats its edge. DeepLOB's smaller edge is gone by about 29 ms (a seventh of the 200 ms horizon). LightGBM's larger edge is still positive at 100 ms, half the horizon. The more accurate the model, the slower the system it can afford.

**4. But better signals decay faster, so they need lower latency.** On BTC, the most confident signals lose almost three times as much to a one-second delay as average signals. Improving the model moves the edge toward shorter horizons, where speed matters more. Accuracy and latency are complements, not substitutes: a more accurate model makes faster execution *more* valuable.

**5. The best models for low latency are the simple ones that are still accurate.**
- At short horizons, simple models match or beat deep ones: logistic regression with history on FI-2010 at k = 10–20, LightGBM on BTC, the one-parameter micro-price rule on large-tick stocks.
- These models also run in microseconds (LightGBM compiled to C++) or nanoseconds (the micro-price rule), where DeepLOB takes milliseconds on a GPU.
- Deep models win only at longer horizons (FI-2010 k = 50–100), and that is where their millisecond inference is affordable.
- So the accuracy–latency frontier is not one curve that forces a trade-off. *Which* model is best depends on the horizon, and the horizon sets the latency budget.

**6. For passive strategies, speed becomes the edge itself.** On large-tick stocks the same quote earns +0.7 to +1.2 bp at the front of the queue and loses about 2 bp at the back. Prediction helps decide *when* to quote (the adverse-selection model cut BTC market-making losses by 86% or more), but queue position, which is a latency race, decides whether quoting pays.

### 6.3 The balancing rule

Putting it together, for a strategy with signal horizon $h$, system latency $L$ and round-trip cost $C$:

1. **Check feasibility first:** is the typical move $m$ over horizon $h$ comfortably larger than $C$? If not, no model or system will work, so change the instrument, horizon or execution style.
2. **Choose the model by accuracy at that horizon,** among models whose inference fits well inside $h$. At $h$ of a few seconds, every model here fits. Around 100 ms only microsecond models fit, and at longer horizons deep models become worth their cost.
3. **Spend on latency in proportion to how fast the signal decays.** Keep $L$ a small fraction of $h$. The fraction must be smaller for better signals, because their edge is concentrated in the first moments.
4. **For passive execution,** latency decides queue position and dominates. Prediction is the filter, speed is the edge.

---

## 7. Limitations

- **Small real samples.** BTC covers about 12 days from one exchange, and LOBSTER one day per stock (with the test period in the last hour). Treat the LOBSTER numbers as a case study.
- **Coarse BTC data.** With 1 s snapshots, the "1 s delay" is the data's resolution, not a system latency. Sub-second effects on BTC are invisible.
- **Queue position is unobserved,** so passive fills are reported as bounds (pessimistic / queue / optimistic), not point estimates.
- **No persistent price impact:** our orders walk the snapshot they arrive on, but do not deplete later snapshots.
- **Synthetic latency sweep:** the generator plants the signal, so the $/ms figures show the mechanism, not real-market magnitudes. The test windows are short (about 90 s per seed), so the P&L is noisy.
- **C++ timings** were taken on a busy machine and are an upper bound. They cover inference only, not network or feed handling ("wire-to-order").
- **Limited tuning** of the deep models: few epochs and little hyperparameter search.
- **Many comparisons** in the regime study (54 cells), so only patterns that hold across segments and models are reported.

## 8. Future work

- **Lower costs before better models:** passive execution on large-tick stocks with a queue-position model, using full LOBSTER order-by-order data over many days.
- **Deep models on sub-second LOBSTER data,** to test whether the deep-wins-at-long-horizons pattern holds on Nasdaq.
- **The systems half:** an order-book engine, an ITCH feed parser and kernel-bypass networking to measure real wire-to-order latency (planned in `cpp/LEARNING_PATH.md`), plus an idle-machine rerun of the C++ model benchmark.
- **A volatility-normalised history feature set** for FI-2010, to test the distribution-shift explanation.

## 9. Conclusion

Short-horizon price moves in limit order books are predictable: significantly, but slightly. Tree models on hand-crafted microstructure features match or beat deep sequence models at the horizons that matter for trading. A one-parameter micro-price rule is as good as LightGBM on large-tick stocks a few seconds ahead.

Turning prediction into profit is a three-way balance:
- **Costs** decide whether a profit is possible at all. On BTC as a taker, even a perfect forecast loses.
- **Accuracy** decides whether you clear the bar. At millisecond latencies it is worth far more than inference speed.
- **Latency** matters in proportion to how fast the signal decays, and better signals decay faster.

The practical recipe that falls out:
1. Pick a market and execution style where the typical move exceeds the cost.
2. Pick the most accurate model whose inference fits well inside the signal's horizon. At short horizons that is a simple model compiled to fast code.
3. Invest in speed in proportion to how concentrated the signal is in its first moments.

---

## Reproducibility

- **Run everything:** `notebooks/colab_run_all.ipynb` on Colab (GPU steps); CPU experiments on a laptop (`README.md` → *Installation & Local Quickstart*).
- **Tests:** `python -m pytest tests` (40 tests). Automatic result checks for Experiment K: `python scripts/check_microprice_results.py` (19 checks).
- **Numbers in §5.2:** `python scripts/decompose_btc_edge.py` → `artifacts/btc/tradability/edge_decomposition.csv`.
- **Figures:** `python scripts/make_real_data_figures.py`.
- **Full results, per experiment:** `README.md` (Experiments A–K, FI-2010, Optiver, C++).

## References

1. Cont, R., Kukanov, A., & Stoikov, S. (2014). The Price Impact of Order Book Events. *Journal of Financial Econometrics*, 12(1), 47–88.
2. Easley, D., López de Prado, M., & O'Hara, M. (2012). Flow Toxicity and Liquidity in a High-Frequency World. *Review of Financial Studies*, 25(5), 1457–1493.
3. Stoikov, S. (2018). The Micro-Price: A High-Frequency Estimator of Future Prices. *Quantitative Finance*, 18(12), 1959–1966.
4. Zhang, Z., Zohren, S., & Roberts, S. (2019). DeepLOB: Deep Convolutional Neural Networks for Limit Order Books. *IEEE Transactions on Signal Processing*, 67(11), 3001–3012.
5. Ntakaris, A., Magris, M., Kanniainen, J., Gabbouj, M., & Iosifidis, A. (2018). Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods. *Journal of Forecasting*, 37(8), 852–866.
6. López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
7. Tran, D. T., Kanniainen, J., Gabbouj, M., & Iosifidis, A. (2021). Data Normalization for Bilinear Structures in High-Frequency Financial Time-Series. *ICPR 2020*.
