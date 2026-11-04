# 6. Metrics: how results are measured

Every result in the project is a number from one of the metrics below. Knowing what each one rewards, and what it hides, is the difference between reading the results and being fooled by them.

Code: [src/utils/metrics.py](../../src/utils/metrics.py) for classification, [exp_k_microprice.py](../../src/experiments/exp_k_microprice.py) for regression, [exp_j_attribution.py](../../src/experiments/exp_j_attribution.py) for significance tests.

## Classification metrics (UP / FLAT / DOWN)

### The confusion matrix: where everything starts

Count how often each true class gets each predicted class. Example with 100 test rows:

|  | predicted FLAT | predicted UP | predicted DOWN |
|:---|:---:|:---:|:---:|
| **true FLAT** (50) | 30 | 10 | 10 |
| **true UP** (25) | 5 | 15 | 5 |
| **true DOWN** (25) | 5 | 5 | 15 |

Every metric below is computed from a table like this.

### Accuracy

Share of rows predicted correctly: (30 + 15 + 15) / 100 = **60%**.

**The trap:** if 70% of rows are FLAT, "always FLAT" scores 70% and is useless. That's why accuracy is *not* the main metric here, and why learned models with balanced weights often show *lower* accuracy than naive flat.

### Precision, recall and F1, per class

For one class, say UP:
- **Precision** = of the rows we called UP, how many were UP? 15 / (10 + 15 + 5) = 0.50.
- **Recall** = of the rows that were UP, how many did we catch? 15 / 25 = 0.60.
- **F1** = their harmonic mean = 2 × 0.50 × 0.60 / (0.50 + 0.60) = **0.545**. It's high only if both are high.

### Macro-F1: the project's main metric

The plain average of F1 over the three classes. Each class counts equally, however rare it is, so a model can't score well by ignoring UP and DOWN.

**Reference points:**
- Random guessing with three equal classes ≈ **0.333** (that's why the BTC threshold makes one third of rows FLAT).
- "Always FLAT" ≈ 0.15–0.27 (F1 is 0 for UP and DOWN).
- BTC LightGBM: **0.467**. That's 0.13 above random, a real but modest edge.

### Weighted-F1

Averages the per-class F1 weighted by how many rows each class has, so it's dominated by the biggest class. The DeepLOB paper reported this, so we report it on FI-2010 to compare. It flatters models when FLAT is common: our FI-2010 DeepLOB has weighted-F1 0.82 but macro-F1 0.72 at k = 10.

### MCC (Matthews correlation coefficient)

A correlation between predictions and truth that corrects for chance and class imbalance. **0** = no better than chance (always-FLAT scores exactly 0), **1** = perfect, negative = worse than chance. Used in the validation report and the live demo.

### Cross-entropy (log-loss)

Judges the predicted *probabilities*, not just the chosen class. Being confidently wrong is punished heavily. Lower is better. Naive flat scores very badly (11.2 on BTC) because it puts 100% on FLAT and is often wrong.

## Regression metrics (predicting a number)

Used in Experiment K (future price change), Optiver RVP (volatility) and Optiver Close (60-second move).

### Information coefficient (IC)

The **correlation** between the forecast and what actually happened. It ignores scale: a forecast that's always 10 times too big but points the right way still has a high IC.

How to read it: 0 = useless. In trading, an IC of **0.05** at high frequency is already useful. Our large-tick micro-price rule reaches **0.29** at 5 s, which is strong.

### R² (out-of-sample)

The share of the variation in the outcome that the forecast explains, compared with forecasting zero:

```
R² = 1 − (sum of squared errors of the model) / (sum of squared errors of always predicting 0)
```

0 = no better than predicting zero, 1 = perfect, negative = worse than zero. For price changes, R² of a few percent is typical even for good models; moves are mostly noise.

### Hit rate

The share of rows where the forecast had the right **sign** (up vs down), counting only rows where the price actually moved. 50% = coin flip. The large-tick micro-price rule is right 64–82% of the time.

### RMSPE (Optiver Realized Volatility)

Root mean squared **percentage** error: √ average of ((actual − predicted) / actual)². Lower is better. Errors on calm stocks (small actual values) count as much as errors on volatile ones. Predicting "the same volatility as the last 10 minutes" scores 0.337; our best honest model 0.223.

### MAE (Optiver Close)

Mean absolute error, in basis points. Lower is better. Predicting zero scores 5.994; our best model 5.848, a 2.4% improvement.

## Is the difference real? Significance tests

A model can beat another on one test set by luck. Two tests check whether a difference is real.

### Block bootstrap confidence intervals

1. Cut the test period into blocks of consecutive rows (5 minutes on BTC). Blocks, not single rows, because neighbouring rows are correlated; resampling single rows would make the results look more certain than they are.
2. Build a new "fake test set" by drawing blocks at random, with replacement, until it's the original length.
3. Compute the difference in macro-F1 between the two models on that fake test set.
4. Repeat many times (2,000 in Experiments G and J, 1,000 in K). The middle 95% of those differences is the **95% confidence interval**.

**How to read it:** if the whole interval is above 0, the advantage is real. LightGBM vs DeepLOB on BTC: +0.058, interval **0.049 to 0.068**, so clearly real. LightGBM vs Random Forest: +0.004, interval 0.001 to 0.007, real but tiny.

### McNemar's test

Looks only at rows where the two models **disagree**: rows only model A got right vs rows only model B got right. If the models were equally good, those two counts would be about equal. The p-value is the probability of a gap this large by chance. LightGBM vs Random Forest: 9,403 vs 9,072 rows, **p = 0.015**, unlikely to be chance but a small effect.

**p-value in one line:** a small p (below 0.05) means "this would rarely happen if there were no real difference". It does *not* say how big or important the difference is.

## Why did the model predict that? Attribution

### Split gain (LightGBM)

How much each feature's splits reduced the training loss, summed over all trees. Quick, but it splits credit arbitrarily between correlated features.

### SHAP (LightGBM)

For each prediction, SHAP divides the model's output among the features so the pieces add up exactly to the prediction. Averaging the absolute shares over many rows says how much the model relies on each feature. More trustworthy than gain. On BTC: the deeper book 39%, level-1 lags 35%, level-1 snapshot 14%, VPIN 7%, levels 2–3 5%.

### Gradient × input (DeepLOB)

For a neural network: how much would the predicted class's score change if an input changed slightly (the gradient), multiplied by the input's value. Summing over many windows shows which levels and which time steps the network focuses on. Its weakness is that it depends on the inputs' scale, so compare like with like (level vs level, time step vs time step).

## Trading metrics

- **Net P&L:** profit or loss in dollars after all costs, over the test window.
- **Fantasy (academic) P&L:** profit if you could trade at the mid with no costs. Many papers report only this.
- **Gross edge per round trip (bp):** average profit per buy-then-sell (or sell-then-buy) before fees.
- **Markout:** for a passive fill, how much the price moved in your favour (positive) or against you (negative) some seconds after the fill.
- **Sharpe ratio:** return divided by its volatility, annualised. It's **not** reported in the tables, because annualising a 90-second test window gives meaningless numbers in the hundreds.

---

## Check yourself

1. A model has 72% accuracy and macro-F1 0.28 on data where 70% of rows are FLAT. What is it probably doing?
2. Why does the bootstrap resample blocks of rows instead of single rows?
3. Model A beats model B with p = 0.001 and a macro-F1 difference of +0.002. Is A "much better"?
4. A forecast has IC 0.25 but negative R². How is that possible?

<details><summary>Answers</summary>

1. Mostly predicting FLAT. High accuracy with near-chance macro-F1 means it barely catches UP or DOWN.
2. Consecutive rows are correlated, so single-row resampling would pretend there's more independent evidence than there is and make intervals too narrow.
3. No. The difference is very likely real (small p) but tiny. Significance and size are different things.
4. The forecast points the right way (correlation) but its scale is wrong, for example 10 times too large, so its squared errors are bigger than predicting zero. That's why Experiment K rescales forecasts with β before computing R².

</details>

Next: [7. Results, explained simply](07_results.md)
