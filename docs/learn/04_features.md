# 4. Features

A **feature** is a number computed from the data that the model uses to make its prediction. The tabular models (logistic regression, Random Forest, LightGBM) see 27 hand-made features per row. The deep models (LSTM, DeepLOB, TLOB) see none of these; they get the raw 40-column book over the last 100 rows and learn their own features.

All 27 come from one place: [src/microstructure/pipeline.py](../../src/microstructure/pipeline.py), helped by [microprice.py](../../src/microstructure/microprice.py), [ofi.py](../../src/microstructure/ofi.py) and [vpin.py](../../src/microstructure/vpin.py).

**Every feature is causal**: the value at row t uses only rows up to t. The few things that are "fitted" (the PCA and the normaliser) are fitted on training rows only.

## The 27 features at a glance

| # | Name | Family | One-line meaning |
|:---:|:---|:---|:---|
| 1 | `imbalance_l1` | Snapshot | Bid vs ask size at the best prices |
| 2–4 | `imbalance_l3`, `imbalance_l5`, `imbalance_l10` | Deeper book | The same over the first 3, 5, 10 levels |
| 5 | `spread_ticks` | Snapshot | Spread measured in ticks |
| 6 | `micro_deviation` | Snapshot | How far the size-weighted price sits from the mid |
| 7–8 | `slope_bid`, `slope_ask` | Deeper book | How thin or thick each side of the book is |
| 9 | `vpin` | Trade flow | How one-sided recent trading has been |
| 10 | `ofi_l1` | Snapshot | Net buying pressure at the best prices since the last row |
| 11–13 | `ofi_pca1`–`ofi_pca3` | Deeper book | Order flow across all 10 levels, compressed to 3 numbers |
| 14–18 | `mlofi_lvl_1` … `mlofi_lvl_5` | Levels 1–5 | Order flow at each of the first 5 levels |
| 19–27 | `ofi_lag_{1,2,5}`, `imbalance_lag_{1,2,5}`, `micro_dev_lag_{1,2,5}` | Recent history | The level-1 signals 1, 2 and 5 rows ago |

One quirk worth knowing: `mlofi_lvl_1` and `ofi_l1` are the **same number** (both are order flow at level 1), so there are really 26 distinct features. It's harmless for trees, but it's a small redundancy.

---

## 1. Queue imbalance: `imbalance_l1`

```
imbalance_l1 = (bid_size_1 − ask_size_1) / (bid_size_1 + ask_size_1)        range −1 to +1
```

**Intuition.** The mid moves when a best queue is used up. If the bid queue is big and the ask queue small, the ask is more likely to run out first, so the next move is more likely up.

**Example.** Bid 900, ask 100 → (900 − 100) / 1000 = **+0.8**: strong buying side, "up" more likely. Bid 100, ask 900 → −0.8.

This is the single most studied order-book predictor. On large-tick stocks it's strong; on BTC it's weaker because the queues are small and change every second.

## 2–4. Deeper imbalances: `imbalance_l3`, `imbalance_l5`, `imbalance_l10`

The same formula, but adding up the sizes over the first 3, 5 or 10 levels on each side. They capture pressure building further from the best price. On the synthetic data, `imbalance_l3` was by far the most important feature (49% of LightGBM's gain), because that's where the generator's informed trader leaves its footprint.

## 5. `spread_ticks`

```
spread_ticks = round((best_ask − best_bid) / tick)
```

1 means the tightest possible market. A wider spread usually means less liquidity or more uncertainty.

**Why ticks and not a percentage?** The first version used spread ÷ mid. With a fixed tick, that number shrinks as the price rises, so it secretly told the model the price *level*, and the model learned to use that instead of real order-book dynamics. It was LightGBM's top feature for the wrong reason. Measuring in ticks removes the problem.

## 6. `micro_deviation` (the micro-price)

```
micro_price     = (bid_size_1 × ask_price_1 + ask_size_1 × bid_price_1) / (bid_size_1 + ask_size_1)
micro_deviation = micro_price − mid
                = imbalance_l1 × spread / 2
```

**Intuition.** The mid ignores the queue sizes. The micro-price weights each price by the *opposite* side's size, so a heavy bid pulls the estimate toward the ask, where the price is likely to go.

**Example.** Ask 100.02, bid 100.00, bid size 900, ask size 100. Imbalance = 0.8, spread = 0.02, so micro − mid = 0.8 × 0.01 = **+0.008**. The "fair" price is 100.018, not 100.01.

Experiment K studies this feature on its own (guide 7).

## 7–8. Book slopes: `slope_bid`, `slope_ask`

```
slope_bid = (bid_price_1 − bid_price_10) / (total bid size over 10 levels)
slope_ask = (ask_price_10 − ask_price_1) / (total ask size over 10 levels)
```

Price distance covered per unit of size. A **steep** slope means the book is thin: a big order would move the price a lot. A **flat** slope means deep liquidity. On real BTC data these were LightGBM's top two features (about 12% each by SHAP).

## 9. `vpin` (Volume-Synchronized Probability of Informed Trading)

**Idea** (Easley, López de Prado and O'Hara, 2012): measure how one-sided trading has been, on a *volume* clock rather than a time clock.

1. Group trades into buckets of equal volume V. On BTC, V = the segment's average traded volume per minute (training period); on synthetic data, 1,000 shares.
2. In each bucket, compute |buy volume − sell volume| / V.
3. VPIN = the average of that over the last 50 buckets. Range 0 (balanced) to 1 (entirely one-sided).
4. Each order-book row gets the VPIN of the latest bucket that had **finished** by then, so it's causal.

**Example.** Buckets of 1,000 with buy volumes 500, 800, 900 → imbalances 0, 600, 800 → VPIN = 1,400 / 3,000 ≈ 0.47.

High VPIN is supposed to signal "toxic", informed flow. Our results don't support that it helps prediction (guide 7), which matches published criticism of VPIN.

## 10–18. Order Flow Imbalance (OFI)

**Idea** (Cont, Kukanov and Stoikov, 2014): instead of looking at what's resting in the book, count what **changed** since the last row. For each level, from one row to the next:

```
Bid side, ΔW:  if the bid price went up   → +new bid size        (new buying interest)
               if it stayed the same       → new size − old size
               if it went down             → −old bid size        (bid was eaten or pulled)

Ask side, ΔV:  if the ask price went down  → +new ask size        (new selling interest)
               if it stayed the same       → new size − old size
               if it went up               → −old ask size        (ask was eaten or pulled)

OFI = ΔW − ΔV        positive = net buying pressure
```

**Example.** The bid stays at 100.00 but grows from 500 to 650 (ΔW = +150); the ask stays at 100.02 but shrinks from 300 to 100 (ΔV = −200). OFI = 150 − (−200) = **+350**: strong buying pressure.

- `ofi_l1`: OFI at level 1.
- `mlofi_lvl_1` … `mlofi_lvl_5`: OFI at levels 1–5 separately ("multi-level OFI").
- `ofi_pca1`–`ofi_pca3`: OFI is computed at all 10 levels, then **PCA** (principal component analysis) compresses those 10 correlated numbers into 3 that keep most of the information. The PCA is fitted on training rows only. The first component is roughly "total order flow across the book".

## 19–27. Lags: recent history

`ofi_lag_1` is the value of `ofi_l1` one row ago; `_2` two rows ago; `_5` five rows ago. The same for `imbalance_l1` and `micro_deviation`. On BTC one row = one second, so these give the model the last few seconds of trend.

They matter a lot. In the BTC ablation, adding these 9 lags to the 4 snapshot features was the biggest single improvement (+0.024 macro-F1), and SHAP gives them 35% of LightGBM's attention.

---

## How much each group helps (ablation, BTC, k = 20 s)

Train LightGBM with growing feature sets and watch macro-F1:

| Feature set | Count | Macro-F1 |
|:---|:---:|:---:|
| A: spread only | 1 | 0.237 |
| B: level-1 snapshot (spread, L1 imbalance, L1 OFI, micro-deviation) | 4 | 0.436 |
| C: B + the 9 lags | 13 | 0.460 |
| D: C + levels 2–3 (`imbalance_l3`, `mlofi_lvl_2`, `mlofi_lvl_3`) | 16 | 0.464 |
| E: all 27 | 27 | 0.467 |

**In plain words:** the top of the book does most of the work, a few seconds of history adds a clear bit, and everything deeper adds almost nothing. (0.333 is random guessing.)

---

## What the deep models see instead

DeepLOB, TLOB and the LSTM get the **raw 40 columns for the last 100 rows**: a 100 × 40 "image" of the book's recent history. Before that, the numbers are normalised ([src/data/normalizers.py](../../src/data/normalizers.py)) so they're on similar scales:

```
price  → (price − P0) / P0            P0 = average best-ask price over the first 100 training rows
size   → log(1 + size / V0)           V0 = median best-ask size over the first 100 training rows
```

So prices become "percentage distance from a fixed reference" (small numbers like 0.01) and sizes become log-scaled numbers around 1. Both P0 and V0 come from training data only.

The idea behind deep models is that they'll *learn* things like imbalance and OFI from this raw input by themselves. DeepLOB's first layers are designed to do exactly that.

## Extra features used in specific experiments

- **Order-flow features** (Experiments H, I, J): for each side, how much was added (limit), traded (market) and cancelled at level 1 relative to the level-1 depth, the log of each over levels 1–5, and the cancel-to-add ratio at level 1, 14 numbers in total ([exp_h_market_making.py](../../src/experiments/exp_h_market_making.py), `flow_features`).
- **FI-2010**: the dataset's own 144 features (40 raw book columns plus 104 hand-made ones from the original paper). For the "+ history" models we add **240 history features**: how each of the 40 raw columns changed over the last 1, 5, 10, 20, 50 and 100 rows (40 × 6), so the tabular models see the same past that DeepLOB sees.
- **Optiver Realized Volatility**: per 10-minute window, the realised volatility of the weighted mid price (WAP) at levels 1 and 2, spread, depth, imbalance, trade volume and order counts, also over the later parts of the window; plus the "neighbour" features built from similar windows (guide 7).
- **Optiver Close**: about 93 features, including imbalance ratios, `market_urgency` (spread × liquidity imbalance), price changes over the last 1–6 snapshots, the winner's "seconds-group" features, ranks across all stocks at the same moment, and the previous day's target.

---

## Check yourself

1. Bid size 300, ask size 700, spread 2 ticks of $0.01. What are `imbalance_l1` and `micro_deviation`?
2. From one row to the next the best bid price rises and its new size is 200; the ask is unchanged at size 400 → 400. What is `ofi_l1`?
3. Why are relative spreads (spread ÷ price) avoided?
4. Why do the lag features help so much?

<details><summary>Answers</summary>

1. imbalance = (300 − 700) / 1000 = −0.4; micro − mid = −0.4 × 0.02 / 2 = −0.004, so the fair price sits slightly below the mid.
2. ΔW = +200 (bid price went up), ΔV = 400 − 400 = 0, so OFI = +200.
3. With a fixed tick, they encode the price level, which lets the model memorise where the price is instead of learning order-book behaviour.
4. Price moves over the next seconds partly continue the last few seconds' pressure. A single snapshot can't show a trend; the lags can.

</details>

Next: [5. Models and how they were trained](05_models.md)
