# 7. Results, explained simply

Each experiment below has the same four parts: the **question**, **what we did**, the **key numbers**, and **what it means in plain words**. Every number comes from a CSV in `artifacts/` (linked), and the [final notebook](../../submission/Final_Project_Notebook.ipynb) shows all of them.

Reminder for reading macro-F1: **0.333 = random guessing**, and the naive momentum rule is the real bar to beat.

---

## Synthetic A: which model predicts best? (5 sessions, k = 20 events)

**Question:** on a market where we planted a signal, which model finds it best?

| Model | Accuracy | Macro-F1 |
|:---|:---:|:---:|
| Naive flat | 0.698 | 0.274 |
| Naive momentum | 0.645 | 0.519 |
| Logistic regression | 0.587 | 0.564 |
| Random Forest | 0.609 | 0.578 |
| **LightGBM** | 0.620 | **0.584** |
| LSTM | 0.552 | 0.521 |
| DeepLOB | 0.597 | 0.559 |
| TLOB | 0.535 | 0.492 |

[csv](../../artifacts/experiment_a_results.csv)

**In plain words:** every learned model except TLOB beats momentum, so the pipeline finds the planted signal. LightGBM is best, Random Forest is basically tied with it, and the deep models are no better than the trees. Notice that naive flat has the *highest* accuracy and the *worst* macro-F1: exactly the trap explained in guide 6.

## Synthetic B: how fast does the signal fade?

| k (events) | LightGBM macro-F1 | Momentum |
|:---:|:---:|:---:|
| 10 | 0.521 | 0.438 |
| 20 | **0.581** | 0.519 |
| 50 | 0.534 | 0.467 |
| 100 | 0.453 | 0.413 |

**In plain words:** skill peaks at 20 events and fades after. The planted informed trader acts for 15–60 events, so this is exactly the shape we should see. The pipeline finds what's there and nothing more. (At k = 10 most moves are too small to count, so 88% of labels are FLAT and UP/DOWN are hard to catch.)

## Synthetic D: does the prediction make money?

Trading LightGBM's signals over each ~89-second test window:

| How you trade | Net profit |
|:---|:---:|
| At the mid, no costs ("fantasy") | $1,858 |
| As a taker (spread + 1.5 bp fee) | $457 |
| As a passive maker (honest fills) | $2 |

Plus a latency sweep: LightGBM's taker profit falls from $508 at 0 ms to $119 at 100 ms; DeepLOB's from $155 to −$247.

**In plain words:** costs eat about three quarters of the profit. Every millisecond of delay costs about $4. LightGBM beats DeepLOB by ~$350 even at zero latency, so its advantage comes from better predictions, not from being faster. More in guide 8.

---

## BTC A: which model predicts best on real data? (4 segments, k = 20 s)

| Model | Macro-F1 (mean ± std) |
|:---|:---:|
| Naive flat | 0.152 |
| Naive momentum | 0.413 ± 0.021 |
| Logistic regression | 0.448 ± 0.016 |
| Random Forest | 0.465 ± 0.017 |
| **LightGBM** | **0.467 ± 0.021** |
| LSTM | 0.397 ± 0.044 |
| DeepLOB | 0.401 ± 0.048 |
| TLOB | 0.449 ± 0.025 |

[csv](../../artifacts/btc/experiment_a_results.csv)

**In plain words:** on real Bitcoin, the order book does predict the next 20 seconds, but only a little: the best model is 0.13 above random and 0.05 above "the last move continues". Trees on hand-made features win; DeepLOB and the LSTM fall *below* momentum, and their scores swing more between segments (±0.045). Section "J" below confirms these gaps are real, not luck.

## BTC B: how fast does it fade?

| Horizon | LightGBM | Momentum |
|:---:|:---:|:---:|
| 10 s | 0.540 | 0.463 |
| 20 s | 0.467 | 0.413 |
| 50 s | 0.395 | 0.367 |
| 100 s | 0.350 | 0.358 |

**In plain words:** the shorter the horizon, the better. By 100 seconds the model is no better than momentum: whatever the book knows, it's about the next few seconds.

## BTC C: which features matter?

The level-1 snapshot alone gets 0.436; adding the last few seconds of history gets 0.460; everything else adds about 0.007. **In plain words:** the top of the book and the last few seconds carry nearly all the signal (details in guide 4).

## BTC D: does it make money?

| How you trade (LightGBM) | Net profit per ~10.7 h test |
|:---|:---:|
| At the mid, no costs | **+$8,055** |
| As a taker (4 bp fee, 1 s delay) | **−$36,394** |
| As a passive maker | −$478 |

**In plain words:** the predictions are worth about $0.52 per trade at the mid, but each trade costs about $2.41 in fees. Trading on every signal loses heavily. Changing the delay from 0 to 30 seconds barely matters next to the fees.

## E: can we trade only the confident signals?

Trade only the top x% most confident signals, as complete round trips. Best result: **0.70 bp per round trip** before fees (cost-aware labels, 60-second hold, top 2%), against **8 bp** of fees per round trip at 4 bp each way. [csv](../../artifacts/btc/tradability/tradability_summary.csv)

**In plain words:** being picky doubles the edge per trade, but it's still about ten times too small to pay the fees.

## Edge decomposition: where exactly does the edge go?

For the top 2% of signals (20-second hold):

```
 +2.23 bp   edge measured at the decision mid        (right 72% of the time)
 −0.75 bp   cost of entering 1 second later
 −1.01 bp   cost of crossing the spread and walking the book
 ───────
 +0.47 bp   before fees
 −8.00 bp   taker fees (4 bp in, 4 bp out)
 ───────
 −7.53 bp   per round trip
```

[csv](../../artifacts/btc/tradability/edge_decomposition.csv)

**In plain words:** even a **perfect** forecast on these rows would earn about 4.3 bp, the average size of the move, still less than the 8 bp fee. So no model, however good, could profit as a taker here. A second finding: the 1-second delay costs the confident signals almost three times as much as average signals, because confident signals predict the moves that happen soonest.

## F: does Ethereum's order book help predict Bitcoin?

Adding ETH features: **+0.016 macro-F1 at 10 s**, +0.012 at 20 s, +0.007 at 50 s, better on all 4 segments. [csv](../../artifacts/btc/cross_asset/cross_asset_summary.csv)

**In plain words:** a little, consistently, and only for a few seconds. That fits ETH moving slightly before BTC.

## G: is there any market condition where the edge survives?

Split trades by VPIN, spread and recent volatility into low / mid / high. Best cell: +1.43 bp gross. Low volatility is positive on all 4 segments; VPIN makes no difference to accuracy (0.467–0.468 in every tercile). [csv](../../artifacts/btc/regimes/regime_tradability.csv)

**In plain words:** no condition beats the costs. The edge is most reliable in **calm** markets, not in "toxic" ones as the original plan expected.

## H: can we make money by being the market maker?

Quoting on both sides of the BTC book:

| Strategy | Pessimistic fills | Optimistic fills |
|:---|:---:|:---:|
| Always quote | −$4,677 | −$4,509 |
| Rule from the research plan | −$1,562 | −$1,445 |
| LightGBM decides when to quote | −$670 | +$81 |

[csv](../../artifacts/btc/market_making/mm_summary.csv)

**In plain words:** BTC's spread is 0.0017 bp, so a market maker earns nothing for the risk of getting picked off. The model cuts the losses by 86% or more but only breaks even if you assume you're always first in the queue. Bad fills are hard to predict (AUC 0.55–0.58, barely above a coin flip's 0.5).

## I: Nasdaq stocks, large-tick vs small-tick (LOBSTER)

Best model's macro-F1 minus momentum:

| Stock | 100 ms | 1 s | 5 s | 10 s |
|:---|:---:|:---:|:---:|:---:|
| INTC (large) | −0.004 | +0.050 | **+0.113** | +0.098 |
| MSFT (large) | −0.005 | **+0.071** | +0.042 | +0.008 |
| AMZN (small) | −0.066 | −0.008 | −0.042 | −0.029 |
| GOOG (small) | −0.085 | −0.027 | −0.035 | −0.036 |

Market-making fills on INTC: **+1.24 bp** at the front of the queue, **−2.22 bp** at the back.

[csv](../../artifacts/lobster/lobster_decay.csv)

**In plain words:** on large-tick stocks the book predicts the next 1–10 seconds clearly; at 100 ms nothing moves, so there's nothing to predict. On small-tick stocks the models mostly can't beat momentum. For a market maker on a large-tick stock, being first in the queue is the difference between winning and losing, which is why speed matters so much in real trading.

## J: are the model differences real? What do the models look at?

| LightGBM vs | Difference | 95% interval | McNemar p |
|:---|:---:|:---:|:---:|
| Momentum | +0.059 | 0.050 to 0.068 | < 0.0001 |
| Logistic regression | +0.019 | 0.015 to 0.024 | < 0.0001 |
| Random Forest | +0.004 | 0.001 to 0.007 | 0.015 |
| DeepLOB | +0.058 | 0.049 to 0.068 | < 0.0001 |

[csv](../../artifacts/btc/attribution/significance_vs_lightgbm.csv)

What they rely on: LightGBM (SHAP) uses the deeper book 39% and the level-1 history 35%. DeepLOB puts twice as much weight on level 1 as on any deeper level, and 57% of its weight on the last 10 of its 100 rows.

**In plain words:** LightGBM really is better than every other model, though only barely better than Random Forest. Both LightGBM and DeepLOB end up relying on the same things: the top of the book and the last few seconds. DeepLOB also has a bias: it calls 41% of FLAT rows DOWN.

## K: the micro-price as a predictor and as a target

On large-tick stocks the mid moves toward the micro-price: by 10 s the gap has closed (β = 0.93). A one-number rule, "forecast = β × gap", scores IC **0.290** at 5 s against LightGBM's **0.209**. Training models on future *micro-price* changes instead of mid changes was never better and significantly worse in 21 of 37 cases. [csv](../../artifacts/microprice/microprice_summary.csv)

**In plain words:** on large-tick stocks, the micro-price is a simple, excellent predictor, as good as a 40-feature model. But using it as the thing to train on doesn't help. On small-tick stocks and BTC it carries little information.

---

## FI-2010: comparison with the published DeepLOB (accuracy, 3 seeds)

| k | Logistic + history | LightGBM + history | DeepLOB (ours) | DeepLOB (paper) |
|:---:|:---:|:---:|:---:|:---:|
| 10 | **86.1%** | 78.0% | 83.1% | 84.5% |
| 20 | **78.3%** | 56.6% | 74.1% | 74.8% |
| 50 | 66.5% | 64.4% | **78.2%** | 80.5% |
| 100 | 60.2% | 75.7% | **77.5%** | — |

[csv](../../artifacts/fi2010/fi2010_setup2_summary.csv)

**In plain words:** our DeepLOB matches the paper, so it's built correctly. At short horizons, a plain linear model that's allowed to see the recent history beats it: DeepLOB's short-term edge came from seeing history, not from being a deep network. At longer horizons DeepLOB clearly wins. LightGBM's collapse at k = 20 isn't a bug: the test days were calmer than the training days, and validation couldn't detect that.

## Optiver Realized Volatility

| Model | RMSPE (lower is better) |
|:---|:---:|
| Predict last 10 minutes' volatility | 0.337 |
| LightGBM | 0.235 |
| + neighbour features, past only | 0.223 |
| + neighbour features, winner's version (uses the future) | 0.214 |
| LightGBM with shuffled cross-validation | 0.214 |

[csv](../../artifacts/optiver_rvp/rvp_summary.csv)

**In plain words:** volatility is much more predictable than direction (30% better than the naive guess). About 59% of the winning trick survives without future data; the other 41% came from peeking ahead. And shuffled cross-validation makes a plain model look as good as the leaky one, which is how leakage hides in competitions.

## Optiver Trading at the Close

MAE: predict zero 5.994 → LightGBM 5.856 → retrained every 12 days + demeaned **5.848**. [csv](../../artifacts/optiver_close/close_holdout_results.csv)

**In plain words:** a 2.4% improvement over predicting nothing, about the same relative edge the competition winners had. Closing-auction moves are almost unpredictable, and the skill is measured in fractions of a basis point.

---

## The whole project in five sentences

1. Order books do predict the next few seconds, significantly but only slightly.
2. Simple models on good features match or beat deep networks at short horizons; deep networks win only at longer ones.
3. On BTC, trading costs are larger than the typical move, so no model can profit as a taker.
4. Accuracy matters far more than inference speed, but the best signals fade fastest, so speed still matters for them.
5. Honest validation changes conclusions: leaky features and shuffled cross-validation make results look much better than they are.

---

## Check yourself

1. LightGBM's BTC macro-F1 is 0.467. Explain to a friend in one sentence whether that's good.
2. Why doesn't the BTC edge decomposition just say "the model isn't accurate enough"?
3. On FI-2010, what experiment showed that DeepLOB's short-horizon advantage came from history rather than architecture?
4. What's the practical lesson of the LOBSTER market-making markouts?

<details><summary>Answers</summary>

1. It's clearly better than guessing (0.333) and than "the last move continues" (0.413), but it's a small edge, too small to pay trading costs.
2. Because even a perfect forecast would earn only about 4.3 bp, less than the 8 bp fee. The problem is cost, not accuracy.
3. Giving logistic regression 240 history features (changes over the last 1–100 rows): it then beat DeepLOB at k = 10 and 20.
4. On a large-tick stock, the same quote earns about +1 bp at the front of the queue and loses about 2 bp at the back, so being fast enough to be first matters more than prediction.

</details>

Next: [8. Execution, costs and latency](08_execution_and_latency.md)
