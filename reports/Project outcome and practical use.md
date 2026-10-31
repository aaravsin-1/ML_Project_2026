# LOB project: outcome and practical use

*As of 2026-10-31. FI-2010 numbers are means over 3 training seeds. LOBSTER results come from one trading day.*

## Bottom line

Order-book data predicts very short-term price direction slightly better than chance, mostly through simple, well-designed features. Once validation is leak-free and trading costs are realistic, that edge does not survive retail fees.

The project's main value is a rigorous evaluation of a field full of inflated results. As a trading strategy it does not work. Its practical use is helping an existing intraday trade execute more cheaply, not generating trades on its own.

## What the project shows

| # | Finding | Evidence |
| --- | --- | --- |
| 1 | Short-term direction is predictable, but only slightly | Real Coinbase BTC: models beat random guessing (1/3 macro-F1) on every one of the 4 segments |
| 2 | Predictable is not the same as profitable | Best case after paying the spread: about 0.7 bp per round trip (cost-aware labels, 60 s hold, top 2% of signals). Break-even at 4 bp taker fees is 8 bp. Every setting loses money even at 1 bp fees per side |
| 3 | Simple models with good features match deep learning at short horizons only | On BTC, LightGBM with OFI, imbalance and micro-price features does as well as or better than DeepLOB, LSTM and TLOB. On FI-2010, logistic regression with price history beats DeepLOB at k=10 (86.1% vs 83.1 ± 0.3% ours, 84.5% published) and k=20 (78.3% vs 74.1 ± 0.5%), but DeepLOB wins clearly at k=50 (78.2% vs 66.5%) and k=100 (77.5% vs 75.7% for LightGBM with history). 3 seeds; the gaps are several times the seed-to-seed spread |
| 4 | Many headline results in this field are partly leakage, and we measured how much | Optiver volatility, 4 forward folds: RMSPE 0.235 without neighbour features, 0.223 with past-only neighbours, 0.214 with the leaky 1st-place-style version. About 59% of the gain survives without future data |
| 5 | Speed is not the bottleneck at this data frequency; fees are | The trained model compiled to C++ predicts in microseconds and matches LightGBM exactly. With 1-second data, a few milliseconds of latency barely matter |
| 6 | Ethereum's order book gives a small, consistent boost for Bitcoin | Macro-F1 +0.016 at 10 s, +0.012 at 20 s, +0.007 at 50 s; improved on all 4 segments at every horizon. The lead fades within seconds |
| 7 | No market regime rescues the edge, and it lives in calm markets, not toxic ones | BTC trades split by VPIN, spread and 60 s volatility: the best regime earns about 1.4 bp per round trip before fees, below break-even even at 1 bp per side. Low volatility is the only regime that is positive on all 4 segments; high VPIN is not consistently better |
| 8 | Market making does not work on BTC; on stocks it depends on tick size and queue position | BTC's spread is 0.0017 bp ($0.01 on ~$60,000), so quoting both sides loses about $10,400 per day. An adverse-selection model cuts that by 86% or more but only breaks even with optimistic fills. On large-tick Nasdaq stocks (INTC, MSFT) a fill earns +0.7 to +1.2 bp at the front of the queue and loses about 2 bp at the back |
| 9 | Large-tick stocks are more predictable, and the signal peaks at 1–5 seconds | LOBSTER, one day, 100 ms clock: learned models beat naive momentum on INTC and MSFT from 500 ms to 10 s (INTC at 5 s: macro-F1 0.457 vs 0.344), but mostly not on AMZN and GOOG. At 100 ms a large-tick price almost never moves |
| 10 | The model ranking is statistically real, but the margin over Random Forest is tiny | Block-bootstrap 95% CIs for LightGBM's macro-F1 lead on BTC: +0.059 (0.050 to 0.068) over momentum, +0.019 (0.015 to 0.024) over logistic regression, +0.004 (0.001 to 0.007) over Random Forest |

Finding 8 is the classic result that market making is a queue-priority business: the same quote makes or loses money depending on how many shares are ahead of it, which favours firms that are fastest to join a new price level.

Finding 3 partly matches the LOBCAST benchmark (Prata et al., 2024), which found state-of-the-art LOB deep models degrade on new data. Our FI-2010 test adds that DeepLOB's short-horizon advantage comes from its inputs, while at longer horizons its architecture adds real value.

## Can retail traders make money with this?

Not by trading the signal directly. The edge is about 1 bp per trade, while a round trip at 4 bp taker fees costs 8 bp.

The 4 bp assumption is generous. Retail fee tiers on most crypto apps are several times higher. Retail traders also see data later and pay the full spread on market orders.

The signal would only pay with near-zero or negative (rebate) maker fees and exchange-level data access. That is how professional high-frequency firms operate.

Making markets instead does not fix this for a retail trader. On BTC there is no spread to earn. On large-tick stocks the spread is worth earning only from the front of the queue, and a retail order joining a queue of thousands of shares is almost always at the back, where the average fill loses about 2 bp.

## Using it for intraday trading

The findings are useful as an overlay on an intraday trade you already planned, not as the reason to trade. Four uses, most promising first:

1. **Execution timing.** Once you've decided to buy, OFI and micro-price say whether to buy now or wait a few seconds. If they point up, buy now, because the price is likely to tick against you. If they point down, wait or post a limit order. Saving 1 to 2 bp on every trade you were already making is how professional execution algorithms use these signals.
2. **Forecast volatility, not direction.** The Optiver results show volatility is far more predictable than direction. Use the forecast to size positions (smaller when high volatility is expected), set stop-losses that fit conditions, and skip bad periods.
3. **Trust the signal more in calm markets.** On BTC the short-term signal was most reliable when recent volatility was low, and weakest when it was high. A wide spread or thin book still means fills cost more, but high VPIN on its own was not a useful warning in our tests.
4. **Prefer limit orders, but know where you stand in the queue.** A maker pays lower fees and can earn the spread, but only when the spread is wide relative to the price step (large-tick stocks, or small-tick stocks when the touch is wide) and when the order is not stuck behind a long queue. On a tiny-tick asset like BTC, a resting order mostly gets filled just before the price moves against it.

**The limit:** these signals fade within seconds to about a minute (on Nasdaq stocks they peak at 1–5 s). They say nothing about where the price will be in an hour. For holds of minutes to hours, the direction call must come from elsewhere. This project helps you enter and exit that trade more cheaply and size it sensibly.

This describes research findings and is not personal trading advice.

## Proposed next experiment: smart execution

Test whether the model cuts execution cost on a fixed task, using the existing BTC execution simulator: buy 1 BTC within the next 5 minutes, repeated across the test periods.

| Method | How it buys |
| --- | --- |
| Immediate | Market order at the start of the window |
| TWAP | Equal slices spread evenly over the 5 minutes (standard benchmark) |
| Model-timed | Waits while OFI and micro-price point down, buys when they point up or time runs out |

The measure is average cost in bp against the mid price at the start of the window. If the model-timed method beats TWAP across all 4 segments, that is a result a retail trader could actually use.

## Caveats and what is still open

- **Small sample:** the BTC data covers a few days on one exchange, split into 4 segments, so error bars are wide.
- **Coarse data:** BTC has 1-second snapshots, not individual order events, and its per-level market-order columns cover only about 30–40% of traded volume. Market-making results are therefore given under three fill rules (pessimistic, queue, optimistic) that bracket the truth.
- **LOBSTER is one day:** five stocks on 21 June 2012, with the test period in the last hour of the session. It shows the tick-size pattern clearly, but the exact numbers would move on other days.
- **Many comparisons:** the regime analysis tested 54 cells, so a few results look significant by chance. Only patterns that hold across segments and models are reported as findings.
- **DeepLOB significance and attribution** need a GPU run (the Colab notebook's `deeplob_attribution` step) and are not in yet.
- **FI-2010 regime shift:** the test days are calmer than the training days (FLAT is 62% of test labels at k=20 against 49% in training), and the validation days do not reveal it. LightGBM with history features is hit hardest: 56.6% at k=20, with validation accuracy near 88% at every setting. A diagnostic run ruled out early stopping as the cause. Volatility-normalized history features are the natural fix to test.
- **Simulator depth (fixed):** orders larger than the visible book used to fill the remainder at the last price plus $0.05, which understated cost. The simulator now extends the book past its last level using the visible level spacing and size, or optionally fills partially or rejects the order. It also counts how often an order runs past the visible depth, so the 1-5 BTC execution test can report it. Past 0.1 BTC results would change only if one of those orders ran past the visible book, which is unlikely at that size; the counter will confirm this on the next rerun. One gap remains: our orders do not deplete the book in later snapshots.
- **Not yet redone:** the C++ speed test was measured while other jobs used the CPU and should be rerun on an idle machine.
