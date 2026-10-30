# LOB project: outcome and practical use

*As of 2026-10-30. Numbers marked provisional come from runs that are still finishing.*

## Bottom line

Order-book data predicts very short-term price direction slightly better than chance, mostly through simple, well-designed features. Once validation is leak-free and trading costs are realistic, that edge does not survive retail fees.

The project's main value is a rigorous evaluation of a field full of inflated results. As a trading strategy it does not work. Its practical use is helping an existing intraday trade execute more cheaply, not generating trades on its own.

## What the project shows

| # | Finding | Evidence |
| --- | --- | --- |
| 1 | Short-term direction is predictable, but only slightly | Real Coinbase BTC: models beat random guessing (1/3 macro-F1) on every one of the 4 segments |
| 2 | Predictable is not the same as profitable | Gross edge about 1 bp per round trip; break-even at 4 bp taker fees is 8 bp. More selective trading and cost-aware labels have not closed the gap so far (provisional) |
| 3 | Simple models with good features match or beat deep learning | On BTC, LightGBM with OFI, imbalance and micro-price features does as well as or better than DeepLOB, LSTM and TLOB. On FI-2010 at k=10, logistic regression with price history gets 86.1% accuracy vs DeepLOB's published 84.5% (provisional, more seeds running) |
| 4 | Many headline results in this field are partly leakage, and we measured how much | Optiver volatility fold 3: past-only neighbour features RMSPE 0.2023 vs 0.1937 for the leaky 1st-place-style version and 0.2175 without them. Most of the gain survives without future data |
| 5 | Speed is not the bottleneck at this data frequency; fees are | The trained model compiled to C++ predicts in microseconds and matches LightGBM exactly. With 1-second data, a few milliseconds of latency barely matter |
| 6 | Ethereum's order book gives a small, consistent boost for Bitcoin | About +0.02 macro-F1 on the segments finished so far, e.g. 0.518 to 0.540 at k=10 (provisional) |

Finding 3 matches the LOBCAST benchmark (Prata et al., 2024), which found state-of-the-art LOB deep models degrade on new data.

## Can retail traders make money with this?

Not by trading the signal directly. The edge is about 1 bp per trade, while a round trip at 4 bp taker fees costs 8 bp.

The 4 bp assumption is generous. Retail fee tiers on most crypto apps are several times higher. Retail traders also see data later and pay the full spread on market orders.

The signal would only pay with near-zero or negative (rebate) maker fees and exchange-level data access. That is how professional high-frequency firms operate.

## Using it for intraday trading

The findings are useful as an overlay on an intraday trade you already planned, not as the reason to trade. Four uses, most promising first:

1. **Execution timing.** Once you've decided to buy, OFI and micro-price say whether to buy now or wait a few seconds. If they point up, buy now, because the price is likely to tick against you. If they point down, wait or post a limit order. Saving 1 to 2 bp on every trade you were already making is how professional execution algorithms use these signals.
2. **Forecast volatility, not direction.** The Optiver results show volatility is far more predictable than direction. Use the forecast to size positions (smaller when high volatility is expected), set stop-losses that fit conditions, and skip bad periods.
3. **Know when not to trade.** A wide spread, a thin book or high VPIN all signal toxic conditions where fills tend to go against you. Avoiding these periods improves results on its own.
4. **Prefer limit orders.** A maker earns the spread instead of paying it and usually pays lower fees. With a maker fee near zero, the economics change a lot.

**The limit:** these signals fade within seconds to about a minute. They say nothing about where the price will be in an hour. For holds of minutes to hours, the direction call must come from elsewhere. This project helps you enter and exit that trade more cheaply and size it sensibly.

This describes research findings and is not personal trading advice.

## Proposed next experiment: smart execution

Test whether the model cuts execution cost on a fixed task, using the existing BTC execution simulator: buy 1 BTC within the next 5 minutes, repeated across the test periods.

| Method | How it buys |
| --- | --- |
| Immediate | Market order at the start of the window |
| TWAP | Equal slices spread evenly over the 5 minutes (standard benchmark) |
| Model-timed | Waits while OFI and micro-price point down, buys when they point up or time runs out |

The measure is average cost in bp against the mid price at the start of the window. If the model-timed method beats TWAP across all 4 segments, that is a result a retail trader could actually use.

## Caveats and what is still running

- **Small sample:** the BTC data covers a few days on one exchange, split into 4 segments, so error bars are wide.
- **Coarse data:** 1-second snapshots, not individual order events, and no queue-position model, so passive (maker) fills are approximate.
- **FI-2010 needs more work:** the history-feature result needs seeds 1 and 2 to finish. LightGBM with history scored only 56.7% at k=20 against 77.4% at k=10, which looks like a training problem to check before writing it up.
- **Still running:** the BTC tradability test, ETH cross-asset test and Optiver volatility rerun should finish within about 30 minutes. The FI-2010 seed runs will take a few more hours. After that, the C++ speed test is rerun on an idle machine and all numbers here are updated.
