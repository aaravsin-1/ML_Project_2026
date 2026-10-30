# LOB project: outcome and practical use

*As of 2026-10-30. FI-2010 numbers are from seed 0 only (the extra-seed runs were stopped).*

## Bottom line

Order-book data predicts very short-term price direction slightly better than chance, mostly through simple, well-designed features. Once validation is leak-free and trading costs are realistic, that edge does not survive retail fees.

The project's main value is a rigorous evaluation of a field full of inflated results. As a trading strategy it does not work. Its practical use is helping an existing intraday trade execute more cheaply, not generating trades on its own.

## What the project shows

| # | Finding | Evidence |
| --- | --- | --- |
| 1 | Short-term direction is predictable, but only slightly | Real Coinbase BTC: models beat random guessing (1/3 macro-F1) on every one of the 4 segments |
| 2 | Predictable is not the same as profitable | Best case after paying the spread: about 0.7 bp per round trip (cost-aware labels, 60 s hold, top 2% of signals). Break-even at 4 bp taker fees is 8 bp. Every setting loses money even at 1 bp fees per side |
| 3 | Simple models with good features match deep learning at short horizons only | On BTC, LightGBM with OFI, imbalance and micro-price features does as well as or better than DeepLOB, LSTM and TLOB. On FI-2010, logistic regression with price history beats DeepLOB at k=10 (86.1% vs 82.9% ours, 84.5% published) and k=20 (78.3% vs 73.5%), but DeepLOB wins clearly at k=50 (78.2% vs 66.5%) and k=100 (77.1% vs 60.2%). Seed 0 only |
| 4 | Many headline results in this field are partly leakage, and we measured how much | Optiver volatility, 4 forward folds: RMSPE 0.235 without neighbour features, 0.223 with past-only neighbours, 0.214 with the leaky 1st-place-style version. About 59% of the gain survives without future data |
| 5 | Speed is not the bottleneck at this data frequency; fees are | The trained model compiled to C++ predicts in microseconds and matches LightGBM exactly. With 1-second data, a few milliseconds of latency barely matter |
| 6 | Ethereum's order book gives a small, consistent boost for Bitcoin | Macro-F1 +0.016 at 10 s, +0.012 at 20 s, +0.007 at 50 s; improved on all 4 segments at every horizon. The lead fades within seconds |

Finding 3 partly matches the LOBCAST benchmark (Prata et al., 2024), which found state-of-the-art LOB deep models degrade on new data. Our FI-2010 test adds that DeepLOB's short-horizon advantage comes from its inputs, while at longer horizons its architecture adds real value.

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
- **FI-2010 is one seed:** the history-feature runs for seeds 1 and 2 were stopped. LightGBM with history scored only 56.7% at k=20 against 77.4% at k=10, which looks like a training problem to check.
- **Simulator limit:** orders larger than the 10 visible levels fill the remainder at the last price plus $0.05, which understates cost. It does not affect the 0.1 BTC trades used so far, but must be fixed before the 1-5 BTC execution test.
- **Not yet redone:** the C++ speed test was measured while other jobs used the CPU and should be rerun on an idle machine.
