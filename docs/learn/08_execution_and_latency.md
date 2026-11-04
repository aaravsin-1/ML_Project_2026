# 8. Execution, costs and latency

A prediction is only worth something if you can act on it. This guide explains how the project turns predictions into simulated trades, where the money goes, and how speed fits in.

Code: [src/execution/simulator.py](../../src/execution/simulator.py) (fills), [strategies.py](../../src/execution/strategies.py) (trading rules), [metrics.py](../../src/execution/metrics.py) (P&L), plus Experiments D, E, G, H and I.

## The simulator: replaying the recorded book

The simulator doesn't invent prices. It replays the real recorded order book and asks: if I had sent this order at this moment, what would I have paid?

### Taker orders (crossing the spread)

1. **Latency.** The order is decided at time t but arrives at the first snapshot whose timestamp is at least t + latency. On BTC, with one snapshot per second, even 1 ms of latency means the *next* second's book.
2. **Walking the book.** A buy takes the best ask. If the order is bigger than the size there, it takes the next level, and so on. Example: buy 0.1 BTC when the best ask has 0.04 BTC at $60,000.00 and the next level 0.10 at $60,000.50 → 0.04 at $60,000.00 + 0.06 at $60,000.50, average $60,000.30.
3. **Orders bigger than the visible book.** By default the book is extended past level 10 using its average spacing and size; it can also fill partially or reject the order (`overflow_policy`). The simulator counts how often this happens.
4. **Fees.** Taker fee in bp of the traded value: 4 bp on BTC (Coinbase's top tier in 2021), 1.5 bp on synthetic data.
5. **Slippage** = how much worse than the mid at decision time the average fill price was.

### Passive (maker) orders

A resting order's fill depends on **queue position**: how many orders are ahead of yours at the same price. Snapshots don't show that, so the project uses three rules that bracket the truth:

| Fill rule | When the order counts as filled | Bias |
|:---|:---|:---|
| **Pessimistic** (trade-through) | Only if the price trades *beyond* your level | Assumes you're last in the queue |
| **Queue** | Once traded volume at your price exceeds the size that was ahead of you | Middle |
| **Optimistic** | As soon as anyone trades at your price | Assumes you're first in the queue |

Reporting all three is honest: the truth lies between them.

**Why pessimistic matters:** in the first version, passive orders filled instantly whenever the price touched them, and the passive strategy "earned" more than the no-cost fantasy. That was impossible. Requiring a trade-through fixed it.

## The trading strategies

| Strategy | Rule | Used in |
|:---|:---|:---|
| **Fantasy (academic)** | Trade at the mid, no fees, no delay | D, as an upper bound |
| **Aggressive taker** | On each UP signal buy one lot, on each DOWN sell one, within an inventory limit (±500 shares synthetic, ±0.5 BTC) | D |
| **Passive maker with VPIN gate** | Join the best price on the predicted side; cancel if the signal flips, VPIN is high, or 50 events pass | D |
| **Round trips** | Enter as taker, exit k seconds later, only on the most confident signals | E, G |
| **Market making** | Quote both sides for up to 30 s, with inventory limits; "always", "rule" or "model" decides when | H, I |

## Where the money goes (BTC)

From the edge decomposition (guide 7), for the most confident 2% of signals:

```
          ┌─────────┐
 +2.23 bp │ signal  │  what the prediction is worth at the decision moment
          └────┬────┘
 −0.75 bp      │ latency: the price moves before you can act (1 s on this data)
 −1.01 bp      │ execution: crossing the spread and walking the book
 −8.00 bp      │ fees: 4 bp in + 4 bp out
          ┌────┴────┐
 −7.53 bp │  net    │
          └─────────┘
```

**The break-even rule.** With hit rate p and average move m, the edge is about (2p − 1) × m. Breaking even after cost C needs p > 0.5 + C / (2m).

| Fee per side | Hit rate needed (top 2%) | Hit rate achieved |
|:---:|:---:|:---:|
| 0 bp | 70% | 72% ✓ |
| 1 bp | 94% | 72% ✗ |
| 4 bp | over 100%, impossible | 72% ✗ |

So with zero fees the confident signals *would* make money; with real fees, no model could.

## Latency: how much does speed matter?

**Synthetic book, taker, net P&L by added delay:**

| Delay | 0 ms | 1 ms | 10 ms | 25 ms | 50 ms | 100 ms |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| LightGBM | $508 | $457 | $427 | $369 | $290 | $119 |
| DeepLOB | $155 | $104 | $71 | $14 | −$76 | −$247 |

Three lessons:
1. **Accuracy matters more than inference speed.** Both models lose about $4 per ms. DeepLOB takes about 2.3 ms longer to compute a prediction than LightGBM, which costs about $10, while LightGBM's better predictions are worth about $350. That's 35 times more.
2. **Accuracy buys time.** DeepLOB's profit runs out at about 29 ms of delay; LightGBM is still profitable at 100 ms.
3. **But better signals fade faster.** On BTC, a 1-second delay costs the most confident signals almost three times what it costs average ones. Improving the model makes speed *more* valuable, not less.

**How long a prediction takes to compute:**

| Model | Time for one prediction |
|:---|:---:|
| DeepLOB on a GPU | 2.4–5.7 ms |
| LightGBM from Python | 0.14 ms |
| The BTC LightGBM compiled to C++ | 10.5 µs (median) |
| The micro-price rule | a few arithmetic operations |

The C++ version is produced by [scripts/lgbm_to_cpp.py](../../scripts/lgbm_to_cpp.py), which turns every tree into nested if/else statements. It gives exactly the same scores as LightGBM (max difference 0, same predicted class on every row). The 10.5 µs was measured while the machine was busy, so it's an upper bound, and it covers only the model, not receiving market data or sending the order.

## The balancing rule

1. **Check the costs first.** Is the typical move over your horizon comfortably bigger than the round-trip cost? If not, change the market, the horizon or the way you trade.
2. **Pick the most accurate model whose computation fits well inside the horizon.** At a few seconds, anything fits. At 100 ms only microsecond models fit. At longer horizons, deep models become affordable and that's where they win.
3. **Spend on speed in proportion to how fast the signal fades.** Better signals need it more.
4. **For passive trading, speed is the edge.** Queue position decides whether a fill makes or loses money; the prediction only decides when to quote.

## Limits of the simulation

- **No market impact afterwards.** Our order walks the book it arrives at, but the next snapshot is the recorded one, as if we'd never traded. Fine for small orders, optimistic for big ones.
- **One-second BTC data** hides everything faster than a second; the "1 s delay" is the data's resolution, not a real system's speed.
- **Queue position is unknown,** hence the three fill rules.
- **Simple strategies:** no position sizing or clever exits.

---

## Check yourself

1. Why does even 1 ms of latency on BTC mean waiting a whole second?
2. Why are passive fills reported under three rules instead of one?
3. A model is right 72% of the time on moves averaging 4.3 bp, and a round trip costs 1.76 bp before fees. What's the most fee per side it could afford?
4. If you made your model much more accurate, would latency matter more or less? Why?

<details><summary>Answers</summary>

1. The book is only recorded once per second, so the earliest snapshot after the decision is the next one.
2. Queue position isn't observed. The pessimistic and optimistic rules bound the truth, and the queue rule estimates it.
3. Edge ≈ (2 × 0.72 − 1) × 4.3 ≈ 1.9 bp, minus 1.76 bp ≈ 0.13 bp left, so only about 0.07 bp per side. In practice even 1 bp per side is far too much.
4. More. More accurate signals tend to predict the moves that happen soonest, so they lose more value per unit of delay.

</details>

Back to the [index](README.md).
