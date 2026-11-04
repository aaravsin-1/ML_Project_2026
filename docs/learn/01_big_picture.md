# 1. The big picture

## The question

Can a machine learning model predict where a price goes in the next few seconds by looking at the **order book**, and can that prediction actually make money once you pay to trade?

Those are two different questions, and the project's main finding is that the answer to the first is "yes, slightly" while the answer to the second is "not with these costs".

## What an order book is

An exchange keeps a list of everyone who wants to buy or sell, and at what price. That list is the **limit order book** (LOB).

```
            ASKS (people selling)
   price     size
   100.04     300      <- level 3 ask
   100.03     150      <- level 2 ask
   100.02      80      <- best ask (level 1): the cheapest you can buy right now
   ─────────────────   spread = 100.02 − 100.00 = 0.02
   100.00     500      <- best bid (level 1): the highest you can sell for right now
    99.99     700      <- level 2 bid
    99.98     400      <- level 3 bid
            BIDS (people buying)
```

- **Bid** = an offer to buy. **Ask** = an offer to sell.
- **Level 1** is the best price on each side; level 2 is the next, and so on. We use 10 levels per side.
- **Spread** = best ask − best bid.
- **Mid price** = (best ask + best bid) / 2 = 100.01 here. This is what we try to predict.
- **Size** (also called depth or quantity) is how much is waiting at that price.

How the price moves: if someone **buys** 80 shares with a market order, they eat the whole best-ask queue. The best ask jumps to 100.03 and the mid rises. So the mid moves when a best queue is used up. That is the key intuition: in the book above, the ask queue (80) is much smaller than the bid queue (500), so the next move is more likely to be **up**. Most of our features are versions of that idea.

## Two ways to trade

- **Taker** (aggressive): send a market order that trades immediately at the best price on the other side. You pay the spread and a fee.
- **Maker** (passive): place a limit order in the book and wait. You don't pay the spread (you may even earn a small rebate), but you might not get filled, and you tend to get filled exactly when the price is about to move against you. This is called **adverse selection**.

## Units you'll see everywhere

- **bp (basis point)** = 0.01% of the price. On a $60,000 bitcoin, 1 bp = $6.
- **Tick** = the smallest allowed price step. $0.01 for both BTC and US stocks.
- **Large-tick vs small-tick.** If the tick is big compared with the price (INTC at $26: one cent ≈ 3.5 bp), prices rarely change and queues get long; the queue sizes are informative. If the tick is tiny compared with the price (BTC at $60,000: one cent = 0.0017 bp), the price jitters constantly. Many results in this project depend on which kind of market you're in.
- **Horizon k** = how far ahead we predict. On BTC, k is in 1-second rows, so k = 20 means 20 seconds ahead.

## The pipeline, end to end

```
raw data ──► 40-column order book ──► features ──► labels ──► train models ──► predictions ──► simulate trading
 (guide 2)       (guide 2)            (guide 4)   (guide 3)    (guide 5)        (guide 6, 7)     (guide 8)
```

1. Load a dataset and turn every snapshot into the same 40 numbers: price and size for 10 ask levels and 10 bid levels.
2. Compute 27 hand-made features from those numbers.
3. Label each row UP, FLAT or DOWN depending on what the mid does over the next k rows.
4. Split the rows in time (first 70% train, next 15% validation, last 15% test) and train models.
5. Score the predictions on the test rows.
6. Feed the predictions to a trading simulator that pays the spread, fees and latency.

## The one equation to remember

For one trade held for a horizon h:

```
net profit  =  edge from being right   −  cost of acting late  −  cost of crossing the spread  −  fees
               (accuracy: the model)      (latency: the system)    (the market)                   (the exchange)
```

If the model is right a fraction p of the time and moves average m bp, the edge is roughly (2p − 1) × m. To break even after a total cost C you need

```
p  >  0.5 + C / (2m)
```

When the cost C is bigger than the typical move m, that threshold is above 100%, so **no model can win**. That is exactly what happens on BTC with a 4 bp taker fee (guide 8).

---

## Check yourself

1. In the book drawn above, which side would a buyer's market order eat, and what would the new mid be after someone buys 80 shares?
2. Why would a market maker's buy order tend to get filled right before the price falls?
3. A model is right 60% of the time on moves that average 4 bp. Trading costs 3 bp per round trip. Does it make money?

<details><summary>Answers</summary>

1. The ask side. After 80 shares are bought, the 100.02 queue is empty, the best ask becomes 100.03 and the mid becomes (100.03 + 100.00) / 2 = 100.015.
2. Because the people who sell to it are disproportionately the ones who know (or guess) the price is about to fall. That is adverse selection.
3. Edge ≈ (2 × 0.6 − 1) × 4 = 0.8 bp per trade, cost 3 bp, so it loses about 2.2 bp per trade. The break-even hit rate would be 0.5 + 3 / 8 = 87.5%.

</details>

Next: [2. The data](02_data.md)
