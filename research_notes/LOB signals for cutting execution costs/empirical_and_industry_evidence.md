# Empirical and industry evidence: do short-term microstructure signals (OBI / OFI / micro-price) cut execution costs?

Scope: cost reduction for a trade that would happen anyway (an execution overlay), not standalone alpha. Project context: Coinbase BTC-USD 1-second LOB snapshots (Kaggle martinsn, about 2021), a LightGBM model on OFI and micro-price features with about 0.7 bp edge, and a planned test of buying 1 BTC within 5 minutes three ways: immediate, TWAP, and model-timed.

Credibility labels used below:
- **[PEER]**: peer-reviewed.
- **[WP]**: working paper or preprint with a reproducible method.
- **[VENDOR]**: marketing or vendor claim, not independently verified.
- **[BLOG]**: practitioner blog or personal project.
- **[2ND]**: number seen only in a search-engine summary or secondary source. I did not read the primary text.

Research date: 2026-10-30.

## Q1. Academic studies measuring execution-cost savings from imbalance / OFI / short-term alpha in placement or timing

### Takeaway
Few academic studies report savings for a trade that would happen anyway, measured on real data against a naive benchmark. They agree on a modest size: an imbalance-timed policy saves a fraction of one bid-ask spread per order, about 7% to 35% of the spread versus TWAP depending on latency and horizon. The savings fall sharply as latency moves from about 1 ms to about 1 to 5 s. Several authors say outright that the signal is too weak for standalone stat-arb but can "spare some basis points" for a trade already decided.

### Cited Findings
- **Stoikov & Waeber, "Reducing transaction costs with low-latency trading algorithms" (SSRN 2661618, Sept 2015) [WP; later published in Quantitative Finance].** This is the closest analogue to the project's test.
  - **Setup:** sell one lot within a short horizon T using top-of-book imbalance I = B/(A+B), solved as optimal stopping (trade / no-trade regions), and backtested with market orders only.
  - **Data:** 5-year on-the-run US Treasuries on eSpeed, Level-I data with millisecond timestamps, 10:30am to 3pm over the first 10 trading days of July 2010. Spread is almost always 1 tick, and periods with a wider spread were filtered out.
  - **Headline:** the policy "significantly outperforms a TWAP algorithm... saving on average approximately 1/3 of the spread per share if trades are executed with low latency (≈1 millisecond)."
  - Savings vs TWAP as a fraction of the spread (Table 1):

    | T (horizon per trade) | L = 1 ms | L = 100 ms | L = 1000 ms | L = 5000 ms |
    |---|---|---|---|---|
    | 10 s | 0.20 | 0.17 | 0.11 | 0.05 |
    | 1 min | 0.32 | 0.26 | 0.17 | 0.07 |
    | 5 min | 0.33 | 0.35 | 0.18 | 0.07 |

  - Savings rise with the horizon and fall with latency. At L = 1 s the 5-minute saving is about 18% of the spread.
  - The authors convert this to money: at one trade per minute and L = 1 s, 0.17 × 270 × $78 = $3,580 per day.
  - **Caveats:** in-sample calibration, a 1-tick-spread asset, no limit orders, no impact modelled, and only 10 days of data. — [SSRN 2661618](https://ssrn.com/abstract=2661618)
- **Lehalle & Mounjid, "Limit order strategic placement with adverse selection risk and the role of latency" (arXiv 1610.00261, 2016–2018; later in Market Microstructure and Liquidity) [WP/PEER].**
  - **Data:** labelled trade data; the figures appear to use European equities such as AZN.
  - **On standalone value:** imbalance effects "are usually not strong enough to be the source of a statistical arbitrage: the expected value of buying and selling back using accurate predictions based on sizes at first limits does not beat transaction costs."
  - **On execution value:** "For an investor who already took the decision to buy or to sell, this information can spare some basis points."
  - **Model result:** optimal control (cancel or stay, conditioned on imbalance) beats a blind "join the bid" strategy. The price improvement is near 0 when imbalance is favourable and turns positive when imbalance is strongly adverse, because the gain comes from avoiding adverse selection. The results are model and simulation based and reported as figures; I found no headline bp number. — [arXiv 1610.00261](https://arxiv.org/abs/1610.00261)
- **Lipton, Pesavento & Sotiropoulos, "Trade arrival dynamics and quote imbalance in a limit order book" (arXiv 1312.0514, Dec 2013) [WP; authors at BofA Merrill Lynch at the time].**
  - **Data:** VOD.L, all trading days in Q1 2012. Other liquid stocks were reported as "qualitatively similar."
  - **Passive fills underdeliver:** "passive fills rarely achieve their expected savings of one full spread." The queue-depletion and trade-flow interaction alone "is responsible for the loss of about 60% of the theoretical spread captured by a passive fill." That loss comes on top of the adverse selection usually blamed.
  - **Imbalance and adverse moves:** the probability of an unfavourable price move rises as the book gets heavier on the near side, "reaching almost 90% in cases of high imbalance."
  - **Suggested policy:** keep the order posted on the near side when imbalance is moderate, and cross the spread when the book is highly imbalanced. The paper derives no optimal policy and gives no realised-savings number. — [arXiv 1312.0514](https://arxiv.org/abs/1312.0514)
- **Cartea, Donnelly & Jaimungal, "Enhancing trading strategies with order book signals" (Applied Mathematical Finance 25(1), 2018) [PEER].**
  - **Data:** Nasdaq, 11 equities, with INTC and ORCL as the main examples. The model is calibrated on Jan–Jun 2014 and tested out-of-sample on Jul–Dec 2014.
  - **Result:** adding volume imbalance "considerably boosts the profits of the strategy," because it "reduces adverse selection costs and positions LOs in the book to take advantage of favorable price movements."
  - **Benchmark:** naive spread capture at the best quotes had negative Sharpe ratios [2ND, CXO Advisory summary].
  - **Caveat:** this is a liquidity-provision strategy, not an execution overlay. I could not extract the table values. — [ORA Oxford PDF](https://ora.ox.ac.uk/objects/uuid:006addde-3a03-4d75-89c1-04b59026e1c0/files/me4008e0ecca779b45d59231ebca3e69c); [IDEAS](https://ideas.repec.org/a/taf/apmtfi/v25y2018i1p1-35.html); [CXO Advisory](https://www.cxoadvisory.com/?p=27752)
- **Bechler & Ludkovski, "Optimal execution with dynamic order flow imbalance" (arXiv 1409.2618; later in SIAM J. Financial Math) [WP/PEER].**
  - **Method:** OFI is modelled as a mean-reverting process that drives price. The study uses Nasdaq ITCH data (TEVA, MSFT, BBBY, May 2011) only to illustrate order flow.
  - **Result:** in 2,000 simulated paths, the fully dynamic OFI-aware policy has an expected cost of 4.257 against 4.547 for the constant-rate (VWAP/TWAP-like) strategy. The paper calls this an "approximately 6.8%" improvement. The cost units are model units, not bp, and the result is simulation only. — [arXiv 1409.2618](https://arxiv.org/html/1409.2618v2)
- **Kolm, Turiel & Westray, "Deep order flow imbalance: Extracting alpha at multiple horizons from the limit order book" (Mathematical Finance 33(4), 2023) [PEER].** The study uses 115 Nasdaq stocks and shows that OFI-based inputs beat raw order-book states for forecasting returns over short horizons. It measures forecast accuracy, not execution-cost savings. I found no execution-cost bp figure in it. — [IDEAS](https://ideas.repec.org/a/bla/mathfi/v33y2023i4p1044-1081.html); [Quantpedia summary](https://vvv.quantpedia.com/tag/order-flow/)
- **Cont, Kukanov & Stoikov, "The price impact of order book events" (2011/2014; NYSE TAQ, 50 US stocks) [PEER].** Over short intervals, price changes are mainly driven by OFI. This is foundational for prediction and contains no execution-savings test. — [IDEAS](https://ideas.repec.org/p/hal/wpaper/hal-00545745.html)
- **Fill probability depends on imbalance.** An interpretable-ML study finds that limit-order fill probability is symmetric and monotone in limit-order-flow imbalance [2ND]. — [arXiv 2307.04863](https://arxiv.org/pdf/2307.04863). A 2025 market-maker paper adds that fill probability is "not a function of imbalance alone." — [arXiv 2502.18625](https://arxiv.org/html/2502.18625v2)

### Inferences
- The best direct evidence (Stoikov–Waeber) puts the value of imbalance timing at about 0.2 to 0.35 spreads per order at low latency. It falls to about 0.07 to 0.18 spreads at 1 to 5 s latency.
- The project's 1-second snapshots mean effective decision latency is at least about 1 s. The L = 1000 ms column (about 11% to 18% of a spread) is the realistic academic prior, not the 1/3-spread headline.
- Across papers the mechanism is the same: the gain comes mostly from avoiding adverse selection, either by not posting or not waiting when imbalance is against you and crossing before the move. Capturing extra spread contributes little. Lipton et al. show that naive passive posting loses most of the spread it nominally earns.
- Most academic "savings" are in-sample, model-calibrated, or simulated, and they ignore the trader's own impact and the queue position of their own order. Treat them as upper bounds.

### Gaps
- No exact out-of-sample P&L or Sharpe values for Cartea–Donnelly–Jaimungal; the tables were not extractable.
- Cont & de Larrard and Gould & Bonart ("Queue imbalance as a one-tick-ahead price predictor") are prediction papers. I found no execution-cost bp numbers in either.
- Avellaneda–Reed–Stoikov ("Forecasting prices from Level-I quotes in the presence of hidden liquidity") is also prediction-focused. I found no reported cost savings.
- Cartea–Jaimungal "Incorporating order-flow into optimal execution" (2016) was not retrieved, so I have no numbers from it.

## Q2. What brokers / execution providers publish (typical reported savings in bp)

### Takeaway
Public broker material almost never isolates how much a short-term-signal overlay saves. The few quantified claims are vendor A/B results in the sub-1-bp to tens-of-percent-of-slippage range, with undisclosed methodology. None is independently verified.

### Cited Findings
- **BestEx Research homepage [VENDOR].**
  - **IS Zero algorithm:** "reducing IS slippage over VWAP by 37% in equities A/B testing across tens of thousands of parent orders." A chart caption gives a sample of about 140k orders. The headline also says "cost reduction vs. arrival," so the benchmark is ambiguous.
  - **Queue-jumping and strategic limit-order routing:** "0.65 bps IS savings at parent order level." No sample, period, or method is given; the context is US equities.
  - **Pulse futures cost model:** realised and projected costs differ by about 1 tick on average, across more than 72k parent orders. This is a model-accuracy claim, not a performance claim. — [bestexresearch.com](https://www.bestexresearch.com/)
- **BestEx "time of day" research [VENDOR]:** executing in the last hour is consistently cheaper, and IS Zero shifts trading into low-cost intervals. This is a scheduling effect, not a microstructure signal. — [BestEx insight](https://www.bestexresearch.com/insights/optimizing-trading-cost-by-leveraging-the-time-of-day-effect)
- **Quantitative Brokers (Almgren's firm) [VENDOR].**
  - **"Extended microprice performance"** (Zhou, May 2026) is a whitepaper on a micro-price signal that reads imbalance deeper in the book, using results from the first 15 futures where it was enabled. I saw no numbers on the insights page. — [QB insights](https://www.quantitativebrokers.com/insights/); [PDF](https://www.quantitativebrokers.com/wp-content/uploads/2026/10/Extended_Microprice_Performance_WebReadyLinked.pdf)
  - **Related QB whitepapers:** queue-depletion time-to-fill for Treasuries SOR (Jan 2026) and XT-roll queue tracking. — same page
- **Stoikov & Waeber note** that broker algorithm names ("React", "Bolt", etc.) market speed, "but there is no consensus on whether or not such investments make economic sense." — [SSRN 2661618](https://ssrn.com/abstract=2661618)
- **US patent family "Methods and systems related to securities trading" (e.g., US 8412617, 8296221, 8301548, 8433645) [VENDOR/patent].** It splits implementation shortfall into adverse selection versus opportunistic savings. It claims an algorithm-switching engine "can eliminate 70% of adverse selection costs with only a small reduction in opportunistic savings," and describes "alpha-profile" strategies with tactical limits. This is a patent assertion and is not evidence [2ND]. — [USPTO 8412617](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8412617)
- **Maglaras (Columbia), Imperial College lectures, May 2015 [academic/practitioner slides]:** short-term alpha signals are used in IS algorithms to adjust speed and limit-order placement in real time. No bp figures were visible. — [Imperial PDF](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/cfm-imperial-institute-of-quantitative-finance/events/distinguished-lectures/Maglaras-Lectures.pdf)
- **Broker TCA methodology:** a Bayesian broker-ranking paper notes that IS performance mixes "volume and price timing, spread capture, adverse selection, short-term impact and reversion, venue selection and HFT anti-gaming logic." That makes it hard to attribute savings to a signal. — [arXiv 1904.01566](https://arxiv.org/pdf/1904.01566)

### Inferences
- The quantified vendor numbers (about 0.65 bp IS savings from smarter limit placement; a 37% cut in slippage relative to a VWAP baseline) are in line with the academic "fraction of a spread" result. They are marketing claims, though, with undisclosed baselines, periods, and selection.
- The project's 0.7 bp edge is the same order of magnitude as BestEx's 0.65 bp placement claim. That is encouraging, but a gross signal edge is not realised savings after fills, fees, and latency.

### Gaps
- I found no public, quantified "alpha-aware" or "smart limit" savings figures from J.P. Morgan, Goldman Sachs, Citi, Virtu/ITG, Bloomberg, Abel Noser, Instinet, Kepler Cheuvreux, or Quod within this search budget. These are usually in client-only decks.
- I did not open the QB extended-microprice PDF, so its tick or bp results are unknown.

## Q3. Crypto-specific execution cost / slippage studies and algorithms

### Takeaway
Crypto execution evidence is mostly vendor TCA and preprints. Its main lesson is that TWAP on liquid BTC pairs is already cheap: from about a fraction of a bp to low single-digit bp versus arrival, before fees, with market drift dominating realised slippage. Learned or order-book-aware policies beat TWAP by tenths of a bp to about 0.7 bp at best, and results flip sign across time windows.

### Cited Findings
- **Talos, "Execution insights through TCA: benchmarks and slippage" (Eliad Hoch, Apr 3, 2025) [VENDOR, real orders].**
  - **Sample:** more than 1,000 TWAP parent orders, about $1B notional, 85% buys, average duration about 100 minutes.
  - **Arrival slippage:** paid "nearly 13 bps from arrival" while the market rose about 42 bps on average during the orders. Most of the slippage was trend, not spread.
  - **Maker rate and participation:** 75% maker rate, assumed participation about 5%.
  - **Recommendation:** for liquid, tight-spread pairs, taking liquidity more often can improve arrival slippage in trending cases.
  - **Arrival benchmark:** the median 1-second mid-quote, from 2024 BTC-USDT data at 1-second intervals. — [Talos](https://www.talos.com/insights/execution-insights-through-transaction-cost-analysis-tca-benchmarks-and-slippage)
- **Anboto Labs (crypto algo vendor), Medium TCA post [VENDOR, 2ND: page returned 403 and figures come from a search summary].**
  - **Slippage:** TWAP slippage about −0.25 bp and arrival slippage about −0.58 bp at about 2.15% participation.
  - **TradFi comparison:** typical TradFi TWAP slippage of −1 to −2 bp.
  - **Related post:** a late-2023 post, "How the Anboto TWAP Algorithm saves Trading Costs," could not be opened. — [Anboto Medium](https://medium.com/@anboto_labs/slippage-benchmarks-and-beyond-transaction-cost-analysis-tca-in-crypto-trading-2f0b0186980e)
- **Zheng et al., "Optimal execution using reinforcement learning" (arXiv 2306.17178, 2023) [WP, 2ND for numbers].**
  - **Setup:** an RL agent with cross-exchange order-book-imbalance features in crypto.
  - **Results:** TWAP IS about −3.1 bp including a 3 bp fee; the RL agent improves on TWAP by about 0.68 bp.
  - **Caveat:** the summary noted two TWAP rows (−3.10 and −1.48 bp), so the exact setting needs checking. — [arXiv 2306.17178](https://arxiv.org/html/2306.17178)
- **"Mixture-of-Experts for cryptocurrency order execution" (arXiv 2610.03369, Oct 2026) [WP, 2ND].**
  - **Base configuration:** every learned policy had higher mean shortfall than TWAP (TWAP 0.39 bp).
  - **After annealing:** some variants fell a few tenths of a bp below TWAP.
  - **TWAP variability:** TWAP shortfall itself ranged from +0.39 bp over all 500 windows to −1.66 bp on the first 120 windows. — [arXiv 2610.03369](https://arxiv.org/html/2610.03369)
- **Binance aggTrades execution project (GitHub, randomaban-beep/QR-Execution-Impact) [BLOG, 2ND].**
  - **Claim:** with zero drift and sublinear impact, TWAP "is very hard to beat" on expected IS, and Almgren–Chriss at the risk-neutral limit statistically ties with it.
  - **VWAP:** a causal prior-7-day VWAP underperforms TWAP by about 0.2 to 7 bp. — [GitHub](https://github.com/randomaban-beep/QR-Execution-Impact)
- **CoinRoutes TCA explainer [VENDOR]:** crypto TCA is harder because there is no consolidated best-price record, so the benchmark has to be built from aggregated venue data. — [CoinRoutes](https://coinroutes.com/academy/transaction-cost-analysis-tca-digital-assets/)
- **Kaiko liquidity reports [data vendor].**
  - **Method:** Kaiko measures slippage with simulated market orders of $50k to $100k.
  - **March 2023 stress:** the BTC $100k sell slippage was about 0.1%, and Coinbase BTC-USD slippage stayed about 2.5 times its start-of-month level [2ND via Liquidity Finder]. — [Kaiko chartbook Dec 2022](https://marketing.kaiko.com/hubfs/Kaiko_Weekly_Cryptocurrency_Chartbook_19_12_2022-1.pdf); [Liquidity Finder Q1 2023](https://liquidityfinder.com/insight/liquidity/the-stateof-liquidityin-crypto-markets-q12023)

### Inferences
- On BTC, TWAP over minutes to hours has small spread-plus-impact cost. Realised arrival slippage is dominated by drift during the window (Talos: 13 bp paid against a 42 bp move). For a 5-minute, 1 BTC test, the variance of the outcome will be driven by the 5-minute price move, not by microstructure costs. A paired design is needed: the same start times, with model-timed minus TWAP.
- The best crypto preprint result (+0.68 bp versus TWAP) is nearly identical to the project's 0.7 bp model edge. The MoE paper shows that such edges can vanish or flip across sample windows.

### Gaps
- I found no published Coinbase Prime TWAP/VWAP TCA statistics, Binance algo-order performance data, Paradigm execution studies, or Amberdata/CCData slippage studies with numbers.
- No peer-reviewed crypto paper was found that tests OFI or micro-price timing of a fixed parent order against TWAP on Coinbase data.

## Q4. Size of spread/impact for ~1 BTC on Coinbase BTC-USD and the realistic ceiling on savings

### Takeaway
For 1 BTC on Coinbase BTC-USD, spread plus book-walking costs only about a fraction of a bp to a few bp; I found no direct source for 1 BTC. Fees dominate: the 2021-era entry tier was 40 bp maker and 60 bp taker. The ceiling for timing savings is therefore about the half-spread plus small depth costs plus whatever short-term drift the model predicts. The maker-versus-taker fee difference matters far more than any signal.

### Cited Findings
- **Fees in the dataset era:** entry-tier fees on Coinbase Pro / Advanced Trade (30-day volume up to $10k) were 0.60% taker and 0.40% maker. Coinbase's own comparison lists at most 0.4% maker and at most 0.6% taker. Fees at higher tiers are lower; I did not verify them. — [Motley Fool](https://fool.com/the-ascent/cryptocurrency/coinbase-vs-coinbase-pro/); [Coinbase](https://www.coinbase.com/en-gb/advanced-markets)
- **Tick size:** the minimum tick on Coinbase Pro BTC-USD is $0.01, and the average BTC-USD spread on the exchanges studied was below 10 bp. Coinbase Pro and Bitfinex were the most liquid venues irrespective of order size [peer-reviewed, 2ND via snippet]. — [ScienceDirect, "Bitcoin unchained: Determinants of cryptocurrency exchange liquidity," J. Empirical Finance 2022](https://www.sciencedirect.com/science/article/pii/S0927539822000822)
- **Spread levels in bp:**
  - Earlier Kaiko research put the Coinbase BTC spread near 0.3 bp, and a January 2026 snapshot gives about 0.12 bp across venues [2ND, blog citing Kaiko]. — [coincub](https://coincub.com/blog/crypto-exchange-spreads-vs-trading-fees-how-to-calculate-the-real-cost-of-a-crypto-trade/)
  - A data vendor lists Coinbase BTC-USD average spread at 0.028 bp (undated, likely recent). — [CryptoStruct](https://cryptostruct.com/pairs/btc-usd)
  - Kaiko defines spread in bp as (ask − bid)/mid × 10,000. — [Kaiko factsheet Feb 2022](https://cdn.shopify.com/s/files/1/1365/1139/files/Kaiko_Cryptocurrency_Factsheet_28_02_2022.pdf?v=1646059182)
- **A recent, unverified third-party table** gives Coinbase slippage for a $10k BTC/USDT order of 0.02% to 0.05% (2 to 5 bp) [low credibility]. — [onebullex](https://www.onebullex.com/explore/deribit-spot-routing-coinbase-liquidity-slippage-confirmation)
- **Coinbase publishes its full book,** so depth figures for it are complete. — [Glassnode](https://studio.glassnode.com/charts/market.OrderbookDepthQuote?a=BTC)

### Inferences
- At a BTC price of $30k to $60k in 2021, a $0.01 tick is about 0.002 to 0.003 bp, so the spread is almost never tick-constrained. Reported average spreads of about 0.03 to 0.3 bp imply a taker half-spread of roughly 0.01 to 0.15 bp. A 1 BTC market order usually has to walk past the top level, which may add up to about 1 bp in normal 2021 conditions (my estimate, not sourced).
- The project's own snapshots are the authoritative source. Compute the time-averaged quoted spread in bp and the VWAP cost of sweeping 1 BTC from the 1-second book to fix the ceiling.
- Using Stoikov–Waeber's L = 1 s, T = 5 min figure (about 18% of a spread), timing alone saves well under 0.1 bp on the spread component. Any larger realised saving must come from the model predicting mid-price drift over seconds to minutes. A 0.7 bp edge is plausible only if it is directional drift capture, and it needs a large sample of paired trials to separate it from the 5-minute volatility of BTC (several bp to tens of bp).
- With fees of 40 bp maker and 60 bp taker, a 20 bp fee gap per trade dwarfs any signal edge. A model-timed strategy that changes the maker/taker mix will see its result dominated by fees, so report results gross and net of fees separately.

### Gaps
- I found no source giving the actual 2021 Coinbase BTC-USD average spread in bp, or the slippage for exactly 1 BTC. A Kaiko price-slippage query (exchange cbse, btc-usd, size 1 BTC) or computing it from the Kaggle data would close this gap.

## Q5. Practitioner write-ups on using imbalance to time child orders

### Takeaway
Practitioner write-ups agree with the academic picture. Imbalance is mainly useful for deciding when to stop waiting passively and cross, because passive fills are adversely selected. Most of the alpha is realised within seconds, so latency and decision cadence matter. Quantified claims are mostly from self-run backtests.

### Cited Findings
- **The Microstructure Lab (Substack), "Execution simulation for HF directional alphas" [BLOG].** Passive fills are biased toward signals where price moved against the prediction. Adverse selection and alpha decay while waiting "more than offset the fee and spread savings." Passive patience was set to 5 s because "64% of the alpha materialises by then." — [Substack](https://themicrostructurelab.substack.com/p/execution-simulation-for-hf-directional)
- **Maciejewski, "Model-free passive execution via order-level shadowing" (arXiv 2609.18019, Sept 2026) [WP].** Uses a full year of replayed CME ES futures. It positions a passive POV ("Shadow-PPOV") as a model-free benchmark "against which a predictive placement model can be scored." A search summary [2ND] said that over about 245 sessions and about 1.6M fills, passive and aggressive slippage were close, and the apparent passive edge was "the signature of adverse selection." — [arXiv 2609.18019](https://arxiv.org/abs/2609.18019)
- **HFT Advisory Substack, "Six market microstructure signals that fire before the price print" [BLOG]:** a practitioner framework for using pre-trade microstructure signals for execution quality. No verified numbers were retrieved. — [Substack](https://hftadvisory.substack.com/p/six-market-microstructure-signals)
- **Open-source intraday-alpha-platform TCA module [BLOG].** Measures mid-price markouts after each fill from 100 ms to 5 minutes, the standard adverse-selection check. Its own report shows passive execution at −1.39 ± 0.07 bp per parent versus aggressive, a small self-generated sample. — [GitHub PR](https://github.com/AshJha0/intraday-alpha-platform/pull/22)
- **Lipton et al. (practitioner-authored, BofA Merrill Lynch):** keep the order posted at moderate imbalance and cross when the book is highly imbalanced against you. — [arXiv 1312.0514](https://arxiv.org/abs/1312.0514)

### Inferences
- For the 1 BTC / 5-minute test, practitioner guidance suggests three comparisons:
  1. The model-timed market-order strategy against TWAP market orders (the Stoikov–Waeber design).
  2. Optionally, a passive-first strategy with an imbalance-triggered cross.
  3. Markouts on every fill to show whether the savings come from avoiding adverse moves.
- With 1-second snapshots, the relevant signal is what remains predictable 1 s or more ahead. The sub-second component that HFT-latency papers exploit is not available.

### Gaps
- I found no Robert Almgren / Quantitative Brokers public talk or blog with quantified savings from imbalance-timed child orders; the QB whitepaper numbers were not retrieved.
- I found no Hudson River Trading talk content on this topic.
