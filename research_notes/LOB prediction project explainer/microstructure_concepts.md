# Market Microstructure Concepts Behind LOB Short-Horizon Price Prediction

Scope: intuitive explanations, formulas, original sources, and critiques for the microstructure concepts used in an LOB short-horizon prediction project (FI-2010 / Optiver-style data). Research date: 2026-10-30.

Verification legend:
- **[verified]**: read in the primary source full text this session (PDF text extracted from arXiv / CREATES working paper).
- **[abstract/secondary]**: confirmed from abstract pages, bibliographic records or search summaries, not the full text.
- **[unverified - from background knowledge]**: standard textbook material I could not fetch this session. Check before quoting.

Worked examples are illustrative arithmetic I made up. They are not taken from any paper.

---

## 1. LOB mechanics: continuous double auction, price-time priority, levels, spread, mid-price, tick size, large- vs small-tick, and why queue depth matters

### Takeaway
An LOB is a continuous double auction. Resting limit orders sit at discrete price levels (multiples of the tick) and are matched by price-time priority. For **large-tick** assets the spread is almost always one tick, so the price can only move once a best queue is depleted. That makes **queue sizes at the best quotes** the main state variable for short-horizon prediction. For **small-tick** assets much more activity happens inside the spread and deeper in the book, and best-queue signals predict less well.

### Intuition and worked example
- Picture two sorted lists of standing offers. Buyers' bids are sorted high to low and sellers' asks low to high. The **best bid** b(t) is the highest bid and the **best ask** a(t) is the lowest ask. Spread = a − b. Mid-price = (a + b)/2.
- **Level k** is the k-th best distinct price on a side. **Depth / queue size** at a level is the total shares resting there.
- **Price-time priority**: a better price executes first. At the same price, the earliest order executes first (FIFO). A market order "eats" the best queue. When that queue is gone, the best price moves to the next level.
- Example (large-tick stock, tick = $0.01). Bid $10.00 × 5,000 shares, ask $10.01 × 300 shares. The spread is already at its minimum of one tick, so the price cannot "drift" smoothly. The mid only moves when a queue empties, or when someone posts inside the spread, which is impossible here. The ask queue (300) is much smaller than the bid queue (5,000), so it is far more likely to be depleted first. The next mid move is therefore more likely up. This is the whole intuition behind queue-imbalance predictors.
- For a small-tick stock (say a $3,000 share with a $0.01 tick), the spread is typically many ticks wide. Orders keep arriving inside the spread, so the "which queue empties first" logic is much weaker.

### Cited Findings
- An LOB is a continuous double-auction mechanism where traders submit orders specifying price and size. Gould et al. (2013) is the standard survey: "Limit order books," *Quantitative Finance* 13(11):1709–1742. [verified title/venue via citation in Gould & Bonart; survey PDF downloaded] — [Gould, Porter, Williams, McDonald, Fenn & Howison, arXiv:1012.0349](https://arxiv.org/abs/1012.0349)
- On Nasdaq, each stock trades in a separate LOB with price-time priority, a lot size of 1 share and a tick size of $0.01. [verified] — [Xu, Gould & Howison 2019, arXiv:1907.06230](https://arxiv.org/abs/1907.06230)
- Eisler, Bouchaud & Kockelkoren split stocks into **large tick** stocks, "such that the bid-ask spread is almost always equal to one tick," and small tick stocks, whose spreads are typically a few ticks. For large-tick stocks, events that change the best price have low probability (about 3% of events altogether). They also find that a simple model in which all impacts are permanent "works well for large tick stocks, for which the bid-ask spread is nearly constant, with no gaps in the order book." [verified] — [Eisler, Bouchaud & Kockelkoren, "The price impact of order book events: market orders, limit orders and cancellations," arXiv:0904.0900](https://arxiv.org/abs/0904.0900) (published in *Quantitative Finance* 2012 [unverified - venue from background knowledge, though Xu et al. cite it as "Eisler et al. [2012]"])
- Dayri & Rosenbaum ("Large tick assets: implicit spread and optimal tick size," *Market Microstructure and Liquidity*, 2015; arXiv 2012) show that many microstructure features of large-tick assets can be summarised by one easily computed parameter **η**. η measures how "constrained" the price is by the tick: low η means very large tick and few price reversals. They define an **implicit spread** 2ηα that replaces the quoted spread in the spread–volatility relation, and find that the optimal tick size corresponds to η = 1/2. [abstract/secondary] — [arXiv:1207.6325](https://arxiv.org/abs/1207.6325); [HAL record](https://hal.archives-ouvertes.fr/hal-01263181)
- Order-flow location depends on tick size (Nasdaq, LOBSTER data). For large-tick stocks (ORCL, CSCO, MU), only about 0.6–1.6% of order-flow activity occurs inside the spread and about 65–73% at the best quotes. For small-tick stocks (AMZN, TSLA, NFLX), about 14–27% occurs inside the spread, and "in all cases more than half of all order flow occurs beyond the level-1 bid- and ask-prices." [verified, Table 2] — [Xu, Gould & Howison 2019](https://arxiv.org/abs/1907.06230)
- The tick-to-price ratio, or equivalently the fraction of time the spread equals one tick, is used as the standard way to classify large vs small tick (Curato & Lillo, cited in CCZ). [verified as cited] — [Cont, Cucuringu & Zhang, arXiv:2112.13213](https://arxiv.org/abs/2112.13213)
- Textbooks covering these mechanics: Bouchaud, Bonart, Donier & Gould, *Trades, Quotes and Prices* (Cambridge UP, 2018) — [CUP](https://doi.org/10.1017/9781316659335); Cartea, Jaimungal & Penalva, *Algorithmic and High-Frequency Trading* (Cambridge UP, 2015) — [CUP](https://www.cambridge.org/9781107091146). [unverified - bibliographic details from background knowledge]

### Inferences
- Queue depth matters for predictability because, for large-tick assets, the mid-price is a **discrete, queue-depletion-driven process**. The relative size of the two best queues is a direct proxy for "which side breaks first." For small-tick assets the price process is closer to continuous and the best queues hold less information. Expect any best-level feature (QI, OFI-L1, micro-price) to work much better on large-tick names, and expect deeper-level features to matter more on small-tick names.
- FI-2010 (Nasdaq Nordic, 5 stocks, 10 levels) and Optiver RVP (2 levels, anonymised and normalised prices) do not expose absolute tick sizes directly. A practical proxy is the fraction of snapshots where spread equals the minimum observed spread.

### Gaps
- I did not fetch the full Dayri–Rosenbaum paper, so the exact estimator for η (based on counts of continuations vs alternations of price moves) is not reproduced here.
- I did not open the textbooks themselves this session. Chapter references are not given.

---

## 2. Order Flow Imbalance (OFI): Cont, Kukanov & Stoikov (2014)

### Takeaway
OFI is the net change in supply and demand at the best quotes. It adds up limit-order arrivals, cancellations and market orders, each signed by its effect on the best bid and ask queues. Over 10-second intervals, mid-price changes are **linear** in OFI with an average **R² of 65%** (50 S&P 500 stocks, April 2010). The slope (price-impact coefficient) is roughly **inversely proportional to depth**. OFI beats trade imbalance (R² 65% vs 32%). Note that this is a **contemporaneous** relationship, not a forecast.

### Definition
For each quote update n, with best bid price/size (P^B_n, q^B_n) and best ask price/size (P^A_n, q^A_n), the per-event contribution is:

```
e_n =  1{P^B_n >= P^B_{n-1}} q^B_n  -  1{P^B_n <= P^B_{n-1}} q^B_{n-1}
     - 1{P^A_n <= P^A_{n-1}} q^A_n  +  1{P^A_n >= P^A_{n-1}} q^A_{n-1}
OFI_k = sum of e_n over events n in interval [t_{k-1}, t_k]
ΔP_k  = β · OFI_k + ε_k          (ΔP in ticks; 10-second intervals)
β_i   = c / AD_i^λ + noise       (AD_i = average best-queue depth in half-hour window i; λ ≈ 1)
```
In the stylized model (depth D at every level, events only at the best quotes), ΔP = OFI/(2D) + truncation error, with OFI = L^b − C^b − M^s − L^s + C^s + M^b. Here L means limit orders, C cancellations and M market orders. [verified, Eq. (1)–(3) of arXiv v3; the event-level e_n formula is the paper's Eq. (1), which pdftotext garbled, so it is reproduced from the standard statement and should be cross-checked]

### Intuition and worked example
- A market sell of 200 and a cancel of 200 on the bid side have the **same effect**: both shrink the bid queue by 200. OFI treats them identically. The paper notes this explicitly: "this variable treats a market sell and a cancel buy of the same size as equivalent."
- Example. Start: bid $10.00 × 500, ask $10.01 × 300.
  1. A buyer adds 200 at $10.00. Bid price is unchanged and q^B goes 500→700, so e = 700 − 500 = **+200**.
  2. A market buy of 200 hits the ask. Ask price is unchanged and q^A goes 300→100, so e = −100 + 300 = **+200**.
  3. Someone cancels 50 on the bid. q^B goes 700→650, so e = **−50**.
  OFI = +350, which is net buying pressure. If average depth D ≈ 400 shares, the stylized model gives ΔP ≈ 350/(2·400) ≈ 0.44 ticks expected upward move. A thin book (small D) means the same OFI moves the price more.

### Cited Findings
- Published in *Journal of Financial Econometrics* 12(1):47–88 (2014), DOI 10.1093/jjfinec/nbt003. arXiv:1011.6402. [abstract/secondary] — [arXiv:1011.6402](https://arxiv.org/abs/1011.6402)
- "Over short time intervals, price changes are mainly driven by the order flow imbalance, defined as the imbalance between supply and demand at the best bid and ask prices… a linear relation between order flow imbalance and price changes, with a slope inversely proportional to the market depth." [verified] — [CKS 2014](https://arxiv.org/abs/1011.6402)
- Data: one month (April 2010) of TAQ data for 50 randomly selected S&P 500 stocks. Regressions on 10-second mid-price changes, estimated in 273 half-hour subsamples per stock (about 180 observations each). The first and last 30 minutes of the day are excluded. [verified] — [CKS 2014](https://arxiv.org/abs/1011.6402)
- Average **R² = 65%**. Adding a quadratic term OFI·|OFI| raises R² only from 65% to 68%, and the quadratic coefficient is insignificant in most samples. The authors conclude there is no evidence of nonlinearity. [verified] — [CKS 2014](https://arxiv.org/abs/1011.6402)
- Tautology check: OFI includes the very events that change the price. Excluding price-changing events from OFI lowers R², "but remained in the 35%-60% region." [verified] — [CKS 2014](https://arxiv.org/abs/1011.6402)
- **OFI vs trade imbalance:** average R² 65% for OFI vs **32%** for trade imbalance (TI). When both are used, TI becomes statistically insignificant. "Trades seem to carry little to no information about price changes after the simultaneous order flow imbalance is taken into account." [verified] — [CKS 2014](https://arxiv.org/abs/1011.6402)
- **Depth:** λ = 1 "appears to be a good approximation for most of the stocks." Near the open, depth is about half its average and the impact coefficient is about twice its average. Price impact is about 5× higher at the open than at the close. [verified] — [CKS 2014](https://arxiv.org/abs/1011.6402)
- **Square-root law:** the linear OFI model plus a scaling argument implies the empirical "square-root" relation between price change and volume. The authors argue the volume–price relation is noisy and "a statistical artifact due to the aggregation of data." [verified] — [CKS 2014](https://arxiv.org/abs/1011.6402)
- In the TAQ data the authors computed OFI from changes in best quotes and sizes (Eq. 1), not from individual order messages, because order-level data were unavailable. [verified, as described in CCZ footnote 4] — [Cont, Cucuringu & Zhang](https://arxiv.org/abs/2112.13213)

### Inferences
- For a **prediction** project, the 65% R² is a misleading headline. It explains price changes using order flow from the **same** interval. Lagged OFI has far weaker predictive power (see Section 3: CCZ report negative out-of-sample R² for 1-minute-ahead returns). OFI's main practical value as a predictor is (a) as a feature summarising recent pressure and (b) scaled by depth (OFI/depth), which the β ∝ 1/depth result motivates.
- Because the slope varies with depth and time of day, models should either normalise OFI by recent depth or include time-of-day / depth features.

### Gaps
- The exact per-event e_n equation could not be cleanly extracted (PDF math was garbled). The version above is the widely reproduced form and should be checked against Eq. (1) of the paper before formal citation.
- The R² for different interval lengths (the paper's robustness across time scales) was not extracted.

---

## 3. Multi-level OFI: Xu, Gould & Howison (2019) and Cont, Cucuringu & Zhang (2023) integrated OFI via PCA

### Takeaway
Extending OFI to deeper levels (MLOFI, a vector of per-level OFIs) improves out-of-sample fit at **every** added level. The improvement is about **65–75% for large-tick** and **15–30% for small-tick** stocks in RMSE terms. Per-level OFIs are highly collinear, so ridge regression or PCA is needed. CCZ compress the top 10 levels into the **first principal component** (which explains about 89% of variance), called **integrated OFI**. It raises contemporaneous out-of-sample R² from about 65% to about 84%. Yet **forecasting** 1-minute-ahead returns with OFI gives **negative** out-of-sample R².

### Definition
- MLOFI_m(t_{k−1}, t_k): the same bookkeeping as OFI, applied at the m-th level on each side, for m = 1..M. The model is ΔP_k = α + Σ_m β_m · MLOFI_m + ε. Xu et al. give careful rules for when levels shift (e.g. an order arriving inside the spread becomes the new level 1 and the old level 1 becomes level 2). [verified] — [Xu, Gould & Howison, arXiv:1907.06230](https://arxiv.org/abs/1907.06230)
- Integrated OFI (CCZ): ofi^I = w₁ᵀ · ofi^(1..10) / ‖w₁‖₁, where w₁ is the first principal vector estimated from historical data and normalised by its l1 norm. [verified] — [Cont, Cucuringu & Zhang, arXiv:2112.13213](https://arxiv.org/abs/2112.13213)

### Intuition
- Level-1 OFI misses pressure building **behind** the best quote. Suppose 10,000 shares are added at level 2 of the bid. A market sell that clears level 1 then meets a wall, so the price falls only one tick and bounces. Deeper flow tells you how resilient the book is.
- Why PCA: OFIs at levels 1–10 move together (correlations above 75%), so putting all 10 in a regression is unstable (multicollinearity, overfitting). The first PC is essentially a weighted "total net order flow across the top of the book." It keeps most of the information in one stable feature.

### Cited Findings
- Xu, Gould & Howison: 6 liquid Nasdaq stocks (LOBSTER). "For all 6 stocks… out-of-sample goodness-of-fit… improves with each additional price level." [verified] — [arXiv:1907.06230](https://arxiv.org/abs/1907.06230) (arXiv preprint; v2 Oct 2019. I did not confirm a journal version.)
- Xu et al.: OLS fits on MLOFI suffer from multicollinearity, which makes deep-level coefficients look insignificant. With **ridge regression**, "even the parameters that correspond to price levels deep into the LOB are strongly statistically significant." Out-of-sample RMSE improves by "about 65–75% for large-tick stocks and about 15–30% for small-tick stocks." [verified] — [arXiv:1907.06230](https://arxiv.org/abs/1907.06230)
- Xu et al. note that CKS (2014, Appendix B3) had briefly tried OFI at 5 levels. Robustness: windows of 30–60 min and Δt of 5–40 s give qualitatively similar results for large-tick stocks. For small-tick stocks, adjusted R² rose slightly with longer windows (up to about 10% variation for AMZN). [verified] — [arXiv:1907.06230](https://arxiv.org/abs/1907.06230)
- CCZ data: Nasdaq ITCH via LOBSTER, top 100 S&P 500 components, 2017-01-01 to 2019-12-31, minute-level. [verified] — [arXiv:2112.13213](https://arxiv.org/abs/2112.13213) (v4, June 2023. The arXiv page lists no journal. The commonly cited venue, *Quantitative Finance* 2023, is [unverified].)
- Explained-variance ratio of PCs of 10-level OFIs: **89.06%**, 4.99%, 2.28%, 1.28%, … The best-level OFI has the **smallest** weight in PC1 but the highest cross-stock variability. [verified, Table 2] — [CCZ](https://arxiv.org/abs/2112.13213)
- Contemporaneous fits: in-sample R² is 71.16% with best-level OFI (PI[1]) vs **87.14%** with integrated OFI (PI^I). Out-of-sample R² is 64.64% vs **83.83%**. Integrated OFI also outperforms using all 10 multi-level OFIs separately out of sample (Appendix B). [verified, Tables 3 and 5] — [CCZ](https://arxiv.org/abs/2112.13213)
- Cross-impact: with best-level OFI, adding other stocks' OFIs (LASSO-selected) helps slightly. Once integrated OFI is used, contemporaneous cross-impact adds essentially nothing (OS R² 83.83% vs 83.62%). Lagged cross-asset OFIs **do** improve forecasting of future returns, but this predictability "decays quickly through time" over horizons of up to a few minutes. [verified] — [CCZ](https://arxiv.org/abs/2112.13213)
- **Key caveat for prediction:** out-of-sample R² for **one-minute-ahead** returns is negative for all models. Values are −0.37% for best-level own-OFI, −0.36% for integrated own-OFI, −0.36% for AR on returns, and −0.10% for the cross-impact versions. The authors note negative R² "do[es] not imply that the forecasts are economically meaningless" and show a profitable forecast-based strategy. [verified, Table 8] — [CCZ](https://arxiv.org/abs/2112.13213)

### Inferences
- **Contemporaneous explanation and future prediction are very different tasks.** Multi-level/integrated OFI is excellent at explaining same-interval price changes. Forecasting is near the noise floor at 1-minute horizons in large US equities. Short-horizon LOB classification tasks (e.g. FI-2010 next-k-event mid-direction) work at much shorter horizons, measured in events, where state features such as queue imbalance retain more signal.
- PCA weights should be fitted on training data only (CCZ use historical data). Fitting PCA on the full sample would leak test information into features.

### Gaps
- Exact R² values per level in Xu et al. were not extracted (only the summary RMSE improvement ranges).
- I did not verify the journal publication of either paper.

---

## 4. Order-book / queue imbalance as a predictor: Gould & Bonart (2016); Cartea, Donnelly & Jaimungal (2018)

### Takeaway
Queue imbalance I = (n_b − n_a)/(n_b + n_a) at the best quotes is a statistically significant predictor of the **direction of the next mid-price move** for all 10 Nasdaq stocks studied. It is strong for large-tick stocks and weak for small-tick ones. Cartea, Donnelly & Jaimungal build the same quantity into an optimal-trading model and find it materially improves strategy P&L by reducing adverse selection.

### Definition
- Gould & Bonart: I(t) = (n_b(b_t,t) − n_a(a_t,t)) / (n_b(b_t,t) + n_a(a_t,t)) ∈ [−1, 1], where n_b and n_a are the best-bid and best-ask queue volumes. (Yang & Zhu 2015 use n_b/(n_b+n_a) ∈ [0,1], a linear rescaling.) [verified] — [Gould & Bonart, arXiv:1512.03492](https://arxiv.org/abs/1512.03492)
- Cartea et al.: volume imbalance ρ = (V_b − V_a)/(V_b + V_a) at the best quotes. Values near +1 or −1 mean strong buy or sell pressure. They discretise it into regimes in a Markov-chain-modulated model. [abstract/secondary] — [IDEAS record, Applied Mathematical Finance 25(1):1–35, 2018](https://ideas.repec.org/a/taf/apmtfi/v25y2018i1p1-35.html)

### Intuition and worked example
- Bid queue 900 shares and ask queue 100 shares give I = (900 − 100)/1000 = **+0.8**. Only 100 shares need to trade or cancel to clear the ask, while 900 would be needed to clear the bid. Absent other information, the next mid move is much more likely to be **up**. The logistic model is P(up | I) = 1/(1 + e^{−(x₀ + x₁ I)}).

### Cited Findings
- Data: 10 liquid Nasdaq stocks, all of 2014 (LOBSTER). Large-tick: MSFT, INTC, MU, CSCO, ORCL. Small-tick: GOOG, AMZN and others. The first and last 30 minutes are excluded. [verified] — [Gould & Bonart](https://arxiv.org/abs/1512.03492)
- Method: logistic regression from I to the next mid-price direction (binary classifier evaluated with out-of-sample ROC/AUC; probabilistic classifier evaluated with proper scoring). Local logistic regression is used as a semi-parametric check. The null model is P(up) = 1/2 regardless of I. [verified] — [Gould & Bonart](https://arxiv.org/abs/1512.03492)
- Results: there is a "strongly statistically significant relationship" for every stock. Binary classification improves on the null by **about 50–60% for large-tick stocks and 10–30% for small-tick stocks**. Probabilistic predictions improve by **about 20–30% (large-tick) and 2–6% (small-tick)**. Local logistic regression slightly outperforms logistic regression at higher computational cost. [verified] — [Gould & Bonart](https://arxiv.org/abs/1512.03492)
- Earlier evidence that queues are coupled, contradicting independent-queue models such as Cont & de Larrard (2013), comes from Garèche et al. (2013), as discussed by Gould & Bonart. [verified as cited] — [Gould & Bonart](https://arxiv.org/abs/1512.03492)
- Cartea, Donnelly & Jaimungal: calibrated on 10 relatively large-tick Nasdaq stocks (Jan–Jun 2014, tested Jul–Dec 2014). Including volume imbalance in the optimisation "considerably boosts" profits, attributed to lower adverse-selection costs and better positioning of limit orders. [abstract/secondary] — [IDEAS](https://ideas.repec.org/a/taf/apmtfi/v25y2018i1p1-35.html); [Oxford ORA copy](https://ora.ox.ac.uk/objects/uuid:006addde-3a03-4d75-89c1-04b59026e1c0); [Cartea & Jaimungal PIMS slides](https://www.math.ualberta.ca/~cfrei/PIMS/OrderImbalance.pdf)
- Caveat raised in a secondary summary: LOB behaviour in 2014 may not represent other liquidity regimes, and imbalance can be **spoofed** (orders posted and cancelled quickly). [abstract/secondary] — [CXO Advisory summary](https://www.cxoadvisory.com/?p=27752)

### Inferences
- Queue imbalance is the natural **baseline** for a deep-learning LOB classifier. A model that cannot beat a one-feature logistic regression on I (especially on large-tick names) is not adding value.
- The "next mid move" target is event-driven. Results do not transfer directly to fixed clock horizons (e.g. 10 s or 1 min), where the moves are a mix of several queue depletions.
- Displayed imbalance is gameable (spoofing, hidden/iceberg orders). This is a real-world limitation for live use, but less relevant for offline datasets.

### Gaps
- Exact AUC values per stock were not extracted (only the summary improvement ranges).
- I did not read the full Cartea–Donnelly–Jaimungal paper. Quantitative P&L improvements are not reported here.

---

## 5. Stoikov (2018) micro-price vs mid-price vs weighted mid

### Takeaway
The mid-price ignores the queues, so it is a poor fair-value estimate in large-tick markets. The **weighted mid** (WAP-style) uses imbalance but is noisy and over-reacts. Stoikov's **micro-price** is defined as the limit of expected future mid-prices given the current imbalance and spread. It is estimated from data via a Markov chain, and is designed to be a martingale.

### Definitions
- Mid: M = (P_b + P_a)/2. Ignores sizes.
- Imbalance: I = Q_b/(Q_b + Q_a) ∈ [0,1].
- Weighted mid: W = I·P_a + (1 − I)·P_b. Note that the ask price is weighted by the **bid** size: a big bid pulls the estimate toward the ask.
- Micro-price: P^micro = lim_{i→∞} E[M_{τ_i} | I_t, S_t] = M_t + g(I_t, S_t). Here S is the spread, τ_i are successive mid-price change times, and g is the sum of adjustments g¹ + g² + … computed from an estimated discrete Markov chain on (I, S, ΔM). [unverified - from background knowledge of the paper; SSRN page returned 403]
- Paper: S. Stoikov, "The micro-price: a high-frequency estimator of future prices," *Quantitative Finance* 18(12):1959–1966 (2018). SSRN 2970694. [venue/pages unverified - background knowledge] — [SSRN 2970694](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694); code and data (BAC and CVX, March 2011) at [github.com/sstoikov/microprice](https://github.com/sstoikov/microprice) [verified: repo describes microprice as "An estimator of the fair price, given the state of the order book"]

### Intuition and worked example
- Bid $10.00 × 900, ask $10.01 × 100. Mid = $10.005. I = 0.9, so W = 0.9·10.01 + 0.1·10.00 = **$10.009**.
- The mid says fair value sits halfway, but the queues say an up-tick is likely. W corrects this, but too aggressively and too jumpily. If 50 shares are added to the ask, W jumps, even though the true probability of an up-move has barely changed. W is also not a martingale. The micro-price instead asks: "historically, when I ≈ 0.9 and spread = 1 tick, where did the mid end up on average?" The answer is typically somewhere between M and W (e.g. ≈ $10.007 in this hypothetical).

### Cited Findings
- The micro-price is "an estimator of the fair price, given the state of the order book." The repo includes a notebook and BAC/CVX data. [verified] — [GitHub sstoikov/microprice](https://github.com/sstoikov/microprice)
- Claims usually attributed to the paper: (i) the mid ignores imbalance, (ii) the weighted mid is noisy and not a martingale, (iii) the micro-price is a martingale by construction and better predicts future mids at short horizons. [unverified - background knowledge; I could not retrieve the SSRN abstract or full text (HTTP 403)]

### Inferences
- In Optiver-style feature engineering, "WAP" **is** the weighted mid W (cross-weighted). It is a cheap, imbalance-aware price proxy, so log returns of WAP reflect order-book pressure as well as trades. The micro-price is the statistically principled refinement. For a prediction project, (micro-price − mid) is a natural engineered feature and a strong baseline for direction prediction.

### Gaps
- I could not verify the paper's abstract, empirical results or exact Markov-chain estimation details (SSRN blocked). Treat the formulas above as background knowledge to check against the paper or notebook.

---

## 6. VPIN: Easley, López de Prado & O'Hara (2012), the Flash Crash claim, and the Andersen & Bondarenko critique

### Takeaway
VPIN measures order-flow "toxicity" (how badly market makers are being adversely selected). It averages absolute buy–sell volume imbalance over equal-volume buckets, with trades classified by **bulk volume classification**. Its authors claimed it signalled the 6 May 2010 Flash Crash in advance. Andersen & Bondarenko found that VPIN reached its historical high only **after** the crash, is mechanically driven by trading intensity, and is a poor predictor of short-run volatility once that link is removed. VPIN is best thought of as a summary of volume-time order imbalance and activity, not a validated crash or volatility forecaster.

### Definition
- **Volume clock:** split trading into buckets of equal volume V instead of equal time. Busy periods produce more buckets.
- **Bulk volume classification (BVC):** within each time (or volume) bar i with volume V_i and price change ΔP_i, the estimated buy volume is V_i^B = V_i · Z(ΔP_i / σ_ΔP) and the sell volume is V_i^S = V_i − V_i^B. Z is the standard normal CDF and σ_ΔP is the standard deviation of bar price changes. This classifies volume probabilistically instead of trade by trade.
- **VPIN** = Σ_{τ=1}^{n} |V_τ^S − V_τ^B| / (n·V), a rolling average over the last n buckets.
- Formulas above are [unverified - standard statement from background knowledge; the RFS/SSRN pages returned 403]. Andersen & Bondarenko confirm the structure: order imbalance per bucket OI = |V^S − V^B|/V, averaged over buckets, with "TR-VPIN" (tick rule) and "BV-VPIN" (bulk volume) variants built on time bars inside volume buckets. [verified] — [Andersen & Bondarenko, CREATES RP 2011-50](https://repec.econ.au.dk/repec/creates/rp/11/rp11_50.pdf)

### Intuition and worked example
- A market maker quotes both sides and profits from the spread if buyers and sellers arrive in balance. If flow becomes one-sided (e.g. informed sellers), the maker keeps buying just before the price drops. That flow is "toxic." VPIN estimates the share of one-sided volume.
- Example: bucket size V = 1,000 contracts, n = 3 buckets with estimated buys of 500, 800 and 900. Imbalances are |500−500| = 0, |200−800| = 600 and |100−900| = 800. VPIN = (0 + 600 + 800)/(3·1000) = **0.47**. Rising VPIN means increasingly one-sided flow.

### Cited Findings
- Easley, López de Prado & O'Hara, "Flow Toxicity and Liquidity in a High-Frequency World," *Review of Financial Studies* 25(5):1457–1493 (2012), DOI 10.1093/rfs/hhs053. Order flow is toxic "when it adversely selects market makers, who may be unaware they are providing liquidity at a loss." VPIN is based on "volume imbalance and trade intensity," is "updated in volume-time," does not require numerical estimation of unobservable parameters, introduces bulk volume classification, and is "a useful indicator of short-term, toxicity-induced volatility." [abstract/secondary] — [IDEAS/RePEc](https://ideas.repec.org/a/oup/rfinst/v25y2012i5p1457-1493.html); [SSRN 1695596](https://papers.ssrn.com/abstract=1695596); [NYU Stern PDF](https://www.stern.nyu.edu/sites/default/files/assets/documents/con_035928.pdf)
- The Flash Crash application is in the companion paper "The Microstructure of the 'Flash Crash': Flow Toxicity, Liquidity Crashes and the Probability of Informed Trading" (*Journal of Portfolio Management*, 2011). It claims VPIN reached historically high levels in the hours before the 6 May 2010 crash. [title verified via SSRN listing; the specific "over an hour before" timing is unverified - background knowledge] — [SSRN 1695041](https://papers.ssrn.com/abstract=1695041)
- **Andersen & Bondarenko (working paper version, CREATES RP 2011-50, May 2013), studying E-mini S&P 500 futures:**
  - "Finding 1: The level of TR-VPIN just prior to the flash crash is elevated, but it is not at a historical high. The TR-VPIN metric only achieves a historical high after the crash has subsided." [verified]
  - The time-bar construction mechanically ties VPIN to trading intensity. The same sequence of trades can produce bucket order imbalance anywhere from 0 to 1 depending only on how fast trading occurs. "Finding 4: All else equal, the expected value of OI, as well as the associated tick rule VPIN measure, is increasing in the degree of volume heterogeneity across time bars." [verified]
  - "Finding 7: Time bar TR-VPIN is strongly correlated with volume and weakly correlated with VIX. In contrast, FB-VPIN [fixed volume bin] is weakly, and predominantly negatively, correlated with volume, and strongly negatively correlated with VIX." Once the mechanical volume link is removed, the correlation with volatility reverses sign. [verified]
  - TR-VPIN correlations with future volatility are "uniformly below" those of a benchmark (U1-VPIN) that assumes **random** trade classification. In other words, the actual buy/sell signing adds no forecasting power beyond trading intensity. [verified]
  - — [Andersen & Bondarenko, CREATES RP 2011-50](https://repec.econ.au.dk/repec/creates/rp/11/rp11_50.pdf)
- The published version is Andersen & Bondarenko, "VPIN and the Flash Crash," *Journal of Financial Markets* 17 (2014) 1–46. It concludes VPIN is "a poor predictor of short run volatility," that its peak came after the crash, and that its apparent predictive power reflects its mechanical relation with trading intensity. [abstract/secondary] — [Kellogg listing](https://www.kellogg.northwestern.edu/faculty/research/detail/2014/vpin-and-the-flash-crash/)
- **Reply:** Easley, López de Prado & O'Hara, "VPIN and the Flash Crash: A rejoinder," *JFM* 17 (2014) 47–52. They argue the critique attacks "a methodology we do not advocate, an analysis we never performed, and conclusions we did not draw." In particular, they say A&B mainly analysed tick-rule VPIN with time bars rather than their recommended BVC setup. [abstract/secondary] — [UTS OPUS record](https://opus.cloud1.lib.uts.edu.au/handle/10453/118180); also "VPIN and the Flash Crash: A Comment" — [SSRN 2062450](https://papers.ssrn.com/abstract=2062450). The same JFM issue carried an editorial piece, "Reflecting on the VPIN dispute." [abstract/secondary]

### What VPIN is and is not good for (inference)
- **Reasonable uses:** a real-time, volume-time summary of signed-flow imbalance plus activity. It is a plausible feature for regime detection (stressed vs calm), and the volume clock and BVC are useful tools on their own.
- **Not established:** a reliable leading indicator of crashes, or a better volatility forecaster than simple activity measures (volume, trade count, realized volatility). Its signal is confounded by trading intensity and depends heavily on parameters (bucket size, number of buckets, bar length, classification rule).
- **For this project:** if VPIN is used as a feature, include raw volume/intensity as well so the model can separate the two. Do not claim predictive value without ablating against intensity-only features.

### Gaps
- I could not access the RFS or JFM full texts (SSRN 403). Exact parameter choices in ELO (e.g. 50 buckets per day) and the rejoinder's specific counter-evidence are not verified here.
- A Retraction Watch item (April 2013) appeared in search results in connection with the dispute. I did not read it and cannot say what it concerned — [Retraction Watch](https://retractionwatch.com/2013/04/11/markets-crash-and-so-does-a-a-paper-explaining-why).

---

## 7. Adverse selection basics: Glosten–Milgrom and Kyle

### Takeaway
The bid–ask spread and price impact exist partly because liquidity providers trade against people who may know more. Each order is a noisy signal of private information, so prices move toward it. This is the theoretical root of "toxicity" (VPIN) and of the empirical fact that order flow (OFI) moves prices.

### Intuition
- **Glosten–Milgrom (1985):** a competitive market maker faces a mix of informed and uninformed traders, one trade at a time. A buy order raises the probability that the asset is worth more. The maker sets ask = E[value | a buy arrives] and bid = E[value | a sell arrives]. The spread is the break-even compensation for losing to informed traders, and it widens with the share of informed traders. Prices update in a Bayesian way with each trade, so trades have permanent impact.
- **Kyle (1985):** one insider, noise traders, and market makers who see only **net** order flow y. The equilibrium price is linear: Δp = λ·y. "Kyle's lambda" λ is price impact per unit of net order flow, and **1/λ is market depth**. The insider trades gradually to hide information inside the noise.
- Link to CKS: the CKS regression ΔP = β·OFI with β ∝ 1/depth is an empirical, quote-level analogue of Kyle's Δp = λ·y. CKS explicitly explain intraday patterns in impact through **observable depth**, rather than through unobservable information asymmetry.

### Cited Findings
- Glosten, L. & Milgrom, P. (1985), "Bid, ask and transaction prices in a specialist market with heterogeneously informed traders," *Journal of Financial Economics* 14(1):71–100. [unverified - bibliographic details from background knowledge] — [DOI 10.1016/0304-405X(85)90044-3](https://doi.org/10.1016/0304-405X(85)90044-3)
- Kyle, A. S. (1985), "Continuous auctions and insider trading," *Econometrica* 53(6):1315–1335. [unverified - bibliographic details from background knowledge] — [DOI 10.2307/1913210](https://doi.org/10.2307/1913210)
- CKS relate intraday patterns in price impact and volatility to market depth, "as opposed to unobservable parameters previously invoked in the literature, such as information asymmetry or informativeness of trades." They also note that low morning depth may itself reflect liquidity providers' awareness of information asymmetry, since with a one-cent spread they can avoid being "picked off" only by posting less. [verified] — [CKS 2014](https://arxiv.org/abs/1011.6402)
- ELO frame toxicity as adverse selection of market makers. [abstract/secondary] — [IDEAS](https://ideas.repec.org/a/oup/rfinst/v25y2012i5p1457-1493.html)

### Inferences
- Any predictive signal in LOB data is, economically, either (a) mechanical, as in queue depletion for large-tick assets, or (b) informational, where order flow carries private information that prices absorb with a lag. Mechanical signals tend to be very short-lived and hard to monetise after costs, because crossing the spread costs about one tick, which is about the size of the predicted move.

### Gaps
- No full-text verification of the 1985 papers this session. This is standard, well-established material.

---

## 8. Realized volatility and WAP as used in the Optiver competitions

### Takeaway
In Optiver Realized Volatility Prediction (Kaggle, 2021), the price proxy is the **WAP**: the best-level prices cross-weighted by the opposite side's size, which equals the weighted mid. Log returns are taken between consecutive book snapshots. **Realized volatility** = sqrt(sum of squared log returns) over a 10-minute bucket. The target is RV of the **next** 10 minutes per (stock_id, time_id), scored by **RMSPE**.

### Definitions
```
WAP1 = (bid_price1 * ask_size1 + ask_price1 * bid_size1) / (bid_size1 + ask_size1)
WAP2 = same with level-2 prices/sizes
r_t  = log(WAP_t / WAP_{t-1})
RV   = sqrt( sum_t r_t^2 )
RMSPE = sqrt( mean( ((y - y_hat) / y)^2 ) )
```
- The cross-weighting (bid price × ask size) is deliberate. A large bid queue pulls WAP toward the ask, the same intuition as the weighted mid and micro-price.
- Optiver "Trading at the Close" (Kaggle, 2023) also supplies a `wap` column with the same cross-weighted definition. [unverified - background knowledge; not fetched this session]

### Worked example
- Snapshots: WAP = 1.0000, 1.0004, 0.9998, 1.0001. Log returns ≈ +0.0004, −0.0006, +0.0003. Squares ≈ 1.6e-7, 3.6e-7, 0.9e-7. Sum = 6.1e-7. **RV ≈ 0.00078.**
- RMSPE penalises **relative** error, so under-predicting a tiny-volatility bucket by 50% costs as much as under-predicting a large one by 50%. This is why many competitors trained with sample weights 1/y².

### Cited Findings
- The competition task is to predict the realized volatility of a set of stocks for given time IDs using the information from a 10-minute window. It is a tabular time-series problem optimised for RMSPE. [abstract/secondary] — [GitHub writeup, Mrutyunjay01](https://github.com/Mrutyunjay01/Optiver-realized-volatility-prediction-kaggle21)
- The data provide bid_price1/2 and ask_price1/2 (normalised prices of the most and second-most competitive levels) plus corresponding sizes. The stock price is approximated by WAP. Log returns are computed between successive WAP values, and RV is the square root of the sum of squared log returns. The target is RV over the 10-minute window **following** the feature window for the same stock_id and time_id. [abstract/secondary] — [Erdős Institute project slides](https://erdosinstitute.org/_files/ugd/be9ca7_e4c07cbf1fc445d69dc482e5884016ca.pdf); [Erdős summary](https://erdosinstitute.org/_files/ugd/be9ca7_20723d941cbd4ef8a665834f2c7e692a.pdf)
- Community code defines `calc_wap1` as (bid_price1·ask_size1 + ask_price1·bid_size1)/(bid_size1 + ask_size1). Some notebooks also use a same-side-weighted variant, which is a different quantity. [abstract/secondary] — [BulletTech writeup](https://raw.githubusercontent.com/BulletTech/BulletTech/main/docs/Main_Course/Machine_Learning/2021-08-27-Kaggle-optiver.md)
- Optiver's data scientists published an official tutorial notebook explaining these concepts. I could not retrieve its text (Kaggle renders client-side). — [Kaggle: Introduction to financial concepts and data](https://www.kaggle.com/code/jiashenliu/introduction-to-financial-concepts-and-data)
- Reported baselines from a student project (not comparable to the leaderboard): predicting the mean gives RMSPE ≈ 1.11. Using current-window RV from WAP1 as the prediction gives ≈ 0.341, and from WAP2 ≈ 0.705. A LightGBM+NN ensemble paper reports 0.2176 on its test set. [abstract/secondary; different validation setups] — [Erdős project slides](https://erdosinstitute.org/_files/ugd/be9ca7_e4c07cbf1fc445d69dc482e5884016ca.pdf)

### Inferences
- The "naive persistence" baseline (next RV = current RV, computed from WAP1) is the right baseline for RVP. Volatility clustering makes it strong.
- Optiver prices are normalised per (stock, time_id) and time_id order is shuffled. This blocks naive cross-time features, but competitors partially reconstructed the time order, a known leakage issue. Any reported score should state whether such reconstruction was used. [Inference; the time-order reconstruction episode is background knowledge, unverified this session]

### Gaps
- I could not quote the official Optiver/Kaggle tutorial text verbatim. The WAP/RV formulas are corroborated by two independent secondary sources, and are standard.
- I did not fetch the "Trading at the Close" competition definitions.
