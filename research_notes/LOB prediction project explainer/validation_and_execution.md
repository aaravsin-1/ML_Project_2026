# Rigorous Validation for Financial ML (A) and Execution Realism in HFT Simulation (B)

Context for the report writer: these notes explain why a student limit-order-book (LOB) prediction project (FI-2010, Optiver Realized Volatility, Optiver Trading-at-the-Close, C++ inference) would use purged/embargoed splits, multi-seed comparisons, macro-F1/MCC, conservative fill and fee assumptions, and why its Sharpe numbers and backtests should be read cautiously. Researched 2026-10-30. Flag key: **[UNVERIFIED]** = from background knowledge or a secondary source not checked against the primary text in this session; **[FORMULA FROM MEMORY]** = standard formula stated from the literature but not re-checked against the fetched PDF.

---

## A1. Look-ahead and leakage types in time-series ML

### Takeaway
Leakage is any path by which information unavailable at prediction time reaches the model. In LOB work the four common forms are: (1) feature leakage (features computed with future data), (2) label overlap (train labels share future windows with test labels), (3) normalization leakage (scalers or z-scores fit on the full sample or on test data), and (4) shuffled-ID leakage (supposedly anonymous or shuffled IDs whose order or content can be reverse-engineered). Each of these inflates out-of-sample scores, and purging, embargo and strictly past-only preprocessing are the standard defenses.

### Cited Findings
- Purging means removing from the training set every observation whose label overlaps in time with a label in the test set. — [Wikipedia: Purged cross-validation](https://en.wikipedia.org/wiki/Purged_cross-validation); [LuxAlgo concept page](https://www.luxalgo.com/library/concept/purged-cross-validation/)
- An embargo adds a buffer after the test set to handle serial correlation. One rule of thumb is about 1% of bars. Another source says there is no closed form and the buffer should grow if features or labels have long memory. — [LuxAlgo](https://www.luxalgo.com/library/concept/purged-cross-validation/); [temporalcv docs](https://temporalcv.readthedocs.io/en/latest/api/cv_financial.html)
- Purging does not fix selection bias from trying many features, labels or models, so multiple-testing corrections and forward confirmation are still needed. — [Wikipedia: Purged cross-validation](https://en.wikipedia.org/wiki/Purged_cross-validation)
- Data leakage across ML-based science is a documented driver of irreproducible results. Kapoor & Narayanan's taxonomy includes "no separate test set", "pre-processing on train+test", "temporal leakage" and "illegitimate features". **[UNVERIFIED, not fetched this session]** — [Kapoor & Narayanan 2023, arXiv 2207.07048](https://arxiv.org/abs/2207.07048)
- Shuffled-ID leakage in the Optiver Realized Volatility Kaggle competition: time_ids were shuffled to hide chronology. Participants reportedly recovered the time order by reconstructing absolute prices from tick sizes and ordering time_ids by an embedding/nearest-neighbour chain, then exploited cross-time_id information. **[UNVERIFIED: community-reported, no primary write-up fetched]** — [Kaggle competition page](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction)

### Inferences
- **Feature leakage** typically comes from centered rolling windows, `shift(-k)` errors, or volatility computed over a window that overlaps the target window. The fix is to compute features only from data with timestamps at or before t.
- **Normalization leakage** comes from fitting a `StandardScaler` or min-max scaler on all data, or z-scoring by full-day statistics. The fix is to fit the scaler on train only, or use a trailing or expanding window. FI-2010 ships pre-normalized versions (z-score, min-max, decimal-precision). Whether those normalizations use only past days is a known question to check in the dataset paper. **[UNVERIFIED for FI-2010 specifics]**
- **Label overlap** arises because a "mid-price direction over the next k events" label at t and at t+1 share k−1 future events. Random K-fold therefore puts near-duplicate labels on both sides of the split, and the model is rewarded for memorization (see A2).
- **Shuffled-ID leakage** is a reason to use group-aware splits (GroupKFold on time_id or stock_id) and to check whether the ordering of IDs is itself predictive. If shuffled IDs can be re-ordered, a "random" split is actually interleaved in time.

### Gaps
- I found no primary, citable write-up of the Optiver time_id reverse-engineering technique; it rests on Kaggle discussion threads that were not fetched.
- I did not verify FI-2010's normalization procedure (Ntakaris et al. 2018) in this session.

---

## A2. Purged K-fold, embargo, CPCV; why overlapping labels make naive CV optimistic

### Takeaway
When labels span intervals [t, t+h], ordinary K-fold leaks future information: a test label's outcome window overlaps neighbouring training labels, and serially correlated features let the model "look up" the answer. Purged K-fold removes overlapping training samples, and the embargo removes a buffer after each test fold. Combinatorial purged CV (CPCV) evaluates all C(N, k) test-group combinations, producing many backtest paths and a distribution of out-of-sample performance rather than one number.

### Cited Findings
- CPCV was introduced by López de Prado in *Advances in Financial Machine Learning* (Wiley 2018). It splits history into blocks, evaluates every combination of test blocks with purging and embargo at each boundary, and yields multiple backtest paths and a performance distribution. — [Wikipedia: Purged cross-validation](https://en.wikipedia.org/wiki/Purged_cross-validation); [skfolio CombinatorialPurgedCV](https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html)
- Implementations exist: skfolio `CombinatorialPurgedCV`, temporalcv `CombinatorialPurgedCV`, and the `purgedcv` conda package. — [skfolio](https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html); [temporalcv](https://temporalcv.readthedocs.io/en/latest/api/cv_financial.html); [conda-forge purgedcv](https://anaconda.org/conda-forge/purgedcv)
- Embargo sizing is heuristic (about 1% of bars), with no closed form. — [LuxAlgo](https://www.luxalgo.com/library/concept/purged-cross-validation/)

### Inferences
- **Mechanics [FORMULA FROM MEMORY, AFML ch. 7 and 12]:** Each label i has a span [t_i,0, t_i,1]. Training sample i is purged if [t_i,0, t_i,1] ∩ [min test t_0, max test t_1] ≠ ∅. Embargo: also drop training samples with t_i,0 in (max test t_1, max test t_1 + h_embargo].
- **CPCV path count [FORMULA FROM MEMORY]:** With N groups and k test groups per split there are C(N, k) splits and φ = (k/N)·C(N, k) complete backtest paths. Example: N = 6, k = 2 gives 15 splits and 5 paths.
- **Why naive CV is optimistic, intuitively:** with horizon-h labels and persistent features, the nearest training neighbour of a test point is often its own time-neighbour with a nearly identical label. Random K-fold therefore measures interpolation, not forecasting. The gap grows with h and with feature persistence.
- **For a student project, the minimum viable version is:** chronological train/validation/test splits by day, a gap of at least h events (or one full bucket) between them, and scalers fit on train only. Full CPCV is the stronger option when comparing many model variants.

### Gaps
- I did not fetch López de Prado's book text directly; the formulas above should be checked against AFML ch. 7 (purged K-fold) and ch. 12 (CPCV).

---

## A3. Backtest overfitting: PBO, Deflated Sharpe Ratio, HLZ t > 3, multiple testing

### Takeaway
If you try N configurations and report the best, the expected maximum Sharpe ratio under the null grows with N, so a "good" backtest is expected by chance alone. PBO (via CSCV) estimates the probability that the in-sample winner is below median out of sample. DSR raises the Sharpe hurdle for the number of trials and for non-normal returns. Harvey, Liu & Zhu argue a new factor needs t > 3.0 rather than 2.0.

### Cited Findings
- **PBO:** Bailey, Borwein, López de Prado & Zhu propose combinatorially symmetric cross-validation (CSCV) to estimate the probability of backtest overfitting. The paper argues that standard hold-out is unreliable for investment backtests. It was published in the *Journal of Computational Finance* 20(4), 2017, DOI 10.21314/JCF.2016.322. — [SSRN 2326253](https://papers.ssrn.com/abstract=2326253); [eScholarship PDF](https://escholarship.org/content/qt4w1110bb/qt4w1110bb.pdf); [Risk.net](https://www.risk.net/journal-of-computational-finance/2471206/the-probability-of-backtest-overfitting)
- An R package `pbo` on CRAN implements the method. — [pbo README](https://packages.oit.ncsu.edu/cran/web/packages/pbo/readme/README.html)
- **DSR:** Bailey & López de Prado, "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality," *Journal of Portfolio Management* 40(5):94–107, 2014. It adjusts for selection bias under multiple testing and for non-normal returns. With a single trial the DSR reduces to the Probabilistic Sharpe Ratio. — [SSRN 2460551](https://papers.ssrn.com/abstract=2460551); secondary: [quantstrat deflatedSharpe docs](https://www.rdocumentation.org/packages/quantstrat/versions/0.16.7/topics/deflatedSharpe)
- **HLZ:** Harvey, Liu & Zhu, "...and the Cross-Section of Expected Returns," *Review of Financial Studies* 29(1):5–68 (2016), DOI 10.1093/rfs/hhv059. Given the hundreds of factors already tested, a new factor needs t > 3.0. Their multiple-testing framework allows correlated tests and missing data, and they conclude that most claimed findings in financial economics are likely false. — [NBER w20592](https://www.nber.org/papers/w20592); [SSRN 2513152](https://papers.ssrn.com/abstract=2513152)
- **Counterpoint:** a 2022 re-analysis (Chen) argues the false discovery rate for |t| > 2 in the cross-section is fairly small, which is more optimistic than HLZ. — [arXiv 2206.15365](https://arxiv.org/pdf/2206.15365)

### Inferences
- **CSCV/PBO recipe [FORMULA FROM MEMORY]:**
  1. Build a T×N matrix of per-period returns for the N trials.
  2. Split it row-wise into S equal blocks and form all C(S, S/2) IS/OOS partitions.
  3. In each partition, pick the IS-best trial n* and compute its OOS relative rank ω̄ ∈ (0, 1).
  4. Compute the logit λ = ln(ω̄ / (1 − ω̄)).
  5. PBO = P(λ ≤ 0), the fraction of partitions where the IS winner is at or below the OOS median.
- **PSR [FORMULA FROM MEMORY]:** PSR(SR\*) = Φ( (SR̂ − SR\*)·√(T−1) / √(1 − γ̂₃·SR̂ + ((γ̂₄ − 1)/4)·SR̂²) ), where γ₃ is skewness, γ₄ is kurtosis (non-excess), and SR̂ is non-annualized.
- **DSR [FORMULA FROM MEMORY]:** DSR = PSR(SR₀\*), where SR₀\* = √V[{SR̂_n}] · ( (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ). Here γ ≈ 0.5772 (Euler–Mascheroni), N is the number of independent trials, and V is the variance of the trial Sharpe ratios. The intuition is that the expected maximum of N null Sharpe ratios grows roughly like √(2 ln N).
- **Relevance:** every hyperparameter sweep, architecture swap, label horizon and threshold the student tried counts toward N. Reporting the count of trials, and using a held-out test touched once, is the cheap honest version of DSR and PBO.
- HLZ's t > 3 maps roughly onto a Bonferroni-style correction: with about 300 tests at α = 5%, the per-test α is about 1.7e-4, i.e. |z| ≈ 3.8. HLZ's adjusted methods (Holm, BHY) land near 3.

### Gaps
- PSR and DSR formulas were not re-checked against the PDF this session (the SSRN pages were reached only via search snippets).

---

## A4. Why annualizing Sharpe from short high-frequency samples misleads; Sharpe estimation error (Lo 2002)

### Takeaway
A Sharpe ratio is an estimate with sizeable standard error. Scaling by √(periods per year) assumes i.i.d. returns, and serial correlation (common in HF P&L, inventory and microstructure noise) biases the annualized figure. Lo reports up to a 65% overstatement for annualized SRs of funds with autocorrelated monthly returns. A few days of HF data multiplied by √(252·23400) can produce absurd Sharpes (double digits) that say little about deployability.

### Cited Findings
- Lo, "The Statistics of Sharpe Ratios," *Financial Analysts Journal* 58(4):36–52, July/Aug 2002, DOI 10.2469/faj.v58.n4.2453. It derives the sampling distribution of SR under i.i.d., stationary and time-aggregated assumptions. — [CFA Institute](https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios); [PDF copy](https://traders.studentorg.berkeley.edu/papers/The-Statistics-of-Sharpe-Ratios.pdf); [RePEc](https://ideas.repec.org/a/taf/ufajxx/v58y2002i4p36-52.html)
- Serial correlation in monthly returns can overstate annual Sharpe ratios by as much as 65%. Monthly SRs cannot in general be annualized by multiplying by √12, and the paper gives the correct conversion for stationary returns. — [CFA Institute summary](https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios)

### Inferences
- **i.i.d. standard error [FORMULA FROM MEMORY]:** SE(SR̂) ≈ √( (1 + SR²/2) / T ), with SR per period. Example: per-period SR = 0.05 over T = 10,000 one-second bars gives SE ≈ 0.01, a 95% CI of [0.03, 0.07]. Naively annualized by √(252·23,400) ≈ 2,428, that CI spans roughly [73, 170]. The point estimate is "precise" only under i.i.d. assumptions and covers a few hours of one regime.
- **Lo's autocorrelation-adjusted annualization [FORMULA FROM MEMORY]:** SR(q) = η(q)·SR, with η(q) = q / √( q + 2·Σ_{k=1}^{q−1} (q − k)·ρ_k ). Positive ρ_k makes the true η smaller than √q, so naive √q overstates the annualized figure.
- **Practical issues beyond statistics:**
  - Gross-of-cost P&L (see B1).
  - Short samples cover a single volatility regime.
  - HF strategies are capacity-limited, so Sharpe does not scale with capital (see B3).
  - The overlap of positions held over multi-step horizons creates autocorrelated P&L.
- **Recommendation:** report per-trade or per-day statistics with a CI, net of fees and spread, and avoid annualizing from less than many weeks of out-of-sample data.

### Gaps
- Full text of Lo (2002) was not fetched (paywall); the formulas above should be checked against the PDF copy linked.

---

## A5. Effective sample size with autocorrelated/overlapping labels; seed variance; paired comparisons

### Takeaway
One million overlapping LOB snapshots are not one million independent observations. Overlapping horizon-h labels and persistent features shrink the effective sample size by roughly a factor of h or more, so naive standard errors and significance tests are far too optimistic. Separately, deep-learning results vary materially across random seeds, so model comparisons should use several seeds and paired tests on identical data splits.

### Cited Findings
- López de Prado's framework (AFML ch. 4) uses "concurrency" and "average uniqueness" of labels to weight samples and set bootstrap draws, because overlapping labels are not independent. This is a secondary description; the book was not fetched. **[UNVERIFIED]** — [Wikipedia: Purged cross-validation](https://en.wikipedia.org/wiki/Purged_cross-validation) (context on overlapping labels)
- In deep RL, results can differ significantly across random seeds, and too few seeds can produce misleading comparisons. **[UNVERIFIED, not fetched]** — [Henderson et al. 2018, "Deep Reinforcement Learning that Matters", arXiv 1709.06560](https://arxiv.org/abs/1709.06560)
- ML benchmark comparisons should account for variance from data sampling, initialization and hyperparameter optimization, and should randomize multiple sources of variation. **[UNVERIFIED, not fetched]** — [Bouthillier et al. 2021, "Accounting for Variance in Machine Learning Benchmarks", arXiv 2103.03098](https://arxiv.org/abs/2103.03098)

### Inferences
- **Effective sample size [FORMULA FROM MEMORY, standard time-series result]:** N_eff ≈ N / (1 + 2·Σ_{k≥1} ρ_k). For non-overlapping-information labels built from a sum of h increments, the label autocorrelation is about ρ_k = 1 − k/h for k < h, giving 1 + 2Σρ_k ≈ h and N_eff ≈ N/h. Feature persistence reduces it further.
- **Practical corrections:**
  - Subsample every h-th label for testing.
  - Use block bootstrap with block length ≥ h.
  - Use Newey–West or HAC standard errors.
  - Use day-level or bucket-level aggregation, treating each day or stock-bucket as one unit.
- **Seed protocol:** run each model with S ≥ 5 seeds on the same chronological splits. For model A vs B, compute per-seed (or per-day) metric differences d_s and test mean(d) with a paired t-test or Wilcoxon signed-rank. Pairing removes shared split difficulty. Report mean ± std (or a CI) across seeds, not the best seed. Picking the best seed is itself multiple testing (ties to A3).
- If two architectures differ by less than seed std on FI-2010, claims that one is better are not supported.

### Gaps
- Henderson et al. and Bouthillier et al. were cited from memory; their URLs were not fetched this session.
- I found no LOB-specific study quantifying seed variance on FI-2010 in this session.

---

## A6. Metrics for imbalanced 3-class direction prediction (macro vs weighted F1, MCC, kappa)

### Takeaway
With down/stationary/up labels, the class mix depends on the horizon and threshold, and the "stationary" class often dominates at short horizons. Accuracy and weighted-F1 reward predicting the majority class. Macro-F1 weights every class equally, and MCC and Cohen's kappa correct for chance agreement, so these are more honest headline metrics. None of these metrics, though, measures tradable value.

### Cited Findings
- On imbalanced data, accuracy gives an overoptimistic estimate driven by the majority class. MCC scores high only if the prediction does well on all four confusion-matrix categories. F1 is not invariant to swapping the positive class and ignores true negatives. MCC was the main metric in FDA-led MAQC/SEQC projects. — [Chicco & Jurman 2020, BMC Genomics](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6941312/)
- Caveat: MCC can be undefined or fluctuate heavily in extreme imbalance. — [Chicco & Jurman 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC6941312/)

### Inferences
Formulas for C = 3 classes (standard definitions, **[FORMULA FROM MEMORY]**):
- **Macro-F1** = (1/C)·Σ_c F1_c, where F1_c = 2·P_c·R_c / (P_c + R_c). **Weighted-F1** = Σ_c (n_c/N)·F1_c, so it tracks the majority class.
- **Multiclass MCC (Gorodkin R_K):** MCC = (c·s − Σ_k p_k·t_k) / √( (s² − Σ_k p_k²)·(s² − Σ_k t_k²) ), where c is the number correct, s the total, p_k the count predicted as k, and t_k the count truly k. It ranges over [−1, 1] and is 0 for chance or constant predictions.
- **Cohen's kappa:** κ = (p_o − p_e) / (1 − p_e), where p_e = Σ_k (p_k/s)·(t_k/s).

Worked example: if 70% of labels are "stationary", always predicting stationary gives accuracy 0.70, weighted-F1 ≈ 0.58, macro-F1 ≈ 0.27, MCC = 0 and κ = 0.

For trading, also report:
- precision on the up and down classes (wrong-sign trades are costly);
- calibration;
- a cost-aware P&L (B1), since a model can raise macro-F1 by improving "stationary" recall without any tradable edge.

### Gaps
- The multiclass MCC (Gorodkin 2004) and kappa sources were not fetched; the definitions are standard but stated from memory.

---

## B1. Spread-crossing cost, taker vs maker fees, crypto fee tiers

### Takeaway
A strategy that crosses the spread pays roughly the half-spread plus the taker fee on every fill, so a round trip costs about the full spread plus 2× the taker fee. On Binance's base tier that is at least 20 bps in fees alone, more than most short-horizon mid-price moves. Predicting mid-price direction is therefore not the same as profiting from it. Maker orders avoid the spread and pay lower fees, but they face fill uncertainty and adverse selection (B2).

### Cited Findings
- **Binance spot schedule** (official page, fetched 2026-10-30):

  | Tier | Maker / Taker |
  |---|---|
  | Regular | 0.100% / 0.100% |
  | VIP 1 | 0.090% / 0.100% |
  | VIP 2 | 0.080% / 0.100% |
  | VIP 3 | 0.040% / 0.060% |
  | VIP 5 | 0.025% / 0.031% |
  | VIP 9 | 0.011% / 0.023% |

  Paying fees in BNB gives 25% off, so Regular becomes 0.075% / 0.075%. — [Binance fee schedule](https://www.binance.com/en/fee/trading)
- A third-party article claims Binance moved to 0 maker fees across tiers. This conflicts with the official page above; treat it as unverified or limited to specific pairs or promotions. — [TradersUnion (unverified)](https://tradersunion.com/news/brokers-news/show/2241303-binance-revises-trading-fee-structure/)
- Binance.US (a separate entity) launched 0% maker / 0.02% taker spot fees. — [Binance.US blog](https://blog.binance.us/zero-fee-trading/)
- **Coinbase Advanced (conflicting sources; the official page returned HTTP 403):**
  - Secondary sources say Coinbase restructured its Advanced tiers on about 16 Sept 2026. They report US entry fees of 0.50% maker / 0.90% taker, falling to 0.35% / 0.75% above $10k of 30-day volume and 0% maker at $50M. They report EU/UK entry at 0.25% / 0.50%, and Brazil, India and others at 0.09% / 0.10%. — [Altrady (Oct 2026)](https://www.altrady.com/blog/crypto-trading-platforms/coinbase-advanced-trade); [Diario Bitcoin](https://www.diariobitcoin.com/exchanges/coinbase-reduce-sus-comisiones-de-advanced-y-baja-el-umbral-inicial-a-usd-10-000/)
  - The older, widely cited schedule is under $10k: 0.60% taker / 0.40% maker, down to 0.05% taker / 0.00% maker at $250M+. — [Financer review](https://financer.com/review/coinbase)
  - Coinbase's own page headline says "We've lowered spot fees on Coinbase Advanced", and that "lower rates kick in sooner". That sits oddly with a reported 0.90% US entry taker fee. **[CONFLICT: verify on Coinbase's official fee page before quoting]** — [Coinbase Advanced markets page](https://www.coinbase.com/en-gb/advanced-markets)
  - Tiers are based on trailing 30-day USD volume and update hourly. — [Analytics Insight](https://www.analyticsinsight.net/amp/story/cryptocurrency-analytics-insight/coinbase-offers-fee-upgrade-for-high-volume-traders)

### Inferences
- **Per-trade cost for a taker [standard decomposition]:** cost ≈ s/2 + f_taker·P + impact, where s is the quoted spread. Round trip: s + 2·f_taker·P (+ impact). For Binance Regular: 2 × 10 bps = 20 bps.
- A mid-price direction classifier must therefore predict moves larger than about (s + 2f)/P to be net-profitable as a taker. Short-horizon labels (k = 10–100 events) on liquid books usually move only a tick or so, often far less than 20 bps. That is why "accuracy" does not equal "profit".
- The breakeven hit rate for symmetric ±m moves with round-trip cost c is p\* = (m + c)/(2m). For example, m = 5 bps and c = 20 bps gives p\* = 2.5, which is impossible, so the strategy cannot work as a taker at that cost level.
- FI-2010 (Nasdaq Nordic equities) and the Optiver datasets have different fee and tick regimes from crypto. The crypto numbers illustrate the order of magnitude only.

### Gaps
- I could not fetch Coinbase's official fee page (HTTP 403), so current Coinbase tiers remain unverified and the sources conflict.
- No authoritative source was retrieved for typical BTC-USD quoted spreads on these venues.

---

## B2. Adverse selection, fill probability, queue position; conservative fill assumptions

### Takeaway
Passive (maker) orders earn the spread but get filled disproportionately when the price is about to move against them, which is adverse selection. Whether and when they fill depends on queue position. Naive backtests that assume a limit order fills whenever the price touches its level overstate both fill rates and P&L. Queue models (Cont–de Larrard, Huang–Lehalle–Rosenbaum, Moallemi–Yuan) and conservative "trade-through" rules are the standard remedies.

### Cited Findings
- **Moallemi & Yuan** (Columbia working paper, 2016) value limit orders by queue position in large-tick stocks. They decompose value into static and dynamic (option-like) parts and find queue value can be the same order of magnitude as the bid–ask spread. A later paper describes it as about half a spread. — [Moallemi & Yuan PDF](https://drupalgsb-test.paas.cc.columbia.edu/sites/default/files-efs/pubfiles/25461/queue-value-2016.pdf); [Guo, Ruan & Zhu, arXiv 1505.04810](https://arxiv.org/pdf/1505.04810)
- **Cont & de Larrard** model level-1 bid/ask queues with Poisson limit orders, market orders and cancellations, a Markovian queueing system. They obtain closed-form expressions for quantities such as the time to the next price change and the probability of an up move, plus a diffusion limit. SIAM J. Financial Math., DOI 10.1137/110856605. — [arXiv 1104.4596](https://www.arxiv.org/abs/1104.4596); follow-up [arXiv 1202.6412](https://www.arxiv.org/abs/1202.6412)
- **Huang, Lehalle & Rosenbaum, the queue-reactive model:**
  - Between changes of the reference (mid) price, the LOB is treated as a Markov queueing system whose order-flow intensities depend on the current queue state.
  - It reproduces stylized facts and is used as a simulator for placement tactics and passive-order execution probabilities.
  - It suits large-tick assets.
  - Published in JASA (2015).
  - A 2026 follow-up by Noble, Rosenbaum & Souilmi simplifies the state to imbalance and spread.

  — [arXiv 1312.0563](https://arxiv.org/abs/1312.0563); [arXiv 2202.12137 (survey/extension)](https://arxiv.org/pdf/2202.12137)

### Inferences
- **Adverse selection intuition (Glosten–Milgrom logic):** a resting bid is hit exactly when a seller arrives, and sellers are disproportionately informed when the price is about to fall. So E[mid_{t+Δ} − fill price | filled] < spread/2, and it can be negative. A backtest that credits a maker with the full half-spread overstates P&L.
- **Fill-model hierarchy, from most to least optimistic:**
  1. Touch-fill: fills when the price touches the level. Badly optimistic.
  2. Trade-through: fills only if a trade prints strictly through your price, guaranteeing that the queue at your level was exhausted. Conservative; it undercounts fills but rarely over-credits.
  3. Queue-position simulation: join at the back of the queue. Volume ahead of you decreases with trades at your level and with a fraction of cancellations (cancellations assumed uniformly or proportionally distributed in the queue). You fill once your queue-ahead reaches 0.
- Open-source HF backtesters such as `hftbacktest` implement probabilistic queue models with latency. **[UNVERIFIED, not fetched: github.com/nkaz001/hftbacktest]**
- **Data limitation:** with aggregated snapshots (FI-2010 10-level, or Optiver 1-second book snapshots), individual trade prints and the queue-ahead evolution are not observed. Queue-position modeling is therefore largely infeasible, and trade-through or taker-only assumptions are the defensible choice.

### Gaps
- The journal version of Moallemi & Yuan was not located; only the working paper was.
- `hftbacktest` was not verified this session.

---

## B3. Market impact: Almgren–Chriss, square-root law, temporary vs permanent impact

### Takeaway
Your own trading moves prices. Temporary impact (paid while trading, decaying afterwards) and permanent impact (a lasting information-driven shift) scale with order size relative to liquidity. Almgren–Chriss trades off impact cost against timing risk. Empirically, metaorder impact grows like the square root of size rather than linearly, so strategy P&L does not scale linearly with capital.

### Cited Findings
- Almgren & Chriss, "Optimal execution of portfolio transactions," *Journal of Risk* 3(2):5–39 (Winter 2000/2001), DOI 10.21314/JOR.2001.041. They minimize a combination of volatility risk and transaction costs from permanent and temporary impact. With linear impact the efficient frontier of liquidation strategies is explicit, and they introduce "liquidity-adjusted VaR". — [Risk.net](https://www.risk.net/journal-risk/2161150/optimal-execution-portfolio-transactions); [DOI](https://www.doi.org/10.21314/JOR.2001.041); [Wikipedia: Almgren–Chriss model](https://en.wikipedia.org/wiki/Almgren%E2%80%93Chriss_model)
- Tóth, Lempérière, Deremble, de Lataillade, Kockelkoren & Bouchaud, "Anomalous Price Impact and the Critical Nature of Liquidity in Financial Markets," *Physical Review X* 1, 021006 (2011):
  - The average price change over a metaorder of size Q follows the "square-root" law, which contradicts Kyle's linear impact.
  - Their explanation is that latent liquidity vanishes near the current price (a V-shaped supply/demand profile), so markets sit near a critical point.

  — [arXiv 1105.1694](https://www.arxiv.org/abs/1105.1694v3); [DOAJ/PRX](https://doaj.org/article/aa335a8de8954ff386b6bb094d176225)

### Inferences
- **Almgren–Chriss setup [FORMULA FROM MEMORY]:**
  - Unaffected price dynamics: S_k = S_{k−1} + σ·√τ·ξ_k − τ·g(n_k/τ), where g(v) = γ·v is permanent impact.
  - Execution price: S̃_k = S_{k−1} − h(n_k/τ), where h(v) = ε·sgn(v) + η·v is temporary impact.
  - Objective: minimize E[cost] + λ·Var[cost].
  - Optimal holdings: x_j = X·sinh(κ(T − t_j)) / sinh(κT), with κ² ≈ λσ²/η̃.
  - λ → 0 gives a straight-line TWAP-like schedule; a large λ front-loads trading.
- **Square-root law [FORMULA FROM MEMORY]:** ΔP/P ≈ Y·σ_daily·√(Q/V_daily), where Y is an O(1) constant (often quoted as roughly 0.5–1, **[UNVERIFIED]**). Example: trading 1% of daily volume with 2% daily vol gives an impact of about Y × 0.2%.
- For a single-snapshot LOB predictor, small orders sit well inside the best-level depth, so impact ≈ spread + fees. For larger notional, consuming multiple levels costs Σ (level price − mid)·q_level, which can be computed directly from the 10-level book. That is a cheap, honest "walk-the-book" cost model.
- Replay backtests ignore permanent impact and the reaction of other participants entirely (see B5).

### Gaps
- I did not fetch Bouchaud's broader reviews (e.g. *Trades, Quotes and Prices*, 2018) for the Y constant or a decay kernel; the impact-decay (propagator) model is not covered here.

---

## B4. Latency: latency arbitrage, cost of latency, why 1-second snapshots cannot resolve sub-second effects

### Takeaway
In continuous limit order books, speed advantages of milliseconds or less earn mechanical arbitrage rents (Budish–Cramton–Shim), and the cost of latency for execution is economically significant (Moallemi–Saglam). A model trained on 1-second snapshots cannot see what happens within the second: which side traded first, quote flickers, queue changes. It also implicitly assumes its order would act at the snapshot price with zero latency. Any edge that disappears with a realistic delay of one or more snapshots is not real.

### Cited Findings
- Budish, Cramton & Shim, "The High-Frequency Trading Arms Race: Frequent Batch Auctions as a Market Design Response," *QJE* 130(4):1547–1621 (2015), DOI 10.1093/qje/qjv027:
  - Using millisecond data, cross-market correlations (e.g. between related instruments) break down at high-frequency horizons, creating mechanical arbitrage opportunities that go to the fastest trader.
  - The resulting arms race is socially wasteful and raises spreads.
  - They propose frequent batch auctions (e.g. every 0.1 s to 1 s).

  — [NBER conference PDF](https://conference.nber.org/conf_papers/f69665.pdf); [UMD publication page](https://econ.umd.edu/publication/high-frequency-trading-arms-race-frequent-batch-auctions-market-design-response-0)
- The often-quoted ES–SPY arbitrage duration statistics (median falling from about 97 ms in 2005 to about 7 ms in 2011) are **[UNVERIFIED: from memory of BCS, not checked in the fetched text]**. — [NBER conference PDF](https://conference.nber.org/conf_papers/f69665.pdf)
- Moallemi & Sağlam, "OR Forum—The Cost of Latency in High-Frequency Trading," *Operations Research* 61(5):1070–1086 (2013), DOI 10.1287/opre.2013.1165:
  - A dynamic-programming model gives a closed-form latency cost in terms of asset parameters.
  - On NYSE stocks in 1995–2005, median latency cost roughly tripled.
  - Going from human to low-latency time scales saves an amount comparable to other execution costs of efficient institutional investors.

  — [Columbia Business School page](https://business.columbia.edu/faculty/research/cost-latency-high-frequency-trading); [Moallemi latency PDF](https://drupalgsb-test.paas.cc.columbia.edu/sites/default/files-efs/pubfiles/25476/Moallemi_latency.pdf); [INFORMS](https://pubsonline.informs.org/doi/suppl/10.1287/opre.2013.1165)
- ABIDES allows per-pair configurable latency between agents and the exchange. — [ABIDES, arXiv 1904.12066](https://arxiv.org/pdf/1904.12066)

### Inferences
- **Sampling argument:** with Δ = 1 s snapshots, any phenomenon on time scale ≪ Δ is aliased or invisible. Examples are the order of events within a second, quote fading and latency-arbitrage races. The model also cannot tell whether a signal at t was actionable before competitors reacted.
- Optiver's Realized Volatility dataset provides book snapshots indexed by `seconds_in_bucket` within 10-minute buckets, i.e. about 1-second resolution. **[UNVERIFIED in this session]** FI-2010 is event-based: snapshots after every 10 events, with no wall-clock timestamps in the normalized files. **[UNVERIFIED]**
- **Latency stress-test:** shift the execution to snapshot t + L for L = 1, 2, 5 and recompute net P&L. Edges that are monotonically and steeply eroded by L are speed-dependent and probably not capturable without colocation.
- The project's C++ AVX2 inference speed of about 271 ns per prediction (from the repo's commit message) is far below network and exchange latencies, which are typically microseconds colocated and milliseconds retail. **[UNVERIFIED typical latency figures]** End-to-end latency, not model inference time, would dominate in practice.

### Gaps
- I did not retrieve BCS's specific arbitrage-duration numbers from the paper text.
- No authoritative source was fetched for typical retail-to-exchange crypto API latencies.

---

## B5. Market-replay vs agent-based simulators (ABIDES and extensions; JAX-LOB); why replay ignores own impact

### Takeaway
Market replay re-plays historical messages or snapshots. It is realistic for the market as it was, but it is non-reactive: other participants do not respond to your orders, so only direct (mechanical) impact is captured and permanent, reactive impact is missed. Agent-based simulators such as ABIDES model the order flow as reactive agents, which captures feedback at the cost of calibration or realism risk. Hybrids replay real flow plus reactive background agents. JAX-LOB moves LOB simulation to GPU for large-scale RL.

### Cited Findings
- **ABIDES (Byrd, Hybinette, Balch; arXiv 2019; SIGSIM-PADS 2020):**
  - An open-source agent-based interactive discrete event simulator.
  - It supports tens of thousands of agents interacting with an exchange agent.
  - Its messaging is modeled after NASDAQ ITCH/OUCH, and latency between each pair of agents is configurable.

  — [arXiv 1904.12066](https://arxiv.org/pdf/1904.12066); [NSF PAR record](https://par.nsf.gov/biblio/10185795-abides-towards-high-fidelity-multi-agent-market-simulation)
- **JAX-LOB (Frey et al., 2023):**
  - A GPU-accelerated LOB simulator for large-scale RL. It was designed for LOBSTER data but can process messages from any source, including strategic agents.
  - The paper notes that replayed historical order flow is unchanged by the agent, so only direct market impact is accounted for, whereas strategic agents' order flow responds to intervention.
  - It cites hybrids, e.g. Karpe et al. (DDQN in ABIDES, replaying real messages plus reactive heuristic agents) and Fang et al. (PPO with replay in ABIDES).

  — [arXiv 2308.13289](https://arxiv.org/pdf/2308.13289)
- ABIDES-Gym (Amrouni et al. 2021), an OpenAI-Gym wrapper for RL on ABIDES, is **[UNVERIFIED, not fetched]**. — [arXiv 2110.14771](https://arxiv.org/abs/2110.14771)
- Realism metrics for simulated LOBs are **[UNVERIFIED, not fetched]**. — [Vyetrenko et al. 2020, "Get Real: Realism Metrics for Robust Limit Order Book Market Simulations", arXiv 1912.04941](https://arxiv.org/abs/1912.04941)

### Inferences
- **Why replay ignores own impact:** if your simulated market order consumes 2 BTC at the ask, the next replayed snapshot still shows the historical ask queue, as if you had never traded. Your liquidity removal is not persisted, and nobody re-prices in response. Replay also cannot capture other traders reacting to your quotes, such as being picked off or front-run. Replay P&L is therefore an upper bound that is roughly valid only for orders much smaller than the top-of-book depth.
- **Snapshot replay (FI-2010, Optiver) is weaker still than message replay (LOBSTER, ITCH).** Without individual messages, even mechanical queue depletion cannot be tracked.
- **Spectrum for "what more could be done":**
  1. Conservative snapshot replay: taker-only, walk-the-book, latency shift, fees. The minimum.
  2. Message-level replay with queue-position models (B2).
  3. Hybrid replay plus reactive agents (ABIDES-style).
  4. A calibrated queue-reactive simulator (B2) or a fully agent-based one, with realism checks (stylized facts: heavy tails, volatility clustering, spread and depth distributions).

### Gaps
- I did not verify the ABIDES extensions (ABIDES-Gym, ABIDES-Markets) or recent 2024–2026 simulators (e.g. newer JAX-LOB versions or generative LOB models) in this session.
