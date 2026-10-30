# New Research Directions for the LOB Prediction Project

Scope: promising next steps for a student LOB project that already has OFI / multi-level OFI+PCA / VPIN / micro-price / depth-imbalance features; naive, logistic, RF, LightGBM, LSTM, DeepLOB and TLOB models; a replay execution simulator; a C++ random-tree latency benchmark; and synthetic, Coinbase BTC 1-s 15-level, FI-2010, Optiver RVP and Optiver Close datasets. Labels are currently 3-class smoothed mid-price moves at 10-100 events.

How sources were checked. Items marked **[verified]** were confirmed in this session through search results or by fetching the page. Items marked **[bg]** are classic or well-known references. Their URLs and arXiv IDs come from background knowledge and were not re-fetched here, so spot-check them before citing formally.

---

## 1. Better targets (regression / multi-horizon, ordinal, cost-aware, triple-barrier, TLOB label)

### Takeaway
Changing the target is the cheapest and highest-value next step. TLOB's own paper shows that labels with a cost-aware threshold make the task much harder, and that profitability is untested. Two benchmarks (LOBCAST, Briola et al.) show that high F1 does not turn into tradable signals. The project already has a replay simulator, so it is well placed to close that gap.

### Cited Findings
- **Kolm, Turiel & Westray, "Deep order flow imbalance: Extracting alpha at multiple horizons from the limit order book."** Mathematical Finance 33(4):1044-1081, 2023 (SSRN 2021). The paper forecasts a *vector* of returns over several horizons for 115 Nasdaq stocks. Models trained on order flow (OFI) significantly beat models trained on raw order-book states. The effective forecast horizon is about two average price changes. **[verified]** — [SSRN](https://papers.ssrn.com/abstract=3900141); [IDEAS/RePEc journal record](https://ideas.repec.org/a/bla/mathfi/v33y2023i4p1044-1081.html)
- **TLOB label definition (Berti & Kasneci 2025, arXiv 2502.15757).** The future average is w₊(t,h,k) = mean of p(t+h−i) for i=0..k. The past average is w₋(t,h,k) = mean of p(t−i) for i=0..k. The change is l = (w₊−w₋)/w₋. The label is UP if l>θ, DOWN if l<−θ, and STABLE otherwise. The key change from the Tsantekidis/DeepLOB/FI-2010 smoothing is that the smoothing window k is *decoupled* from the horizon h. The paper argues that "a key drawback is that the window length k coincides with the prediction horizon h", which under-smooths short horizons and over-smooths long ones. θ is set to the mean absolute percentage change, chosen for class balance. **[verified]** — [arXiv HTML v3](https://arxiv.org/html/2502.15757v3); [code](https://github.com/LeonardoBerti00/TLOB)
- TLOB with a **spread-based threshold** (θ = average spread as % of mid) was tested only on Tesla at h=50/100/200. F1 fell to 41.39 / 36.48 / 30.82. Intel was excluded because 99.99% of its labels would be "stable". The paper does not test profitability and lists it as future work. **[verified]** — [arXiv HTML v3](https://arxiv.org/html/2502.15757v3)
- TLOB reports F1 on BTC (Binance BTCUSDT perpetual, 250 ms sampling, Jan 2023) of 74.7 / 61.74 / 48.54 / 41.49 at h=10/20/50/100. F1 falls steeply with horizon. It also reports that predictability on Intel at h=50 dropped from 66.87 (2012) to 60.19 (2015). That comparison uses only one day per period. **[verified]** — [arXiv HTML v3](https://arxiv.org/html/2502.15757v3)
- **Briola, Bartolucci & Aste, "Deep limit order book forecasting: a microstructural guide."** Quantitative Finance 25(7), 2025 (arXiv 2403.09267). The study covers 15 NASDAQ stocks. It finds that "high forecasting power does not necessarily yield actionable trading signals", that standard ML metrics are inadequate, and proposes measuring the probability of correctly forecasting *complete transactions*. The authors release the LOBFrame codebase. **[verified]** — [arXiv](https://arxiv.org/abs/2403.09267v1); [PMC full text](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12315853/)
- **Prata et al., LOBCAST benchmark** (Artificial Intelligence Review 2024, arXiv 2308.01915). The authors re-run 15 deep LOB models. These models do well on FI-2010 but "performance drops significantly" on unseen LOBSTER data. The released framework includes profit analysis through trading simulation. **[verified]** — [arXiv](https://arxiv.org/abs/2308.01915v1); [Springer](https://link.springer.com/article/10.1007/s10462-024-10715-4)
- **Triple-barrier labelling and meta-labelling.** López de Prado, *Advances in Financial Machine Learning* (Wiley 2018), ch. 3. Each event gets an upper (profit-take) barrier, a lower (stop-loss) barrier and a vertical (time-out) barrier. The label is whichever barrier is hit first. Meta-labelling trains a second model to decide whether to act on a primary signal. **[bg]** — [Wiley book page](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
- **Ordinal classification.** UP/FLAT/DOWN is ordered. Frank & Hall (ECML 2001), "A simple approach to ordinal classification", splits it into K−1 binary "y > k" problems. CORAL (Cao, Mirjalili & Raschka 2020, arXiv 1901.07884) gives rank-consistent ordinal heads for neural nets. **[bg]** — [CORAL arXiv](https://arxiv.org/abs/1901.07884)
- **Multi-horizon seq2seq for LOBs.** Zhang & Zohren (2021, arXiv 2105.10430), "Multi-horizon forecasting for limit order books: novel deep learning approaches and hardware acceleration using intelligent processing units", uses encoder-decoder and attention to predict several horizons together. **[bg]** — [arXiv](https://arxiv.org/abs/2105.10430)

### Inferences (experiment design, difficulty, risks)
- **What it is.** Instead of asking "did the smoothed mid go up by an arbitrary θ?", ask "how much will the price move?" (regression, multi-horizon), "which way, respecting order?" (ordinal), or "would a trade here have made money after costs?" (tradable or triple-barrier).
- **Experiment A (multi-horizon regression, after Kolm et al.).**
  - Targets: log mid-returns at h ∈ {1, 2, 5, 10, 20, 50, 100} events, all predicted at once.
  - Models: a multi-output LightGBM (one model per horizon) and an LSTM/TLOB with a 7-dim linear head and MSE loss. Compare OFI-input against raw-LOB-input versions; this replicates Kolm's headline result on Coinbase BTC.
  - Metrics: out-of-sample R² per horizon, IC (Spearman), and the "R² vs horizon" curve. Kolm et al. found the curve peaks at about 2 price changes.
  - Difficulty: low. These are mostly label-pipeline changes. Compute is CPU-feasible.
- **Experiment B (cost-aware / tradable labels).**
  - UP if the future best bid at t+h minus the current best ask exceeds the fee, i.e. a long entered by crossing the spread is profitable. DOWN is symmetric. FLAT otherwise. This is stricter than TLOB's "θ=spread" test and matches the project's spread-crossing simulator.
  - Report class balance per dataset. Expect "FLAT" to dominate on Coinbase BTC at 1 s, as TLOB found for Intel.
  - Evaluate with simulator PnL per trade and hit rate on the trades taken, not F1.
- **Experiment C (triple barrier).**
  - Barriers at ±k·(rolling σ of mid returns) or ±(spread + fees), with a vertical barrier at h events.
  - Add meta-labelling: the primary model is the current LightGBM direction model, and a secondary binary model predicts "is this signal profitable?" using features plus the primary model's probability.
  - This connects naturally to direction 5 (abstention).
- **Experiment D (TLOB label ablation).**
  - Grid over k ∈ {1, 5, 10, h} at fixed h, and show that F1 and simulator PnL move differently.
  - This is a clean, publishable-in-a-blog ablation, because TLOB itself did not test profitability.
- **Experiment E (ordinal).** Replace softmax cross-entropy with CORAL or the Frank & Hall decomposition in the LSTM and LightGBM models. Measure macro-F1 and the "UP predicted as DOWN" error rate, which is the costly error.
- **Risks.**
  - Tradable labels are extremely imbalanced. Use class weights or focal loss and report precision on the minority classes.
  - Regression R² at these horizons will be tiny (often < 1-5%). That is normal, but it can look like failure to a reader.
  - Triple-barrier labels overlap in time, which creates label concurrency. Use purged / embargoed CV (also López de Prado).

### Gaps
- I did not fetch the Kolm et al. full text, so exact network architecture and R² magnitudes are not reported here.
- I could not verify any paper that evaluates TLOB-style labels with a full execution simulator. This looks like an open gap the project could fill.

---

## 2. Point processes and queue models (Hawkes, queue-reactive)

### Takeaway
Hawkes processes and the queue-reactive (QR) model give the project two things. They supply *interpretable intensity features* (e.g. the current excitation level of market-buy arrivals) for the classifiers. More importantly, they provide a **generative order-flow model** that can make the replay simulator reactive (the market responds to the agent's orders), instead of a pure historical replay.

### Cited Findings
- **Bacry, Mastromatteo & Muzy (2015), "Hawkes processes in finance."** Market Microstructure and Liquidity 1(1). This is the standard review of self- and mutually-exciting point processes for order arrivals, price changes and market impact. **[bg]** — [arXiv 1502.04592](https://arxiv.org/abs/1502.04592)
- **Huang, Lehalle & Rosenbaum (2015), "Simulating and analyzing order book data: The queue-reactive model."** JASA 110(509). Arrival rates of limit orders, cancellations and market orders at each queue depend on the current queue sizes, which are estimated from data. The authors use it as a market simulator, for example to evaluate execution tactics. **[bg]** — [arXiv 1312.0563](https://arxiv.org/abs/1312.0563)
- JAX-LOB (Frey, Nagy et al. 2023) is a GPU limit-order-book simulator that processes thousands of books in parallel. It is a candidate engine to drive with Hawkes/QR-generated flow. **[verified]** — [arXiv 2308.13289](https://arxiv.org/abs/2308.13289)

### Inferences (experiment design, difficulty, risks)
- **What it is.** A Hawkes process says that each event raises the short-term probability of more events of the same or other types, with the effect decaying exponentially. Order flow clusters this way. The QR model says that the probability of the next order or cancel at a level depends on how long that queue currently is.
- **Experiment A (Hawkes features).**
  - Fit a multivariate exponential-kernel Hawkes model to event types {market buy, market sell, best-bid add/cancel, best-ask add/cancel}. Use the `tick` library (X-DataInitiative) or a hand-rolled MLE.
  - Coinbase 1-s snapshots are not event-level, so use event-level data from direction 10 (tardis/LOBSTER), or approximate events from snapshot diffs.
  - Features: the current conditional intensity λ_i(t) for each type, and the buy-minus-sell intensity imbalance. Add them to LightGBM and run a SHAP ablation (direction 7).
- **Experiment B (QR-driven simulator).**
  - Estimate QR intensities, i.e. rate tables indexed by queue size bucket, from event-level data. Use them to simulate *counterfactual* fills for passive orders. This replaces the "trade-through" heuristic with a model where the agent's own order sits in a queue that depletes stochastically.
  - Validate stylized facts: spread distribution, queue-size distribution, signature plot.
- **Experiment C.** Use the Hawkes branching ratio over rolling windows as a regime / endogeneity indicator, feeding direction 6.
- **Difficulty.** Medium. The Hawkes MLE is standard. The QR model needs event-level (L3 or L2-incremental) data and careful bookkeeping. CPU compute is enough.
- **Risks.**
  - On 1-s snapshot data the event types are not observed, which biases the fitted kernels.
  - Hawkes models with exponential kernels misfit the power-law memory of real flow.
  - A QR simulator does not model informed traders, so strategy PnL can still be optimistic.

### Gaps
- I did not verify current maintenance status of the `tick` library or alternatives (e.g. `hawkeslib`).
- I did not fetch a 2023-2026 paper using Hawkes intensities as features for deep LOB classifiers. Treat novelty claims cautiously.

---

## 3. Foundation / large models for LOB and financial time series

### Takeaway
LOB-specific transformers and generative models are very active (2025-2026): TLOB, LiT, LOBERT, LOBS5, TRADES and LOB-Bench. General time-series foundation models (Chronos, TimesFM, Moirai) have mixed-to-poor zero-shot evidence on financial returns. The project's best angle is (a) adding a generative model as a *simulator* or data augmenter and evaluating it with LOB-Bench-style metrics, and (b) a small, honest zero-shot TSFM baseline.

### Cited Findings
- **TLOB** (Berti & Kasneci, arXiv 2502.15757, Feb 2025, rev. May 2025) is a dual-attention (spatial + temporal) transformer. It claims average SOTA gains of 3.7 F1 on FI-2010, 1.3 on Tesla, 7.7 on Intel and 1.1 on BTC. A simple **MLPLOB** baseline from the same paper is roughly equal on FI-2010 (e.g. 81.64 vs 81.55 at h=10). **[verified]** — [arXiv](https://arxiv.org/abs/2502.15757); [GitHub](https://github.com/LeonardoBerti00/TLOB)
- **LiT: limit order book transformer** (Xiao, Ventre, Wang, Li, Huan, Liu). Frontiers in AI, 13 Oct 2025, DOI 10.3389/frai.2025.1616485. It uses structured patches with self-attention followed by LSTM layers and no CNN. The authors report that removing the CNN does not hurt performance. It exists; I found no arXiv version. **[verified]** — [Frontiers](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/abstract); [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12555381/)
- **LOBERT: Generative AI Foundation Model for Limit Order Book Messages** (Linna, Baltakys, Iosifidis, Kanniainen). arXiv 2511.12563, Nov 2025, v2 Sep 2026, NeurIPS 2025 GenAI in Finance workshop. It is an encoder-only BERT adaptation that treats each whole multi-dimensional message as one token, with continuous price, volume and time. Downstream tasks are mid-price movement and next-message prediction. The abstract claims leading performance with shorter context. I found no code link on the abstract page. **[verified]** — [arXiv](https://arxiv.org/abs/2511.12563)
- **Nagy et al. (2023), "Generative AI for End-to-End Limit Order Book Modelling: A Token-Level Autoregressive Generative Model of Message Flow Using a Deep State Space Network" (LOBS5).** It is trained on LOBSTER NASDAQ data with an LLM-style tokenizer for messages. Generated messages are applied to a JAX-LOB book. Out-of-sample perplexity is low, and mid-price returns from generated flow correlate significantly with real ones. The authors propose it as a world model for RL. **[verified]** — [arXiv 2309.00638](https://arxiv.org/abs/2309.00638v1)
- **LOB-Bench (2025)** is a benchmark for generative LOB models, co-authored by Nagy. LOBS5 is state of the art on it. **[verified]** — [arXiv 2502.09172](https://arxiv.org/pdf/2502.09172)
- **TRADES** (Berti, Prenkaj, Velardi; arXiv 2502.07071, v3 Nov 2025) is a transformer-based denoising diffusion model that generates order flow conditioned on the past. It ships with **DeepMarket**, an open-source Python framework for deep-learning market simulation, and a synthetic LOB dataset. **[verified]** — [arXiv HTML](https://arxiv.org/html/2502.07071v3)
- **ByteGen** (arXiv 2508.02247) is a tokenizer-free generative model for order-book events in byte space. It was seen only in search results; I did not read it. — [arXiv](https://arxiv.org/pdf/2508.02247)
- **Evidence on time-series foundation models in finance.**
  - Rahimikia, Ni & Wang, "Re(Visiting) Time Series Foundation Models in Finance" (arXiv 2511.18578): zero-shot Chronos-large with 512 lags gives out-of-sample R² = −1.37% and directional accuracy just above 51% on excess returns. Long-short portfolios lose money. Models pretrained on financial data do better. **[verified via search snippet]** — [arXiv](https://arxiv.org/html/2511.18578v1)
  - Noguer i Alonso & Franklin (arXiv 2606.27100, 2026): Moirai-2.0 and TimesFM-2.5 rank best against from-scratch baselines on five large US stocks. Per a secondary summary, they evaluate forecast error only, not trading. **[partially verified]** — [arXiv](https://arxiv.org/pdf/2606.27100); [Emergent Mind summary](https://www.emergentmind.com/papers/2606.27100)
  - Chronos (Ansari et al. 2024, arXiv 2403.07815), TimesFM (Das et al. 2023/ICML 2024, arXiv 2310.10688) and Moirai (Woo et al. 2024, arXiv 2402.02592) are the original model papers. **[bg]** — [Chronos](https://arxiv.org/abs/2403.07815); [TimesFM](https://arxiv.org/abs/2310.10688); [Moirai](https://arxiv.org/abs/2402.02592)

### Inferences (experiment design, difficulty, risks)
- **What it is.** One family is bigger, pretrained or attention-based models that read LOB snapshots or messages. The other is generative models that *write* realistic order flow and can act as simulators.
- **Experiment A (TSFM zero-shot baseline).**
  - Feed Chronos-Bolt-small or TimesFM the last 512 mid-returns on Coinbase BTC. Map the forecast median at horizon h to UP/FLAT/DOWN with the project's θ.
  - Compare against the naive and LightGBM models. The expected result, consistent with Rahimikia et al., is near-random. That is still a valuable negative result.
  - Difficulty: low. Inference only. A Colab GPU is enough.
- **Experiment B (MLPLOB sanity check).** Implement TLOB's MLPLOB baseline. If an MLP matches TLOB on the project's data, that is a strong "architecture isn't the bottleneck" finding.
- **Experiment C (generative simulator).**
  - Train TRADES via DeepMarket, which is open source, on event-level data. Evaluate generated flow with LOB-Bench-style distributional metrics: spread, order-size and inter-arrival distributions, and the correlation of returns with real data.
  - Then test whether augmenting training data with synthetic sequences improves LightGBM/TLOB on real test days.
  - Difficulty: high. Needs a GPU for hours to days and event-level data.
- **Experiment D (LOBERT-style message tokens).** Feasible only with message-level data (LOBSTER or tardis incremental L2). Treat it as a stretch goal.
- **Risks.**
  - Generative models trained on a few days can memorise, and a high-perplexity mismatch is hard to diagnose.
  - Published LOB transformer gains are small (about 1 F1 on BTC) and may not survive the project's purged evaluation.
  - TSFMs are trained mostly on non-financial series, so zero-shot results are likely noise.

### Gaps
- Whether LOBERT code or weights are released is unconfirmed.
- I did not verify LOB-Bench's exact metric list or code URL. Check the arXiv PDF.

---

## 4. Reinforcement learning for execution and market making

### Takeaway
The replay simulator with spread crossing, book walking, fees and latency is already most of an RL environment. The classic Avellaneda-Stoikov (AS) and Guéant-Lehalle-Fernandez-Tapia (GLFT) closed-form quoting policies are strong, cheap baselines. A natural student project plugs the project's *direction forecast* into the AS/GLFT reservation price and compares that against a deep-RL agent.

### Cited Findings
- **Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book."** Quantitative Finance 8(3):217-224. The market maker quotes around a *reservation price* r = s − q·γ·σ²·(T−t), where q is inventory, with an optimal spread that depends on risk aversion γ and order-arrival intensity parameters (A, k). **[bg]** — [author PDF (NYU)](https://www.math.nyu.edu/~avellane/HighFrequencyTrading.pdf)
- **Guéant, Lehalle & Fernandez-Tapia (2013), "Dealing with the inventory risk: a solution to the market making problem."** Mathematics and Financial Economics. It gives closed-form asymptotic quotes for AS with inventory limits. **[bg]** — [arXiv 1105.3115](https://arxiv.org/abs/1105.3115)
- **Spooner, Fearnley, Savani & Koukorinis (2018), "Market making via reinforcement learning."** AAMAS 2018. It uses TD learning with tile coding on a LOB simulator built from historical data, with an asymmetrically dampened PnL reward. **[bg]** — [arXiv 1804.04216](https://arxiv.org/abs/1804.04216)
- **Nevmyvaka, Feng & Kearns (2006), "Reinforcement learning for optimized trade execution."** ICML 2006. It uses Q-learning on a NASDAQ (INET) book with state = (time left, inventory left, market variables) and reports large cost reductions versus submit-and-leave. **[bg]** — [ICML / author PDF](https://www.cis.upenn.edu/~mkearns/papers/rlexec.pdf)
- **Ning, Lin & Jaimungal, "Double deep Q-learning for optimal execution."** arXiv 1812.06600 (Applied Mathematical Finance 2021). **[bg]** — [arXiv](https://arxiv.org/abs/1812.06600)
- **JAX-LOB** is a GPU-accelerated LOB simulator explicitly aimed at "large scale reinforcement learning for trading". **[verified]** — [arXiv 2308.13289](https://arxiv.org/abs/2308.13289)
- **ABIDES-Gym** (Amrouni et al., ICAIF 2021) provides OpenAI-Gym wrappers around the ABIDES agent-based market simulator, with execution and market-making environments. **[bg]** — [arXiv 2110.14771](https://arxiv.org/abs/2110.14771)
- Nagy et al. suggest LOBS5 as a world model for high-frequency RL. **[verified]** — [arXiv 2309.00638](https://arxiv.org/abs/2309.00638v1)

### Inferences (experiment design, difficulty, risks)
- **What it is.** Instead of predicting direction and then applying a fixed rule, learn a *policy*: where to quote, or how to slice a parent order, to maximise PnL minus inventory risk.
- **Experiment A (forecast-skewed market making).**
  - Implement AS/GLFT in the replay simulator on Coinbase BTC. Calibrate σ from rolling returns and the A, k intensity parameters from fill rates versus distance-from-mid.
  - Variant: shift the reservation price by α·E[Δmid], where E[Δmid] comes from the project's LightGBM regression output (direction 1).
  - Metrics: PnL, Sharpe, inventory variance, fill rate and adverse selection (mid move after fill).
  - Difficulty: low-medium. CPU is enough.
- **Experiment B (RL execution).**
  - Liquidate Q BTC within T seconds. State = (time left, inventory left, spread, OFI, model probabilities). Actions = {cross now, post at best, post 1 tick deeper, wait}.
  - Train DQN or PPO (stable-baselines3) on the replay simulator. Baselines: TWAP and immediate execution.
  - Difficulty: medium. Hours of CPU/GPU.
- **Experiment C.** Port the environment to JAX-LOB or ABIDES-Gym for comparison. Medium-high effort.
- **Risks.**
  - **Replay simulators have no market impact.** The book does not react to the agent, so RL will learn to exploit replay artefacts. Mitigate with small order sizes, conservative queue-position fills, and the QR model from direction 2.
  - RL is sample-hungry and unstable. Always report AS/TWAP baselines.

### Gaps
- I did not verify current maintenance status of ABIDES-Gym or JAX-LOB repos.
- I did not search for 2024-2026 deep-RL market-making surveys. Recent references for this section are thin.

---

## 5. Uncertainty and abstention (conformal prediction, calibration, trade only when confident)

### Takeaway
Since most high-frequency predictions are near coin-flips, the most practical gain often comes from *selective prediction*: trade only when calibrated confidence is high. Adaptive conformal inference handles the distribution shift that standard conformal assumes away.

### Cited Findings
- **Gibbs & Candès (2021), "Adaptive conformal inference under distribution shift."** NeurIPS 2021. ACI updates the miscoverage level online: α_{t+1} = α_t + γ(α − err_t). This gives long-run coverage guarantees without exchangeability. **[bg]** — [arXiv 2106.00170](https://arxiv.org/abs/2106.00170)
- **Zaffran et al. (2022), "Adaptive conformal predictions for time series" (AgACI).** ICML 2022. It aggregates ACI over several γ values. **[bg]** — [arXiv 2202.07282](https://arxiv.org/abs/2202.07282)
- **Angelopoulos, Bates, Malik & Jordan (2020), "Uncertainty sets for image classifiers using conformal prediction" (RAPS).** This is a conformal prediction-set method for classification. **[bg]** — [arXiv 2009.14193](https://arxiv.org/abs/2009.14193)
- Meta-labelling in López de Prado (2018) is the finance-native version of "a second model decides whether to act". **[bg]** — [Wiley](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
- Briola et al. find high forecasting accuracy does not imply actionable signals. This motivates evaluating only the trades actually taken. **[verified]** — [arXiv 2403.09267](https://arxiv.org/abs/2403.09267v1)

### Inferences (experiment design, difficulty, risks)
- **What it is.** Each prediction is wrapped in a calibrated uncertainty statement, such as a prediction set {UP} versus {UP, FLAT, DOWN}. The strategy acts only when the set is a single non-FLAT class.
- **Experiment A (calibration).** Report reliability diagrams and ECE for LightGBM, DeepLOB and TLOB. Apply temperature scaling (NNs) or isotonic regression (GBDT) on a time-ordered calibration split.
- **Experiment B (conformal sets with ACI).**
  - Nonconformity score = 1 − p̂(true class), or RAPS. Run ACI online over the test period with γ ∈ {0.005, 0.01, 0.05}.
  - Trade only when the set is a singleton UP or DOWN.
  - Plot coverage over time, the fraction of time traded, and simulator PnL per trade versus the always-trade baseline.
- **Experiment C (accuracy-coverage curve).**
  - Rank test points by max probability or margin, then plot accuracy and PnL/trade against the % of points traded.
  - This is cheap and highly interpretable. It often shows that the top 5-10% most confident predictions carry most of the edge, but that has to be checked on the project's data.
- **Difficulty.** Low. Pure post-processing, CPU only.
- **Risks.**
  - Coverage guarantees are marginal, not conditional. Coverage can fail exactly in volatile regimes.
  - Abstaining reduces the trade count, so PnL estimates get noisy. Report confidence intervals and bootstrap by day.

### Gaps
- I did not search for a 2023-2026 paper applying ACI specifically to LOB direction classification. Novelty is unverified.

---

## 6. Regime detection and non-stationarity

### Takeaway
LOB relationships drift. LOBCAST shows models collapse on new data, and TLOB reports predictability declining over years. Cheap remedies are rolling or online retraining and regime-conditioned models. Simple regime features from an HMM or change-point detector can also be fed to the GBDT.

### Cited Findings
- LOBCAST: the best FI-2010 models' "performance drops significantly" on unseen LOBSTER data. **[verified]** — [arXiv 2308.01915](https://arxiv.org/abs/2308.01915v1)
- TLOB: Intel h=50 F1 fell from 66.87 (2012) to 60.19 (2015), using a single day per period. **[verified]** — [arXiv HTML](https://arxiv.org/html/2502.15757v3)
- **Optiver Trading at the Close.**
  - Kaggle competition, Sep-Dec 2023, on Nasdaq closing-auction data. **[verified]** — [Optiver blog](https://optiver.com/kaggle-and-optiver-predicting-nasdaqs-closing-cross-auction-movements/)
  - A GitHub project re-implements "top solutions ... with ML engineering best practices extension" under the name "retrain-monitoring". **[verified, existence only]** — [GitHub](https://github.com/GrigoriiTarasov/Optiver-Trading-at-the-Close-retrain-monitoring)
  - **I could not retrieve the 1st-place write-up itself**, so the claim that the winner used online learning / periodic retraining during the forecasting phase is *unverified* in this session. It is commonly reported, but should be checked on the Kaggle discussion tab.
- Change-point detection: Adams & MacKay (2007), "Bayesian online changepoint detection", arXiv 0710.3742. The `ruptures` Python library implements offline change-point methods (Truong, Oudre & Vayatis 2020, Signal Processing). **[bg]** — [BOCPD arXiv](https://arxiv.org/abs/0710.3742); [ruptures docs](https://centre-borelli.github.io/ruptures-docs/)

### Inferences (experiment design, difficulty, risks)
- **What it is.** Markets switch between states such as calm/volatile or trending/mean-reverting. A model trained on one state can fail in another.
- **Experiment A (retraining schedule).** On Coinbase BTC (~12 days) and Optiver Close, compare:
  - a static model;
  - an expanding-window model retrained daily;
  - a sliding-window model retrained daily;
  - LightGBM warm-start (`init_model`) online updates every N minutes.

  Report per-day metrics as a time series, not one pooled number.
- **Experiment B (HMM regimes).**
  - Fit a 2-3 state Gaussian HMM (`hmmlearn`) on rolling realised vol, spread and OFI variance, fitted on train only, with forward filtering for the test period.
  - Two options: (i) add the filtered state probabilities as features, or (ii) train one model per regime.
- **Experiment C (change-point triggered retrain).** Run BOCPD on the model's rolling log-loss and retrain when a change-point is detected. Compare compute used and accuracy against a fixed schedule.
- **Experiment D (domain adaptation across assets).** Train on BTC and test on ETH/ADA (same Kaggle dataset, direction 8). Try per-asset feature normalisation (z-scored by rolling stats, spread in ticks) and report the transfer gap.
- **Difficulty.** Low-medium, CPU.
- **Risks.**
  - HMM look-ahead: smoothing (forward-backward) leaks the future. Use filtering only.
  - Twelve days of data give few regime changes, so conclusions are fragile.

### Gaps
- The Optiver Close 1st-place method details are not verified (see above).

---

## 7. Cross-asset and cross-impact features

### Takeaway
Cont, Cucuringu & Zhang (2023) find that lagged cross-asset OFI improves *future* return forecasts, though the effect is short-lived. The project's Kaggle crypto dataset reportedly also contains ETH and ADA, so a BTC↔ETH lead-lag study is a nearly free experiment.

### Cited Findings
- **Cont, Cucuringu & Zhang (2023), "Cross-impact of order flow imbalance in equity markets."** Quantitative Finance 23(10):1373-1393, arXiv 2112.13213.
  - (i) Integrated OFI (PCA over levels) explains price impact better than best-level OFI.
  - (ii) For *contemporaneous* impact, cross-impact terms add little once multi-level OFI is used (shown by LASSO sparsity).
  - (iii) *Lagged* cross-asset OFIs "do improve the forecast of future returns, both in terms of R² and economic incentives", mainly at short horizons, decaying rapidly.

  **[verified]** — [arXiv](https://arxiv.org/abs/2112.13213v4); [IDEAS](https://ideas.repec.org/a/taf/quantf/v23y2023i10p1373-1393.html)
- The Kaggle dataset "High Frequency Crypto Limit Order Book Data" by martinsn has about 12 days of LOB data for BTC, ETH and ADA, with the 15 best bid/ask levels (seen via a third-party mirror). **[verified via mirror; sampling frequency not confirmed]** — [Baselight mirror of Kaggle dataset](https://baselight.app/u/kaggle/dataset/martinsn_high_frequency_crypto_limit_order_book_data)

### Inferences (experiment design, difficulty, risks)
- **What it is.** Order flow in one asset (e.g. BTC) may predict short-term moves in a related asset (ETH), because traders and arbitrage bots link them.
- **Experiment A (lead-lag).** Align BTC and ETH on timestamps.
  - Compute the cross-correlation of 1-s mid returns at lags −10..+10 s.
  - Compute the Hayashi-Yoshida estimator if timestamps are asynchronous.
  - Run Granger tests for OFI_BTC(t−k) → r_ETH(t).
- **Experiment B (cross-OFI features, after Cont et al.).**
  - For the ETH target, add lagged BTC integrated-OFI (the project already has multi-level OFI+PCA) at lags 1, 2, 5, 10 s.
  - Use LASSO and LightGBM. Compare out-of-sample R² / F1 and simulator PnL with and without the cross features.
  - Reproduce the "contemporaneous: no gain; lagged: small gain, fast decay" pattern.
- **Experiment C.** Run the same test on Optiver RVP / Close, which have many stocks: compute a sector/market-wide OFI aggregate as a feature.
- **Difficulty.** Low. The data is already present (to be confirmed), CPU only.
- **Risks.**
  - Timestamp misalignment of even 1 s can create fake lead-lag.
  - Within-exchange crypto pairs may be arbitraged within milliseconds, so 1-s data may show only contemporaneous correlation.

### Gaps
- I did not confirm on Kaggle itself that the ETH/ADA files share timestamps and 1-s sampling with the BTC file.

---

## 8. Interpretability (SHAP for GBDT, attention analysis)

### Takeaway
TreeSHAP gives exact, fast feature attributions for LightGBM and is the standard tool. Attention weights in TLOB/DeepLOB are a weaker form of explanation and should be checked with perturbation tests.

### Cited Findings
- **Lundberg et al. (2020), "From local explanations to global understanding with explainable AI for trees."** Nature Machine Intelligence. Introduces TreeSHAP (exact Shapley values for tree ensembles in polynomial time) and SHAP interaction values. **[bg]** — [arXiv 1905.04610](https://arxiv.org/abs/1905.04610)
- **Jain & Wallace (2019), "Attention is not Explanation"**, NAACL. Rebutted in part by Wiegreffe & Pinter (2019), "Attention is not not Explanation". Together they show attention weights are not reliable importance scores without further tests. **[bg]** — [arXiv 1902.10186](https://arxiv.org/abs/1902.10186); [arXiv 1908.04626](https://arxiv.org/abs/1908.04626)
- TLOB's dual attention runs over features (spatial) and time (temporal), so attention maps can be extracted for both axes. **[verified, architecture only]** — [arXiv](https://arxiv.org/abs/2502.15757)

### Inferences (experiment design, difficulty, risks)
- **Experiment A (TreeSHAP).** Use `shap.TreeExplainer` (or LightGBM's `pred_contrib=True`) on the test set. Produce:
  - global importance;
  - dependence plots for OFI, micro-price deviation and VPIN;
  - how importance changes by horizon (h=10 vs 100) and by regime (direction 6).
  - Hypothesis to test: OFI dominates short horizons, and depth imbalance or VPIN matter more at long horizons.
- **Experiment B (attention vs perturbation).** For TLOB, extract temporal attention averaged over the test set. Compare it with occlusion importance, i.e. zeroing or permuting one time step or level and measuring the change in logit. Report rank correlation between the two.
- **Experiment C.** Use SHAP to prune features, then re-run the C++ latency benchmark with the smaller model (direction 9).
- **Difficulty.** Low, CPU.
- **Risks.** Correlated features (OFI levels, PCA components) split credit. Group features before interpreting.

### Gaps
- I did not search for 2023-2026 LOB-specific interpretability papers.

---

## 9. Fast inference: benchmarking the actual trained model

### Takeaway
The project's C++ benchmark uses *random* trees. Replace them with the trained LightGBM model compiled via lleaves (LLVM) or Treelite→TL2cgen (C source), and compare against native LightGBM and ONNX Runtime at batch size 1. Published numbers suggest compiled trees run about 2-5x faster than LightGBM at batch 1.

### Cited Findings
- **lleaves** (Simon Boehm) is an LLVM-based compiler for LightGBM and a drop-in `predict` replacement. Its vendor benchmark (NYC taxi, i7-4770, minimum of 20k runs) gives:

  | Batch size | LightGBM | Treelite | lleaves |
  |---|---|---|---|
  | 1 | 52.31 µs | 28.03 µs | 9.61 µs |
  | 100 | 441.15 µs | 94.14 µs | 31.88 µs |

  Compiled models can be cached with `compile(cache=...)`. **[verified]** — [lleaves article](https://siboehm.com/articles/21/lleaves); [GitHub](https://github.com/siboehm/lleaves); [PyPI](https://pypi.org/project/lleaves)
- **Treelite 4.0+ no longer generates C code**; that moved to **TL2cgen**. The workflow is: load the LightGBM model with Treelite, call `tl2cgen.generate_c_code(model, dirpath=...)` to get C99 .h/.c files, then `tl2cgen.export_lib()` or `create_shared()` for a .so/.dll, or `export_srcpkg()` for a source package with a Makefile/CMake. **[verified]** — [TL2cgen docs](https://tl2cgen.readthedocs.io/en/latest/index.html); [Treelite migration note](https://tl2cgen.readthedocs.io/en/latest/_sources/treelite-migration.rst.txt)
- ONNX path: LightGBM → ONNX via `onnxmltools.convert_lightgbm`, run with ONNX Runtime. Microsoft Hummingbird compiles trees to tensor operations for PyTorch/ONNX/TVM. **[bg]** — [ONNX Runtime](https://onnxruntime.ai/); [onnxmltools](https://github.com/onnx/onnxmltools); [Hummingbird](https://github.com/microsoft/hummingbird)
- DeepLOB on GPU: export PyTorch → ONNX → TensorRT (FP16/INT8). Zhang & Zohren (2021) benchmarked LOB models on Graphcore IPUs. **[bg]** — [TensorRT docs](https://docs.nvidia.com/deeplearning/tensorrt/); [arXiv 2105.10430](https://arxiv.org/abs/2105.10430)

### Inferences (experiment design, difficulty, risks)
- **Experiment design.**
  1. Save the trained LightGBM model (`model.txt`).
  2. Build four backends:
     - (a) LightGBM C API `LGBM_BoosterPredictForMatSingleRowFast`;
     - (b) TL2cgen-generated C compiled with `-O3 -march=native` and linked into the existing C++ harness;
     - (c) a lleaves-compiled .so called from C, or through Python with overhead measured separately;
     - (d) ONNX Runtime C++ API with one intra-op thread.
  3. Measure p50/p99/p99.9 latency at batch 1 over 1e5-1e6 real feature rows from the test set (not random inputs, since branch prediction depends on the data). Pin a core, warm up, and use `rdtsc` or `std::chrono::steady_clock`.
  4. Check numerical equality of predictions across backends (max abs diff).
  5. Compare against the existing hand-written AVX2 random-tree result (271 ns), using matched tree count and depth.
  6. Also time **feature computation** (OFI, micro-price from a book update), since in practice it can dominate model latency.
- **DeepLOB/TLOB.** ONNX Runtime CPU against TensorRT GPU at batch 1. Report that GPU batch-1 latency includes PCIe transfer.
- **FPGA.** Mention it only as context. Tree inference maps well to FPGAs (e.g. via hls4ml / conifer for BDTs from the HEP community), but this is out of scope for a student project. **[bg]** — [conifer GitHub](https://github.com/thesps/conifer)
- **Difficulty.** Low-medium. Mostly build-system work on Windows/MSVC or WSL.
- **Risks.**
  - Vendor benchmarks are from old hardware.
  - lleaves supports only LightGBM, and some objectives or categorical handling may differ.
  - On Windows, LLVM and shared-library toolchains can be fiddly; WSL is safer.

### Gaps
- I did not verify current ONNX Runtime / onnxmltools LightGBM converter support for multiclass models, though it is long-standing.
- I did not verify conifer's current status.

---

## 10. Data: other free LOB datasets and event-level data

### Takeaway
Event-level data is the biggest enabler for directions 2, 3 and 4. tardis.dev reportedly gives free first-of-month crypto data, which needs re-confirming. LOBSTER sells NASDAQ message/book data and offers sample files. Coinbase/Binance public WebSocket feeds let the project record its own L2 data for free.

### Cited Findings
- **tardis.dev.** A ~2022 Deribit blog post states that tick-level data for the "first day of each month" is available via the API without an API key. **This could not be confirmed against current tardis docs in this session.** **[partially verified, dated]** — [Deribit Insights](https://insights.deribit.com/?p=7828); [tardis docs](https://docs.tardis.dev/api)
  - Tardis docs describe preserving individual exchange events and book changes (incremental L2). Their data types include `incremental_book_L2`, `book_snapshot_25`, `trades` and `quotes` (type names are [bg]). — [Tardis market data primer](https://docs.tardis.dev/market-data-primer); [CSV docs](https://docs.tardis.dev/downloadable-csv-files)
- **LOBSTER** (lobsterdata.com) provides NASDAQ message + orderbook files reconstructed from ITCH. Access is account-based (the R package `lobsteR` requires login). Free sample files for a few tickers on one day (e.g. AAPL, AMZN, GOOG, INTC, MSFT, 2012-06-21) are listed on the "Data Samples" page per [bg]. The page is a JS app, so I could not fetch it. **[partially verified]** — [lobsteR CRAN](https://cran.r-project.org/web/packages/lobsteR/refman/lobsteR.html); [LOBSTER samples page](https://lobsterdata.com/info/DataSamples.php)
- **Hyperliquid L4 order-book data paper** ("An Open Book: Level 4 Order Book Data from the Hyperliquid Exchange") was found on SSRN. It suggests on-chain DEX order books provide full order-level data. **[verified existence only]** — [SSRN](https://papers.ssrn.com/abstract=6465720)
- TRADES/DeepMarket releases a synthetic LOB dataset. **[verified]** — [arXiv](https://arxiv.org/html/2502.07071v3)
- Message-level deep learning: Zhang & Zohren, "Deep learning for market by order data" (2021). **[bg]** — [arXiv 2102.08811](https://arxiv.org/abs/2102.08811)
- Binance public data dumps (data.binance.vision) provide free trades/aggTrades/klines. Spot L2 book snapshots/diffs are *not* in the public dumps, per [bg], so they would need to be recorded live via WebSocket. Coinbase Advanced Trade WebSocket `level2` channel is free. **[bg, unverified]** — [Binance public data GitHub](https://github.com/binance/binance-public-data)

### Inferences (experiment design, difficulty, risks)
- **Experiment A (record your own).** Run a small asyncio recorder for the Coinbase `level2` + `matches` channels (BTC-USD, ETH-USD) for 1-2 weeks, storing to Parquet. This gives true event-level data with trades, which enables Hawkes fits, QR estimation, LOBERT-style tokens and realistic queue simulation. Difficulty: medium, mostly reliability and gap handling.
- **Experiment B (tardis free day).** Download one free day of Binance-futures `incremental_book_L2` + `trades` and rebuild the book. This is directly comparable to TLOB's BTC setup (Binance BTCUSDT.P).
- **Experiment C (LOBSTER sample).** Run the full pipeline on a LOBSTER sample day to test cross-market generalisation, the LOBCAST-style robustness check.
- **Risks.**
  - Self-recorded feeds have sequence gaps, so the snapshots must be resynced.
  - One free day per month is too little for training, but fine for testing.
  - LOBSTER licence terms restrict redistribution.

### Gaps
- Current tardis free-data policy, exact LOBSTER sample tickers/dates, and Binance public L2 availability were not confirmed against live pages in this session.

---

## Suggested priority (inference, not sourced)
1. **Directions 1 + 5 together.** Use cost-aware / triple-barrier / multi-horizon targets, evaluated with the existing simulator plus confidence-based abstention. This is cheapest and directly answers the "F1 ≠ profit" critique from Briola et al., LOBCAST and TLOB's own admission.
2. **Direction 9.** Benchmark the *trained* LightGBM through TL2cgen and lleaves in the existing C++ harness. Low effort, and it fixes a known weakness of the project.
3. **Directions 7 + 8.** Add BTC→ETH cross-OFI and SHAP. Low effort, using data the project already has.
4. **Direction 6.** Per-day evaluation plus rolling retraining.
5. **Direction 4A.** AS/GLFT market making skewed by forecasts.
6. **Stretch goals:** event-level data (10A), then Hawkes/QR (2), then generative models (3C) and RL execution (4B).
