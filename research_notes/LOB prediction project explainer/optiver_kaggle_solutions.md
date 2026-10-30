# Optiver Kaggle Competitions (2021 Realized Volatility, 2023-24 Trading at the Close): Tasks, Top Solutions, and Lessons for LOB Research

Method note: Kaggle pages are JS-rendered, so most facts below were pulled directly from Kaggle's own JSON backend (forum topic text, competition overview pages, final leaderboards) and then cited by their public Kaggle URL. Leaderboard numbers are copied from the final private leaderboards that Kaggle served on 2026-10-30. Public-LB scores now show as 0 on both leaderboards because Kaggle wiped them at the end, so any public-LB figures come from forum posts written during the competitions.

## Q1. Optiver Realized Volatility Prediction (2021): task, data, metric, scores, 1st place and other top solutions

### Takeaway
The task was to forecast the next 10 minutes of realized volatility from 10 minutes of 2-level book and trade data, scored by RMSPE. The naive "use the current RV" baseline scores about 0.34 RMSPE, and the winner scored 0.19548 private. Most of the gap between about 0.22 (an honest feature model) and about 0.195 (the top) came from exploiting the shuffled `time_id`s. Competitors recovered real prices from the tick size, reconstructed the time order, and pooled features from "neighbouring" time_ids, which can include future time_ids. The hosts called this "future information" and later tried to weaken it by adding noise to the forecasting-phase data, but they reverted the noise after community pushback.

### Cited Findings

**Competition setup**
- Dates: launched June 28, 2021. Entry and team-merger deadline Sept 20, 2021. Final submission deadline Sept 27, 2021. A forecasting phase with leaderboard updates "roughly every two weeks" followed, and the competition ended Jan 10, 2022. — [Timeline](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/overview/timeline)
- Prizes: $100k in total ($25k / $20k / $15k / $10k, then $5k each for 5th–10th). It was a code competition with notebooks limited to ≤9h CPU/GPU and internet off. — [Prizes / Code Requirements](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/overview)
- Participation (Kaggle competition metadata): 3,852 teams, 4,395 competitors, 21,295 joined users. The final private LB lists 3,809 ranked teams. — [Leaderboard](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/leaderboard)
- Data files:
  - `book_[train/test].parquet` (partitioned by `stock_id`) holds the top 2 levels: `bid_price1/2`, `ask_price1/2`, `bid_size1/2`, `ask_size1/2`, plus `seconds_in_bucket`.
  - `trade_[train/test].parquet` holds `price` (VWAP within the second), `size`, `order_count`.
  - Prices are **normalized**. "Time IDs are not necessarily sequential but are consistent across all stocks."
  - The hidden test set produces about 150,000 target values. During training most of the test set was filler. The private LB used only data collected after the deadline, so public and private had zero overlap.
  - The sample submission benchmark is the median train target.
  - Source: [Data page](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data)
- Host FAQ, part 1 (data structure):
  - There are 112 unique `stock_id`s, and they are the same real stocks throughout.
  - Each `time_id` is a 20-minute window. The first 10 minutes are given as features and the second 10 minutes are used to build the target, with no overlap.
  - "time_id is randomly shuffled, so it will not contain any information other than serving as a bridge between different dataset."
  - Source: [Competition data FAQ (host)](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/249752)
- Host FAQ, part 2 (granularity and gaps):
  - Book rows are the last snapshot within each second. Trade rows aggregate all trades within the second.
  - Missing seconds in the book can be forward-filled. Missing trade seconds mean no trade happened.
  - Source: [Competition data FAQ (host)](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/249752)
- Target: realized volatility over the 10-minute window after the feature data. The official tutorial computes WAP from level-1 book prices and sizes, takes log returns, and computes RV = sqrt(sum of squared log returns). — [Data page](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data); [Optiver tutorial notebook](https://www.kaggle.com/code/jiashenliu/introduction-to-financial-concepts-and-data)
- Metric: RMSPE = sqrt(mean(((y − ŷ)/y)²)). — [Evaluation](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/overview/evaluation)
  - A participant noted this equals RMSE weighted by 1/y². A weighted polynomial regression scored 0.23855, versus >0.29 without the weights. Using 1/y² sample weights became standard practice. — [Custom Evaluation thread](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/250324)
- **Naive baseline (verified):** the Optiver tutorial predicts the next-10-min RV with the current 10-min RV. It reports "R2 score: 0.628, RMSPE: 0.341" on train and "The naive submission scored a RMSPE 0.327 on public LB". — [Optiver tutorial notebook](https://www.kaggle.com/code/jiashenliu/introduction-to-financial-concepts-and-data)

**Final private leaderboard (verified from Kaggle LB JSON)**

| Rank | Team | Private RMSPE |
|---|---|---|
| 1 | nyanp | 0.19548 |
| 2 | YoonSoo | 0.19720 |
| 3 | Eduardo Peynetti | 0.19883 |
| 4 | [PKSHA]life is volatile | 0.19899 |
| 5 | jiebao | 0.19905 |
| 6 | Light | 0.19960 |
| 7 | Michael Poluektov | 0.20013 |
| 10 | keks | 0.20418 |

Medal cutoffs: gold up to rank 17 at 0.21299, silver up to rank 192 at 0.22016, bronze up to rank 385 at 0.22169. Field quantiles: 10th pct 0.22163, median 0.22756, 75th pct 0.25218. — [Leaderboard](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/leaderboard)

Public-LB scores during the training phase were lower than the private scores:
- nyanp's final ensemble had public 0.18324 and private 0.19545 (late-submission rerun). — [Ablation study](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/302626)
- A GNN team reported LB 0.18345 and PB 0.20139. — [GNN short gold solution](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/275185)

**1st place: nyanp, "Nearest Neighbors"**

Source for this subsection unless noted: [1st Place Solution – Nearest Neighbors](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970)

- Three key points as stated by the author:
  1. Time-series CV built on a reverse-engineered time_id order.
  2. Nearest-neighbor aggregation features, which took the score "from 0.21 to 0.19".
  3. A blend of LightGBM, an MLP, and the 1D-CNN from the MoA 2nd-place solution.
- **Price recovery from tick size:** the real price is recovered as 0.01 / (the smallest gap between distinct normalized prices). He credits "someone pointed out in the discussion" for the idea.
- **Time-order recovery:** the time_id × stock price matrix is compressed with 1-D **t-SNE** (perplexity 400), and the time_ids are sorted along that axis. The direction is fixed using a known stock ("id61 = AMZN").
  - Checked against yfinance, the training data spans **2020-01-01 to 2021-03-31**: 443 trading days and 3,833 time_ids, about 8.6 per day. His guess is 30-minute slots between about 10:30 and 15:00.
- **Validation:** a 4-fold time-series CV with 10% of the data held out for validation in each fold gave "good enough (though not perfect)" CV/LB correlation.
  - He did **not** use the recovered order directly as a test-set feature, "because there is no guarantee that t-SNE will be able to sort the time-id order correctly."
  - He used the order for adversarial validation, to detect features whose distribution shifts over time.
- **Nearest-neighbor (NN) features:**
  - The motivating observation: the next time_id's training window starts just after the current target window, so the RV of adjacent time_ids is a powerful proxy for the target.
  - Generalized version: find the N most similar time_ids (N = 2, 3, 5, 10, 20, 40) under several distance metrics and average features such as RV and size across them.
  - The same aggregation is done across similar stock_ids, and combined: e.g. "the average tau of 20 similar stocks with similar volatility in 5 closest time-ids."
  - About 600 features in total, 360 of them NN features.
  - His own caveat: "I don't think we can use future information in a real Optiver's scenario."
- **Feature processing:** `trade.order_count` and `book.total_volume` aggregates drifted over time, so they were converted to ranks within each time_id. `log1p` was applied to skewed features.
- **What didn't help:** beta-type domain features, TabNet (too slow), training the NN on the residual against current RV, and dimensionality-reduction features.

**1st place ablation (late submissions, by the winner)**

Source: [Ablation study](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/302626)

- A **single LightGBM** scored 0.19699 private, which alone would have ranked #1.
  - 3×CNN: 0.19644.
  - Full ensemble (5×LGBM + 3×CNN + 3×MLP): 0.19545.
- There were 7 neighbor types:
  - time_id neighbors by real price (Canberra and Mahalanobis distances), by volatility (Manhattan), and by trade size (Canberra and Mahalanobis);
  - stock_id neighbors by real price and by volatility.
- Single-LGBM private RMSPE by feature set:

  | Feature set | Private RMSPE | Rank it would get |
  |---|---|---|
  | No NN features (GroupKFold) | 0.22492 | #1358 |
  | Volatility + trade-size NN only, GroupKFold ("tick-size leak-free") | 0.20367 | #9 |
  | Volatility + trade-size + real-price NN, GroupKFold | 0.19734 | #1 |
  | Same, with time-series CV | 0.19699 | #1 |
  | Stock-id NN only | 0.22018 | — |

- Author's conclusions:
  - "feature diversity was more important than model diversity."
  - Rank normalization had almost no private-LB effect (0.19708 without it vs 0.19699 with it). His explanation: short-horizon volatility-NN features already track drift.

**Other top solutions**
- **4th place, "[PKSHA]life is volatile"** (posted as "Tentative 3rd Place Solution (6th in Public)"; the final LB shows them 4th at 0.19899). — [Writeup](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/278676)
  - time_id nearest-neighbor features found independently, plus the distance ratio between the 1st and k-th neighbor. No time-series CV, because they had not reverse-engineered the order.
  - **Target transformation:** predict target / RV(0–600s), i.e. the *ratio* of future to current volatility, for robustness to non-stationary volatility levels.
  - **"300-second model":** concatenate train + test, predict RV(300–600) from 0–300s, then apply the model to 300–600s data to predict 600–900. This is effectively self-supervised training on test-period data, used as a meta-feature.
  - **"Macro" model:** predicts the mean volatility across stocks per time_id.
  - Final model: LightGBM × ANN stacking.
- **7th place, Michael Poluektov** (0.20013 private):
  - He attributes his public placement mainly to "The Leak". Credit goes to a since-deleted post (topic 256725 is now "[Deleted Topic]"), and "such a solution could not actually be implemented in production."
  - Other ideas: fitting a linear trend (mean/slope/error) over 100-second sub-window features instead of passing them raw; dropping `stock_id` as a categorical (arguing it won't generalize to the future); stock aggregations over the N closest time_ids; BorutaShap feature selection; 50/50 LGBM + FFNN.
  - Source: [7th place main ideas](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/276506)
  - His repo describes time-order recovery as a graph problem: edge weights are the distances between one interval's end prices and the next interval's start prices, and the order comes from an approximate shortest Hamiltonian path. — [GitHub michaelpoluektov/orvp](https://github.com/michaelpoluektov/orvp)
- **Stas Sl ("1st public → 154 private")** — [Writeup](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/279170)
  - An RNN with **attention across all 112 stocks**, taking input of shape (N, 112, 600, F). Each higher-dimensional version (single stock → all stocks → all stocks + NN time_ids) added about 0.015 public LB (0.215 → 0.20 → 0.185).
  - The best version without nearest neighbors scored 0.198 public.
  - Models were tiny (about 60K parameters) to avoid overfitting, since there are only about 3,830 effective samples.
  - Per-stock feature normalization and LayerNorm over stocks × features both helped.
  - Diagnosed collapse: GE's 1-for-8 reverse split (July 30, 2021) changed stock 31's sizes in the forecasting data. Dividing stock 31's sizes by 8 in an offline test degraded his model 0.198 → 0.22, similar to his private drop.
- **GNN gold:** a graph over time_id distance with TransformerConv and 2–3 neighbors scored 0.18345 public and 0.20139 private. — [GNN post](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/275185)

**Common (non-leak) features, as documented in writeups and the tutorial**
- WAP, log returns, and realized volatility (tutorial).
- RV over 100-second sub-windows and trends across them (7th).
- Size, order-count, and volume aggregates (1st).
- Stock- and time_id-level aggregations or clustering (Stas Sl; 1st).
- Current RV as the dominant feature (1st).
- Sources: [tutorial](https://www.kaggle.com/code/jiashenliu/introduction-to-financial-concepts-and-data); [7th](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/276506); [1st](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970)
- Before the leak became widespread, a competitor judged that "in order to get below 0.20XX there is no real 'magic'" and that a single LightGBM with GroupKFold on time_id worked. — [RDizzl3 observations](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/263308)

### Inferences
- The best leak-free figure verified here is 0.20367 (nyanp's GroupKFold LGBM with volatility and trade-size neighbors). Volatility-similarity neighbors do not need price recovery, but they do pool information across time_ids, including future ones. Without any cross-time_id information the best he reports is 0.22018–0.22492, roughly bronze-to-median territory. So the realistic, deployable improvement over the 0.34 naive baseline is about 0.34 → 0.22. The extra 0.22 → 0.195 depended on cross-time_id pooling.
- For a student LOB project, the two transferable ideas are:
  - modelling the volatility *ratio* (target / current RV) instead of its level;
  - cross-sectional pooling over stocks at the **same** timestamp, which is legitimate.

### Gaps
- No writeups were found for the 2nd (YoonSoo, 0.19720), 3rd (Eduardo Peynetti, 0.19883), 5th (jiebao), or 6th (Light) private-LB teams. Searches of the competition forum returned nothing under those names.
- The original post revealing the tick-size price recovery (topic 256725) is deleted, so its author and date cannot be verified.
- No source was found for the claim that the hosts "declined to use a time-series API". What is verified is the noise episode in Q3. RVP used a code competition with test parquet files, while TATC explicitly used a time-series API.
- No Optiver blog post about either competition's results was checked.

## Q2. Optiver – Trading at the Close (2023–24): task, data, metric, scores, 1st place and other top solutions

### Takeaway
The task was to predict each stock's 60-second WAP move relative to a synthetic Nasdaq index during the 10-minute closing auction, scored by MAE in basis points. Score ranges were extremely compressed:
- predicting all zeros: about 5.465 on the public test period;
- winner: 5.4030 private (hyd);
- 10th place: 5.4379.

The winning recipe was:
- strict time-based holdout validation;
- about 300 features, including cross-sectional (same-timestamp) ratio and rank features;
- CatBoost + GRU + Transformer;
- **online retraining** on targets revealed during the forecasting phase;
- post-processing that forces the index-weighted mean prediction to zero, using reverse-engineered index weights.

### Cited Findings

**Competition setup**
- Dates: Sept 20, 2023 start. Entry and team-merge deadline Dec 13, 2023. Final submission Dec 20, 2023. Forecasting phase with updates about every two weeks. Planned end date March 20, 2024; the final scores were published about March 22, 2024. — [Timeline](https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview/timeline); [Final update post](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/485903)
- The first leaderboard update covered trading days through January 5, 2024. — [First LB update](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/468464)
- Participation: 20,639 registrations, 4,374 participants, 3,225 teams, 94 countries. Prizes were the same as RVP ($25k for 1st). — [Kaggle recap](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/486883); [Overview](https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview)
- Data fields:
  - Identifiers: `stock_id`, `date_id` (sequential), `seconds_in_bucket` (seconds since the start of the closing auction).
  - Auction fields: `imbalance_size` and `matched_size` (USD); `imbalance_buy_sell_flag` (+1/−1/0); `reference_price`, `far_price` (auction interest only), `near_price` (auction + continuous orders).
  - Continuous-book fields: `bid/ask_price`, `bid/ask_size` (USD notional), `wap`.
  - All prices are expressed relative to the stock's WAP at the start of the auction.
  - Source: [Data page](https://www.kaggle.com/competitions/optiver-trading-at-the-close/data)
- Target = (StockWAP_{t+60}/StockWAP_t − IndexWAP_{t+60}/IndexWAP_t) × 10,000 bps. The index is "a custom weighted index of Nasdaq-listed stocks constructed by Optiver." — [Data page](https://www.kaggle.com/competitions/optiver-trading-at-the-close/data)
- Submission rules:
  - Predictions go through a **time-series API** "which ensures that models do not peek forward in time."
  - `revealed_targets` delivers the previous day's full targets at `seconds_in_bucket == 0` of each new day, which is what enables lag-target features and online learning.
  - Source: [Evaluation](https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview/evaluation); [Data page](https://www.kaggle.com/competitions/optiver-trading-at-the-close/data); [revealed_targets mechanism](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/443275)
- Dimensions:
  - 200 stocks, 55 snapshots per day (every 10 s from 0 to 540 s), training date_ids 0–480 (481 days). — [Index weights post](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/442851); [revealed_targets mechanism](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/443275)
  - The 1st place used "first 400 days … last 81 days", which is consistent with 481 days. — [1st place](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/487446)
  - The training set has 17 columns and more than 5.2M rows. — [Dataset guide](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/445025)
  - Probing showed the public test period was 46 days (2,530 time steps, 506,000 rows). — [Obtain information by submitting](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/443195)
- Metric: MAE. — [Evaluation](https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview/evaluation)
- **Zero baseline (public LB only):**
  - "After submitting an all zeros submission, the score is **5.465**." — [Pims](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/443195)
  - Another participant cited "naïve baseline … 5.46 by predicting everything as 0" against a then-top public score of 5.3205 (Nov 2023). — [Jason Feng](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/452872)

**Final private leaderboard (verified from Kaggle LB JSON)**

| Rank | Team | Private MAE |
|---|---|---|
| 1 | hyd | 5.4030 |
| 2 | (team name shown as a URL) | 5.4041 |
| 3 | (⊙﹏⊙)Rib~ | 5.4172 |
| 4 | M | 5.4261 |
| 5 | Yusef + Ivan + Adam('s GPU) | 5.4270 |
| 6 | Daniel FG | 5.4285 |
| 7 | NimaShahbazi | 5.4300 |
| 8 | Stay hungry, Stay foolish | 5.4301 |
| 9 | ADAM. | 5.4352 |
| 10 | HAO | 5.4379 |

Medal cutoffs: gold up to rank 18 at 5.4476, silver up to rank 221 at 5.4723, bronze up to rank 443 at 5.4734. Quantiles: 10th pct 5.4732, 25th pct 5.4737, median 5.4955, 90th pct 5.6036. — [Leaderboard](https://www.kaggle.com/competitions/optiver-trading-at-the-close/leaderboard)

**1st place: hyd (solo)**

Source for this subsection: [1st place solution](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/487446)

- **Final blend:** CatBoost 0.5, GRU 0.3, Transformer 0.2 (weights searched on the validation set). CV 5.8117, private 5.4030. All three models share the same **300 features**.
- **Validation:** train on the first 400 days, hold out the last 81. "The CV score aligns with leaderboard score very well."
- **"Magic" features:**
  - Aggregates within `seconds_in_bucket_group` (<300 s, 300–480 s, >480 s): ratio of the first value in the group to the current value, and ratio of a rolling mean to the current value, computed per date/group/stock.
  - **Cross-sectional features per (date_id, seconds_in_bucket):** ratio of the cross-stock mean to the stock's value, and the stock's rank divided by the count.
- **Feature selection:** top 300 by CatBoost importance, needed to fit memory and time for online retraining.
- **Models:**
  - GRU: input (batch, 55 time steps, features) → 4 GRU layers → 55 outputs, i.e. one day of one stock as a sequence.
  - Transformer: input (batch, **200 stocks**, features) → 4 encoder layers → 200 outputs, i.e. attention *across stocks* at one timestamp. The output is forced to zero mean (`out - out.mean(1)`).
  - Sample weights were also used.
- **Online learning:** retrain every 12 days, 5 times in total. To survive memory limits he saved training data one file per day and loaded it day by day. His best run went over time, so it actually got 4 updates; he estimates about 5.400 without the overrun.
- **Ablation on the private test set (all with post-processing):**

  | Model | No online learning | One online update | Five online updates |
  |---|---|---|---|
  | CatBoost | 5.4523 | 5.4291 | 5.4165 |
  | Blend | 5.4438 | 5.4157 | 5.4030 |

  So online learning was worth about 0.03–0.04 MAE, larger than the gap between 1st and 10th place.
- **Post-processing:** subtract the *index-weighted* mean of predictions ("better than average-mean since metric already told").
- **What didn't help:** 1D-CNN or MLP in the ensemble, multi-day GRU input, larger transformers (DeBERTa), predicting the target bucket mean with GBDT.

**Index-weight discovery**
- Regressing the implied index return (stock return minus target) on the 200 stock returns recovers weights with R² ≈ 0.99999999. Rounded to 3 decimals they take values in {0.001, 0.002, 0.004, 0.006, 0.008, 0.02, 0.04} and sum to 1.0. Posted Sept 24, 2023, four days after launch. — [Weights of the Synthetic Index](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/442851)
- These weights power the post-processing used by 1st, 9th and the 5th-place team: the "weighted sum of target for all stocks should be zero." — [9th place](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/486868); [5th-public writeup](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/462743)

**Other top solutions**
- **5th place team** (the "[5th Public LB]" writeup by Yusef A., posted Dec 21, 2023, before the final; the private LB's 5th team is "Yusef + Ivan + Adam('s GPU)"). — [Writeup](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/462743)
  - 5× LightGBM with continuous updates plus 5× CatBoost on 216 features, with index-weighted sum-to-zero post-processing.
  - LightGBM was updated two ways: `lgb.train(init_model=…)` adds trees, and `booster.refit()` re-estimates leaf values with a decay rate k (new = k·old + (1−k)·new).
  - Feature groups: stock-specific, intraday rolling, market-wide (including a weighted index price), imbalance features (e.g. "market_urgency").
  - Many shallow models with `extra_trees=True` were more robust.
  - An MLP and a TCN were dropped as unstable.
- **6th, Daniel FG** (5.4285), deep learning only. — [Writeup](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/486040)
  - Seq2seq Transformers (3 models) and a GRU, with stock-wise attention layers and an encoder taking 2 days of history. Outputs have shape (batch, stocks, 55).
  - About 35 features; imputation with zeros and standard scaling.
  - Online incremental learning every day on the newly revealed data.
  - Time split of 359 train / 121 validation days, with an EMA of the validation metric used to smooth epoch noise. Online learning was validated on the last 20 days.
  - Outputs were constrained to sum to zero.
- **7th, Nima Shahbazi** (5.4300). — [Writeup](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/486169)
  - LightGBM + LSTM + ConvNet with minimal NN feature engineering.
  - Key features: "deviation from median" of raw features within the same (date_id, seconds_in_bucket), i.e. a cross-sectional demeaning.
  - Online learning for both the NNs and LGBM. "Deviation features and online learning helped to reduce the error by a big margin."
  - Simple time-based split.
- **9th, ADAM.** (5.4352). — [Writeup](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/486868); [code](https://github.com/ChunhanLi/9th-kaggle-optiver-trading-close)
  - XGBoost with 3 seeds and 157 features, an MAE objective, and 1.5× sample weight on the latest 45 days.
  - Retrained only twice online (at the start of the private period and 30 days later).
  - Features: imbalance (imb1/imb2), market urgency, diffs and shifts, rolling mean/std, MACD, size scaled by each stock's median, and a reconstructed 6-second-earlier target from historical WAP.
  - Group-wise forward feature selection.
  - Noted "a really stable correlation between local cv and lb."
- **14th (gold):** an XGBoost + LightGBM ensemble with 170 features, "with as much refitting as we could spare." — [Writeup](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/485985)

**How much online learning mattered (community analysis)**
- A participant plotted public vs. update scores. The cluster that contained all prize-winning teams was about 0.015 better than the main cluster after update 1, and a further ≈0.007 apart after update 2. The author attributes this mainly to online training. — [How significant is the impact of online learning?](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/473085)
- In this reading, the "magic" was online learning on date_id > 480, which only fits the 9-hour limit with fewer features and fast models. — [ARMADA](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/462642)
- A widely discussed piece of advice: hold out about 45 days (the private period length) as one fold. — [go, modeling ideas](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/450626)

### Inferences
- The signal is tiny relative to noise:
  - The winner improved on the zero baseline by about 1–1.2% MAE. This compares a public-period zero score of 5.465 with a private-period winner; the two periods differ, so it is only approximate.
  - The gap from 1st to the bronze cutoff is only about 0.07 MAE.
  - Many teams cluster at 5.4732–5.4737, which looks like a copied public notebook, though this is unverified.
  - For a student project, report improvements in relative terms against a zero or persistence baseline, and test significance (e.g. Diebold–Mariano tests or a block bootstrap on daily MAE).
- Two structural tricks generalize to any relative-to-index LOB target:
  - cross-sectional normalization at the same timestamp (rank, mean ratio, median deviation);
  - an output constraint (zero weighted mean). Both are leak-free because they use only same-timestamp information.

### Gaps
- No public writeups were found for private 2nd (5.4041), 3rd (5.4172), or 4th (5.4261). Forum searches for "2nd/3rd/4th place" returned only unrelated topics.
- The private-period zero-prediction score was not found. Only the public-period 5.465 is verified.
- An ESANN 2024 paper (ES2024-159) appears to study this dataset and the value of feeding back revealed targets, per a search snippet. The PDF could not be parsed, so its claims are unverified. — [ESANN 2024 PDF](https://www.esann.org/sites/default/files/proceedings/2024/ES2024-159.pdf)

## Q3. Cross-cutting lessons: leakage from shuffled IDs, public/private shakeups, time-based validation, online retraining

### Takeaway
"Shuffled" IDs do not hide time when prices and tick sizes leak the real price level. Leakage can be subtle: pooling across time_ids, or computing global statistics over train+test. Forecasting-phase leaderboards shake up badly when solutions are brittle to distribution shift (stock splits, missing stocks, noise). In the 2023 competition, a strict temporal holdout plus online retraining was what separated the winners.

### Cited Findings

**Leakage**
- **Shuffled IDs leak order (RVP):** the time order was recovered via tick size → real price → t-SNE or Hamiltonian path. It was validated against real prices (2020-01 to 2021-03) and used both for CV and for "future" neighbor features. — [1st place](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970); [7th place](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/276506); [orvp repo](https://github.com/michaelpoluektov/orvp)
- **Host response (RVP):**
  - After update 2, Kaggle added noise "mapping to an error on realized volatility on the order of 10^-6", "expected to degrade a few that relied strongly on a very precise expectation of stock tick sizes … to make the competition more fair for participants who did not attempt to utilize future information." — [Second LB update](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/282791)
  - Many former top-10 teams dropped to ranks 1,800–3,800 in that update (e.g. nyanp 2 → 3836). — [Shakeup #2 plot](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/283843)
  - Kaggle then reverted: "I made an error in judgment when I proceeded with adding noise … We will be reverting this change." — [Rerun without noise](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/284121)
- **Global-statistics leakage (TATC):** many public notebooks computed per-stock global statistics (median/std of sizes and prices) over the whole dataset. These should be fit on the training split only and saved with the model. — [Why there's leakage in most notebooks](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/463826)
- **Stock-level drift (RVP):** per-stock aggregates computed on test vs. train features gave the same CV but public LB 0.19664 vs 0.19906. Stock characteristics drift, so CV alone could not validate this choice. — [AmbrosM, "The puzzle's pieces"](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/273887)

**Shakeups and fragility**
- RVP's final public-vs-private scatter was "somewhat volatile." — [Shakeup dataset](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/300785)
- Kaggle noted public/private deltas were "not meaningful" because the test periods differed entirely. — [Final update](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/300023)
- Documented RVP collapses:
  - Stas Sl: public 1st → 154th private (1259th after the first update), linked to GE's 1-for-8 reverse split changing stock 31's sizes. — [Writeup](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/279170)
  - Fritz Cremer: 37th → 2786th (≈0.195 → ≈0.246) from a missing-stock handling bug that only triggered when stocks were absent. Without the bug he would have scored 0.21828. — [Writeup](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/300746)
  - SUNFADED rose from public 1398th to private 24th. — [Writeup](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/300348)
- TATC: the winner and the 9th-place team both report close CV/LB agreement under a temporal holdout. — [1st](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/487446); [9th](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/486868)

**Validation**
- RVP 1st: reverse-engineered time-series 4-fold CV gave "good enough (though not perfect)" correlation. — [1st](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970)
- RVP ablation: time-series CV gave a slightly better private score than GroupKFold (0.19699 vs 0.19734). — [Ablation](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/302626)
- TATC validation splits:
  - 1st: 400 train / 81 holdout days. — [1st](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/487446)
  - 6th: 359 / 121, plus a last-20-days check for online learning. — [6th](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/486040)
  - Community advice: one 45-day fold matching the private horizon. — [go](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/450626)

**Online retraining under non-stationarity (TATC)**
- 1st place: online learning cut CatBoost MAE from 5.4523 to 5.4165 and the blend from 5.4438 to 5.4030. — [1st](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/487446)
- 6th refit daily; 7th: "online learning helped … by a big margin"; 9th retrained twice; 14th refit as much as time allowed. — [6th](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/486040); [7th](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/486169); [9th](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/486868); [14th](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/485985)
- In RVP, where online learning was not possible, the 4th-place team transformed the target to a ratio over current RV specifically because volatility *levels* are non-stationary. — [4th](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/278676)

### Inferences
Lessons for a student LOB project:
1. **Audit IDs and any normalized fields for recoverable time order.** The tick-size trick was found by the community within weeks. Any dataset that is "shuffled" but contains price levels should be assumed orderable.
2. **Prefer purged, walk-forward time splits** whose holdout length matches the deployment horizon, and fit every scaler and aggregate on train only.
3. **Same-timestamp cross-sectional features are legitimate and strong.** Cross-time pooling is not, unless it only looks backward.
4. **Report against naive baselines:** current RV for volatility (RMSPE 0.341 on train) and zero for relative returns (≈5.465 public MAE).
5. **Online or refit evaluation is a first-class experimental variable.** Simulate it with an expanding-window refit every k days.
6. **Robustness to corporate actions and missing instruments** (splits, delistings) matters as much as model choice. Per-stock normalization and rank features help.
7. **Feature diversity beat model diversity** in RVP, and a single GBDT was enough to win. A carefully engineered GBDT baseline is a must-have comparator for any deep LOB model.

### Gaps
- The quantitative impact of the RVP noise episode on the final standings could not be isolated, because the noise was reverted and all scores were rerun.
- No academic paper that cleanly re-benchmarks either dataset was verified. The ESANN 2024 paper (TATC) and some theses exist, but their content was not checked.
