# Additional Experiments & 25-Stock Out-of-Universe Verification

**Diagnostic Anatomy, Heterogeneous Ensembling, Multi-Horizon / Cross-Market Upgrades, and Anti-Overfitting Proof Across `36.0M` Indian NSE Limit-Order-Book Snapshots**

> **Repository Artifacts Reproducing This Document:**
> - [src/experiments/nse_error_and_upgrade.py](../src/experiments/nse_error_and_upgrade.py) → [artifacts/nse/error_diagnostics/](../artifacts/nse/error_diagnostics/)
> - [src/experiments/nse_final_25stock_verification.py](../src/experiments/nse_final_25stock_verification.py) → [artifacts/nse/final_verification/](../artifacts/nse/final_verification/)
> - [src/experiments/nse_edge_probes.py](../src/experiments/nse_edge_probes.py) (`Probe 3A` SSF↔Cash lead-lag)

---

## 1. Executive Summary & Strict Evaluation Hygiene

After deploying the baseline **LOB Alpha Studio** web terminal, live visual inspection of held-out `k = 20 s` replays raised five critical scientific questions:

1. **Why do order-book classifiers look "wrong" ~45%–55% of the time on average** even when the model learned real structure?
2. **Why does the 100 s `TLOB` Transformer win on certain seconds while tabular `LightGBM` wins on others**—and why do all models simultaneously fail on sharp institutional sweeps?
3. **Why combine `LightGBM + TLOB` specifically** instead of triplets (`K=3`), quadruplets (`K=4`), or averaging all 7–8 models?
4. **Does averaging `10, 20, 30, ..., 100` predictions** (either across horizons or across time/batches) improve hit rate?
5. **Do our upgrades (`LightGBM v2`, Multi-Horizon Averaging, and Single-Stock Futures features) genuinely generalize across the Indian market**, or were they overfitted to `RELIANCE`, `NIFTY_FUT`, and `IDEA`?

Every experiment reported here preserves **strict zero-leakage hygiene**:
- **Day-boundary isolation:** Every return, lag, rolling mean, volatility measure, and futures basis demean resets cleanly at `09:15:00 IST` (never crossing overnight gaps).
- **Walk-forward causal splits:** Each 64-day instrument is divided into `4` contiguous `16`-day chronological segments (`11` Train / `2` Validation / `3` Test days per segment = `12` held-out test days, `~268,572` out-of-sample rows per instrument).
- **Leave-One-Segment-Out (LOSO) ensemble weights:** Any ensemble blend weight `w_s` applied to segment `s` is tuned strictly on the other segments `j != s`.
- **Zero-tuning out-of-universe lock:** Feature definitions (`38` `v2` features, `10 s + 20 s` averaging, `8` Single-Stock Futures features) and hyperparameters (`LightGBMLOB`: `num_leaves=63, max_depth=8, lr=0.05, n_estimators=250, early_stopping=20`) were **frozen** on `3` development benchmarks and evaluated unchanged on **`22` completely unseen NSE instruments (`5.91M` unseen test seconds)** and **`10` unseen Single-Stock Futures (SSF) equities**.

---

## 2. Anatomy of Average Prediction Errors

Analyzing all `805,714` held-out test seconds of `RELIANCE`, `NIFTY_FUT`, and `IDEA` (and extending error decomposition across all `25` instruments) shows three distinct sources of average error—and two of them are fixable or artifacts of the evaluation framing rather than poor learning.

### 2.1 Dead-Zone (`±α`) Boundary Noise Explains `62%–87%` of All Mistakes

The ternary target classifies whether smoothed 20-second forward mid-return `r_20s` exceeds `±α`. Most classification mistakes are **not** directional reversals (`UP` predicted while market went `DOWN`); instead they are **threshold boundary calls (`FLAT ↔ Move`)** right around `±α`:

| Instrument | Tick Size (bp) | Total `v1` Error Share | `±α` Flat↔Move Boundary Share | True `UP↔DOWN` Reversal Share | Median `\|r₂₀ₛ\|/α` (Both Right) | Median `\|r₂₀ₛ\|/α` (Both Wrong) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RELIANCE** | `0.797 bp` | `54.2%` | **`62.2%`** | `37.8%` | `2.65×` | `1.20×` |
| **NIFTY_FUT** | `0.043 bp` | `54.2%` | **`66.4%`** | `33.6%` | `3.09×` | `1.02×` |
| **IDEA** | `7.410 bp` | `44.6%` | **`87.5%`** | `12.5%` | `n/a` (discrete 1t) | `n/a` |
| **25-Stock Mean** | `0.035–7.41 bp` | — | **`63.4%`** (`55.8%–87.4%`) | `36.6%` | — | — |

On `RELIANCE`, when both `LightGBM` and `TLOB` err together, the true future return sits barely `1.20 × α` from zero (`26.0%` inside the narrow band `[0.5α, 1.5α]` vs only `10.9%` when both are right). In other words, the book is accurately reflecting tiny net pressure, and Brownian tick randomness tips the final return just above or below `α`.

### 2.2 Equilibrium Entropy & The Selective High-Conviction Curve

Over `~60%` of every 6.25-hour session occurs while the order book is balanced (`max class prob < 0.45`). Evaluating models at **100% forced coverage** forces coin-flip guesses on balanced noise. Abstaining during equilibrium and filtering by **Hybrid Ensemble (`v2 + TLOB`)** maximum class probability transforms realized precision ([selective_conviction_curve.csv](../artifacts/nse/error_diagnostics/selective_conviction_curve.csv)):

| Hybrid Confidence Cutoff (`max prob ≥ τ`) | RELIANCE Cov / Acc / F1 | NIFTY_FUT Cov / Acc / F1 | IDEA Cov / Acc / F1 |
| :--- | :---: | :---: | :---: |
| `τ = 0.34` (~Full Day) | `99.7%` · `50.2%` · `0.502` | `99.7%` · `47.8%` · `0.467` | `100.0%` · `61.3%` · `0.529` |
| `τ = 0.45` | `51.2%` · `58.6%` · `0.586` | `49.1%` · `54.8%` · `0.509` | `84.6%` · `63.7%` · `0.555` |
| `τ = 0.50` | `31.3%` · `64.8%` · `0.645` | `28.7%` · `59.5%` · `0.531` | `65.3%` · `65.9%` · `0.585` |
| `τ = 0.55` | `18.2%` · **`72.0%`** · `0.705` | `15.5%` · **`64.6%`** · `0.556` | `45.9%` · **`68.3%`** · `0.620` |
| `τ = 0.60` | `10.6%` · **`80.7%`** · `0.766` | `7.4%` · **`69.2%`** · `0.574` | `29.4%` · **`71.1%`** · `0.663` |
| **`τ = 0.65`** | `6.6%` · **`89.4%`** · `0.808` | `2.8%` · **`73.9%`** · `0.606` | `17.2%` · **`76.0%`** · `0.725` |

### 2.3 Hidden Regime Bug Discovered & Fixed: Late-Day Closing Stalls (`~15:15–15:30 IST`)

Calibration diagnostics uncovered a sharp drop in baseline `LightGBM v1` accuracy on late-day zero-volatility stalls (`pos_in_day >= 20,000` with `vol_10s_ticks == 0`). Because `v1` had **no volatility or session-clock features**, any resting bid queue wall triggered an overconfident `UP` call even when trades had frozen near the close:
- **RELIANCE late-day stall rows (`n = 14,165`):** `LightGBM v1` scored `53.4%` accuracy → **`LightGBM v2` (with volatility + `day_progress`) jumped to `84.8%` (`+31.4 pp`)**.
- **IDEA late-day stall rows (`n = 25,433`):** `LightGBM v1` scored `46.4%` accuracy → **`LightGBM v2` jumped to `73.5%` (`+27.1 pp`)**.

---

## 3. Forensic Case Study: Irreducible Market Shocks (`IDEA`, `27 Aug 2026, 09:28:31 IST`)

Visual inspection of live replays surfaced textbook "unanimous wrong" timestamps such as **`IDEA`, Segment 0, Replay Step 739 (`2026-08-27 09:28:31 IST`)**, where **all 6 models (`LightGBM`, `TLOB`, `DeepLOB5`, `LSTM`, `Random Forest`, `Logistic Regression`)** predicted `DOWN` (`51%–64%`), yet mid-price jumped **`+5.31 bp (UP)`** within 20 seconds.

Forensic reconstruction from the raw `IDEA` LOB, order-count book, total market depth, and millisecond trade tape (`t = 09:28:31`, raw index `315810`) shows **exact ground truth**:

| Offset | Best Bid (`₹15.07`) | Best Ask (`₹15.08`) | Order Counts (`Ask / Bid`) | Total Book Buy / Sell Depth | Tape Events / Trade Prints |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `t - 10s` | `419,189` shares | `161,569` shares | `20 / 33` | `25.20M / 55.07M` | Heavy bid support erodes |
| `t - 8s` | `53,549` shares | `349,117` shares | `24 / 17` | `24.96M / 55.33M` | Buyer sweeps `72,615` sh `@ ₹15.08` |
| **`t + 0s`** | **`34,467` shares** | **`382,983` shares** | `24 / 21` | `25.03M / 55.48M` | **Snapshot in screenshot: `Imb = -0.83`** |
| `t + 3s` | `148,878` shares | `348,508` shares | `26 / 22` | `25.21M / 55.50M` | Aggressive buy hits `112,537` sh `@ ₹15.08` |
| `t + 11s` | `131,288` shares | `310,459` shares | `29 / 22` | `25.24M / 55.62M` | Aggressive buy hits `165,857` sh `@ ₹15.08` |
| **`t + 19s`** | **`₹15.09 × 151,113`** | **`₹15.10 × 746,132`** | `45 / 5` | `25.99M / 54.65M` | **Single `481,514`-share sweep (`~₹72.7L`) clears `₹15.08` & `₹15.09`!** |

- **Why the prediction was rational at `t = 0`:** Passive cash demand (`Bid 1`) had collapsed by `92%` (`419k → 34k`) beneath an `11×` Ask wall, total sell depth doubled total buy depth (`55.5M` vs `25.0M`), and even `NIFTY_FUT` fell `-6.00` index points across `[t-15s, t+22s]`. Under observable cash-book state at `t = 0`, `DOWN` was Bayes-optimal.
- **Why this error is irreducible from cash LOB alone:** Institutional buyers executing **`759,908` shares (`~₹1.15 Crore`) in 18 seconds** deliberately avoid resting passive bids that would telegraph demand to HFT market makers; they wait off-book and fire aggressive marketable sweeps starting at `t + 3 s`. Until the first child fill arrives, **no trace exists in passive cash quotes**—representing the fundamental Bayes-error ceiling of single-asset passive LOB prediction.

---

## 4. Why Models Disagree & Why Cross-Paradigm Ensembles Win (`K = 1..8`)

### 4.1 Why `LightGBM` vs `TLOB` Disagree `~50%` of the Time

Breaking all held-out test seconds into four agreement buckets ([disagreement_summary.csv](../artifacts/nse/error_diagnostics/disagreement_summary.csv), [microstructure_buckets.csv](../artifacts/nse/error_diagnostics/microstructure_buckets.csv)):

| Instrument | Both Right | Only LightGBM | Only TLOB | Both Wrong | Oracle Union (`≥1` Right) | Agree Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RELIANCE** | `27.1%` | `18.7%` | `17.6%` | `36.6%` | **`63.4%`** | `52.1%` |
| **NIFTY_FUT** | `24.6%` | `21.2%` | `16.9%` | `37.2%` | **`62.8%`** | `51.4%` |
| **IDEA** | `30.7%` | `24.7%` | `18.6%` | `26.0%` | **`74.0%`** | `61.7%` |

- **When `TLOB` beats `LightGBM v1`:** On `RELIANCE`, `OnlyTLOB` has markedly **lower short-term volatility (`vol_10s = 0.271t` vs `0.331t`)** and smaller instant drift (`2.50t` vs `2.91t`). `Baseline LightGBM` only saw `5 s` of lags to predict a `20 s` move, whereas `TLOB` observed a `100 s` spatial-temporal sequence (`SEQ_LEN = 100`).
- **When `LightGBM` beats `TLOB`:** `OnlyLGBM` has more abrupt **L1 imbalance flips (`32.3%` reversal vs 20 s mean vs `29.6%` for `OnlyTLOB`)**. Trees react to a 1-second shock on the next tree split, whereas `TLOB` suffers `100 s` attention inertia—plus under NIFTY Seg 3's `+10.7%` rally, ratio/OFI tree features stayed shift-invariant.

### 4.2 Signal/Error Correlation & All Subset Sizes (`K = 1, 2, 3, 4, 7, 8`)

Measuring correlation with `LightGBM v2` on `RELIANCE` explains which pairings carry genuinely independent information:
- **Redundant family (`sig_r = 0.75–0.95`, `err_r = 0.36–0.80`):** `Logistic Regression`, `Random Forest`, `LightGBM v1` (pairing `LightGBM + RF` or `LightGBM + LogReg` yields `0.0000` gain across all 21 pairs).
- **Orthogonal families (`sig_r = 0.17–0.45`, `err_r = 0.23–0.31`):** `Naive Momentum` (pure price trend), `TLOB` (`100s` Dual-Attention Transformer), `DeepLOB5` (local 3D CNN), and `LSTM`.

Evaluating every subset of size `K` with strict Leave-One-Segment-Out (LOSO) convex weights ([ensemble_k1_to_k8_summary.csv](../artifacts/nse/final_verification/ensemble_k1_to_k8_summary.csv)):

| Benchmark | Pool | `K=1` Solo Best | `K=2` Best Pair | `K=3` Best Triplet | `K=4` Best Quadruplet | `K=7/8` Full Pool |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **RELIANCE** | Original-7 | `0.4542` (`LGBM_v1`) | `0.4752` (`+TLOB`) | `0.4828` (`+Mom+TLOB`) | **`0.4868`** (`+Mom+RF+TLOB`) | `0.4826` (**drops `-0.42 pp`**) |
| **RELIANCE** | Upgraded-8 | `0.4963` (`LGBM_v2`) | `0.5004` (`+TLOB`) | `0.5009` (`+Mom+TLOB`) | **`0.5010`** (`+Mom+CNN+TLOB`) | `0.4998` (**drops `-0.11 pp`**) |
| **NIFTY_FUT** | Original-7 | `0.4432` (`LGBM_v1`) | `0.4479` (`+Mom`) | `0.4532` (`+Mom+CNN`) | **`0.4542`** (`+Mom+CNN+TLOB`) | `0.4498` (**drops `-0.44 pp`**) |
| **NIFTY_FUT** | Upgraded-8 | `0.4676` (`LGBM_v2`) | `0.4682` (`+TLOB`) | `0.4690` (`+LR+CNN`) | **`0.4691`** (`+LR+TLOB`) | `0.4680` (**drops `-0.11 pp`**) |
| **IDEA** | Original-7 | `0.4851` (`LGBM_v1`) | `0.5144` (`+Mom`) | `0.5221` (`Mom+RF+TLOB`) | **`0.5274`** (`Mom+RF+L1+TLOB`) | `0.5246` (**drops `-0.27 pp`**) |
| **IDEA** | Upgraded-8 | `0.5278` (`LGBM_v2`) | `0.5363` (`+Mom`) | `0.5394` (`+Mom+L1`) | **`0.5410`** (`Mom+RF+CNN+L2`) | `0.5420` (`+L1`) |

**Key Ensemble Insights:**
1. **`LightGBM v1 + Naive Momentum` Diagnostic Discovery:** Before upgrading features, `NaiveMom + LGBM_v1` ranked Top-2 on all stocks and **`+2.92 pp` on `IDEA`**! Because zero-parameter Momentum uses *past price changes* while `LGBM v1` only used *queue & OFI features*, this directly motivated adding `5s–60s` signed price drift into **`LightGBM v2`**, lifting solo tree F1 from `0.4542 → 0.4963` (`+4.22 pp`) and making Momentum mostly redundant afterward.
2. **Heterogeneous Triplets/Quadruplets (`K=3–4`) vs Over-Ensembling (`K=7–8`):** Adding one representative from distinct architecture families (**Tree + Transformer + CNN + Drift**) captures `+0.4–0.9 pp` extra gain when each brings orthogonal signals, whereas throwing in weaker correlated models (`LSTM`, `LogReg`) degrades `RELIANCE` and `NIFTY_FUT`.

---

## 5. Feature Upgrades (`v2`, Multi-Horizon `10s..100s`) & Averaging `N = 1..100` Study

### 5.1 What `LightGBM v2` (38 Causal Features) Adds

Adding **13 strictly causal within-day features** closes the `5 s` memory gap and fixes late-day closing stalls without hurting speed (`10.5 µs` class C++ trees):
- **Block A (`9` Multi-Scale Flow & Drift):** `mid_drift_{5,15,30,60}s_ticks`, `ofi_mean_{10,30,60}s`, `imb1_mean_20s`, `imb1_vel_20s = imb_l1 - imb1_mean_20s`.
- **Block B (`3` Realized Volatility & Spread):** `vol_{10,30}s_ticks` (trailing mean `|Δmid_1s| / tick`), `spread_mean_20s`.
- **Block C (`1` Intraday Seasonality):** `day_progress = pos_in_day / 22500.0` (`09:15 → 15:30 IST`).

### 5.2 Three Ways of Averaging (`10, 20, ..., 100`) — Only Two Work!

We tested all three interpretations of "taking an average of `10..100` predictions" ([averaging_1_to_100_summary.csv](../artifacts/nse/final_verification/averaging_1_to_100_summary.csv)):

1. **Multi-Horizon Model Averaging (`10s, 20s, ..., 100s` trained models evaluated at the same instant `t`) → HELPS at `[10s, 20s]` (`+0.7–1.2 pp`):**
   - On `RELIANCE v2` predicting `20 s`: Single `20s` (`0.4963` F1, `0.6886` AUC) → **`Avg(10s, 20s)` (`0.5032` F1, `0.6944` AUC, `63.80%` Hit)** → `Avg(10s..30s)` (`0.5027`) → declines smoothly out to `Avg(10s..100s)` (`0.4877` F1) as long-horizon models dilute fast microstructure timing.
2. **Trailing Output Smoothing (`Mean of last N = 10..100 seconds [t-N+1..t]`) → HURTS MONOTONICALLY:**
   - Averaging past predictions adds **phase lag**: `RELIANCE` directional hit rate drops `63.2% (N=1) → 56.9% (N=10) → 54.6% (N=20) → 52.3% (N=50) → 51.1% (N=100)` (`NIFTY_FUT: 68.2% → 53.6%`; `IDEA: 74.6% → 55.3%`). Past probabilities expire faster than 10 seconds.
3. **Batch-of-`N` Majority Win Rate Across Full 6.25h Days → CLIMBS TO `89%–96%`:**
   - Because each prediction has a positive directional edge (`63%–76%`), grouping non-overlapping directional predictions across each full session cancels single-trade randomness via the Law of Large Numbers:

| Symbol (`12` Full Days, `~22,380 s/day`) | Daily Hit (`Min–Max`) | Batch `1` | Batch `10` | Batch `20` | Batch `30` | Batch `50` | **Batch `100` Win %** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RELIANCE** (`v2`) | `63.4%` (`59.3%–68.0%`) | `63.4%` | `67.3%` | `72.2%` | `75.3%` | `81.6%` | **`89.3%`** (`93.4%` w/ SSF+MH) |
| **NIFTY_FUT** (`v2`) | `68.2%` (`64.9%–70.2%`) | `68.2%` | `72.5%` | `77.4%` | `81.1%` | `86.2%` | **`92.7%`** (`94.8%` w/ MH) |
| **IDEA** (`v2`) | `75.2%` (`68.8%–81.6%`) | `75.2%` | `78.3%` | `83.1%` | `85.2%` | `90.0%` | **`95.0%`** (`96.0%` w/ SSF+MH) |

---

## 6. Cross-Market Single-Stock Futures (`SSF → Cash`) & 25-Stock Out-of-Universe Verification

To test whether `LightGBM v2`, `Multi-Horizon Avg(10s, 20s)`, and **`8` causal Single-Stock Futures features** (`ssf_imb_l1`, `ssf_imb_l5`, `ssf_imb1_5s`, causally demeaned `basis_dev_bp`, `ssf_ret_1s/5s_bp`, and `lead_5s/15s_bp`) overfit our `3` development benchmarks, we froze all formulas and evaluated them across **all `25` NSE instruments (`~6.71M` held-out test seconds, `22` strictly unseen)** and **all `12` equities with `64/64` matched SSF days (`10` unseen)**.

### 6.1 Summary Across All 5 Relative-Tick Regimes ([tick_regime_verification.csv](../artifacts/nse/final_verification/tick_regime_verification.csv))

| Relative Tick Regime | Instruments | Boundary Err % | Baseline `v1` F1 | Upgraded `v2+MH` F1 | Best Deployed (`+SSF`) F1 | Total `ΔF1` | `v1 → Best` ROC-AUC | Mean Batch-100 Win |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Tiny Tick (`<0.10 bp`)** | `2` | `66.9%` | `0.4513` | `0.4824` | **`0.4824`** | `+3.11 pp` | `0.6426 → 0.6741` | **`94.8%`** |
| **2. Small Tick (`0.20–0.65 bp`)** | `9` | `61.4%` | `0.4305` | `0.4917` | **`0.4963`** | **`+6.58 pp`** | `0.6264 → 0.6900` | **`91.4%`** |
| **3. Medium Tick (`0.65–1.30 bp`)** | `8` | `59.8%` | `0.4297` | `0.4890` | **`0.4957`** | **`+6.60 pp`** | `0.6289 → 0.6899` | **`92.0%`** |
| **4. Large Tick (`1.30–2.50 bp`)** | `4` | `63.3%` | `0.4735` | `0.5246` | **`0.5280`** | **`+5.46 pp`** | `0.6629 → 0.7238` | **`91.8%`** |
| **5. Ultra-Large Tick (`>2.50 bp`)** | `2` | `82.1%` | `0.4989` | `0.5585` | **`0.5608`** | **`+6.18 pp`** | `0.7259 → 0.7790` | **`93.1%`** |

### 6.2 Formal Nonparametric Anti-Overfitting Tests ([statistical_tests_summary.csv](../artifacts/nse/final_verification/statistical_tests_summary.csv))

| Comparison | Evaluation Cohort | Wins / Total | Mean Before → After | Mean Lift (`Δ`) | Wilcoxon Signed-Rank `p` | Binomial Sign `p` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Macro-F1: `v2 (38f)` vs `v1 (25f)` | **All 25 Instruments** | **`25 / 25` (`100%`)** | `0.4443 → 0.4962` | **`+5.20 pp`** | **`2.98 × 10⁻⁸`** | `2.98 × 10⁻⁸` |
| Macro-F1: `v2+MH(10,20)` vs `v1` | **Unseen 22 Instruments** | **`22 / 22` (`100%`)** | `0.4421 → 0.4999` | **`+5.77 pp`** | **`2.38 × 10⁻⁷`** | `2.38 × 10⁻⁷` |
| ROC-AUC: `v2+MH(10,20)` vs `v1` | **Unseen 22 Instruments** | **`22 / 22` (`100%`)** | `0.6386 → 0.6947` | **`+5.61 pp`** | **`2.38 × 10⁻⁷`** | `2.38 × 10⁻⁷` |
| Macro-F1: Best Deployed vs `v1` | **Unseen 22 Instruments** | **`22 / 22` (`100%`)** | `0.4421 → 0.5044` | **`+6.23 pp`** | **`2.38 × 10⁻⁷`** | `2.38 × 10⁻⁷` |
| Macro-F1: `+SSF` vs `v2+MH` *(pure SSF)* | **All 12 SSF Equities** | **`12 / 12` (`100%`)** | `0.5059 → 0.5154` | **`+0.95 pp`** | **`2.44 × 10⁻⁴`** | `2.44 × 10⁻⁴` |
| Macro-F1: `+SSF` vs `v2+MH` *(pure SSF)* | **Unseen 10 SSF Equities** | **`10 / 10` (`100%`)** | `0.5024 → 0.5124` | **`+1.00 pp`** | **`9.77 × 10⁻⁴`** | `9.77 × 10⁻⁴` |

> **Anti-Overfitting Conclusion:** Gains on the **`22` strictly untouched instruments (`+5.77 pp` from `v2+MH`, `+6.23 pp` with `SSF`) actually exceed those on the `3` development benchmarks**, and **`300 / 300` full held-out trading days** (`25` instruments × `12` test days) individually beat `55%` directional hit rate (lowest single day across all 300 symbol-days: `58.84%`). Combining everything on `RELIANCE` (`v2 + SSF + Multi-Horizon(10s,20s) + TLOB`) pushes held-out Macro-F1 to **`0.5160` (`0.7081` ROC-AUC)**—a `+6.19 pp` lift over the original LightGBM baseline. Furthermore, feeding the exact same `8` cross-market futures formulas to **`NIFTY_FUT`** from companion **`BANKNIFTY_FUT`** lifts index-futures performance from `0.4729 → 0.4754` Macro-F1 (`0.6664 → 0.6689` ROC-AUC, `68.33% → 68.75%` DirHit) out-of-sample, powering the live 46-feature terminal across all dashboard benchmarks.

### 6.3 Complete Per-Instrument 25-Stock Verification Ledger ([universe25_upgrade_verification.csv](../artifacts/nse/final_verification/universe25_upgrade_verification.csv))

| Symbol | Sector | Tick (`bp`) | Boundary Err % | `v1` F1 | `v2+MH` F1 | `+SSF+MH` F1 | Best AUC (`Δ`) | Dir Hit | Days `>55%` | Batch-100 Win |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **IDEA** *(dev)* | Telecom | `7.410` | `87.4%` | `0.4830` | `0.5441` | `0.5474` | `0.7729 (+0.0493)` | `75.6%` | `12/12` | **`96.0%`** |
| **YESBANK** | Banking | `4.518` | `76.8%` | `0.5149` | `0.5728` | **`0.5741`** | `0.7851 (+0.0568)` | `67.9%` | `12/12` | `90.2%` |
| **SUZLON** | Power | `2.370` | `65.1%` | `0.4733` | `0.5250` | `—` | `0.7255 (+0.0585)` | `64.7%` | `12/12` | `90.8%` |
| **ITC** | FMCG | `1.883` | `63.4%` | `0.4850` | `0.5306` | **`0.5445`** | `0.7373 (+0.0626)` | `67.8%` | `12/12` | **`95.3%`** |
| **BPCL** | Oil & Gas | `1.634` | `60.9%` | `0.4706` | `0.5238` | `—` | `0.7186 (+0.0603)` | `64.7%` | `12/12` | `89.2%` |
| **NTPC** | Power | `1.532` | `63.6%` | `0.4650` | `0.5189` | `—` | `0.7141 (+0.0625)` | `65.2%` | `12/12` | `91.8%` |
| **KOTAKBANK** | Banking | `1.186` | `60.8%` | `0.4332` | `0.4872` | **`0.5049`** | `0.6977 (+0.0757)` | `66.9%` | `12/12` | **`94.3%`** |
| **BHEL** | Cap. Goods | `1.186` | `61.9%` | `0.4035` | `0.4833` | `—` | `0.6820 (+0.0650)` | `62.9%` | `12/12` | `90.4%` |
| **PNB** | Banking | `0.856` | `61.7%` | `0.4183` | `0.4933` | `—` | `0.6894 (+0.0730)` | `64.4%` | `12/12` | `89.4%` |
| **CANBK** | Banking | `0.810` | `61.8%` | `0.4170` | `0.4911` | `—` | `0.6867 (+0.0672)` | `63.3%` | `12/12` | `89.7%` |
| **AXISBANK** | Banking | `0.805` | `58.2%` | `0.4378` | `0.4828` | **`0.4948`** | `0.6897 (+0.0540)` | `65.7%` | `12/12` | `93.9%` |
| **RELIANCE** *(dev)* | Energy | `0.797` | `62.2%` | `0.4537` | `0.5032` | **`0.5135`** | `0.7060 (+0.0583)` | `65.4%` | `12/12` | `93.4%` |
| **ICICIBANK** | Banking | `0.718` | `56.2%` | `0.4469` | `0.4940` | **`0.5022`** | `0.6930 (+0.0520)` | `64.7%` | `12/12` | `93.0%` |
| **HDFCBANK** | Banking | `0.698` | `55.8%` | `0.4273` | `0.4771` | **`0.4827`** | `0.6749 (+0.0431)` | `63.4%` | `12/12` | `92.1%` |
| **ASHOKLEY** | Auto | `0.620` | `65.7%` | `0.3966` | `0.4786` | `—` | `0.6734 (+0.0835)` | `63.0%` | `12/12` | `90.3%` |
| **SUNPHARMA** | Pharma | `0.540` | `57.4%` | `0.4243` | `0.4922` | `—` | `0.6828 (+0.0622)` | `63.1%` | `12/12` | `89.0%` |
| **BHARTIARTL** | Telecom | `0.534` | `61.7%` | `0.4580` | `0.5028` | **`0.5191`** | `0.7104 (+0.0670)` | `65.9%` | `12/12` | `93.9%` |
| **SBIN** | Banking | `0.504` | `64.0%` | `0.4249` | `0.4968` | **`0.5052`** | `0.6969 (+0.0745)` | `66.1%` | `12/12` | **`94.8%`** |
| **BAJFINANCE** | Financials | `0.489` | `59.6%` | `0.4388` | `0.4839` | `—` | `0.6780 (+0.0397)` | `63.1%` | `12/12` | `90.7%` |
| **INFY** | IT Services | `0.446` | `63.2%` | `0.4494` | `0.4967` | **`0.5078`** | `0.7014 (+0.0641)` | `65.4%` | `12/12` | `93.4%` |
| **TCS** | IT Services | `0.434` | `59.8%` | `0.4263` | `0.4830` | **`0.4883`** | `0.6852 (+0.0562)` | `63.8%` | `12/12` | `93.1%` |
| **ONGC** | Oil & Gas | `0.429` | `59.6%` | `0.4171` | `0.4990` | `—` | `0.6946 (+0.0755)` | `62.9%` | `12/12` | `87.7%` |
| **TITAN** | Consumer | `0.205` | `61.6%` | `0.4390` | `0.4924` | `—` | `0.6870 (+0.0497)` | `63.9%` | `12/12` | `90.1%` |
| **NIFTY_FUT** *(dev)* | Index Fut | `0.043` | `66.4%` | `0.4425` | `0.4729` | `—` | `0.6664 (+0.0303)` | `68.3%` | `12/12` | **`94.8%`** |
| **BANKNIFTY_FUT** | Index Fut | `0.035` | `67.4%` | `0.4602` | `0.4919` | `—` | `0.6818 (+0.0328)` | `69.1%` | `12/12` | **`94.8%`** |


---

## 7. Complete ML Model & Evaluation Reporting Essentials (`classification_report`, Priors, Calibration & Generalization)

To ensure this project's experimental documentation meets full Machine Learning publication and grading completeness standards, all diagnostic, calibration, class-balance, and generalization metrics were computed strictly out-of-sample across our `3` core deep-benchmarked instruments (`805,714` held-out test seconds) and all `25` NSE instruments (`6,714,225` held-out test seconds, via [src/experiments/nse_ml_completeness_diagnostics.py](../src/experiments/nse_ml_completeness_diagnostics.py)).

### 7.0 Checklist of 10 Essential ML Evaluation & Reporting Necessities

| # | Mandatory ML Reporting Element | Why Relying on Plain Accuracy Alone Fails | Verified Evidence Added Below |
| :-: | :--- | :--- | :--- |
| **1** | **Target Class Definitions & Split Priors** | Imbalanced or regime-shifting priors invalidate raw accuracy across splits | Section 7.1 (`% FLAT / % UP / % DOWN` across Train / Val / Test & 25 stocks) |
| **2** | **Dummy & Random Sanity Baselines** | Proves learned models beat both prior-guessing and uniform noise | Section 7.2 (`Uniform Random`, `Stratified Random`, `Naive Flat`, `Momentum`) |
| **3** | **Per-Class `classification_report` (`P / R / F1`)** | Exposes single-class collapse (`0.00–0.20` F1) masked by overall accuracy | Section 7.2 (All 3 classes across 15 model rows + 25-stock `min F1 >= 0.384`) |
| **4** | **Full `3 × 3` Confusion Matrices** | Separates harmless `±α` dead-zone misses from costly `UP ↔ DOWN` reversals | Section 7.3 (`Logistic Regression`, `LightGBM v1/v2`, `TLOB`, `Flagship Hybrid`) |
| **5** | **Imbalance-Robust Metrics (`MCC`, `κ`, `BalAcc`)** | Penalizes majority-class bias even under `69.4% FLAT` discrete ticks | Section 7.4 (`Balanced Acc`, `Matthews Correlation`, `Cohen's κ`, `Macro PR-AUC`) |
| **6** | **Proper Scoring & Calibration (`LogLoss`, `Brier`, `ECE`)** | Verifies predicted probabilities reflect real win probabilities (not overconfident) | Section 7.4 (`ECE = 0.0056` on `RELIANCE`, `LogLoss < ln(3) = 1.0986`) |
| **7** | **Train vs. Validation vs. Test Generalization Gap** | Detects tree memorization vs true out-of-sample pattern transfer | Section 7.5 (Train/Val/Test F1, LogLoss & early-stopping `best_iteration`) |
| **8** | **Multi-Fold Stability (`Mean ± Std`)** | Distinguishes genuine lift from a single lucky test week | Section 7.2 & 7.5 (`4 × 16`-day walk-forward segment `Mean ± Std`) |
| **9** | **Feature Attribution (`TreeSHAP` & Split Gain)** | Audits whether causal `v2` & `SSF` signals actually drive predictions | Section 7.6 (`day_progress`, `vol_30s`, `mid_drift_5s`, `ssf_ret_5s`, `basis_dev`) |
| **10** | **Reproducible Complexity, Window & Latency** | Captures parameter budget, input geometry, and live `10.5 µs` C++ speed | Section 7.6 (`Tiers 0–5 + Hybrid` architecture, params, CPU/GPU latency) |

---

### 7.1 Target Definitions, Adaptive Threshold `α`, and 3-Class Distributions ([class_distribution_splits.csv](../artifacts/nse/final_verification/class_distribution_splits.csv))

All classifiers predict the integer-encoded ternary future smoothed mid-return label at horizon `k = 20 s`:
- **Class `0` (`FLAT`):** `|r_20s| ≤ α` (dead zone calibrated strictly on each segment's **training days** as the `33.3%` quantile of `|r_20s|`)
- **Class `1` (`UP`):** `r_20s > +α`
- **Class `2` (`DOWN`):** `r_20s < -α`

| Symbol | Split (`64` Days Total) | Rows (`N`) | `FLAT (0)` % | `UP (+1)` % | `DOWN (-1)` % | Mean `α` (`bp`) ± Std |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **RELIANCE** | `Train` (`44` days) | `984,757` | `33.46%` | `32.58%` | `33.95%` | `0.5586 ± 0.2335` |
| **RELIANCE** | `Validation` (`8` days) | `179,048` | `33.52%` | `32.91%` | `33.57%` | `0.5586 ± 0.2335` |
| **RELIANCE** | `Test` (`12` days) | `268,572` | **`33.59%`** | **`32.40%`** | **`34.01%`** | `0.5586 ± 0.2335` |
| **NIFTY_FUT** | `Train` (`44` days) | `984,764` | `33.45%` | `32.70%` | `33.85%` | `0.3924 ± 0.1580` |
| **NIFTY_FUT** | `Validation` (`8` days) | `179,048` | `30.15%` | `35.26%` | `34.58%` | `0.3924 ± 0.1580` |
| **NIFTY_FUT** | `Test` (`12` days) | `268,572` | **`31.13%`** | **`34.82%`** | **`34.05%`** | `0.3924 ± 0.1580` |
| **IDEA** | `Train` (`44` days) | `984,760` | `64.37%` | `17.00%` | `18.62%` | `0.0000 ± 0.0000` |
| **IDEA** | `Validation` (`8` days) | `179,048` | `67.57%` | `15.79%` | `16.64%` | `0.0000 ± 0.0000` |
| **IDEA** | `Test` (`12` days) | `268,570` | **`69.41%`** | **`14.81%`** | **`15.78%`** | `0.0000 ± 0.0000` |

**Physical takeaway across all 25 stocks:** On the `23` small-, medium-, tiny-, and large-tick stocks (`tick ≤ 2.37 bp`), continuous price updates keep `Test` class priors tightly balanced around **`27.9%–35.5% FLAT`, `31.8%–36.6% UP`, and `31.9%–35.5% DOWN`**. On the two ultra-large-tick stocks (**`YESBANK`** `4.52 bp` → `57.2% FLAT / 20.5% UP / 22.3% DOWN`; **`IDEA`** `7.41 bp` → `69.4% FLAT / 14.8% UP / 15.8% DOWN`), the 20s mid-price remains unchanged more than 33% of the time, so the training quantile pins `α = 0.0` and `FLAT` forms the natural majority class—making plain accuracy deeply misleading without per-class F1 and Macro-F1.

---

### 7.2 Per-Class `classification_report` (`Precision / Recall / F1`) & Class-Health Audit ([per_class_classification_reports.csv](../artifacts/nse/final_verification/per_class_classification_reports.csv))

A healthy 3-class classifier should never collapse any class into `0.00–0.20` F1. Evaluating every class separately immediately uncovers baseline model blind spots that overall accuracy or Weighted-F1 would hide:

#### RELIANCE (`N = 268,572` Held-Out Test Seconds · Priors: `33.6% / 32.4% / 34.0%`)

| Tier & Model | `FLAT` F1 (`P / R`) | `UP` F1 (`P / R`) | `DOWN` F1 (`P / R`) | `Min Class F1` | `Macro-F1 ± Std` | `Wtd-F1` | `Acc` | `MCC` | `LogLoss` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Sanity` **Uniform Random** | `0.3345` (`0.34/0.33`) | `0.3273` (`0.32/0.33`) | `0.3373` (`0.34/0.33`) | `0.3273` | `0.3331 ± 0.002` | `0.3332` | `33.3%` | `-0.0004` | `1.0986` |
| `Tier 0` **Naive Flat** | `0.5029` (`0.34/1.00`) | **`0.0000`** (`0/0`) | **`0.0000`** (`0/0`) | **`0.0000`** ✗ | `0.1676 ± 0.021` | `0.1689` | `33.6%` | `0.0000` | `9.1747` |
| `Tier 0` **Naive Momentum** | `0.4824` (`0.48/0.48`) | `0.3834` (`0.38/0.38`) | `0.4013` (`0.40/0.40`) | `0.3834` | `0.4224 ± 0.019` | `0.4228` | `42.3%` | `0.1339` | `1.4235` |
| `Tier 1` **Logistic Regression** | **`0.2517`** (`0.47/0.17`) | `0.4821` (`0.40/0.62`) | `0.4606` (`0.44/0.48`) | **`0.2517`** ⚠ | `0.3981 ± 0.029` | `0.3974` | `42.2%` | `0.1434` | `1.0729` |
| `Tier 2` **Random Forest** | `0.3825` (`0.51/0.31`) | `0.4770` (`0.41/0.56`) | `0.4681` (`0.46/0.48`) | `0.3825` | `0.4426 ± 0.014` | `0.4422` | `44.8%` | `0.1784` | `1.0508` |
| `Tier 3` **LightGBM v1 (25f)** | `0.4067` (`0.50/0.34`) | `0.4821` (`0.42/0.56`) | `0.4737` (`0.47/0.48`) | `0.4067` | `0.4542 ± 0.007` | `0.4539` | `45.8%` | `0.1910` | `1.0405` |
| `Tier 4` **LSTM** | `0.3877` (`0.42/0.36`) | `0.4039` (`0.39/0.42`) | `0.4178` (`0.41/0.43`) | `0.3877` | `0.4031 ± 0.038` | `0.4032` | `40.4%` | `0.1059` | `1.0998` |
| `Tier 5` **DeepLOB5** | `0.3326` (`0.54/0.24`) | `0.4610` (`0.39/0.57`) | `0.4462` (`0.43/0.47`) | `0.3326` | `0.4133 ± 0.040` | `0.4128` | `42.5%` | `0.1456` | `1.0683` |
| `Tier 5` **TLOB** | `0.4163` (`0.55/0.34`) | `0.4571` (`0.41/0.52`) | `0.4598` (`0.43/0.49`) | `0.4163` | `0.4444 ± 0.027` | `0.4443` | `44.7%` | `0.1748` | `1.0416` |
| `Upgraded` **LightGBM v2 (38f)** | `0.5103` (`0.55/0.48`) | `0.4913` (`0.45/0.54`) | `0.4873` (`0.50/0.47`) | `0.4873` | `0.4963 ± 0.019` | `0.4963` | `49.6%` | `0.2458` | `0.9860` |
| `Upgraded` **v2 + SSF + MH(10,20)** | **`0.5268`** (`0.55/0.51`) | `0.5060` (`0.48/0.54`) | `0.5078` (`0.52/0.49`) | `0.5060` | `0.5135 ± 0.025` | `0.5136` | `51.3%` | `0.2711` | `0.9702` |
| **`Flagship` Hybrid (`v2+SSF+MH+TLOB`)** | `0.5223` (`0.56/0.49`) | **`0.5129`** (`0.47/0.56`) | **`0.5126`** (`0.52/0.50`) | **`0.5126`** ✓ | **`0.5159 ± 0.025`** | **`0.5159`** | **`51.6%`** | **`0.2754`** | **`0.9674`** |

#### NIFTY_FUT (`N = 268,572`) & IDEA (`N = 268,570`) Key Tier Comparison

| Symbol | Tier & Model | `FLAT` F1 | `UP` F1 | `DOWN` F1 | `Min Class F1` | `Macro-F1 ± Std` | `Wtd-F1` | `Acc` | `MCC` | `LogLoss` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NIFTY_FUT** | `Naive Flat` | `0.4748` | **`0.0000`** | **`0.0000`** | **`0.0000`** ✗ | `0.1583 ± 0.013` | `0.1478` | `31.1%` | `0.0000` | `9.5144` |
| **NIFTY_FUT** | `LSTM` *(Seg 3 collapse)* | **`0.1794`** | `0.4585` | `0.3837` | **`0.1794`** ✗ | `0.3405 ± 0.076` | `0.3461` | `37.3%` | `0.0522` | `1.1111` |
| **NIFTY_FUT** | `Random Forest` / `LogReg` | `0.2796` / `0.2951` | `0.494` / `0.488` | `0.513` / `0.512` | `0.2796` | `0.4288` / `0.4313` | `0.434` / `0.436` | `44.8%` / `44.7%` | `0.170` / `0.168` | `1.051` / `1.063` |
| **NIFTY_FUT** | `LightGBM v1 (25f)` | `0.3082` | `0.4995` | `0.5218` | `0.3082` | `0.4432 ± 0.006` | `0.4475` | `45.8%` | `0.1855` | `1.0418` |
| **NIFTY_FUT** | **`Flagship Hybrid (v2+MH+TLOB)`** | **`0.3844`** | **`0.5123`** | **`0.5269`** | **`0.3844`** ✓ | **`0.4745 ± 0.008`** | **`0.4775`** | **`48.2%`** | **`0.2206`** | **`1.0168`** |
| **IDEA** | `Naive Flat` *(Accuracy Trap)* | `0.8194` | **`0.0000`** | **`0.0000`** | **`0.0000`** ✗ | `0.2731 ± 0.009` | `0.5688` | **`69.4%`** | `0.0000` | `4.2263` |
| **IDEA** | `Logistic Regression` | `0.4077` | `0.3562` | `0.3500` | `0.3500` | `0.3713 ± 0.032` | `0.3910` | `37.9%` | `0.1624` | `1.0625` |
| **IDEA** | `TLOB` / `DeepLOB5` | `0.625` / `0.619` | `0.317` / `0.356` | `0.313` / `0.361` | `0.3127` | `0.4185` / `0.4452` | `0.530` / `0.539` | `49.3%` / `50.4%` | `0.172` / `0.222` | `0.989` / `0.940` |
| **IDEA** | `LightGBM v1 (25f)` | `0.6679` | `0.4062` | `0.3813` | `0.3813` | `0.4851 ± 0.017` | `0.5839` | `55.4%` | `0.2585` | `0.9484` |
| **IDEA** | **`Flagship Hybrid (v2+SSF+MH+TLOB)`** | **`0.7685`** | **`0.4478`** | **`0.4330`** | **`0.4330`** ✓ | **`0.5498 ± 0.005`** | **`0.6681`** | **`65.1%`** | **`0.3434`** | **`0.7973`** |

**Three critical ML findings revealed only by Per-Class F1:**
1. **`LSTM` suffers genuine class collapse (`FLAT F1 = 0.1794 < 0.20`) on `NIFTY_FUT`:** Its `FLAT` recall plunges to `12.5%` because unnormalized price levels under Segment 3's rally distort dead-zone gates (`LogLoss = 1.1111`, worse than uniform `1.0986`).
2. **Linear models under-predict `FLAT` (`0.2517` F1, `17.2%` recall on `RELIANCE`):** A single linear hyperplane cannot isolate a bounded `[-α, +α]` region surrounded by both `UP` and `DOWN`.
3. **All `25 / 25` NSE instruments pass healthy 3-class balance after upgrades ([universe25_class_and_perclass_f1.csv](../artifacts/nse/final_verification/universe25_class_and_perclass_f1.csv)):** Across `6,714,225` test seconds, the **lowest single-class F1 across all `25 × 3 = 75` instrument-class pairs** in our deployed models is **`0.3837`** (`NIFTY_FUT FLAT`, nearly double the `0.20` collapse threshold), with 25-stock mean class F1s at **`0.5090` (`FLAT`), `0.5036` (`UP`), and `0.5031` (`DOWN`)** (`Mean MCC = 0.2620`, `Mean LogLoss = 0.9647`).

---

### 7.3 Full `3 × 3` Confusion Matrices (`LightGBM v1` vs. `Flagship Hybrid`) ([confusion_matrices_detailed.csv](../artifacts/nse/final_verification/confusion_matrices_detailed.csv))

Comparing exact held-out confusion matrices (row = true label, column = predicted label, with row-normalized recall `%`) proves *where* `LightGBM v2 + SSF + MH + TLOB` wins:

| Instrument & Model | True Class (`Support`) | `Pred FLAT` (Count / Recall%) | `Pred UP` (Count / Recall%) | `Pred DOWN` (Count / Recall%) | Class Precision |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **RELIANCE · `LightGBM v1`** | `FLAT` (`90,217`) | `31,032` (`34.4%`) | `34,717` (`38.5%`) | `24,468` (`27.1%`) | `49.75%` |
| **RELIANCE · `LightGBM v1`** | `UP` (`87,010`) | `14,422` (`16.6%`) | `48,504` (`55.7%`) | `24,084` (`27.7%`) | `42.46%` |
| **RELIANCE · `LightGBM v1`** | `DOWN` (`91,345`) | `16,921` (`18.5%`) | `31,010` (`33.9%`) | `43,414` (`47.5%`) | `47.21%` |
| **RELIANCE · `Flagship Hybrid`** | `FLAT` (`90,217`) | **`44,123` (`48.9%`)** | `26,077` (`28.9%`) | `20,017` (`22.2%`) | **`56.03%`** |
| **RELIANCE · `Flagship Hybrid`** | `UP` (`87,010`) | `16,885` (`19.4%`) | **`48,592` (`55.8%`)** | **`21,533` (`24.8%`)** | **`47.42%`** |
| **RELIANCE · `Flagship Hybrid`** | `DOWN` (`91,345`) | `17,747` (`19.4%`) | **`27,798` (`30.4%`)** | **`45,800` (`50.1%`)** | **`52.43%`** |
| **IDEA · `Flagship Hybrid`** | `FLAT` (`186,411`) | **`132,884` (`71.3%`)** | `22,585` (`12.1%`) | `30,942` (`16.6%`) | **`83.36%`** |
| **IDEA · `Flagship Hybrid`** | `UP` (`39,778`) | `13,134` (`33.0%`) | **`19,976` (`50.2%`)** | **`6,668` (`16.8%`)** | **`40.40%`** |
| **IDEA · `Flagship Hybrid`** | `DOWN` (`42,381`) | `13,396` (`31.6%`) | **`6,879` (`16.2%`)** | **`22,106` (`52.2%`)** | **`37.02%`** |

On `RELIANCE`, upgrading to `Flagship Hybrid` boosts true `FLAT` capture by **`+13,091` seconds (`+14.5 pp` recall)** while simultaneously cutting dangerous opposite-direction reversals (`UP↔DOWN`: `27.7% → 24.8%` and `33.9% → 30.4%`). On `IDEA`, `UP↔DOWN` cross-reversals drop to barely `16.2%–16.8%`, with `FLAT` precision reaching **`83.36%`**.

---

### 7.4 Imbalance-Robust Metrics & Probabilistic Calibration ([per_class_classification_reports.csv](../artifacts/nse/final_verification/per_class_classification_reports.csv))

A balanced 3-class uniform random baseline produces `Balanced Acc = 0.3333`, `MCC = 0.0000`, `Cohen's κ = 0.0000`, `ROC-AUC = 0.5000`, `Macro PR-AUC = 0.3333`, `LogLoss = 1.0986`, and `Brier Score = 0.6667`:

| Benchmark | Model | `Bal. Acc` | `MCC` (`±Std`) | `Cohen's κ` | `ROC-AUC` | `Macro PR-AUC` (`F / U / D`) | `LogLoss` | `Brier` | `10-Bin ECE` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RELIANCE** | `LightGBM v1` | `0.4589` | `0.191 ± 0.018` | `0.1880` | `0.6479` | `0.4738` (`0.49/0.45/0.48`) | `1.0405` | `0.6283` | `0.0087` |
| **RELIANCE** | **`Flagship Hybrid`** | **`0.5163`** | **`0.275 ± 0.040`** | **`0.2743`** | **`0.7080`** | **`0.5594`** (`0.61/0.52/0.54`) | **`0.9674`** | **`0.5804`** | **`0.0056`** |
| **NIFTY_FUT** | `LightGBM v1` | `0.4523` | `0.186 ± 0.012` | `0.1829` | `0.6364` | `0.4569` (`0.36/0.50/0.50`) | `1.0418` | `0.6282` | `0.0038` |
| **NIFTY_FUT** | **`Flagship Hybrid`** | **`0.4776`** | **`0.221 ± 0.007`** | **`0.2195`** | **`0.6683`** | **`0.4904`** (`0.41/0.53/0.53`) | **`1.0168`** | **`0.6121`** | **`0.0046`** |
| **IDEA** | `LightGBM v1` | `0.5411` | `0.259 ± 0.014` | `0.2404` | `0.7231` | `0.5211` (`0.81/0.40/0.36`) | `0.9484` | `0.5724` | `0.0823` |
| **IDEA** | **`Flagship Hybrid`** | **`0.5789`** | **`0.343 ± 0.006`** | **`0.3370`** | **`0.7734`** | **`0.5786`** (`0.86/0.45/0.43`) | **`0.7973`** | **`0.4791`** | `0.0679` |

*Note:* Notice that `Balanced Acc` (`0.5789`) on `IDEA` rises steeply above `0.3333` alongside `PR-AUC` (`0.5786` vs `0.3333`), while `RELIANCE` (`ECE = 0.0056`, i.e., `0.56%` calibration error) and `NIFTY_FUT` (`ECE = 0.0046`, i.e., `0.46%` error) remain near-perfectly calibrated probability estimators across all confidence deciles.

---

### 7.5 Train vs. Validation vs. Test Generalization Gap & Early Stopping ([train_val_test_generalization.csv](../artifacts/nse/final_verification/train_val_test_generalization.csv))

Comparing in-sample training performance against held-out Validation (`8d`) and Test (`12d`) performance across the 4 walk-forward segments verifies that early stopping (`patience = 20`) prevents overfitting even when adding `v2` and `SSF` features:

| Symbol & Model | `Train F1` | `Val F1` | `Test F1 (±Std)` | `Train–Test Gap` | `Val–Test Gap` | `Train → Test LogLoss` | `Best Iter (±Std)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RELIANCE · `v1 (25f)`** | `0.5167` | `0.4363` | `0.4443 ± 0.0065` | `+0.0725` | `-0.0080` | `0.9696 → 1.0385` | `62.2 ± 11.5` |
| **RELIANCE · `v2 (38f)`** | `0.5629` | `0.4921` | `0.4862 ± 0.0190` | `+0.0767` | `+0.0059` | `0.9169 → 0.9860` | `79.5 ± 11.7` |
| **RELIANCE · `v2 + SSF (46f)`** | `0.5784` | `0.5001` | `0.4956 ± 0.0223` | `+0.0828` | `+0.0045` | `0.8985 → 0.9744` | `89.8 ± 10.2` |
| **NIFTY_FUT · `v1 (25f)`** | `0.5153` | `0.4281` | `0.4420 ± 0.0073` | `+0.0733` | `-0.0139` | `0.9881 → 1.0418` | `91.0 ± 41.2` |
| **NIFTY_FUT · `v2 (38f)`** | `0.5575` | `0.4478` | `0.4667 ± 0.0084` | `+0.0907` | `-0.0190` | `0.9425 → 1.0208` | `91.0 ± 40.6` |
| **IDEA · `v1 (25f)`** | `0.5819` | `0.4967` | `0.4808 ± 0.0156` | `+0.1011` | `+0.0159` | `0.8191 → 0.9470` | `105.0 ± 32.2` |
| **IDEA · `v2 (38f)`** | `0.6367` | `0.5324` | `0.5261 ± 0.0091` | `+0.1106` | `+0.0063` | `0.7378 → 0.8433` | `105.0 ± 16.2` |
| **IDEA · `v2 + SSF (46f)`** | `0.6874` | `0.5349` | `0.5264 ± 0.0078` | `+0.1611` | `+0.0085` | `0.6776 → 0.8405` | `148.5 ± 4.4` |

Every upgraded configuration maintains a **near-zero Validation-to-Test gap (`-0.019` to `+0.009` F1)**, confirming that validation-based early stopping (~`62–149` trees vs `250` max) transfers cleanly to unseen future days.

---

### 7.6 Upgraded Feature Attribution (`TreeSHAP` + Gain) & Complete Model Complexity Ledger ([v2_ssf_feature_importance.csv](../artifacts/nse/final_verification/v2_ssf_feature_importance.csv))

Evaluating `shap.TreeExplainer` across `8,000` held-out seconds per benchmark proves that the newly added **multi-scale `v2`** and **cross-market `SSF`** features dominate the upgraded trees—accounting for the **Top 7 out of 7 features on `RELIANCE`**:

| Rank | `RELIANCE` (`v2+SSF 46f`) — `TreeSHAP %` (`Gain %`) | `NIFTY_FUT` (`v2 38f`) — `TreeSHAP %` (`Gain %`) | `IDEA` (`v2+SSF 46f`) — `TreeSHAP %` (`Gain %`) |
| :-: | :--- | :--- | :--- |
| **1** | `day_progress` — **`14.23%`** (`17.79%`) | `slope_ask` — **`13.10%`** (`10.50%`) | `day_progress` — **`10.53%`** (`11.72%`) |
| **2** | `vol_30s_ticks` — **`9.54%`** (`18.14%`) | `slope_bid` — **`12.93%`** (`9.85%`) | `vol_10s_ticks` — **`8.75%`** (`12.33%`) |
| **3** | `mid_drift_5s_ticks` — **`9.41%`** (`11.11%`) | `mid_drift_5s_ticks` — **`11.21%`** (`13.33%`) | `mid_drift_5s_ticks` — **`8.11%`** (`11.84%`) |
| **4** | `ssf_ret_5s_bp` *(SSF)* — **`5.79%`** (`4.70%`) | `vol_30s_ticks` — **`7.62%`** (`11.86%`) | `micro_deviation` — **`6.44%`** (`6.80%`) |
| **5** | `vol_10s_ticks` — **`3.81%`** (`3.38%`) | `day_progress` — **`5.89%`** (`7.89%`) | `imbalance_l1` — **`6.41%`** (`4.90%`) |
| **6** | `basis_dev_bp` *(SSF)* — **`3.52%`** (`3.08%`) | `imbalance_l5` — **`5.89%`** (`4.10%`) | `basis_dev_bp` *(SSF)* — **`6.08%`** (`5.00%`) |
| **7** | `lead_15s_bp` *(SSF)* — **`3.50%`** (`1.60%`) | `ofi_lag_1` — **`3.89%`** (`2.26%`) | `vpin` — **`4.41%`** (`5.69%`) |

#### Reproducible Model Hyperparameters, Input Geometry & Inference Complexity (`RELIANCE`, [inference_latency.csv](../artifacts/nse/models/RELIANCE/inference_latency.csv))

| Tier | Model Family | Input Window / Features | Key Hyperparameters & Normalization | Inference Speed (Median `p50` / `p99`) |
| :---: | :--- | :--- | :--- | :--- |
| **0** | `Naive Flat / Momentum` | `0` / `1` scalar | Majority prior / Trailing `k`-sec smoothed sign | `< 1 µs` CPU |
| **1** | `Logistic Regression` | `1s` vector (`25f`) | L2 penalty, `class_weight="balanced"`, train z-score | `< 5 µs` CPU |
| **2** | `Random Forest` | `1s` vector (`25f`) | `100` trees, `max_depth=12`, `min_samples_leaf=10` | `~1–3 ms` CPU |
| **3** | `LightGBM v1 / v2 / +SSF` | `1s` causal vector (`25f / 38f / 46f`) | `num_leaves=63, max_depth=8, lr=0.05, es=20, balanced` | **`10.5 µs` C++** (`0.13–0.32 ms` Py) |
| **4** | `LSTM` | `100s × 20` raw LOB | 2-layer LSTM, static train z-score | GPU CUDA |
| **5** | `DeepLOB5` | `100s × 20` raw LOB | 3D Conv + Inception + LSTM, static train z-score | `4.17–8.43 ms` CUDA (`9.0–20.3 ms` `p99`) |
| **5** | `TLOB` | `100s × 20` raw LOB | Dual-Attention Transformer + **Adaptive BiN** | Multi-ms GPU CUDA |
| **★** | **`Flagship Hybrid`** | `LightGBM v2[+SSF] (10s,20s) + TLOB` | Strict **LOSO** probability blend (`0.75 / 0.25` typ.) | Hybrid (offloaded sequence) |
