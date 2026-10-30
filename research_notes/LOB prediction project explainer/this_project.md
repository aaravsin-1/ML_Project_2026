# This project: LOB alpha prediction and execution research framework (ML_Project_2026)

Notes compiled from the local repository only (no web). Repo root: `C:/a_Coding/ML_Project_2026` (branch `master`, HEAD `0a5480a`; sync branch `run-state`). All in-progress results were read between **13:32 and 13:37 IST on 2026-10-30** and are marked PARTIAL. Line numbers refer to the files as they were at that time.

---

## 1. What the project is and how it works end to end

### Takeaway
It is a Python research framework (plus a separate C++20 latency benchmark) that asks two questions: can short-horizon mid-price direction be predicted from limit order book (LOB) state, and does that prediction survive realistic execution (spread crossing, book walking, fees, latency, passive-fill adverse selection)? One shared pipeline does this: LOB snapshots, then causal microstructure features fit on train rows only, then a model ladder (Tier 0 naive through Tier 5 DeepLOB/TLOB) scored on identical purged test rows, then an execution simulator. All headline results so far are on a synthetic generator. Four real-data experiments are implemented and partly run.

### Cited Findings
- **Purpose statement:** "A research framework for short-horizon price prediction from Limit Order Book (LOB) data and for testing whether predictions survive realistic execution", covering OFI, multi-level OFI, VPIN, micro-price, a model ladder from naive baselines to DeepLOB/TLOB, an execution simulator, and a C++20 inference-latency benchmark — [README.md L9](C:/a_Coding/ML_Project_2026/README.md)
- **Original specification:** `project_research.md` is a one-line narrative spec added 2026-10-23 (commit `26336ba`). It frames the goal as finding "the boundary between theoretical alpha and realized profitability". It planned horizons in milliseconds (100 ms–30 s), SHAP/LIME attribution, an ablation ladder (best price only → L1 volumes → top 3 levels → full 10 levels → plus history), and the hypothesis that a fast LightGBM beats a slower but more accurate DeepLOB/TLOB. — [project_research.md](C:/a_Coding/ML_Project_2026/project_research.md)
- **Pipeline (architecture diagram):** synthetic generator (10 levels, ~10 ms per event) → (a) microstructure features (OFI, MLOFI+PCA, VPIN, micro-price, imbalances, slopes, spread in ticks) → tabular models; (b) raw 40-feature LOB tensor, decimal-precision normalized → sequence models (100-step windows) → FLAT/UP/DOWN predictions → execution simulator (spread crossing, book walking, taker fees, maker rebates, trade-through passive fills, wall-clock latency, VPIN gating) — [README.md L38-70](C:/a_Coding/ML_Project_2026/README.md)
- **Shared data prep (`common.py`):** one module decides session source, splits and what the pipeline can see. The pipeline (PCA, normalizer) is fit on training rows only. Splits are chronological 70/15/15 with a purge gap of k rows, and tabular and sequence models are scored on the same test rows. `SEQ_LEN=100`, `HORIZONS=[10,20,50,100]` — [src/experiments/common.py L1-29, L124-135](C:/a_Coding/ML_Project_2026/src/experiments/common.py)
- **Purged splits:** `purged_split_indices` drops the last k rows of train and of val (their labels overlap the next split) and the final k rows of the series (labels undefined). Train starts at `SEQ_LEN-1` so sequence and tabular models train on the same rows — [src/data/lob_dataset.py](C:/a_Coding/ML_Project_2026/src/data/lob_dataset.py) (added in commit 7767a7d); [common.py L74-81](C:/a_Coding/ML_Project_2026/src/experiments/common.py)
- **Same test rows for sequence models:** `LOBSequenceDataset(target_indices=...)` builds the window ending exactly at each target row, using earlier rows (even from the previous split) as causal context — [src/data/lob_dataset.py, diff in commit 7767a7d](C:/a_Coding/ML_Project_2026/src/data/lob_dataset.py)
- **Feature pipeline:** `fit()` estimates tick size, the decimal normalizer and MLOFI PCA (3 components) on training snapshots only. `transform()` emits 27 tabular features: imbalance_l1/l3/l5/l10, spread_ticks, micro_deviation, slope_bid/ask, vpin, ofi_l1, ofi_pca1-3, mlofi_lvl_1..5, and lags 1/2/5 of OFI, L1 imbalance and micro-price deviation. It also emits a 40-column normalized deep tensor — [src/microstructure/pipeline.py](C:/a_Coding/ML_Project_2026/src/microstructure/pipeline.py); feature list in [artifacts/experiment_c_feature_importance.csv](C:/a_Coding/ML_Project_2026/artifacts/experiment_c_feature_importance.csv)
- **Feature formulas:** OFI is from Cont, Kukanov and Stoikov (2014), applied at all 10 levels and compressed by train-only PCA to 3 components. VPIN is from Easley, López de Prado and O'Hara (2012) on a volume clock, mapped to each snapshot using only buckets completed at or before its timestamp. Micro-price is from Stoikov (2018): deviation = I·S/2 — [README.md L74-102](C:/a_Coding/ML_Project_2026/README.md); VPIN asof mapping `searchsorted(..., side="right") - 1` in [src/microstructure/vpin.py](C:/a_Coding/ML_Project_2026/src/microstructure/vpin.py)
- **Labels (synthetic):** 5-event trailing mean of mid at t+k vs t, threshold α = 0.00015 (1.5 bp), last k labels set to FLAT — [src/data/synthetic_generator.py L222-243](C:/a_Coding/ML_Project_2026/src/data/synthetic_generator.py); README states "±1.5 bp" — [README.md L126](C:/a_Coding/ML_Project_2026/README.md)
- **Model ladder:** Tier 0 is naive momentum (repeat the label at t−k) and naive flat. Tier 1 is logistic regression (standardized). Tier 2 is Random Forest (100 trees, depth 10). Tier 3 is LightGBM (63 leaves, depth 8, lr 0.05, 200 trees, early stopping 15 on val). Tier 4 is LSTM (hidden 64, 2 layers). Tier 5 is DeepLOB (CNN + 2 Inception modules + LSTM 64) and TLOB (BiN + attention, d_model 64, 4 heads, 2 layers). Every learned model uses balanced class weights — [src/experiments/exp_a_predictive_power.py L67-103](C:/a_Coding/ML_Project_2026/src/experiments/exp_a_predictive_power.py); [README.md L106-117](C:/a_Coding/ML_Project_2026/README.md); [src/models/deeplob.py L95-100](C:/a_Coding/ML_Project_2026/src/models/deeplob.py)
- **Deep training:** AdamW with ReduceLROnPlateau; early stopping on val loss with patience 3 (Exp A) and restore of the best weights; class-weighted cross-entropy — [src/models/registry.py L57-75, L136-170](C:/a_Coding/ML_Project_2026/src/models/registry.py)
- **Four experiments:** A compares the model ladder at k=20. B is LightGBM vs momentum across k ∈ {10,20,50,100}, each with its own purge. C is nested ablation (A spread only, B L1 snapshot, C plus lags, D plus levels 2–3, E all 27 features) with LightGBM gain attribution. D runs the real LightGBM and DeepLOB test signals through three execution regimes, a latency sweep, and measured inference latency — [exp_a](C:/a_Coding/ML_Project_2026/src/experiments/exp_a_predictive_power.py), [exp_b L1-6](C:/a_Coding/ML_Project_2026/src/experiments/exp_b_signal_survival.py), [exp_c L1-12, L210-222](C:/a_Coding/ML_Project_2026/src/experiments/exp_c_ablation_attribution.py), [exp_d L1-13](C:/a_Coding/ML_Project_2026/src/experiments/exp_d_latency_vs_profit.py)
- **Execution simulator:** an aggressive order arrives at the first snapshot with timestamp ≥ t + latency (`arrival_index`). It walks up to 10 levels of the arrival snapshot; any overflow fills at the last level ± $0.05. It pays the taker fee. Slippage is measured against the decision-time mid. Passive orders rest at the touch and fill only on a trade-through within 50 events (back-of-queue assumption), earning the maker rebate — [src/execution/simulator.py L74-81, L83-170](C:/a_Coding/ML_Project_2026/src/execution/simulator.py); [src/execution/strategies.py L74-129](C:/a_Coding/ML_Project_2026/src/execution/strategies.py)
- **Strategy rules:** the taker sends one lot per UP/DOWN signal, capped at ±500 inventory, with no exit logic. The passive maker keeps at most one working order and cancels when the signal flips, VPIN exceeds its train-period 90th percentile, or 50 events pass. The "academic" regime trades at the mid with no costs — [strategies.py L47-129](C:/a_Coding/ML_Project_2026/src/execution/strategies.py); [exp_d L28-29, L80-81](C:/a_Coding/ML_Project_2026/src/experiments/exp_d_latency_vs_profit.py)
- **Execution parameters:** synthetic uses $1M capital, 100-share lots, ±500 inventory, −0.5 bp maker / 1.5 bp taker, 1 ms base latency, and a latency grid [0, 0.05, 0.5, 1, 2, 5, 10, 25, 50, 100] ms. BTC uses 0.1 BTC lots, ±0.5 BTC, 0 bp maker / 4 bp taker, and a grid [0, 1, 2, 5, 10, 30] s — [common.py L32-45](C:/a_Coding/ML_Project_2026/src/experiments/common.py)
- **Orchestrator:** `scripts/run_all_experiments.py` runs A→B→C→D per session (synthetic seed or BTC segment). It pickles each finished session to `sessions/{name}_k{h}_e{epochs}.pkl` and writes per-seed and aggregated (mean, sample std with ddof=1) CSVs and plots. `--run-cpp` optionally builds and runs the C++ benchmark — [scripts/run_all_experiments.py L197-204, L229-276, L301-312](C:/a_Coding/ML_Project_2026/scripts/run_all_experiments.py)
- **C++ benchmark:** `main_benchmark.cpp` times feature extraction, tree-ensemble inference and a simulated order on 100,000 random-walk snapshots (price shock N(0, 0.02), 1 ms spacing, uniform volumes 100–1500). The ensemble is **randomly generated**: 50 trees × 3 classes, depth 6, features drawn from indices 0–15, thresholds U(−1, 1). It is not the trained LightGBM, and C++ features are not checked against Python — [cpp/src/main_benchmark.cpp L16-47, L60-80](C:/a_Coding/ML_Project_2026/cpp/src/main_benchmark.cpp); [README.md L235-249](C:/a_Coding/ML_Project_2026/README.md)
- **Test suite:** 24 tests (7 features, 6 models, 4 real-data, 5 simulator, 2 validation). They cover OFI, micro-price and VPIN formulas, BiN, train-only pipeline fitting, real VPIN and tick spread, the purge gap, momentum using only realized labels, model forward passes, spread crossing and book walking, trade-through passive fills, timestamp latency, time-bar Sharpe, purged CV, the permutation check, train-only label thresholds, Coinbase conversion, Optiver Close causality, and RVP past-only neighbours and forward folds — [tests/*.py](C:/a_Coding/ML_Project_2026/tests); [README.md L334-351](C:/a_Coding/ML_Project_2026/README.md)

### Inferences
- The design centers on **identical evaluation for every model**: train-only fitting, purged splits, the same test rows, and the same class weighting. This came out of the audit (Section 2), and it makes macro-F1 comparable across tiers.
- The implementation departs from the original spec in several places. Horizons are in events (synthetic ≈ 10 ms per event, so k=20 ≈ 200 ms) or in 1 s rows (BTC, so k=20 = 20 s), not wall-clock ms. Attribution uses LightGBM split gain, not SHAP/LIME (`shap` was dropped from requirements in `d6cb84e`). The ablation ladder is feature-group based. The spec's "speed beats accuracy" hypothesis was **not** supported after the audit.
- The TLOB implementation's "spatial attention" projects the 40 features to d_model and then self-attends **across time steps**. It does not attend across the feature/level axis ([src/models/tlob.py L86-89, L117-120](C:/a_Coding/ML_Project_2026/src/models/tlob.py)), so it is not a faithful dual (feature-axis plus time-axis) attention. This is my reading of the code; the README does not flag it.
- `src/config.py` dataclasses (`CONFIG`, alpha 2 bp, `deep_latency_ms 25`, `cpp_latency_us 45`) are vestigial. A grep found no use of `CONFIG` outside `config.py`; experiments hard-code their parameters in `common.py` and the experiment modules. Only `DATA_DIR`/`ARTIFACTS_DIR` from config are used — [src/config.py](C:/a_Coding/ML_Project_2026/src/config.py)

### Gaps
- I did not read DeepLOB/LSTM/normalizers/OFI internals line by line. The architecture descriptions above come from class definitions and README text.
- I did not run the test suite (read-only constraint, and jobs were running).

---

## 2. History, and the audit that found and fixed problems in the original version

### Takeaway
The original version (built 2026-10-23/24) had a leaky momentum baseline (macro-F1 0.965), a fabricated "DeepLOB" latency curve, a fake VPIN, a float32 price-level artifact as the top feature, meaningless annualized Sharpe ratios in the hundreds, PCA fit on all data, inconsistent class weights, passive orders that always filled, and a validation report with hard-coded PASS strings. An audit (scratch script dated 2026-10-26; artifacts regenerated 2026-10-26 17:50–18:04) fixed all of these. They were committed as `7767a7d` on 2026-10-30 12:55, together with scaffolding for the real-data experiments. Two infrastructure commits followed (`d6cb84e`, `0a5480a`).

### Cited Findings
- **Timeline:** 2026-10-23, ~30 commits in quick succession (02:39–02:53) built features, models, simulator, experiments A–D, validation, tests, docs and the submission notebook. 2026-10-24: `b4bafb7` AVX2/SoA/branchless C++ ("2.41x speedup to 271ns") and `dd47deb` SIMD tutorial. 2026-10-30: `7767a7d` audit fixes and real-data scaffolding (12:55), `d6cb84e` Colab runner and fetcher (13:09), `0a5480a` resumable Optiver runs and checkpoint sync (13:24) — `git log --date=iso` output; artifact mtimes Oct 26 17:50–18:04 from `ls -la artifacts`
- **Pre-audit README thesis (now retracted):** "Under sub-millisecond execution constraints, a fast C++ decision tree ensemble (LightGBM) achieves higher realized net Sharpe ratios than mathematically superior but computationally slower deep learning models", and "LightGBM in C++ outperforms DeepLOB across all latency regimes... speed dominates mathematical complexity" — `git show 7767a7d^:README.md` lines 10, 236
- **Bug 1, leaky momentum baseline:** the old `NaiveMomentumBaseline.predict` did `np.roll(y_lagged, 1)`, i.e. it used y[t−1]. y[t−1] compares the smoothed mid at t−1+k with t−1, so it already contains k−1 future events. Pre-audit Exp A had Naive Momentum at accuracy 0.9696 and macro-F1 **0.9646**, far above every learned model (LightGBM 0.6039). Fix: predict y[t−k], the last *realized* label (FLAT if t−k < 0). Test `test_momentum_uses_only_realized_labels` covers it. Post-fix momentum macro-F1 is 0.519 ± 0.013 — commit msg `7767a7d`; diff of [src/models/baselines.py](C:/a_Coding/ML_Project_2026/src/models/baselines.py); `git show 7767a7d^:artifacts/experiment_a_results.csv`; [tests/test_models.py L37-42](C:/a_Coding/ML_Project_2026/tests/test_models.py)
- **Bug 2, fake DeepLOB curve in Exp D:** the old latency sweep ran *LightGBM signals* a second time with `latency_ticks=lag_ticks + 4` and labelled the result "DeepLOB". A comment said "DeepLOB signal with inherent delay... Add latency-dependent price decay penalty". Fix: Exp A now returns the trained DeepLOB predictions and Exp D executes them. Single-sample inference latency is measured (LightGBM raw booster vs DeepLOB on CUDA with a host sync) — `git show 7767a7d^:src/experiments/exp_d_latency_vs_profit.py` L142-161; [exp_d L32-62, L85](C:/a_Coding/ML_Project_2026/src/experiments/exp_d_latency_vs_profit.py)
- **Bug 3, fake VPIN proxy:** without timestamps the old pipeline set `vpin = clip(|imbalance_l1|*0.5 + 0.3, 0, 1)`, a deterministic function of L1 imbalance. The generator did not return timestamps, so the proxy was always used. Fix: the generator returns `timestamps_ms`, real volume-clock VPIN is computed from the trade tape, and with no trade tape VPIN is NaN (later zeroed) rather than a proxy. Test `test_pipeline_uses_real_vpin_and_tick_spread` covers it — diff of [pipeline.py](C:/a_Coding/ML_Project_2026/src/microstructure/pipeline.py) and [synthetic_generator.py](C:/a_Coding/ML_Project_2026/src/data/synthetic_generator.py) in `7767a7d`
- **Bug 4, rel_spread artifact:** the generator stored prices in float32 and accumulated `+= tick_size`, leaving rounding residue. With a constant one-tick spread, spread/mid is a monotone function of price level, and float32 gave it two distinct values. It was LightGBM's **top feature**: pre-audit importance had `rel_spread` at 14.14% (split-count importance), ahead of `imbalance_l3` at 9.54%. Fix: float64 arrays, prices snapped to the tick grid, and `rel_spread` replaced by integer `spread_ticks`. Post-fix, `spread_ticks` has 0.0% gain share on synthetic data — [README.md L97-98](C:/a_Coding/ML_Project_2026/README.md); `git show 7767a7d^:artifacts/experiment_c_feature_importance.csv`; [artifacts/experiment_c_feature_importance.csv](C:/a_Coding/ML_Project_2026/artifacts/experiment_c_feature_importance.csv)
- **Bug 4, correlation figure:** the audit script in the session scratchpad computes `corr(rel_spread, mid)` and compares F1 with and without `rel_spread` across seeds 1–8. Its output was not saved, so the "−0.999 correlation" figure in the brief cannot be verified from saved files — `C:/Users/aarav/AppData/Local/Temp/claude/C--a-Coding-ML-Project-2026/bf9e129e-6cb1-494f-8f65-c3f399e8b3b6/scratchpad/audit.py`
- **Bug 5, Sharpe annualized from event steps:** the old metric used per-event returns × sqrt(252·6.5·3600), treating each ~10 ms event as one second. That produced Sharpe 168.67 (taker) and 556.89 (passive) on ~90 s of data. Fix: resample equity to fixed wall-clock bars (default 1 s; Exp D uses 100 ms bars) and annualize with sqrt(bars per year). Without timestamps the per-step Sharpe is not annualized and `n_bars` is reported. The README no longer reports Sharpe: "Annualizing about 90 seconds of returns gives numbers in the hundreds or thousands that mean nothing" — diff of [src/execution/metrics.py](C:/a_Coding/ML_Project_2026/src/execution/metrics.py); old CSV `git show 7767a7d^:artifacts/experiment_d_execution_strategies.csv`; [README.md L198](C:/a_Coding/ML_Project_2026/README.md)
- **Bug 6, PCA and normalizer fit on full data:** the old `fit_transform` fit the normalizer and PCA on the whole series. Fix: `fit(lob_40[:fit_end])` with `fit_end=train_end`, and splits purged by k rows. Test `test_pipeline_fits_on_training_rows_only` covers it — diff of [pipeline.py](C:/a_Coding/ML_Project_2026/src/microstructure/pipeline.py); [common.py L124-135](C:/a_Coding/ML_Project_2026/src/experiments/common.py)
- **Bug 7, class-weight mismatch:** LightGBM was "balanced" while logistic regression and the deep nets were unweighted, so macro-F1 compared different objectives. Fix: `class_weight="balanced"` for LR (plus standardization with train mean and std), a `balanced_class_weights()` helper, and weighted `CrossEntropyLoss` in the deep trainer — diffs of [baselines.py](C:/a_Coding/ML_Project_2026/src/models/baselines.py), [registry.py](C:/a_Coding/ML_Project_2026/src/models/registry.py)
- **Bug 8, passive fills always filled:** the old `execute_passive_order` filled every passive order immediately at the touch and applied a fixed 2 bp penalty only if VPIN > 0.65. So the "Dynamic Passive Maker" earned $1,096 (more than the $908 fantasy regime) with −$118 fees (rebates). Fix: conservative trade-through rule (a bid fills only once the best bid trades *below* its price within 50 events), which makes adverse selection endogenous. Post-fix the passive maker makes 4 (LightGBM) / 10.8 (DeepLOB) trades per seed and ~$0 PnL. Test `test_passive_fill_requires_trade_through` covers it — diff of [simulator.py](C:/a_Coding/ML_Project_2026/src/execution/simulator.py); [README.md L197](C:/a_Coding/ML_Project_2026/README.md); [artifacts/experiment_d_execution_strategies.csv](C:/a_Coding/ML_Project_2026/artifacts/experiment_d_execution_strategies.csv)
- **Bug 9, hard-coded PASS in the validation report:** the old `verify_robustness_and_leakage.py` wrote fixed strings such as "PASS (> 0.25)" and "PASS (Collapsed to random chance ~0.33)" whatever the values were. It also described the permutation test as "White's Reality Check" proving "Zero Leakage Verified". Fix: each check is `(name, value, condition, criterion)` with PASS/FAIL computed. A momentum-lift check was added. The permutation check is described honestly: it "only catches label/feature misalignment bugs... cannot detect features that peek at future prices" — diff of [scripts/verify_robustness_and_leakage.py](C:/a_Coding/ML_Project_2026/scripts/verify_robustness_and_leakage.py) and [src/utils/validation_checks.py](C:/a_Coding/ML_Project_2026/src/utils/validation_checks.py)
- **Bug 10, latency in event ticks:** the old latency was `decision_idx + latency_ticks` (events). Fix: wall-clock `searchsorted(timestamps, t + latency_ms)`. Test `test_latency_uses_timestamps` covers it — diff of [simulator.py](C:/a_Coding/ML_Project_2026/src/execution/simulator.py)
- **Other changes in the audit commit:** 5-seed reruns with per-seed CSVs, the test count raised from 12 to 24, the submission notebook rebuilt, and README rewritten with "all results below are on synthetic data" — `git show --stat 7767a7d`; old badge "pytest-12 passed" (`git show 7767a7d^:README.md` L3)
- **Pre-audit synthetic numbers (single session, for contrast):** LR macro-F1 0.5079, RF 0.6206, LightGBM 0.6039, LSTM 0.4195, DeepLOB 0.5515, TLOB 0.4657. Old latency table: "Sharpe" LightGBM 168.67→100.66 vs fake-DeepLOB 148.17→89.36 over 0.05→100 ms. Old validation: permutation F1 0.2964, CV mean F1 0.6006, small-tick transfer F1 0.5623, all marked PASS — `git show 7767a7d^:artifacts/{experiment_a_results,experiment_d_latency_sharpe,overfitting_verification_report}.csv`

### Inferences
- The rel_spread artifact and the leaky momentum baseline both look like predictive skill but are not. The first lets a model memorize the price level; the second is a baseline that peeks at the future. Removing them changed the story from "deep/fast models extract alpha" to "learned models add ~0.04–0.07 macro-F1 over an honest naive rule".
- With the fake DeepLOB curve gone, the latency conclusion reversed. LightGBM's PnL advantage now comes from better signals, not speed.
- **Not stated in the README:** the real-data "Optiver RVP" design (Section 4) also exists to demonstrate leakage, here the 1st-place transductive nearest-neighbour features. Leakage auditing is a theme across the whole project.

### Gaps
- No saved output from the audit script. The exact pre-fix correlation and the per-seed F1 drop from removing rel_spread are not recoverable from files.
- The audit work is dated 2026-10-26 from file mtimes; who ran it, and in what session, is not recorded in git.

---

## 3. Synthetic results (5 seeds, mean ± sample std)

### Takeaway
On synthetic data, LightGBM is the best model (macro-F1 0.584 ± 0.012 at k=20). It beats DeepLOB, LR, LSTM, TLOB and momentum in 5/5 seeds, and Random Forest in 4/5 (+0.006, within noise). The honest momentum baseline reaches 0.519. Skill peaks at k=20, and most of the signal is in levels 1–3 (cumulative L3 imbalance ≈ 49% of gain). Crossing the spread removes ~75% (LightGBM) and ~94% (DeepLOB) of fantasy PnL. The LightGBM–DeepLOB PnL gap exists at zero latency, so it comes from signal quality, not speed.

### Cited Findings
- **Setup:** 5 independent sessions (seeds 0–4), 60,000 events each (~10 min simulated), deep nets capped at 10 epochs (Colab cell uses `--epochs 10`), reproduced with `python scripts/run_all_experiments.py` (~8 min on a CUDA GPU) — [README.md L127, L135](C:/a_Coding/ML_Project_2026/README.md); [notebooks/colab_run_all.ipynb cell 13](C:/a_Coding/ML_Project_2026/notebooks/colab_run_all.ipynb)
- **Exp A (k=20), accuracy / macro-F1 / cross-entropy (mean ± std):**

  | Model | Accuracy | Macro-F1 | CE |
  |---|---|---|---|
  | Naive Flat | 0.6982 ± 0.0133 | 0.2741 ± 0.0031 | 4.812 |
  | Naive Momentum | 0.6454 ± 0.0120 | 0.5192 ± 0.0132 | 0.961 |
  | Logistic Regression | 0.5866 ± 0.0108 | 0.5636 ± 0.0075 | 0.812 |
  | Random Forest | 0.6085 ± 0.0076 | 0.5780 ± 0.0075 | 0.757 |
  | LightGBM | 0.6203 ± 0.0136 | **0.5845 ± 0.0115** | **0.743** |
  | LSTM | 0.5521 ± 0.0485 | 0.5212 ± 0.0233 | 0.859 |
  | DeepLOB | 0.5973 ± 0.0246 | 0.5590 ± 0.0096 | 0.823 |
  | TLOB | 0.5353 ± 0.0261 | 0.4918 ± 0.0177 | 0.965 |

  — [artifacts/experiment_a_results.csv](C:/a_Coding/ML_Project_2026/artifacts/experiment_a_results.csv)
- **Paired per-seed wins (computed from the per-seed CSV), LightGBM minus other model, macro-F1:**

  | vs | Mean difference | Wins |
  |---|---|---|
  | DeepLOB | +0.0254 | 5/5 |
  | Logistic Regression | +0.0209 | 5/5 |
  | Random Forest | +0.0064 | **4/5** |
  | LSTM | +0.0633 | 5/5 |
  | TLOB | +0.0927 | 5/5 |
  | Momentum | +0.0652 | 5/5 |

  Per-seed LightGBM macro-F1: 0.5836, 0.5829, 0.6037, 0.5734, 0.5787 — [artifacts/experiment_a_per_seed.csv](C:/a_Coding/ML_Project_2026/artifacts/experiment_a_per_seed.csv)
- **Why accuracy is below Naive Flat:** balanced class weights trade majority-class accuracy for UP/DOWN recall, so macro-F1 is the comparison metric. The deep nets train ≤10 epochs and "may be under-trained" — [README.md L150](C:/a_Coding/ML_Project_2026/README.md)
- **Exp B (LightGBM by horizon):**

  | k | Macro-F1 | Momentum macro-F1 | Accuracy | Test FLAT share |
  |---|---|---|---|---|
  | 10 | 0.5210 ± 0.0111 | 0.4378 | 0.7155 | 88.3% |
  | 20 | **0.5810 ± 0.0107** | 0.5192 | 0.6140 | 69.8% |
  | 50 | 0.5344 ± 0.0116 | 0.4668 | 0.5422 | 38.7% |
  | 100 | 0.4533 ± 0.0133 | 0.4126 | 0.5075 | 22.1% |

  Skill peaks at k=20, matching the generator's 15–60-event informed bursts. At k=10, 88% FLAT makes UP/DOWN hard to recall — [artifacts/experiment_b_signal_decay.csv](C:/a_Coding/ML_Project_2026/artifacts/experiment_b_signal_decay.csv); [README.md L154-163](C:/a_Coding/ML_Project_2026/README.md)
- **Why Exp B at k=20 (0.5810) differs slightly from Exp A (0.5845):** Exp A's LightGBM uses 200 trees with early stopping 15; Exp B/C use 150 trees with early stopping 10 — [exp_a L84-85](C:/a_Coding/ML_Project_2026/src/experiments/exp_a_predictive_power.py); [exp_b L161-162](C:/a_Coding/ML_Project_2026/src/experiments/exp_b_signal_survival.py)
- **Exp C ablation (LightGBM, k=20):**

  | Model | Features | Macro-F1 |
  |---|---|---|
  | A: spread only | 1 | 0.1605 ± 0.1022 |
  | B: L1 snapshot | 4 | 0.4924 ± 0.0105 |
  | C: + lags | 13 | 0.5408 ± 0.0064 |
  | D: + levels 2–3 | 16 | 0.5703 ± 0.0082 |
  | E: all | 27 | 0.5810 ± 0.0107 |

  Increments: +0.049 from short history, +0.029 from levels 2–3, +0.011 from the rest of the book. Model A carries no information (the spread is always one tick) and collapses to a seed-dependent single class — [artifacts/experiment_c_ablation_results.csv](C:/a_Coding/ML_Project_2026/artifacts/experiment_c_ablation_results.csv); [README.md L169-177](C:/a_Coding/ML_Project_2026/README.md)
- **Exp C gain attribution (share of total split gain, mean ± std over seeds):** imbalance_l3 49.31 ± 1.68%, vpin 7.54%, ofi_pca1 6.31%, slope_bid 5.33%, slope_ask 4.93%, imbalance_l5 4.08%, ofi_lag_1 4.01%, imbalance_l1 3.37%. spread_ticks and mlofi_lvl_1 are 0.0% (mlofi_lvl_1 duplicates ofi_l1). micro_deviation is ~0.02%. The README notes gain is model-internal, not causal, and splits credit arbitrarily among correlated features — [artifacts/experiment_c_feature_importance.csv](C:/a_Coding/ML_Project_2026/artifacts/experiment_c_feature_importance.csv); [README.md L179](C:/a_Coding/ML_Project_2026/README.md)
- **Exp D, execution regimes (1 ms latency; test window 8,980 events ≈ 89 s; mean ± std):**

  | Model | Regime | Net PnL | Trades | Fees | Slippage |
  |---|---|---|---|---|---|
  | LightGBM | Fantasy mid | $1,857.6 ± 135.9 | 658.4 | — | — |
  | LightGBM | Aggressive taker | **$457.2 ± 256.9** | 658.4 | $982.0 | $421.8 |
  | LightGBM | Passive maker | $2.4 ± 30.8 | 4.0 | −$2.0 | — |
  | DeepLOB | Fantasy mid | $1,781.8 ± 141.8 | 796.8 | — | — |
  | DeepLOB | Aggressive taker | $103.6 ± 160.2 | 796.8 | $1,185.9 | $500.6 |
  | DeepLOB | Passive maker | $6.6 ± 173.1 | 10.8 | −$5.4 | — |

  The taker keeps 24.6% (LightGBM) and 5.8% (DeepLOB) of fantasy PnL — [artifacts/experiment_d_execution_strategies.csv](C:/a_Coding/ML_Project_2026/artifacts/experiment_d_execution_strategies.csv); [README.md L185-197](C:/a_Coding/ML_Project_2026/README.md)
- **Per-seed aggressive taker PnL** (LightGBM vs DeepLOB): seed 0 $188.2 vs $195.4 (DeepLOB ahead); seed 1 $639.5 vs $147.4; seed 2 $542.1 vs −$156.3; seed 3 $183.1 vs $73.7; seed 4 $732.9 vs $258.0. LightGBM is ahead in 4/5 — [artifacts/experiment_d_strategies_per_seed.csv](C:/a_Coding/ML_Project_2026/artifacts/experiment_d_strategies_per_seed.csv)
- **Latency sweep, aggressive taker net PnL (mean ± std):**

  | Latency | LightGBM | DeepLOB |
  |---|---|---|
  | 0 ms | $507.8 ± 254.8 | $155.2 ± 156.2 |
  | 0.05 ms | $460.7 | $109.7 |
  | 1 ms | $457.2 | $103.6 |
  | 2 ms | $454.1 | $99.7 |
  | 10 ms | $426.7 | $70.9 |
  | 25 ms | $368.6 | $13.7 |
  | 50 ms | $289.9 | −$75.9 |
  | 100 ms | $118.6 ± 277.7 | −$247.4 ± 209.5 |

  From 0 to 100 ms LightGBM loses ~$389 and DeepLOB ~$403. The 0 → 0.05 ms drop is a discretization effect: at zero latency the order fills on the decision snapshot itself — [artifacts/experiment_d_latency_sharpe.csv](C:/a_Coding/ML_Project_2026/artifacts/experiment_d_latency_sharpe.csv); [README.md L200-213](C:/a_Coding/ML_Project_2026/README.md)
- **Measured inference latency (median across seeds):** LightGBM raw booster 0.138 ms on CPU; DeepLOB 2.365 ms on CUDA with batch 1. Per seed, DeepLOB ranged 2.32–2.64 ms on CUDA. The README interpolates that DeepLOB's extra ~2.3 ms costs ~$10 (≈$109 at 0.14 ms vs ≈$99 at 2.4 ms), against a ~$350 signal-quality gap — [artifacts/experiment_d_inference_latency.csv](C:/a_Coding/ML_Project_2026/artifacts/experiment_d_inference_latency.csv), [..._per_seed.csv](C:/a_Coding/ML_Project_2026/artifacts/experiment_d_inference_latency_per_seed.csv); [README.md L211-213](C:/a_Coding/ML_Project_2026/README.md)
- **Sharpe values in the CSVs:** 100 ms bars, annualized, e.g. LightGBM taker 636.5 ± 359.7. They are kept only for relative comparison and are not reported in the README tables — [experiment_d_execution_strategies.csv](C:/a_Coding/ML_Project_2026/artifacts/experiment_d_execution_strategies.csv); [README.md L198](C:/a_Coding/ML_Project_2026/README.md)
- **Validation report** (single session, seed 42, 60,000 events; from `verify_robustness_and_leakage.py`):

  | Check | Value | Criterion | Result |
  |---|---|---|---|
  | Generalization gap (F1 train − test) | 0.1212 | < 0.25 | PASS |
  | Test MCC | 0.4254 | > 0 | PASS |
  | Cohen's kappa | 0.4001 | > 0 | PASS |
  | Macro-F1 lift over momentum | +0.0956 | > 0 | PASS |
  | Purged CV mean macro-F1 | 0.5879 | fold std < 0.05 | PASS |
  | Purged CV min MCC | 0.3862 | every fold > 0 | PASS |
  | Permutation macro-F1 | 0.2880 | ≤ 0.45 | PASS |
  | Permutation MCC | −0.0255 | \|MCC\| ≤ 0.10 | PASS |
  | Small-tick transfer macro-F1 | 0.5088 | — | PASS |
  | Small-tick transfer MCC | 0.3442 | > 0 | PASS |

  — [artifacts/overfitting_verification_report.csv](C:/a_Coding/ML_Project_2026/artifacts/overfitting_verification_report.csv)
- **Purged CV details:** `PurgedTimeSeriesSplit(n_splits=5, horizon_k=20, embargo_pct=0.01)` yields **4** test folds (i = 1..4). Training uses data before the test fold (purged by k) plus data after the test fold and a 1% embargo. The README calls it "purged 5-fold CV"; the submission README says "4-Fold" — [src/data/purged_cv.py L20-47](C:/a_Coding/ML_Project_2026/src/data/purged_cv.py); [scripts/verify_robustness_and_leakage.py L61-62](C:/a_Coding/ML_Project_2026/scripts/verify_robustness_and_leakage.py); [submission/README.md](C:/a_Coding/ML_Project_2026/submission/README.md)
- **C++ benchmark** (GCC 15.2, `-O3 -march=native`): throughput 398,975 snapshots/s; feature extraction P50 0 (below timer resolution) and P99 0.2 µs; tree inference P50 1.8 µs and P99 6.1 µs; end-to-end P50 1.9, P90 4.6, P99 6.3, P99.9 15.1 µs. Separately, commit `b4bafb7` claims a 2.41× speedup to 271 ns per event for SoA + AVX2 + branchless trees in `optimization_comparison.cpp`, also on a synthetic ensemble — [artifacts/cpp_latency_benchmarks.csv](C:/a_Coding/ML_Project_2026/artifacts/cpp_latency_benchmarks.csv); [README.md L241-249](C:/a_Coding/ML_Project_2026/README.md); commit message `b4bafb7`
- **Submission package:** an executed notebook plus `submission.csv` with 8,980 purged test events of one synthetic session (seed 42). Columns are class probabilities, predicted label, VPIN and a passive-maker action (POST_PASSIVE_BID/ASK, HOLD_FLAT, CANCEL_VPIN_TOXIC) — [submission/README.md](C:/a_Coding/ML_Project_2026/submission/README.md)

### Inferences
- The synthetic conclusions describe the generator, which the README states plainly. The informed regime skews market-order direction (85/15), limit insertions (70/30) and cancellations, so depth imbalance (L3) and true-aggressor VPIN are near-direct readouts of the hidden state. That explains imbalance_l3 at ~49% of gain and VPIN in second place.
- Across seeds, the PnL std (~$257 for the LightGBM taker) is comparable to the mean ($457). Only the sign and relative ordering of the PnL results are meaningful.
- The small-tick transfer check passes on MCC (0.344) even though macro-F1 drops to 0.509. Here "PASS" means positive transfer, not equal performance.

### Gaps
- The README does not record the seeds or settings behind the C++ numbers beyond the compiler flags, and the 271 ns optimization result has no saved CSV.
- No confidence intervals or significance tests beyond per-seed std and win counts.

---

## 4. Real-data results so far (PARTIAL; read 2026-10-30 13:32–13:37 IST)

### Takeaway
At the time of reading: Optiver RVP **finished**. FI-2010 had finished k=10 and was training DeepLOB for k=20. BTC had finished segment 0 of 4 and was midway through segment 1. Optiver Close had trained the static model and the first of six retrain blocks. Early signals:
- **BTC:** learned models beat the naive baselines (LightGBM macro-F1 0.488 vs momentum 0.436 on segment 0), but every taker strategy loses heavily after the 4 bp taker fee.
- **FI-2010:** our DeepLOB comes within ~1.6 pp of the published accuracy at k=10.
- **Optiver RVP:** past-only nearest-neighbour features help a little; the winner's transductive version adds a further gain that comes from future leakage; shuffled K-fold is optimistic.

### Cited Findings
- **Running processes at 13:37 IST:**
  - pid 30048 `run_all_experiments.py --source btc` (started 13:09:26)
  - pid 28980 `fi2010_benchmark --epochs 15` (started 13:09:26)
  - pid 3432 `optiver_close` (started 13:23:04)
  - pid 14296 `sync_checkpoints.py --loop 600` (started 13:23:04)
  - No `optiver_rvp` process (finished). Many joblib/loky worker processes started 13:21 are still alive.
  - All `logs/*.err` files are 0 bytes.

  — `Get-CimInstance Win32_Process` output; [logs/pids.txt](C:/a_Coding/ML_Project_2026/logs/pids.txt) ("btc=30048 fi=28980 ... optiver_close=3432 optiver_rvp=24448 sync=14296")
- **BTC segment 0 (complete; checkpoint `artifacts/btc/sessions/btc_segment_0_k20_e8.pkl`, 13:22):** 257,682 rows (1 s snapshots); train 180,258, val 38,632, test 38,633 rows (test ≈ 38,634 s ≈ 10.7 h). Test class shares FLAT/UP/DOWN 0.300/0.374/0.326 — [logs/btc_run.log](C:/a_Coding/ML_Project_2026/logs/btc_run.log); pickle read with pandas
- **BTC segment 0, Exp A (k=20 rows = 20 s), macro-F1 / accuracy:**

  | Model | Macro-F1 | Accuracy |
  |---|---|---|
  | Momentum | 0.4363 | 0.4386 |
  | Flat | 0.1538 | 0.2999 |
  | Logistic Regression | 0.4572 | 0.4693 |
  | Random Forest | 0.4787 | 0.4917 |
  | LightGBM | **0.4882** | 0.5000 |
  | LSTM | 0.4351 | 0.4804 |
  | DeepLOB | 0.4214 | 0.4214 |
  | TLOB | 0.4684 | 0.4742 |

  DeepLOB is below momentum on this segment — [btc_segment_0_k20_e8.pkl "a"](C:/a_Coding/ML_Project_2026/artifacts/btc/sessions/btc_segment_0_k20_e8.pkl); [logs/btc_run.log](C:/a_Coding/ML_Project_2026/logs/btc_run.log)
- **BTC segment 0, Exp B (LightGBM vs momentum macro-F1):** k=10: 0.5576 vs 0.4895; k=20: 0.4882 vs 0.4363; k=50: 0.4067 vs 0.3923; k=100: 0.3570 vs 0.3598. Skill decays monotonically and LightGBM falls to momentum level at 100 s — same sources
- **BTC segment 0, Exp C:** spread only 0.2501; L1 snapshot 0.4522; + lags 0.4823; + L2–3 0.4850; all 0.4882. Most of the gain is in the L1 snapshot plus history; depth adds only +0.006. Top gain shares: slope_bid 16.56%, slope_ask 11.93%, imbalance_l10 10.14%, vpin 9.58%, ofi_lag_1 4.71%, micro_dev_lag_2 4.68%, imbalance_l1 4.58%; imbalance_l3 only 1.60% — pickle keys "c", "imp"
- **BTC segment 0, Exp D (base latency setting 1 ms, i.e. the next 1 s snapshot):**

  | Model | Regime | Net PnL | Trades | Fees | Slippage |
  |---|---|---|---|---|---|
  | LightGBM | Fantasy mid | **+$7,707.37** | 16,091 | — | — |
  | LightGBM | Aggressive taker | **−$36,830.67** | 16,091 | $38,583.27 | $7,214.84 |
  | LightGBM | Passive maker | −$1,447.64 | 926 | — | — |
  | DeepLOB | Fantasy mid | +$4,420.00 | 8,734 | — | — |
  | DeepLOB | Aggressive taker | −$19,771.37 | 8,734 | $20,911.85 | — |
  | DeepLOB | Passive maker | −$1,426.43 | 694 | — | — |

  VPIN toxic threshold (train q0.90) = 0.592 — pickle key "d"; [logs/btc_run.log](C:/a_Coding/ML_Project_2026/logs/btc_run.log)
- **BTC segment 0 latency sweep (taker PnL):** LightGBM −$35,795.88 (0 s) → −$36,830.67 (1 s) → −$37,348.23 (2 s) → −$38,204.32 (5 s) → −$39,887.48 (10 s) → −$41,879.51 (30 s). DeepLOB −$19,162.10 → −$19,771.37 → −$20,208.79 → −$21,046.28 → −$21,931.11 → −$22,606.06. Inference: LightGBM 0.138 ms, DeepLOB 5.69 ms (CUDA) — pickle keys "lat", "inf"
- **BTC segment 1 (PARTIAL, Exp A done, Exp B running at 13:37):** test FLAT/UP/DOWN 0.269/0.393/0.339. Macro-F1: momentum 0.4221, LR 0.4621, RF 0.4802, LightGBM 0.4822, LSTM 0.3954, DeepLOB 0.4418, TLOB 0.4666. No aggregated `artifacts/btc/*.csv` exists yet (written only after all 4 segments) — [logs/btc_run.log](C:/a_Coding/ML_Project_2026/logs/btc_run.log); [run_all_experiments.py L252-276](C:/a_Coding/ML_Project_2026/scripts/run_all_experiments.py)
- **Earlier partial BTC run (2026-10-26, log at 18:33), same segment 0:** identical tabular numbers. DeepLOB differed: macro-F1 0.4166 vs 0.4214 now; trades 9,604 vs 8,734; taker −$22,245.25 vs −$19,771.37. Inference was 6.70 ms. That run was interrupted during segment 1 DeepLOB — `C:/Users/aarav/AppData/Local/Temp/claude/C--a-Coding-ML-Project-2026/bf9e129e-6cb1-494f-8f65-c3f399e8b3b6/scratchpad/btc_run.log`
- **FI-2010 Setup 2, k=10 (complete, `artifacts/fi2010/setup2_k10_e15.csv`):** 254,750 train rows (days 1–7), 139,587 test rows (days 8–10). Test FLAT/UP/DOWN 0.707/0.152/0.142.

  | Model | Accuracy | Weighted-F1 | Macro-F1 |
  |---|---|---|---|
  | Naive Flat | 0.7066 | 0.5852 | 0.2760 |
  | Momentum (lag k) | 0.5980 | 0.5980 | 0.3965 |
  | Logistic Regression | 0.7080 | 0.6133 | 0.3361 |
  | LightGBM (144 features) | 0.5314 | 0.5689 | 0.4437 |
  | **DeepLOB (ours, ≤15 epochs)** | **0.8288** | **0.8208** | **0.7205** |
  | DeepLOB (published, Zhang et al. 2019) | 0.8447 | 0.8340 | — |

  — [artifacts/fi2010/setup2_k10_e15.csv](C:/a_Coding/ML_Project_2026/artifacts/fi2010/setup2_k10_e15.csv); [logs/fi2010.log](C:/a_Coding/ML_Project_2026/logs/fi2010.log)
- **FI-2010 k=20 (PARTIAL; DeepLOB still training at 13:37):** test FLAT/UP/DOWN 0.621/0.197/0.183. Flat: acc 0.6205, wF1 0.4752, mF1 0.2553. Momentum: 0.5168 / 0.5168 / 0.3907. LR: 0.6410 / 0.5796 / 0.4254. LightGBM: 0.4724 / 0.4918 / 0.4561. Published DeepLOB for comparison: k=20 74.85% / 72.82%; k=50 80.51% / 80.35% — [logs/fi2010.log](C:/a_Coding/ML_Project_2026/logs/fi2010.log); [fi2010_benchmark.py L33-34](C:/a_Coding/ML_Project_2026/src/experiments/fi2010_benchmark.py)
- **Optiver RVP (COMPLETE at ~13:34; `artifacts/optiver_rvp/rvp_summary.csv`, `rvp_folds.csv`), forward expanding-window CV, 4 folds of 10% each on the recovered time order, RMSPE mean ± std:**

  | Model | RMSPE |
  |---|---|
  | Naive (current RV) | 0.3370 ± 0.0174 |
  | LightGBM (book/trade features) | 0.2326 ± 0.0130 |
  | + NN features, past-only | 0.2213 ± 0.0141 |
  | + NN features, transductive (1st-place style, leaks future) | 0.2127 ± 0.0138 |
  | LightGBM, shuffled K-fold CV | 0.2115 ± 0.0070 |

  The ordering transductive < past-only < base holds in all 4 forward folds. Per fold, transductive / past-only / base: 0.2169 / 0.2258 / 0.2349; 0.2234 / 0.2316 / 0.2451; 0.2180 / 0.2274 / 0.2359; 0.1923 / 0.2005 / 0.2143 — [artifacts/optiver_rvp/rvp_summary.csv](C:/a_Coding/ML_Project_2026/artifacts/optiver_rvp/rvp_summary.csv); [artifacts/optiver_rvp/ckpt/fold_results_4folds.csv](C:/a_Coding/ML_Project_2026/artifacts/optiver_rvp/ckpt/fold_results_4folds.csv); [logs/optiver_rvp.log](C:/a_Coding/ML_Project_2026/logs/optiver_rvp.log)
- **Optiver Close (PARTIAL at 13:37):** 93 features; train rows 4,346,893; holdout 890,999 rows (81 days). Tuned `n_best` = 467 trees (early stopping on the last 40 training days). The static model's holdout predictions are saved. Online retrain block 1 was saved at 13:34; blocks 2–6 are pending (81 days / 12 = 7 blocks; block 0 reuses the static predictions). No `close_holdout_results.csv` (MAE) yet — [logs/optiver_close.log](C:/a_Coding/ML_Project_2026/logs/optiver_close.log); `artifacts/optiver_close/ckpt/{n_best.npy, pred_static.npy (890,999 preds), online_block1.npy}`; [optiver_close.py L136-172](C:/a_Coding/ML_Project_2026/src/experiments/optiver_close.py)
- **Optiver Close static-model gain share (top 30 saved):** market_urgency 11.82%, seconds_in_bucket 4.18%, liquidity_imbalance 3.93%, reference_price_wap_imb 2.91%, signed_imbalance 2.08%, reference_price_bid_price_imb 1.85%, reference_price_ask_price_imb 1.80%, matched_size 1.78%, matched_size_xs_mean_ratio 1.72%, stock_id 1.68%, ..., target_prev_day 1.53% — [artifacts/optiver_close/close_feature_importance.csv](C:/a_Coding/ML_Project_2026/artifacts/optiver_close/close_feature_importance.csv)
- **README status line:** "Their results will be added once the runs finish", and "Synthetic results only, so far" — [README.md L11, L357](C:/a_Coding/ML_Project_2026/README.md)

### Inferences
- **BTC economics:** fantasy PnL per trade is ~$0.48 (LightGBM) and ~$0.51 (DeepLOB). The taker fee is ~$2.40 per trade (4 bp × 0.1 BTC × ~$60k). So the gross edge per trade is roughly one fifth of the fee, and turnover (16k trades in ~10.7 h) turns a statistically real edge into large losses. This matches the project's thesis that statistical predictability need not be economically exploitable. Latency in seconds adds only a few thousand dollars of extra loss on top of the fee drag.
- **BTC attribution vs synthetic:** on BTC, book shape (slopes, L10 imbalance) and VPIN dominate, while L3 imbalance is minor. This is a useful contrast to the synthetic generator, where the planted mechanism makes L3 imbalance dominant.
- **FI-2010:** our DeepLOB reproduces the published k=10 result to within 1.59 pp accuracy and 1.32 pp weighted-F1, which suggests the DeepLOB implementation is sound. In contrast, LightGBM and LR on the 144 handcrafted FI-2010 features score poorly (LightGBM accuracy 0.53 is below the 0.71 flat rate even without class weights). On FI-2010 the raw-sequence deep model clearly wins, the opposite of the synthetic and BTC rankings.
- **RVP:** past-only neighbours improve RMSPE by ~0.011 over the base model. The winner's transductive neighbours add another ~0.009 that is attributable to future information. Shuffled K-fold looks ~0.021 better than forward CV. One confound: the shuffled K-fold variant omits the train-only `stock_rv_target_ratio` feature that the forward variants include ([optiver_rvp.py L257, L264](C:/a_Coding/ML_Project_2026/src/experiments/optiver_rvp.py)).
- **Reproducibility:** GPU nondeterminism changes DeepLOB results between runs (segment 0 DeepLOB macro-F1 0.4166 vs 0.4214), while the tree models reproduce exactly.

### Gaps
- BTC segments 1–3, the BTC aggregates, FI-2010 k=20/50/100 DeepLOB, and Optiver Close holdout MAE (zero baseline, static, static+demean, retrained+demean) were not available at reading time.
- I did not compute the Close holdout MAE from `pred_static.npy` myself; that would need loading the 641 MB `train.csv` while jobs were running.

---

## 5. Datasets and how each experiment uses them

### Takeaway
Five data sources:
- A synthetic generator with a planted informed-flow regime (used for all headline results).
- Coinbase BTC-USD 1 s snapshots (Kaggle martinsn, CC0), rerunning Experiments A–D on 4 segments with real fees.
- FI-2010 Setup 2 (published DeepLOB comparison).
- Optiver Realized Volatility (forward-in-time CV and leaky vs past-only neighbour features).
- Optiver Trading at the Close (winner's date split and 12-day retraining).

### Cited Findings
- **Synthetic generator:** 10 levels; clock increments Exp(mean 10 ms); event mix 40% market / 40% limit / 20% cancel. Large-tick regime: mean queue 800, std 250, spread 1 tick, informed-flow start probability 0.08. Small-tick: queue 120, std 60, spread 3 ticks, probability 0.20. Informed bursts last `integers(15, 60)` events, i.e. 15–59. During a burst, buy probability is 0.85 (or 0.15) vs 0.50; limit insertions favour the informed side 70/30; cancellations hit the opposite side. A market order larger than the L1 queue shifts the book by one tick. Prices are snapped to the tick grid; base price 100, tick 0.01 — [src/data/synthetic_generator.py L33-47, L90-181](C:/a_Coding/ML_Project_2026/src/data/synthetic_generator.py)
- **Coinbase BTC-USD:** Kaggle "High Frequency Crypto Limit Order Book Data" (martinsn, CC0), built with sadighian/crypto-rl. The file `BTC_1sec.csv` (1.76 GB) has 1,030,728 rows from 2021-04-07 11:32:42 UTC to 2021-04-19 09:54:22 UTC with 15 levels per side; the loader uses 10. Prices are rebuilt as `round(mid*(1+distance)/0.01)*0.01`. Quantities in coins = level notional / price. The trade tape is rebuilt from per-interval `buys`/`sells` notional ÷ mid, giving up to one aggregated buy and one sell row per second with true aggressor side. Results are cached to `.L10.npz` — [src/data/crypto_loader.py](C:/a_Coding/ML_Project_2026/src/data/crypto_loader.py); `data/raw/crypto/` listing and CSV head/tail
- **BTC sessions:** 4 contiguous non-overlapping segments of 257,682 rows (~3 days each). Each segment has its own 70/15/15 split. Labels use a per-horizon threshold α_k = the 1/3 quantile of |smoothed k-row return| over training rows only, so one third of training labels are FLAT. The VPIN bucket = average training volume per minute. Tick 0.01; fees 0 bp maker / 4 bp taker ("Coinbase Pro fee schedule (April 2021), top volume tier"); 0.1 BTC lots; ±0.5 BTC; latency grid in whole seconds — [common.py L38-45, L90-121](C:/a_Coding/ML_Project_2026/src/experiments/common.py); [src/data/labels.py](C:/a_Coding/ML_Project_2026/src/data/labels.py); [README.md L226](C:/a_Coding/ML_Project_2026/README.md)
- **FI-2010:** "NoAuction, Z-score" files (Ntakaris et al., 2018), 5 Nasdaq Nordic stocks over 10 days, one row per 10 order-book events. Setup 2 trains on `Train_Dst_NoAuction_ZScore_CF_7` (days 1–7) and tests on `Test_..._CF_7/8/9` (days 8–10), "the same files the DeepLOB authors' public code uses". The last 20% of training rows (minus a k-row purge) is held out for early stopping. To match the benchmark, models use **no class weights**, and the headline "F1" is support-weighted. LightGBM/LR use all 144 features (40 LOB + 104 handcrafted); DeepLOB uses the 40 LOB columns. Labels for k ∈ {10, 20, 30, 50, 100} are remapped 1→UP, 2→FLAT, 3→DOWN. The fetcher extracts only the 4 Setup 2 files (~0.9 GB of the 8 GB archive) — [src/experiments/fi2010_benchmark.py L1-12, L37-45, L63-118](C:/a_Coding/ML_Project_2026/src/experiments/fi2010_benchmark.py); [src/data/fi2010_loader.py](C:/a_Coding/ML_Project_2026/src/data/fi2010_loader.py); [scripts/fetch_datasets.py L25-30, L54-61](C:/a_Coding/ML_Project_2026/scripts/fetch_datasets.py)
- **Optiver Realized Volatility (2021):** the task is to predict the next-10-minute WAP realized volatility per (stock_id, time_id) from the previous 10 minutes of 2-level book and trades; the metric is RMSPE. 112 stock partitions are present locally.
  - Features: WAP1/WAP2 log-return RV over full/last 450/300/150 s, spread, depth, imbalance, trade RV and counts, cross-sectional time_id means and ranks.
  - Time order recovered with the 1st-place trick: price ≈ 0.01 / smallest normalized tick, then a 1-D t-SNE (perplexity 400) of the time_id × stock price matrix, oriented so AMZN (stock 61) rises.
  - Forward expanding folds on the recovered order; LightGBM with 1/y² sample weights (equivalent to RMSPE).
  - Nearest-neighbour RV features are built "transductive" (all time_ids, as the winner did, which reaches into the future) vs past-only; shuffled K-fold is added for contrast.
  - The recovered order is checkpointed because t-SNE may not reproduce across machines.

  — [src/experiments/optiver_rvp.py L1-19, L39-134, L137-193, L196-273](C:/a_Coding/ML_Project_2026/src/experiments/optiver_rvp.py)
- **Optiver Trading at the Close (2023):** the task is to predict the 60 s future WAP move vs a synthetic index (bps) in the last 10 minutes before the Nasdaq close; the metric is MAE.
  - Split follows the 1st-place solution: train date_id < 400, holdout the last 81 days.
  - Features (93): auction imbalances, pairwise price imbalances, pressure terms, strictly-past lags within stock-day, the winner's "magic" features (seconds-bucket group first-value ratios and expanding means, cross-sectional mean-ratio and rank), and the previous day's revealed target.
  - Models: zero baseline; LightGBM (MAE objective, 255 leaves) trained once; and retrained every 12 holdout days on all data seen so far.
  - Post-processing subtracts the cross-sectional mean prediction (equal weights, because index weights are not in train.csv).
  - The table is kept in float32 to fit Colab RAM.

  — [src/experiments/optiver_close.py L1-19, L36-104, L107-173](C:/a_Coding/ML_Project_2026/src/experiments/optiver_close.py)
- **Licensing:** Optiver data is competition data, so each competition's rules must be accepted on kaggle.com. Kaggle rules prohibit redistribution, so raw data is never synced to GitHub — [README.md L231, L302](C:/a_Coding/ML_Project_2026/README.md); [scripts/fetch_datasets.py L1-9, L43-46](C:/a_Coding/ML_Project_2026/scripts/fetch_datasets.py)
- **LOBSTER:** a `lobster_loader.py` exists, but no experiment uses LOBSTER data — [src/data/lobster_loader.py](C:/a_Coding/ML_Project_2026/src/data/lobster_loader.py); README repository structure

### Inferences
- Units differ across sources: synthetic k is in events (~10 ms each), BTC k is in 1 s rows, and FI-2010 k is in rows of 10 events. Horizon comparisons across datasets are therefore not like-for-like in wall-clock time.
- BTC VPIN uses true aggressor-side volume, but at 1 s aggregation. Each bucket's composition is coarse, and trade prices are set to the mid, not actual prints.
- The two Optiver experiments are not LOB direction-classification tasks. They test validation hygiene (temporal ordering and transductive leakage) and online retraining on real microstructure data, which fits the project's audit theme.

### Gaps
- The BTC CSV's 15-level schema details beyond the columns used were not inspected. The README's "~12 days (Apr 2021)" matches the timestamps found.
- RVP: the number of time_ids and rows was not printed in the log.

---

## 6. Infrastructure: checkpointing, resume, sync, Colab, fetching, paths

### Takeaway
Every long run checkpoints at a natural boundary: BTC/synthetic per session, FI-2010 per horizon, Close per training stage, RVP per (fold, model) plus the recovered order and neighbour features. `scripts/sync_checkpoints.py` mirrors `artifacts/` to an orphan `run-state` branch through a separate git worktree, so a run can move between the laptop and Colab. Paths are configurable via `LOB_DATA_DIR`/`LOB_ARTIFACTS_DIR`. **There is a blocking bug:** `src/data/labels.py` is git-ignored, so a fresh clone (including Colab) cannot import the experiment code.

### Cited Findings
- **Configurable paths:** `DATA_DIR = LOB_DATA_DIR or <root>/data`; `ARTIFACTS_DIR = LOB_ARTIFACTS_DIR or <root>/artifacts` — [src/config.py L8-17](C:/a_Coding/ML_Project_2026/src/config.py)
- **Checkpoints:**
  - Runner: `sessions/{name}_k{h}_e{epochs}.pkl` holding the A/B/C/D/lat/inf results; `--fresh` ignores it — [run_all_experiments.py L231-250](C:/a_Coding/ML_Project_2026/scripts/run_all_experiments.py)
  - FI-2010: `fi2010/setup2_k{k}_e{epochs}.csv` — [fi2010_benchmark.py L73-78](C:/a_Coding/ML_Project_2026/src/experiments/fi2010_benchmark.py)
  - Close: `ckpt/{n_best, pred_static, online_block{b}}.npy` — [optiver_close.py L120-161](C:/a_Coding/ML_Project_2026/src/experiments/optiver_close.py)
  - RVP: `ckpt/recovered_time_order.csv`, `nn_features_{past,transductive}.parquet`, `fold_results_{n}folds.csv` (one row per fold × model, skipped when done) — [optiver_rvp.py L202-244](C:/a_Coding/ML_Project_2026/src/experiments/optiver_rvp.py)
- **Sync script:**
  - Push copies files ≤50 MB that are newer or differ in size into the worktree `../ML_Project_2026_run_state`, force-adds them, commits "sync checkpoints <timestamp>", and pushes `run-state`. `--loop 600` repeats every 10 minutes.
  - Restore runs `git archive origin/run-state`, extracts it, and copies only files that do not already exist locally. The docstring says it never overwrites newer files; in practice it never overwrites any existing file.
  - It never deletes files from the branch.

  — [scripts/sync_checkpoints.py L30-32, L39-105](C:/a_Coding/ML_Project_2026/scripts/sync_checkpoints.py)
- **Sync state:** `origin/run-state` has one commit, `cc60039` "sync checkpoints 2026-10-30 13:22:28". It contains the synthetic artifacts, the BTC segment 0 pickle, the FI-2010 k=10 CSV, the Close n_best/pred_static/importance, and the RVP ckpt files. `logs/sync.log`: "[13:23:04] no changes", "[13:33:04] pushed" — `git log run-state`, `git ls-tree origin/run-state`; [logs/sync.log](C:/a_Coding/ML_Project_2026/logs/sync.log)
- **Colab notebook** (`notebooks/colab_run_all.ipynb`), cells in order:
  1. Settings (repo `aaravsin-1/ML_Project_2026`, branch master, artifacts on Drive).
  2. Drive mount and env vars (`LOB_ARTIFACTS_DIR` → `MyDrive/ML_Project_2026/artifacts`); Kaggle credentials from Colab Secrets (`KAGGLE_USERNAME`/`KAGGLE_KEY`, optional `GITHUB_TOKEN`).
  3. `pip install`.
  4. "Resume from earlier progress" (`sync_checkpoints.py --restore`).
  5. `fetch_datasets.py`.
  6. pytest.
  7. Experiment cells 5a–5f with rough T4 times: synthetic ~15 min, BTC ~1.5–2.5 h, FI-2010 ~1–2 h, Close ~30–60 min, RVP ~30–60 min, validation ~5 min.
  8. Results tables.
  9. Zip download (excluding `sessions/` and pickles).
  10. Optional push back to `run-state`.

  — notebook cell sources (cells 0–24)
- **Fetcher:** `scripts/fetch_datasets.py` (replaced a Windows-only PowerShell fetcher in `d6cb84e`) calls the Kaggle CLI per dataset: btc `-f BTC_1sec.csv` ~330 MB, fi2010 ~2.1 GB, optiver_close `train.csv` ~200 MB, optiver_rvp ~2.6 GB. It skips datasets whose marker file exists, extracts only what is needed, and prints the rules URL on 403/"accept" errors — [scripts/fetch_datasets.py L32-87](C:/a_Coding/ML_Project_2026/scripts/fetch_datasets.py)
- **Local orchestration this session:** `logs/run_optiver_{close,rvp}.ps1` wait until `train.csv` exists and no `.zip`/`.kaggle-partial` file remains, then run the experiment with `PYTHONUNBUFFERED=1`, logging to `logs/*.log` — [logs/run_optiver_close.ps1](C:/a_Coding/ML_Project_2026/logs/run_optiver_close.ps1), [logs/run_optiver_rvp.ps1](C:/a_Coding/ML_Project_2026/logs/run_optiver_rvp.ps1)
- **BLOCKING BUG, `labels.py` not tracked:** `.gitignore` line 33 has `data/` (meant for raw datasets), which also matches `src/data/`. `src/data/labels.py` was created after that rule and is ignored: `git check-ignore -v` → `.gitignore:33:data/ src/data/labels.py`. It is absent from `origin/master`. Yet `src/experiments/common.py` L22 and `tests/test_real_data.py` L6 import it. Every experiment (all import `common.py`, including synthetic), the runner, the verification script and the test suite would fail with ModuleNotFoundError on a fresh clone, i.e. on the Colab path. The other `src/data/*.py` files are tracked only because they were committed before the rule existed — `git ls-tree -r origin/master`, [.gitignore L32-33](C:/a_Coding/ML_Project_2026/.gitignore), [common.py L22](C:/a_Coding/ML_Project_2026/src/experiments/common.py)

### Inferences
- The local runs work only because `labels.py` exists in the working tree. The fix is to commit it with `git add -f`, or narrow the ignore rule to `/data/`. Until then, the README's Colab instructions do not work.
- The orphan-branch-plus-worktree design keeps `master` clean and avoids redistributing Kaggle data. Checkpoints are keyed only by name, horizon and epochs, so changing other hyperparameters without `--fresh` would silently reuse stale results.

### Gaps
- I could not verify whether the Colab notebook has been run successfully end to end; no Colab outputs are in the repo.

---

## 7. Limitations: acknowledged in the README, plus others observed in the code

### Takeaway
The README acknowledges six limitations:
1. Results are synthetic only.
2. Test windows are ~90 s.
3. The strategy is naive.
4. The book does not react to our orders.
5. Tuning is limited (≤10 epochs).
6. The C++ benchmark runs a random tree ensemble, not the trained model.

The code shows further caveats: the git-ignored `labels.py`, RVP early stopping on the scored fold, TLOB attention not over features, GPU nondeterminism, an unused `noise_volatility`, docstring over-claims, and vestigial config values.

### Cited Findings
- **README "Known Limitations" (verbatim points):**
  - "Synthetic results only, so far... The real-data experiments are implemented, but their results are not in yet."
  - "Short test windows. About 90 s of trading per seed means PnL is noisy and annualized risk metrics are not meaningful."
  - "Simple strategy. One lot per signal up to a ±500 inventory cap, with no position sizing by confidence and no exit logic."
  - "Simple book impact. The book does not react to our own orders beyond walking the snapshot."
  - "Limited tuning. Deep models get at most 10 epochs and little hyperparameter search."
  - "Separate C++ path. The C++ benchmark runs a random ensemble, not the trained model."

  — [README.md L355-362](C:/a_Coding/ML_Project_2026/README.md)
- **Other README caveats:**
  - The permutation check "only detects label/row misalignment. It cannot prove that features don't look ahead" — [README.md L129](C:/a_Coding/ML_Project_2026/README.md)
  - Gain importance is "not a causal attribution" — [README.md L179](C:/a_Coding/ML_Project_2026/README.md)
  - C++ numbers "are not the latency of the model evaluated above, and they do not include network or market-data parsing" — [README.md L239](C:/a_Coding/ML_Project_2026/README.md)
  - "treat the absolute numbers as indicative" — [README.md L213](C:/a_Coding/ML_Project_2026/README.md)
- **RVP validation:** the validation fold is used for LightGBM early stopping and also scored (`eval_set=[(X_va, y_va)]`, then `rmspe(y[va_mask], ...)`), in both the forward folds and shuffled K-fold. All RVP RMSPEs are therefore somewhat optimistic, though equally so across variants — [optiver_rvp.py L186-193, L254-259, L265-267](C:/a_Coding/ML_Project_2026/src/experiments/optiver_rvp.py)
- **Synthetic generator:**
  - `noise_volatility` is defined for both regimes but never used — [synthetic_generator.py L40, L47](C:/a_Coding/ML_Project_2026/src/data/synthetic_generator.py) (grep found no other use)
  - The docstring claims "Poisson & Hawkes self-exciting order arrival", but arrivals are i.i.d. exponential, and the informed regime is a Bernoulli start with a fixed countdown — [synthetic_generator.py L5-6, L90-103](C:/a_Coding/ML_Project_2026/src/data/synthetic_generator.py)
- **Simulator:** book-walk overflow beyond 10 levels fills at a fixed ±$0.05 from the last level, not scaled to the tick or price (tiny for BTC at ~$60k) — [simulator.py L117-134](C:/a_Coding/ML_Project_2026/src/execution/simulator.py)
- **BTC latency:** the "1 ms" base latency maps to the next 1 s snapshot because arrival = first snapshot with timestamp ≥ t + latency. Zero latency fills on the decision snapshot itself — [simulator.py L74-81](C:/a_Coding/ML_Project_2026/src/execution/simulator.py); [common.py L43](C:/a_Coding/ML_Project_2026/src/experiments/common.py)
- **FI-2010:** test files for days 8–10 are concatenated, and sequence windows for the first test rows use the end of training as context. FI-2010 interleaves 5 stocks, so windows can cross stock boundaries (inherited from the standard benchmark files) — [fi2010_benchmark.py L63-67](C:/a_Coding/ML_Project_2026/src/experiments/fi2010_benchmark.py)
- **Vestigial config:** `src/config.py` dataclasses (alpha 2 bp, deep latency 25 ms, C++ 45 µs, adverse penalty) are unused by the current code — [src/config.py L20-90](C:/a_Coding/ML_Project_2026/src/config.py)

### Inferences
- The most consequential unacknowledged issue is the untracked `labels.py` (Section 6). It breaks reproducibility from GitHub.
- The second most consequential is that the Optiver RVP absolute RMSPEs share an early-stopping-on-test bias. The leak and CV-optimism comparisons (relative differences) are still informative.
- On BTC, a natural next step consistent with the project's framing is a strategy with fewer, higher-confidence trades. The current one-lot-per-signal taker trades ~1,500 times per hour against a fee 5× its per-trade edge.

### Gaps
- I did not check whether the README badge "24 passed" currently holds; tests were not run.
- No documentation of which GPU produced the synthetic results, beyond "CUDA". The BTC inference latency (5.7 ms) vs synthetic (2.4 ms) DeepLOB difference may reflect GPU contention from concurrent jobs; that is unverified.
