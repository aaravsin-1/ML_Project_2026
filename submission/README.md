# Quantitative Competition Submission Package

This folder contains the complete, self-contained **Jupyter Notebook submission** and out-of-sample prediction results for high-frequency Limit Order Book (LOB) directional price trend forecasting.

---

## Folder Contents

* **`LOB_Alpha_Prediction_Pipeline.ipynb`**:
  A fully executed Jupyter notebook with all outputs, markdown explanations, training trajectories, and evaluation dataframes populated.
* **`submission.csv`**:
  Prediction file for the 8,980 purged out-of-sample test events of one synthetic session (seed 42): class probabilities, predicted labels, VPIN, and the passive-maker action. `event_id` is the row index in the session.

---

## Notebook Architecture & Pipeline Sections

1. **Section 1: High-Frequency Market Data Simulation**:
   - Generates 60,000 synthetic events (about 10 minutes) across 10 depth levels in a large-tick regime.
   - The generator plants a hidden informed-flow state, so predictability reflects that construction, not a real market.
2. **Section 2: Microstructural Feature Engineering**:
   - Computes Cont et al. Level-1 OFI and Multi-Level OFI with PCA dimensionality reduction.
   - Calculates Easley et al. Volume-Synchronized Probability of Informed Trading (VPIN) on a volume clock.
   - Extracts Stoikov Micro-Price deviation, depth imbalances ($L_1, L_3, L_5, L_{10}$), and book slopes.
3. **Section 3: Model Progression Hierarchy (Tiers 0 to 5)**:
   - Benchmarks Tier 0 (lag-k momentum, flat), Tier 1 (Logistic Regression), Tier 2 (Random Forest), Tier 3 (LightGBM), and Tier 5 (DeepLOB & TLOB), all with balanced class weights on the same purged test rows.
   - Reports out-of-sample Accuracy, Macro-F1, and Matthews Correlation Coefficient (MCC).
4. **Section 4: Quantitative Validation & Zero Lookahead Audit**:
   - Executes 4-Fold Purged Time-Series Cross Validation with $k=20$ horizon purging and a 1% embargo buffer.
   - Runs a target-permutation check, which catches label/row misalignment (it cannot by itself rule out lookahead in features).
5. **Section 5: High-Fidelity Double-Auction Execution Simulation**:
   - Contrasts fantasy mid-price fills vs. an aggressive taker vs. a passive maker with trade-through fills and a VPIN gate.
   - Reports net PnL, trade counts, fees and slippage over the ~90 s test window.
6. **Section 6: Competition Submission File Generation**:
   - Generates `submission.csv` mapping predictions to execution actions.

---

## Submission CSV Format (`submission.csv`)

| Column Name | Type | Description |
|:---|:---:|:---|
| `event_id` | Integer | Unique identifier for each out-of-sample event. |
| `prob_flat` | Float | Model-predicted probability of no significant mid-price movement. |
| `prob_up` | Float | Model-predicted probability of an upward mid-price movement. |
| `prob_down` | Float | Model-predicted probability of a downward mid-price movement. |
| `predicted_label` | Integer | Predicted class (`0: FLAT`, `1: UP`, `2: DOWN`). |
| `vpin_toxicity` | Float | Volume-clock VPIN in $[0, 1]$ as of the event (CANCEL when above its 90th training percentile). |
| `recommended_action` | String | Tactical execution action (`POST_PASSIVE_BID`, `POST_PASSIVE_ASK`, `HOLD_FLAT`, or `CANCEL_VPIN_TOXIC`). |

---

## How to Run the Notebook

```bash
# Launch Jupyter Notebook
jupyter notebook submission/LOB_Alpha_Prediction_Pipeline.ipynb

# Or execute headlessly via CLI
python scripts/build_and_execute_notebook.py
```
