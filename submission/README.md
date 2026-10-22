# Quantitative Competition Submission Package

This folder contains the complete, self-contained **Jupyter Notebook submission** and out-of-sample prediction results for high-frequency Limit Order Book (LOB) directional price trend forecasting.

---

## Folder Contents

* **`LOB_Alpha_Prediction_Pipeline.ipynb`**:
  A fully executed Jupyter notebook with all outputs, markdown explanations, training trajectories, and evaluation dataframes populated.
* **`submission.csv`**:
  Competition prediction file containing 3,000 out-of-sample test events with class probabilities, predicted directional labels, rolling VPIN toxicity metrics, and tactical execution actions.

---

## Notebook Architecture & Pipeline Sections

1. **Section 1: High-Frequency Market Data Simulation**:
   - Generates 20,000 continuous double-auction events across 10 depth levels.
   - Models large-tick market microstructure with realistic spreads and order arrival clustering.
2. **Section 2: Microstructural Feature Engineering**:
   - Computes Cont et al. Level-1 OFI and Multi-Level OFI with PCA dimensionality reduction.
   - Calculates Easley et al. Volume-Synchronized Probability of Informed Trading (VPIN) on a volume clock.
   - Extracts Stoikov Micro-Price deviation, depth imbalances ($L_1, L_3, L_5, L_{10}$), and book slopes.
3. **Section 3: Model Progression Hierarchy (Tiers 0 to 5)**:
   - Benchmarks Tier 0 (Naïve Flat), Tier 1 (Logistic Regression), Tier 2 (Random Forest), Tier 3 (LightGBM), and Tier 5 (DeepLOB & TLOB).
   - Reports out-of-sample Accuracy, Macro-F1, Matthews Correlation Coefficient (MCC), and Log Loss.
4. **Section 4: Quantitative Validation & Zero Lookahead Audit**:
   - Executes 4-Fold Purged Time-Series Cross Validation with $k=20$ horizon purging and a 1% embargo buffer.
   - Runs Target Permutation Reality Check (White's Test) proving zero lookahead leakage.
5. **Section 5: High-Fidelity Double-Auction Execution Simulation**:
   - Contrasts Academic Fantasy Mid-Price vs. Realistic Aggressive Taker vs. Dynamic Passive Maker with VPIN Defense.
   - Calculates Realized Net PnL, Annualized Sharpe Ratio, Max Drawdown, and Fee Schedules.
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
| `vpin_toxicity` | Float | Contemporaneous VPIN toxicity metric in $[0, 1]$. |
| `recommended_action` | String | Tactical execution action (`POST_PASSIVE_BID`, `POST_PASSIVE_ASK`, `HOLD_FLAT`, or `CANCEL_VPIN_TOXIC`). |

---

## How to Run the Notebook

```bash
# Launch Jupyter Notebook
jupyter notebook submission/LOB_Alpha_Prediction_Pipeline.ipynb

# Or execute headlessly via CLI
python scripts/build_and_execute_notebook.py
```
