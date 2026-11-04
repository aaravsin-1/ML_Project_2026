# 3. Labels, splits and leakage

Before any model, three decisions shape everything: **what** we predict (the label), **which rows** train and test the model (the split), and **how we make sure the model can't cheat** (leakage controls). Getting these wrong produces results that look great and mean nothing. The project's first version made exactly those mistakes (see the "ten audit bugs" section of the [explainer](../../reports/LOB%20prediction%20project%20explainer.md)).

## The label: UP, FLAT or DOWN

For each row t and horizon k, the code ([src/data/labels.py](../../src/data/labels.py)) does this:

1. Smooth the mid price with a trailing 5-row average: s(t) = average of mid(t−4) ... mid(t). This removes some bid-ask bounce noise.
2. Compute the forward return: r(t) = (s(t+k) − s(t)) / s(t).
3. Compare it with a threshold α:
   - r > +α → **UP** (label 1)
   - r < −α → **DOWN** (label 2)
   - otherwise → **FLAT** (label 0)

**Choosing α.**
- On **BTC**, α is set separately for each horizon so that exactly **one third of training rows are FLAT** (`calibrate_alpha`). Because it's computed on training rows only, it can't peek at the test period. With roughly a third of rows in each class, random guessing scores about 0.333 macro-F1, an easy reference point.
- On **synthetic** data, α is fixed at 1.5 bp.
- On **LOBSTER**, the same one-third rule, but without smoothing (smoothing = 1).
- On **FI-2010**, the dataset's own labels (±0.2% on the average of the next k mids).

**Worked example.** Suppose k = 20, α = 1.2 bp, and the smoothed mid goes from 60,000.00 now to 60,009.00 twenty seconds later. r = 9 / 60,000 = 1.5 bp > 1.2 bp, so the label is UP.

**Why the label is only known later.** The label at row t uses prices up to t+k. So at time t you only know labels up to row t−k. This matters for the momentum baseline (guide 5) and for the split.

## The split: chronological, with a purge gap

Rows are split **in time order**, never shuffled ([purged_split_indices](../../src/data/lob_dataset.py)):

```
|<────────────── train 70% ──────────────>|gap|<─ val 15% ─>|gap|<─ test 15% ─>|gap|
                                            k                k                 k
```

- **Train**: the model learns from these rows.
- **Validation**: used during training to decide when to stop (early stopping) and to pick thresholds. Never used for the final score.
- **Test**: scored once, at the end.
- **The gap (purge).** The last k rows of train have labels that look k rows into the validation period. If kept, the model would be trained on information from the validation period. So those k rows are dropped, and the same before the test period. The final k rows have no future at all, so they're dropped too.
- **Warm-up.** Training starts at row 99, so every row has 100 rows of history behind it. That lets the deep models (which need a 100-row window) and the tabular models train and test on **exactly the same rows**.

**Why not shuffle?** Markets change over time. A shuffled split puts tomorrow's rows in training and yesterday's in test, so the model gets to see the future. The Optiver experiment shows the effect directly: shuffled cross-validation made a plain model look as good as a leaky one.

On a BTC segment of 257,682 rows at k = 20 this gives about 180,000 training rows, 38,600 validation rows and 38,600 test rows (about 10.7 hours of trading).

## Leakage controls: how the project avoids cheating

**Leakage** = information the model wouldn't have in real life sneaking into training or features. The controls:

| Control | What it prevents | Where |
|:---|:---|:---|
| Features use only rows ≤ t | Features peeking at future prices | [pipeline.py](../../src/microstructure/pipeline.py), checked by `test_pipeline_fits_on_training_rows_only` in [tests/test_features.py](../../tests/test_features.py) |
| Everything "fitted" (normaliser, PCA, label threshold, VPIN bucket size) uses training rows only | Test-period statistics shaping the features | `fit(lob[:train_end])` in [common.py](../../src/experiments/common.py) |
| Purged chronological splits | Labels overlapping across splits | [lob_dataset.py](../../src/data/lob_dataset.py) |
| The momentum baseline uses y(t−k), not y(t−1) | A "naive" rule that secretly reads the future | [baselines.py](../../src/models/baselines.py) |
| Shuffled-target control | Bugs that let models score well without real labels | [validation_checks.py](../../src/utils/validation_checks.py), Experiment K |
| Every model scored on the same test rows | Unfair comparisons | `data.splits(k)` shared by all models |

**The shuffled-target control** is worth understanding. You train the same model on labels that have been randomly shuffled, so they carry no information. If the pipeline is honest, that model must score at chance (macro-F1 ≈ 0.33, information coefficient ≈ 0). If it scores well, something in the pipeline leaks. In this project the controls land where they should: macro-F1 0.288 on synthetic data and an average IC of −0.0005 in Experiment K.

**What the control can't catch.** If a *feature* itself looks ahead, the shuffled model still scores at chance, because shuffling breaks the label, not the feature. That's why there's a separate test (`test_pipeline_fits_on_training_rows_only`) that triples every order size after the training period and checks that no feature before it changes. Similar tests cover the BTC labels, the Optiver Close features and the FI-2010 history features.

---

## Check yourself

1. Why is the BTC FLAT threshold computed from training rows only?
2. With k = 50, how many rows are dropped between train and validation, and why?
3. A "naive momentum" rule that predicts y(t−1) scored 0.96 macro-F1 in the first version of the project. Why was that fake?
4. Your shuffled-target control scores macro-F1 0.70. What do you conclude?

<details><summary>Answers</summary>

1. Otherwise the threshold would depend on the size of moves in the test period, which you can't know in advance.
2. 50 rows, because each of the last 50 training labels looks up to 50 rows ahead, into the validation period.
3. y(t−1) is computed from prices up to t−1+k, so it already contains k−1 rows of future prices. It's almost a copy of the answer.
4. The pipeline leaks: something lets the model succeed without real labels (for example misaligned rows). Stop and find it before trusting any result.

</details>

Next: [4. Features](04_features.md)
