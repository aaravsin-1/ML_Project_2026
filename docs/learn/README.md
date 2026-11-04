# Learning the project, part by part

These guides explain the whole project from the ground up: the data, how each feature is computed, how every model was built and trained, what each metric means, and what every result says in plain words. Each guide links to the exact code it describes, so you can read the explanation and then the implementation.

## Reading order

| # | Guide | What you'll understand after it |
|:---:|:---|:---|
| 1 | [The big picture](01_big_picture.md) | What an order book is, the question the project asks, and the handful of terms everything else uses |
| 2 | [The data](02_data.md) | All six datasets: what one row looks like, how it becomes a 40-column order book, where it lives |
| 3 | [Labels, splits and leakage](03_labels_splits_validation.md) | What the models predict, how train/validation/test are split in time, and how the project avoids cheating |
| 4 | [Features](04_features.md) | All 27 hand-made features, one by one, with formulas and worked examples, plus the raw input the deep models see |
| 5 | [Models and how they were trained](05_models.md) | Every model from "repeat the last move" to DeepLOB and TLOB: how it works and its exact training settings |
| 6 | [Metrics](06_metrics.md) | Accuracy, macro-F1, MCC, IC, R², RMSPE, bootstrap CIs, McNemar's test, SHAP: what each one measures and how to read it |
| 7 | [Results, explained simply](07_results.md) | Every experiment's numbers and what they mean |
| 8 | [Execution, costs and latency](08_execution_and_latency.md) | How trades are simulated, why profit disappears, and how speed fits in |

Each guide ends with **check yourself** questions; the answers are hidden until you click them. Try answering before you open them.

## Other documents

- [submission/PROJECT_REPORT.md](../../submission/PROJECT_REPORT.md): the final report, the shortest complete argument.
- [submission/Final_Project_Notebook.ipynb](../../submission/Final_Project_Notebook.ipynb): every result, runnable.
- [reports/LOB prediction project explainer.md](../../reports/LOB%20prediction%20project%20explainer.md): the long explainer, with literature comparisons, the audit bugs and a glossary.
- [README.md](../../README.md): every table and command.
