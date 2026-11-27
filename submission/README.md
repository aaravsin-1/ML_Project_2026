# Final Submission

**Start with the [final project report](PROJECT_REPORT.md).** It covers the question, data, method and results, and concludes on how prediction accuracy, latency and trading costs have to be balanced for a signal to pay.

**Then the [final notebook](Final_Project_Notebook.ipynb)** ([open in Colab](https://colab.research.google.com/github/aaravsin-1/ML_Project_2026/blob/master/submission/Final_Project_Notebook.ipynb)). It documents every experiment with its results and runs a live demo of the pipeline. **Runtime → Run all** takes a few minutes and needs no GPU or credentials. Each experiment can be recomputed from the notebook's `RERUN` settings. Full per-experiment results are in the [main README](../README.md).

## Folder contents

| File | What it is |
|:---|:---|
| `PROJECT_REPORT.md` | The final report |
| `ADDITIONAL_EXPERIMENTS_AND_VERIFICATION.md` | Additional experiments & complete ML audit: error anatomy, IDEA shock forensics, K=1..8 ensembles, N=1..100 averaging, SSF->Cash, 25-stock verification, 3-class priors, per-class F1 reports, confusion matrices, calibration & generalization gaps |
| `Final_Project_Notebook.ipynb` | The final notebook, executed, with every experiment documented. Rebuild it with `python scripts/build_final_notebook.py` |
