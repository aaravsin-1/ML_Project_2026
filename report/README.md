# Project report (LaTeX)

`LOB_ML_Project_Report.pdf` is the built report. To rebuild from the repository root:

```bash
python report/scripts/btc_extra_metrics.py   # ~25 min CPU: retrains tabular BTC models for AUC, log-loss, calibration, timing
python report/scripts/make_tables.py         # LaTeX tables from report/data/
python report/scripts/feature_corr.py        # feature-correlation heatmap
python report/scripts/make_figures.py        # all other figures (reads artifacts/ and data/raw/)
report/build.sh                              # pdflatex twice
```

Set the course name and guide at the top of `report.tex`.
