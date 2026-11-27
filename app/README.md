# LOB Alpha Studio (Live ML Web Deployment)

Interactive web terminal backed by the **46-feature Flagship Hybrid** (**38f LightGBM v2 + 8f Companion Futures + Multi-Horizon `10s/20s` + `TLOB` Dual-Attention Transformer + Block PCA + TreeSHAP**) in [`app/server.py`](server.py).

## Features
- **Live Order Book Replay:** Second-by-second 5-level LOB replay across `RELIANCE`, `NIFTY_FUT`, `NIFTY_FUT` Seg-3 Distribution Shift, and `IDEA`, showing real-time Flagship vs. `TLOB` vs. `LightGBM v2`/`v1` vs. `LSTM` vs. `DeepLOB5` predictions and jump-to-regime presets (`Confident Signal`, `+SSF Futures Rescue`, `v2 Fixes Old Error`, `TLOB Wins (100s)`, `LGBM Wins (1s Flip)`, `±α Boundary Trap`).
- **Interactive Book Sandbox (`+SSF`):** Real-time 46-feature inference (`38` cash + `8` companion futures inputs) with instant **TreeSHAP** attribution and stress presets (`Futures Sweep Leads Cash`).
- **25-Stock Universe & Tick-Regime Explorer:** Complete `25`-instrument (`22` untouched out-of-universe) and `12`-stock `SSF → Cash` benchmark explorer across 5 relative-tick regimes (`0.035 bp` to `7.41 bp`).
- **Model Ladder & Ensembles:** `K = 1..8` heterogeneous subset search, `N = 1..100` averaging regimes, and `10.5 µs` C++ latency comparison.
- **ML Evaluation & Audit:** Class priors, per-class `Precision/Recall/F1` reports, `3×3` confusion matrices, `MCC`/`ECE`/Calibration, Train-Val-Test generalization gaps, and `TreeSHAP` rankings.

## Run Locally (1 Click or CLI)
Double-click [`run_dashboard.bat`](../run_dashboard.bat), or run from the repository root:
```bash
pip install -r requirements-app.txt
python -m app.server --port 8050
```
Then open `http://127.0.0.1:8050`.

## Rebuild Model Bundle (Optional)
If you retrain models and want to refresh `app/bundle/`:
```bash
python -m app.build_bundle
```

## Deploy to the Cloud (Docker)
1. **Hugging Face Spaces (Docker):** Create a **Docker** Space and push this repository (`Dockerfile` uses port `7860` and lightweight CPU dependencies in `requirements-app.txt`).
2. **Render / Railway / Fly.io:** Connect this GitHub repository and select `Docker` runtime; no external dataset mount is required because `app/bundle/` (~35 MB) is self-contained.
