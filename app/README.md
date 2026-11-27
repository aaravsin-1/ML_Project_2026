# LOB Alpha Studio (Live ML Web Deployment)

Clean, minimal interactive frontend backed by a real **FastAPI + LightGBM + Block PCA + TreeSHAP** deployment (`app/server.py`).

## Run Locally (1 Click or CLI)
Double-click `run_dashboard.bat`, or run:
```bash
python -m app.server --port 8050
```
Then open `http://127.0.0.1:8050`.

## Rebuild Model Bundle (Optional)
If you retrain models and want to refresh `app/bundle/`:
```bash
python -m app.build_bundle
```

## Deploy to the Cloud (Free Public URL in ~2 Minutes)
1. **Hugging Face Spaces (Docker):** Create a new **Docker** Space, push this repository (`Dockerfile` uses port `7860` and lightweight CPU dependencies in `requirements-app.txt`), and it builds automatically into a public URL.
2. **Render / Railway / Fly.io:** Connect this GitHub repository and select `Docker` runtime; no large dataset mount is required because `app/bundle/` (~5 MB) is self-contained.
