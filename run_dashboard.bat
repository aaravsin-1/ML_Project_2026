@echo off
cd /d "%~dp0"
echo Starting LOB Alpha Studio (FastAPI + Live LightGBM/TreeSHAP)...
start "" http://127.0.0.1:8050
python -m app.server --port 8050
