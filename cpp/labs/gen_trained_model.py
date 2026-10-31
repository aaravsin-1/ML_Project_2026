"""Lab 07 setup: regenerate the trained BTC LightGBM model as C++ from the saved model file.

Needs only `pip install lightgbm` (no BTC data, no torch). Writes cpp/generated/btc_lightgbm_model.hpp,
which cpp/src/trained_model_benchmark.cpp includes.

    python cpp/labs/gen_trained_model.py
"""

import sys
from pathlib import Path

import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from lgbm_to_cpp import generate_cpp  # noqa: E402

model_file = ROOT / "artifacts" / "cpp_trained_model" / "btc_lightgbm_segment0_k20.txt"
booster = lgb.Booster(model_file=str(model_file))
out = ROOT / "cpp" / "generated" / "btc_lightgbm_model.hpp"
info = generate_cpp(booster, booster.current_iteration(), out)
print(f"[+] {out}  {info}")
