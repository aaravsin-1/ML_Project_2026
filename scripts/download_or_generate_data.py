"""Dataset Acquisition and Synthetic Generation Utility."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import DATA_DIR
from src.data.synthetic_generator import SyntheticLOBGenerator


def generate_benchmark_datasets(n_events: int = 50000):
    """Generates synthetic high-frequency LOB benchmarks for rapid development and reproduction."""
    print(f"[+] Generating benchmark datasets in {DATA_DIR} ({n_events} events per session)...")

    # 1. Large-Tick Asset Regime (e.g. INTC, MSFT, Liquid Blue Chips)
    print("    - Simulating Large-Tick Asset (Deep queues, 1-tick spread, high SNR)...")
    gen_large = SyntheticLOBGenerator(base_price=100.0, tick_size=0.01, regime="large_tick", seed=42)
    lob_large, labels_large, trades_large, ts_large = gen_large.generate_session(n_events=n_events)

    np.save(DATA_DIR / "synthetic_large_tick_lob.npy", lob_large)
    np.save(DATA_DIR / "synthetic_large_tick_timestamps_ms.npy", ts_large)
    np.savez(DATA_DIR / "synthetic_large_tick_labels.npz", **{str(k): v for k, v in labels_large.items()})
    trades_large.to_csv(DATA_DIR / "synthetic_large_tick_trades.csv", index=False)

    # 2. Small-Tick Asset Regime (e.g. AMZN, TSLA, Crypto, Volatile Assets)
    print("    - Simulating Small-Tick Asset (Shallow queues, 3-tick spread, noisy)...")
    gen_small = SyntheticLOBGenerator(base_price=100.0, tick_size=0.01, regime="small_tick", seed=101)
    lob_small, labels_small, trades_small, ts_small = gen_small.generate_session(n_events=n_events)

    np.save(DATA_DIR / "synthetic_small_tick_lob.npy", lob_small)
    np.save(DATA_DIR / "synthetic_small_tick_timestamps_ms.npy", ts_small)
    np.savez(DATA_DIR / "synthetic_small_tick_labels.npz", **{str(k): v for k, v in labels_small.items()})
    trades_small.to_csv(DATA_DIR / "synthetic_small_tick_trades.csv", index=False)

    print(f"[+] Successfully generated benchmark datasets in {DATA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquire or generate LOB data.")
    parser.add_argument("--events", type=int, default=50000, help="Number of events to generate")
    args = parser.parse_args()
    generate_benchmark_datasets(n_events=args.events)
