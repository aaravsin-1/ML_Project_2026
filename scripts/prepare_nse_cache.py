"""Builds and verifies the 64-day 1-second 5-level LOB cache for 8 NSE instruments."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.config import ARTIFACTS_DIR
from src.data.nse_loader import load_nse_symbol
from src.microstructure.nse_pipeline import build_nse_segments, prepare_nse_segment
from src.models.deeplob5 import DeepLOB5
from src.models.recurrent import LSTMLOBNet
from src.models.tlob import TLOB

NSE_INSTRUMENTS = [
    ("IDEA", "EQUITY", "ultra_large_tick"),
    ("YESBANK", "EQUITY", "large_tick"),
    ("ITC", "EQUITY", "large_tick"),
    ("RELIANCE", "EQUITY", "medium_tick"),
    ("HDFCBANK", "EQUITY", "medium_tick"),
    ("ICICIBANK", "EQUITY", "medium_tick"),
    ("TCS", "EQUITY", "small_tick"),
    ("NIFTY", "DERIVATIVE", "tiny_tick_fut"),
]


def main() -> None:
    out_dir = ARTIFACTS_DIR / "nse"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary = []

    for sym, asset, tick_class in NSE_INSTRUMENTS:
        ts = time.time()
        d = load_nse_symbol(sym, asset=asset, workers=8)
        mid = (d.lob_20[:, 0] + d.lob_20[:, 2]) / 2.0
        sprd = d.lob_20[:, 0] - d.lob_20[:, 2]
        med_mid = float(np.median(mid))
        tick_bp = d.tick_size / med_mid * 1e4
        one_tick = float(np.mean(np.isclose(sprd, d.tick_size, atol=1e-4))) * 100.0
        med_sprd_bp = float(np.median(sprd / mid * 1e4))
        med_bq1 = float(np.median(d.lob_20[:, 3]))
        # 20s abs mid move within same day
        same20 = d.day_id[:-20] == d.day_id[20:]
        m20 = np.abs((mid[20:][same20] - mid[:-20][same20]) / mid[:-20][same20] * 1e4)
        mean_abs_20s_bp = float(np.mean(m20))
        # L1 imbalance vs 20s move correlation
        imb1 = (d.lob_20[:, 3] - d.lob_20[:, 1]) / (d.lob_20[:, 3] + d.lob_20[:, 1] + 1e-8)
        signed20 = (mid[20:][same20] - mid[:-20][same20]) / mid[:-20][same20] * 1e4
        corr_imb_20s = float(np.corrcoef(imb1[:-20][same20], signed20)[0, 1])

        raw_ticks = d.meta["raw_ticks_total"]
        rec = {
            "symbol": d.symbol,
            "asset": asset,
            "tick_class": tick_class,
            "days": len(d.days),
            "start_date": d.days[0],
            "end_date": d.days[-1],
            "rows_1s": len(d.lob_20),
            "raw_ticks": raw_ticks,
            "median_mid_inr": round(med_mid, 2),
            "tick_inr": d.tick_size,
            "tick_bp": round(tick_bp, 3),
            "one_tick_pct": round(one_tick, 1),
            "median_spread_bp": round(med_sprd_bp, 3),
            "median_bq1": round(med_bq1, 1),
            "mean_abs_20s_bp": round(mean_abs_20s_bp, 3),
            "corr_imb1_20s": round(corr_imb_20s, 4),
            "trades_reconstructed": len(d.trades_df),
        }
        summary.append(rec)
        print(
            f"{d.symbol:<12} days={len(d.days):>2} rows={len(d.lob_20):>9,} ticks={raw_ticks:>10,} "
            f"mid=₹{med_mid:>8.2f} tick={d.tick_size:.2f} ({tick_bp:>5.2f}bp) "
            f"1-tick={one_tick:>5.1f}% |20s|={mean_abs_20s_bp:>5.2f}bp corr(I1,20s)={corr_imb_20s:+.3f} "
            f"({time.time()-ts:.1f}s)"
        )

    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(out_dir / "nse_dataset_summary.csv", index=False)

    # Verification checks on RELIANCE segment 0
    rel = load_nse_symbol("RELIANCE", "EQUITY")
    segs = build_nse_segments(rel, n_segments=4)
    prep = prepare_nse_segment(segs[0])
    idx20 = prep.splits(20)
    assert np.all(segs[0].day_id[idx20["test"]] == segs[0].day_id[idx20["test"] + 20]), "Overnight label leak!"
    assert np.all(segs[0].pos_in_day[idx20["test"]] >= 99), "Overnight sequence window leak!"
    assert segs[0].day_id[idx20["train"]].max() < segs[0].day_id[idx20["val"]].min()
    assert segs[0].day_id[idx20["val"]].max() < segs[0].day_id[idx20["test"]].min()
    assert not np.isnan(prep.tab_feat).any() and not np.isnan(prep.deep_feat).any()
    print(
        f"\n[+] Pipeline verified on RELIANCE seg0: feats={prep.tab_feat.shape} "
        f"(25 core + 5 NSE-native), deep={prep.deep_feat.shape}, "
        f"MLOFI PCA var={np.round(prep.pca_explained_var, 3)} (sum={prep.pca_explained_var.sum():.3f}), "
        f"splits(20s)={len(idx20['train']):,}/{len(idx20['val']):,}/{len(idx20['test']):,}"
    )

    # Verify GPU forward passes for 5-level deep models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 100, 20, device=device)
    for name, m in [
        ("LSTM", LSTMLOBNet(num_features=20)),
        ("DeepLOB5", DeepLOB5()),
        ("TLOB", TLOB(num_features=20, seq_len=100)),
    ]:
        m = m.to(device)
        out = m(x)
        n_p = sum(p.numel() for p in m.parameters())
        print(f"    {name:<10} out={tuple(out.shape)} params={n_p:,} device={device.type}")

    print(f"\n[+] Saved {out_dir / 'nse_dataset_summary.csv'} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
