"""Where does the BTC round-trip edge go? Splits Experiment E/G's taker round trips into three parts:

    edge at the decision mid  ->  minus the cost of reacting one snapshot (1 s) later  ->  minus crossing the
    spread and walking the book  =  gross edge before fees

Also reports the break-even hit rate for each fee level. With symmetric moves of average size m and a hit rate p,
the expected edge is (2p - 1) m, so breaking even after a round-trip cost c needs p* = 1/2 + c / (2 m).

Reads the round trips Experiment G saved (artifacts/btc/regimes/sessions/*_trades.csv) and the BTC books.

    python scripts/decompose_btc_edge.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS_DIR  # noqa: E402
from src.experiments.common import btc_sessions  # noqa: E402

FEES_BPS = (0, 1, 4)  # per side; 4 bp is Coinbase's top-tier taker fee in Experiments D and E


def decompose_segment(session, trades: pd.DataFrame) -> pd.DataFrame:
    lob, ts = session.lob_40, session.timestamps_ms
    mid = (lob[:, 0] + lob[:, 2]) / 2
    out = []
    for (signal, k, coverage), g in trades.groupby(["signal", "horizon_k", "coverage"]):
        t = g["row"].to_numpy()
        raw = (mid[t + k] - mid[t]) / mid[t] * 1e4
        # the trade's side is not stored; recover it from the signed mid move (rows where the mid moved)
        ok = np.abs(raw) > 1e-9
        t, raw = t[ok], raw[ok]
        side = np.sign(g["mid_move_bps"].to_numpy()[ok] / raw)
        gross = g["gross_bps"].to_numpy()[ok]
        entry = np.searchsorted(ts, ts[t] + 1.0)      # same rule as the simulator: first snapshot >= t + 1 ms
        exit_ = np.searchsorted(ts, ts[t + k] + 1.0)
        delayed = side * (mid[exit_] - mid[entry]) / mid[entry] * 1e4
        out.append({"segment": session.seed, "signal": signal, "horizon_k": k, "coverage": coverage, "trades": len(t),
                    "edge_at_decision_mid_bps": (side * raw).mean(),
                    "edge_after_delay_bps": delayed.mean(),
                    "gross_bps": gross.mean(),
                    "mean_abs_move_bps": np.abs(raw).mean(),
                    "hit_rate_at_decision_mid": float((side * raw > 0).mean()),
                    "hit_rate_after_delay": float((delayed > 0).mean()),
                    "delay_s": float(np.median(ts[entry] - ts[t]) / 1000)})
    return pd.DataFrame(out)


def main() -> None:
    src = ARTIFACTS_DIR / "btc" / "regimes" / "sessions"
    parts = []
    for session in btc_sessions(4):
        trades = pd.read_csv(src / f"{session.name}_trades.csv")
        trades = trades[trades.regime_var == "vpin"]  # each trade is listed once per regime variable
        parts.append(decompose_segment(session, trades))
    per_seg = pd.concat(parts, ignore_index=True)

    cols = [c for c in per_seg.columns if c.endswith(("_bps", "_mid", "_delay", "_s"))]
    summary = (per_seg.groupby(["signal", "horizon_k", "coverage"])
               .apply(lambda d: pd.Series({"trades": d.trades.sum(),
                                           **{c: np.average(d[c], weights=d.trades) for c in cols}}),
                      include_groups=False)
               .reset_index()
               .sort_values(["signal", "horizon_k", "coverage"], ascending=[False, True, False]))
    summary["delay_cost_bps"] = summary.edge_at_decision_mid_bps - summary.edge_after_delay_bps
    summary["crossing_cost_bps"] = summary.edge_after_delay_bps - summary.gross_bps
    summary["delay_share_of_edge"] = summary.delay_cost_bps / summary.edge_at_decision_mid_bps
    for fee in FEES_BPS:
        # cost of a round trip at the decision mid: delay + crossing + two fees
        cost = summary.delay_cost_bps + summary.crossing_cost_bps + 2 * fee
        summary[f"breakeven_hit_rate_fee{fee}"] = 0.5 + cost / (2 * summary.mean_abs_move_bps)

    out = ARTIFACTS_DIR / "btc" / "tradability"
    out.mkdir(parents=True, exist_ok=True)
    per_seg.to_csv(out / "edge_decomposition_per_segment.csv", index=False)
    summary.to_csv(out / "edge_decomposition.csv", index=False)
    pd.set_option("display.width", 250)
    show = ["signal", "horizon_k", "coverage", "trades", "edge_at_decision_mid_bps", "delay_cost_bps",
            "crossing_cost_bps", "gross_bps", "mean_abs_move_bps", "hit_rate_at_decision_mid",
            "breakeven_hit_rate_fee0", "breakeven_hit_rate_fee1", "breakeven_hit_rate_fee4"]
    print(summary[show].round(3).to_string(index=False))
    print(f"\n[+] {out / 'edge_decomposition.csv'}")


if __name__ == "__main__":
    main()
