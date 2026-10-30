"""Comprehensive Financial and Microstructure Execution Performance Metrics."""

from typing import Dict, Optional
import numpy as np

TRADING_SECONDS_PER_YEAR = 252 * 6.5 * 3600


class PerformanceMetrics:
    """Calculates risk-adjusted returns, drawdowns, and transaction friction metrics."""

    @staticmethod
    def resample_to_bars(
        equity: np.ndarray,
        timestamps_ms: np.ndarray,
        bar_ms: float
    ) -> np.ndarray:
        """Last equity value in each fixed wall-clock bar (bars with no events carry forward)."""
        bar_id = np.floor((timestamps_ms - timestamps_ms[0]) / bar_ms).astype(np.int64)
        n_bars = int(bar_id[-1]) + 1
        bars = np.full(n_bars, np.nan)
        bars[bar_id] = equity  # later events in the same bar overwrite earlier ones
        # Forward-fill empty bars
        for i in range(1, n_bars):
            if np.isnan(bars[i]):
                bars[i] = bars[i - 1]
        return bars

    @staticmethod
    def calculate_metrics(
        equity_curve: np.ndarray,
        initial_capital: float = 1000000.0,
        timestamps_ms: Optional[np.ndarray] = None,
        bar_ms: float = 1000.0
    ) -> Dict[str, float]:
        """Calculates performance statistics.

        Sharpe is computed on returns over fixed wall-clock bars (default 1 second) and
        annualized with sqrt(bars per trading year). Event-level returns are unevenly
        spaced in time, so annualizing them with a per-second factor (as an earlier
        version did) inflates Sharpe by orders of magnitude. Without timestamps the
        Sharpe is reported per step and is NOT annualized.

        Note: a sample covering only minutes of trading gives a very noisy Sharpe
        estimate; `n_bars` is returned so callers can judge that.
        """
        equity = np.array(equity_curve, dtype=np.float64)
        if len(equity) < 2:
            return {
                "net_pnl": 0.0, "total_return_pct": 0.0, "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0, "max_drawdown_pct": 0.0, "bar_volatility": 0.0, "n_bars": 0
            }

        net_pnl = equity[-1] - initial_capital
        total_return = net_pnl / initial_capital

        if timestamps_ms is not None:
            series = PerformanceMetrics.resample_to_bars(equity, np.asarray(timestamps_ms, dtype=np.float64), bar_ms)
            annualization = np.sqrt(TRADING_SECONDS_PER_YEAR * 1000.0 / bar_ms)
        else:
            series = equity
            annualization = 1.0

        bar_returns = np.diff(series) / np.maximum(series[:-1], 1e-8)
        mean_ret = np.mean(bar_returns) if len(bar_returns) else 0.0
        std_ret = np.std(bar_returns) if len(bar_returns) else 0.0

        sharpe = (mean_ret / std_ret) * annualization if std_ret > 1e-12 else 0.0

        downside_returns = bar_returns[bar_returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.sqrt(np.mean(downside_returns ** 2))
            sortino = (mean_ret / (downside_std + 1e-12)) * annualization
        else:
            sortino = sharpe

        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / np.maximum(running_max, 1e-8)
        max_dd = float(np.max(drawdowns))

        return {
            "net_pnl": float(net_pnl),
            "total_return_pct": float(total_return * 100.0),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown_pct": float(max_dd * 100.0),
            "bar_volatility": float(std_ret),
            "n_bars": int(len(bar_returns))
        }
