"""Comprehensive Financial and Microstructure Execution Performance Metrics."""

from typing import Dict, List, Optional, Tuple
import numpy as np


class PerformanceMetrics:
    """Calculates risk-adjusted returns, drawdowns, and transaction friction metrics."""

    @staticmethod
    def calculate_metrics(
        equity_curve: np.ndarray,
        initial_capital: float = 1000000.0,
        annualization_factor: float = np.sqrt(252 * 6.5 * 3600)  # High-frequency scale
    ) -> Dict[str, float]:
        """Calculates institutional performance statistics."""
        if len(equity_curve) < 2:
            return {
                "total_return": 0.0,
                "net_pnl": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0
            }

        equity = np.array(equity_curve, dtype=np.float64)
        net_pnl = equity[-1] - initial_capital
        total_return = net_pnl / initial_capital

        # Step-by-step returns
        step_returns = np.diff(equity) / np.maximum(equity[:-1], 1e-8)

        # Mean and standard deviation
        mean_ret = np.mean(step_returns)
        std_ret = np.std(step_returns)

        # Annualized Sharpe
        if std_ret > 1e-9:
            sharpe = (mean_ret / std_ret) * annualization_factor
        else:
            sharpe = 0.0

        # Downside deviation for Sortino
        downside_returns = step_returns[step_returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            sortino = (mean_ret / (downside_std + 1e-9)) * annualization_factor
        else:
            sortino = sharpe

        # Maximum Drawdown
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / np.maximum(running_max, 1e-8)
        max_dd = float(np.max(drawdowns))

        return {
            "net_pnl": float(net_pnl),
            "total_return_pct": float(total_return * 100.0),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown_pct": float(max_dd * 100.0),
            "step_volatility": float(std_ret)
        }
