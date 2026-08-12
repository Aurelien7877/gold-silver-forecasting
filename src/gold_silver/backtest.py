"""Long/short signal backtesting with explicit transaction costs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import BacktestConfig


@dataclass
class BacktestReport:
    metrics: dict[str, float] = field(default_factory=dict)
    equity: pd.DataFrame = field(default_factory=pd.DataFrame)


def _safe_sharpe(returns: pd.Series, annualization: int) -> float:
    std = returns.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(annualization))


def backtest_predictions(
    predictions: pd.Series | np.ndarray,
    realized_returns: pd.Series | np.ndarray,
    config: BacktestConfig | object,
) -> BacktestReport:
    """Apply sign(prediction) to the next return and charge turnover costs."""
    backtest_config = config if isinstance(config, BacktestConfig) else config.backtest
    pred = pd.Series(predictions, copy=False, dtype=float)
    realized = pd.Series(realized_returns, copy=False, dtype=float)
    if isinstance(predictions, pd.Series) and isinstance(realized_returns, pd.Series):
        joined = pd.concat([pred.rename("prediction"), realized.rename("realized")], axis=1).dropna()
        pred, realized = joined["prediction"], joined["realized"]
    else:
        n = min(len(pred), len(realized))
        pred, realized = pred.iloc[:n], realized.iloc[:n]
    threshold = float(backtest_config.signal_threshold)
    position = pd.Series(np.where(pred > threshold, 1.0, np.where(pred < -threshold, -1.0, 0.0)), index=pred.index)
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * float(backtest_config.transaction_cost_bps) / 10_000.0
    net_return = position * realized - cost
    equity = pd.DataFrame({"prediction": pred, "realized_return": realized, "position": position, "turnover": turnover, "cost": cost, "net_return": net_return})
    cumulative = (1.0 + net_return).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    downside = net_return.where(net_return < 0, 0.0)
    downside_std = downside.std(ddof=1)
    annualization = int(backtest_config.annualization)
    annual_return = float((1.0 + net_return.mean()) ** annualization - 1.0) if len(net_return) else 0.0
    correlation = float(pred.corr(realized)) if len(pred) > 1 and pred.std(ddof=1) > 0 and realized.std(ddof=1) > 0 else 0.0
    metrics = {
        "n_observations": float(len(net_return)),
        "cumulative_return": float(cumulative.iloc[-1] - 1.0) if len(cumulative) else 0.0,
        "annualized_return": annual_return,
        "annualized_volatility": float(net_return.std(ddof=1) * np.sqrt(annualization)) if len(net_return) > 1 else 0.0,
        "sharpe": _safe_sharpe(net_return, annualization),
        "sortino": float(net_return.mean() / downside_std * np.sqrt(annualization)) if downside_std and np.isfinite(downside_std) else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "calmar": float(annual_return / abs(drawdown.min())) if len(drawdown) and drawdown.min() < 0 else 0.0,
        "turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "hit_rate": float((net_return > 0).mean()) if len(net_return) else 0.0,
        "ic": correlation,
    }
    equity["equity"] = cumulative
    equity["drawdown"] = drawdown
    return BacktestReport(metrics=metrics, equity=equity)


def cost_sensitivity(predictions: pd.Series, realized_returns: pd.Series, config: BacktestConfig, costs=(0.0, 5.0, 10.0, 20.0)) -> pd.DataFrame:
    rows = []
    for cost in costs:
        adjusted = BacktestConfig(transaction_cost_bps=float(cost), annualization=config.annualization, signal_threshold=config.signal_threshold)
        report = backtest_predictions(predictions, realized_returns, adjusted)
        rows.append({"transaction_cost_bps": cost, **report.metrics})
    return pd.DataFrame(rows)


def bootstrap_sharpe_ci(
    net_returns: pd.Series | np.ndarray,
    annualization: int = 252,
    n_bootstrap: int = 500,
    block_size: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """Moving-block bootstrap interval for the annualized Sharpe.

    Daily returns are autocorrelated and overlapping market regimes are
    plausible, so independent row resampling is too optimistic.  This is a
    descriptive uncertainty estimate, not a guarantee of future performance.
    """
    values = np.asarray(pd.Series(net_returns, dtype=float).dropna(), dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        return {"sharpe_ci_low": 0.0, "sharpe_ci_high": 0.0, "prob_sharpe_positive": 0.0}
    rng = np.random.default_rng(random_state)
    block_size = max(1, min(int(block_size), len(values)))
    starts = np.arange(len(values))
    sharpe_values = []
    for _ in range(n_bootstrap):
        sampled = []
        while len(sampled) < len(values):
            start = int(rng.choice(starts))
            sampled.extend(values[(start + np.arange(block_size)) % len(values)])
        sample = np.asarray(sampled[:len(values)])
        std = sample.std(ddof=1)
        sharpe_values.append(float(sample.mean() / std * np.sqrt(annualization)) if std > 0 else 0.0)
    quantiles = np.quantile(sharpe_values, [0.025, 0.975])
    return {
        "sharpe_ci_low": float(quantiles[0]),
        "sharpe_ci_high": float(quantiles[1]),
        "prob_sharpe_positive": float(np.mean(np.asarray(sharpe_values) > 0)),
    }


def paired_bootstrap_sharpe_difference(
    net_returns_a: pd.Series | np.ndarray,
    net_returns_b: pd.Series | np.ndarray,
    annualization: int = 252,
    n_bootstrap: int = 1000,
    block_size: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """Block-bootstrap the paired difference in strategy Sharpe ratios."""
    paired = pd.concat(
        [pd.Series(net_returns_a, dtype=float), pd.Series(net_returns_b, dtype=float)],
        axis=1,
    ).dropna()
    values = paired.to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        return {"sharpe_difference": 0.0, "difference_ci_low": 0.0, "difference_ci_high": 0.0, "prob_a_beats_b": 0.0}
    rng = np.random.default_rng(random_state)
    block_size = max(1, min(int(block_size), len(values)))
    starts = np.arange(len(values))

    def sharpe(sample):
        std = sample.std(ddof=1)
        return float(sample.mean() / std * np.sqrt(annualization)) if std > 0 else 0.0

    observed = sharpe(values[:, 0]) - sharpe(values[:, 1])
    differences = []
    for _ in range(n_bootstrap):
        sampled = []
        while len(sampled) < len(values):
            start = int(rng.choice(starts))
            sampled.extend(values[(start + np.arange(block_size)) % len(values)])
        sample = np.asarray(sampled[:len(values)])
        differences.append(sharpe(sample[:, 0]) - sharpe(sample[:, 1]))
    quantiles = np.quantile(differences, [0.025, 0.975])
    return {
        "sharpe_difference": observed,
        "difference_ci_low": float(quantiles[0]),
        "difference_ci_high": float(quantiles[1]),
        "prob_a_beats_b": float(np.mean(np.asarray(differences) > 0)),
    }
