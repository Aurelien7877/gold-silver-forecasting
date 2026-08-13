"""Post-selection robustness diagnostics for financial forecasts.

These diagnostics do not turn one successful period into a claim of universal
SOTA. They quantify data-snooping risk, regime dependence and stability across
historical rolling origins using the same net-return convention as the main
backtest.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .backtest import backtest_predictions
from .config import BacktestConfig


def _sharpe(values: np.ndarray, annualization: int) -> float:
    values = np.asarray(values, dtype=float)
    std = values.std(ddof=1)
    return float(values.mean() / std * np.sqrt(annualization)) if std > 0 else 0.0


def _circular_block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    indices: list[int] = []
    starts = np.arange(n)
    while len(indices) < n:
        start = int(rng.choice(starts))
        indices.extend(((start + np.arange(block_size)) % n).tolist())
    return np.asarray(indices[:n], dtype=int)


def white_reality_check(
    strategy_returns: dict[str, pd.Series],
    benchmark_returns: pd.Series | None = None,
    annualization: int = 252,
    n_bootstrap: int = 1000,
    block_size: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Test the maximum Sharpe among many strategies against a benchmark.

    The bootstrap recenters each strategy's excess return under the null that
    no candidate has skill. Taking the maximum again in every resample accounts
    for the fact that the winner was selected after observing many candidates.
    This is a data-snooping diagnostic, not a guarantee of future performance.
    """
    if not strategy_returns:
        raise ValueError("strategy_returns must contain at least one candidate")
    frame = pd.concat(strategy_returns, axis=1).sort_index().dropna()
    if frame.empty:
        raise ValueError("No common finite observations for the strategy returns")
    if benchmark_returns is None:
        benchmark = pd.Series(0.0, index=frame.index)
    else:
        benchmark = pd.Series(benchmark_returns, dtype=float).reindex(frame.index).fillna(0.0)
    excess = frame.subtract(benchmark, axis=0).to_numpy(dtype=float)
    observed = np.asarray([_sharpe(excess[:, i], annualization) for i in range(excess.shape[1])])
    winner_index = int(np.argmax(observed))
    rng = np.random.default_rng(random_state)
    centered = excess - excess.mean(axis=0, keepdims=True)
    block_size = max(1, min(int(block_size), len(centered)))
    bootstrap_maxima = np.empty(n_bootstrap, dtype=float)
    for iteration in range(n_bootstrap):
        sample = centered[_circular_block_indices(len(centered), block_size, rng)]
        bootstrap_maxima[iteration] = max(
            _sharpe(sample[:, column], annualization) for column in range(sample.shape[1])
        )
    return {
        "winner": str(frame.columns[winner_index]),
        "observed_winner_sharpe": float(observed[winner_index]),
        "observed_max_sharpe": float(observed.max()),
        "bootstrap_max_sharpe_95": float(np.quantile(bootstrap_maxima, 0.95)),
        "p_value_max_sharpe": float(
            (1.0 + np.sum(bootstrap_maxima >= observed.max())) / (n_bootstrap + 1.0)
        ),
        "n_observations": float(len(frame)),
        "n_candidates": float(frame.shape[1]),
        "n_bootstrap": float(n_bootstrap),
        "block_size": float(block_size),
    }


def _group_metrics(frame: pd.DataFrame, config: BacktestConfig) -> dict[str, float]:
    net = frame["net_return"].dropna()
    return {
        "n_observations": float(len(net)),
        "mean_net_return": float(net.mean()) if len(net) else 0.0,
        "compounded_return_on_regime_days": float((1.0 + net).prod() - 1.0) if len(net) else 0.0,
        "sharpe": _sharpe(net.to_numpy(), config.annualization) if len(net) > 1 else 0.0,
        "hit_rate": float((net > 0).mean()) if len(net) else 0.0,
        "mean_turnover": float(frame.loc[net.index, "turnover"].mean()) if len(net) else 0.0,
        "mean_realized_return": float(frame.loc[net.index, "realized_return"].mean()) if len(net) else 0.0,
    }


def regime_performance(
    predictions: pd.Series,
    realized_returns: pd.Series,
    config: BacktestConfig,
    volatility_window: int = 20,
) -> pd.DataFrame:
    """Report selected-strategy performance by ex-post market regimes.

    Direction and volatility labels use realized test outcomes, so this table
    is explicitly descriptive. It is useful for discovering fragility, not for
    tuning the model or making a real-time signal.
    """
    report = backtest_predictions(predictions, realized_returns, config)
    frame = report.equity.copy()
    realized = frame["realized_return"]
    volatility = realized.rolling(volatility_window, min_periods=max(5, volatility_window // 2)).std()
    volatility = volatility.fillna(volatility.median())
    frame["direction"] = np.where(realized >= 0.0, "up", "down")
    frame["volatility"] = np.where(
        volatility >= volatility.median(), "high_volatility", "low_volatility"
    )
    frame["calendar_period"] = frame.index.to_period("Y").astype(str)
    rows: list[dict[str, Any]] = []
    for regime_type in ("direction", "volatility", "calendar_period"):
        for label, group in frame.groupby(regime_type, observed=True):
            rows.append(
                {
                    "regime_type": regime_type,
                    "regime": str(label),
                    **_group_metrics(group, config),
                }
            )
    return pd.DataFrame(rows)


def fixed_origin_stability(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: Any,
    origin_dates: list[str],
    test_window: int,
    gap: int,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Evaluate fixed final parameters on several historical origins.

    This is a stability diagnostic: the parameters were selected once by the
    main development search and are intentionally not retuned per origin.
    Therefore it must not be confused with an independent nested benchmark.
    """
    from sklearn.base import clone

    rows: list[dict[str, Any]] = []
    for origin in origin_dates:
        start = int(X.index.searchsorted(pd.Timestamp(origin)))
        train_end = start - max(0, int(gap))
        stop = start + int(test_window)
        if train_end < 100 or stop > len(X):
            continue
        fitted = clone(estimator).fit(X.iloc[:train_end], y.iloc[:train_end])
        predictions = fitted.predict(X.iloc[start:stop])
        report = backtest_predictions(
            pd.Series(predictions, index=X.iloc[start:stop].index),
            y.iloc[start:stop],
            config,
        )
        rows.append(
            {
                "origin": str(X.index[start].date()),
                "test_end": str(X.index[stop - 1].date()),
                "n_train": float(train_end),
                "n_test": float(stop - start),
                **report.metrics,
            }
        )
    return pd.DataFrame(rows)
