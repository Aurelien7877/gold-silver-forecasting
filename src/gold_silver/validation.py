"""Chronological model search and locked test evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, ParameterGrid, TimeSeriesSplit

from .backtest import backtest_predictions
from .config import ProjectConfig
from .models import candidate_specs


@dataclass
class SearchResult:
    family: str
    best_estimator: Any
    best_params: dict[str, Any]
    best_score: float
    cv_results: pd.DataFrame = field(default_factory=pd.DataFrame)


def net_sharpe_scorer(estimator, X, y) -> float:
    return make_net_sharpe_scorer(ProjectConfig())(estimator, X, y)


def make_net_sharpe_scorer(config: ProjectConfig):
    """Create a scorer that uses the configured costs and signal threshold.

    A closure is necessary because scikit-learn calls scorers with only
    ``(estimator, X, y)``. Keeping the configuration here prevents a search
    from silently reverting to a different backtest protocol.
    """
    def scorer(estimator, X, y) -> float:
        predictions = estimator.predict(X)
        report = backtest_predictions(
            pd.Series(predictions, index=X.index),
            pd.Series(y, index=X.index),
            config.backtest,
        )
        return report.metrics["sharpe"]

    return scorer


def _time_splitter(X: pd.DataFrame, config: ProjectConfig) -> TimeSeriesSplit:
    n = len(X)
    requested = min(config.search.validation_window, max(1, n // (config.search.n_splits + 1)))
    return TimeSeriesSplit(n_splits=config.search.n_splits, test_size=requested, gap=config.search.gap)


def run_walk_forward_search(X: pd.DataFrame, y: pd.Series, model_family: str, config: ProjectConfig, asset: str = "gold") -> SearchResult:
    """Run a compact GridSearchCV using chronological folds and net Sharpe."""
    specs = candidate_specs(
        asset=asset,
        random_state=config.search.random_state,
        include_xgboost=model_family == "xgboost",
        include_foundation_models=config.search.include_foundation_models,
    )
    if model_family not in specs:
        raise ValueError(f"Unknown model family {model_family}; choose from {list(specs)}")
    estimator, grid = specs[model_family]
    if not grid and model_family in {"chronos", "timesfm"}:
        return _run_manual_search(X, y, estimator, [{}], model_family, config)
    if not grid:
        estimator = clone(estimator).fit(X, y)
        score = net_sharpe_scorer(estimator, X, y)
        return SearchResult(model_family, estimator, {}, score, pd.DataFrame([{"mean_test_score": score}]))
    combinations = [
        {key: [value] for key, value in combination.items()}
        for combination in list(ParameterGrid(grid))[: config.search.max_trials]
    ]
    if model_family in {"tsmixer", "patch_tst", "time_mixer", "chronos", "timesfm"}:
        return _run_manual_search(X, y, estimator, combinations, model_family, config)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=combinations,
        scoring=make_net_sharpe_scorer(config),
        cv=_time_splitter(X, config),
        refit=True,
        n_jobs=config.search.n_jobs,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(X, y)
    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    return SearchResult(model_family, search.best_estimator_, search.best_params_, float(search.best_score_), cv_results)


def _run_manual_search(X, y, estimator, combinations, model_family, config):
    """Run the neural grid without joblib orchestration, which is safer on macOS."""
    rows = []
    best_score = float("-inf")
    best_params = {}
    for combination in combinations:
        params = {key: values[0] for key, values in combination.items()}
        fold_scores = []
        for train_idx, valid_idx in _time_splitter(X, config).split(X):
            candidate = clone(estimator).set_params(**params)
            candidate.fit(X.iloc[train_idx], y.iloc[train_idx])
            prediction = candidate.predict(X.iloc[valid_idx])
            fold_scores.append(
                backtest_predictions(
                    pd.Series(prediction, index=X.iloc[valid_idx].index),
                    y.iloc[valid_idx],
                    config.backtest,
                ).metrics["sharpe"]
            )
        score = float(np.mean(fold_scores))
        rows.append({"params": str(params), "mean_test_score": score, "fold_scores": str(fold_scores)})
        if score > best_score:
            best_score, best_params = score, params
    final_estimator = clone(estimator).set_params(**best_params).fit(X, y)
    cv_results = pd.DataFrame(rows).sort_values("mean_test_score", ascending=False)
    return SearchResult(model_family, final_estimator, best_params, best_score, cv_results)


def split_development_test(X: pd.DataFrame, y: pd.Series, config: ProjectConfig):
    split = max(1, int(len(X) * (1.0 - config.search.test_fraction)))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def select_best_family(X_dev: pd.DataFrame, y_dev: pd.Series, config: ProjectConfig, asset: str = "gold") -> tuple[SearchResult, pd.DataFrame]:
    results: list[SearchResult] = []
    families = list(
        candidate_specs(
            asset=asset,
            random_state=config.search.random_state,
            include_xgboost=False,
            include_foundation_models=config.search.include_foundation_models,
        )
    )
    if config.search.include_xgboost:
        try:
            import xgboost  # noqa: F401

            families.append("xgboost")
        except ImportError:
            pass
    baseline = run_walk_forward_search(X_dev, y_dev, "zero", config, asset=asset)
    for family in families:
        # Zero-return is a diagnostic benchmark, never an eligible final model.
        if family == "zero":
            continue
        results.append(run_walk_forward_search(X_dev, y_dev, family, config, asset=asset))
    results.sort(key=lambda result: result.best_score, reverse=True)
    # Keep fitted search results available for a common locked-test comparison
    # without rerunning every hyperparameter search.
    results_by_family = {result.family: result for result in results}
    results[0].candidate_results = results_by_family
    table = pd.DataFrame(
        [
            {"family": baseline.family, "best_score": baseline.best_score, "best_params": str(baseline.best_params), "eligible_final_model": False}
        ]
        + [
            {"family": r.family, "best_score": r.best_score, "best_params": str(r.best_params), "eligible_final_model": True}
            for r in results
        ]
    )
    return results[0], table


def evaluate_locked_test(result: SearchResult, X_dev: pd.DataFrame, y_dev: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, config: ProjectConfig) -> tuple[np.ndarray, dict[str, float]]:
    estimator = clone(result.best_estimator).fit(X_dev, y_dev)
    predictions = estimator.predict(X_test)
    report = backtest_predictions(pd.Series(predictions, index=X_test.index), y_test, config)
    return predictions, report.metrics


def evaluate_fixed_model_walk_forward(estimator, X: pd.DataFrame, y: pd.Series, config: ProjectConfig) -> float:
    """Evaluate fixed hyperparameters on chronological folds without retuning."""
    scores = []
    for train_idx, valid_idx in _time_splitter(X, config).split(X):
        candidate = clone(estimator).fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions = candidate.predict(X.iloc[valid_idx])
        report = backtest_predictions(
            pd.Series(predictions, index=X.iloc[valid_idx].index),
            y.iloc[valid_idx],
            config.backtest,
        )
        scores.append(report.metrics["sharpe"])
    return float(np.mean(scores))


def diebold_mariano_test(
    actual: pd.Series | np.ndarray,
    prediction_a: pd.Series | np.ndarray,
    prediction_b: pd.Series | np.ndarray,
    loss: str = "squared_error",
    hac_lags: int | None = None,
) -> dict[str, float]:
    """One-step Diebold-Mariano test on paired out-of-sample forecasts.

    A negative statistic means model A has lower average loss than model B.
    For a one-day horizon the loss differential has no extra forecast-overlap
    lag correction; this is intentionally a paired test on the locked dates.
    """
    frame = pd.concat(
        [pd.Series(actual, dtype=float), pd.Series(prediction_a, dtype=float), pd.Series(prediction_b, dtype=float)],
        axis=1,
    ).dropna()
    y, a, b = frame.iloc[:, 0].to_numpy(), frame.iloc[:, 1].to_numpy(), frame.iloc[:, 2].to_numpy()
    if loss == "absolute_error":
        differential = np.abs(y - a) - np.abs(y - b)
    elif loss == "squared_error":
        differential = (y - a) ** 2 - (y - b) ** 2
    else:
        raise ValueError("loss must be 'squared_error' or 'absolute_error'.")
    n = len(differential)
    if n > 1:
        # Newey-West long-run variance protects the paired comparison from
        # short-memory autocorrelation in the loss differential.
        lags = min(5, n - 1) if hac_lags is None else min(max(0, int(hac_lags)), n - 1)
        centered = differential - differential.mean()
        gamma0 = float(np.mean(centered * centered))
        long_run_variance = gamma0
        for lag in range(1, lags + 1):
            covariance = float(np.mean(centered[lag:] * centered[:-lag]))
            weight = 1.0 - lag / (lags + 1.0)
            long_run_variance += 2.0 * weight * covariance
        variance = max(0.0, long_run_variance)
    else:
        lags, variance = 0, 0.0
    statistic = float(differential.mean() / np.sqrt(variance / n)) if variance > 0 else 0.0
    return {
        "n_observations": float(n),
        "mean_loss_difference_a_minus_b": float(differential.mean()) if n else 0.0,
        "dm_statistic": statistic,
        "p_value_two_sided": float(2.0 * norm.sf(abs(statistic))),
        "hac_lags": float(lags),
    }


def holm_bonferroni(p_values: pd.Series | np.ndarray) -> np.ndarray:
    """Holm step-down correction for a family of held-out comparisons."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted
