#!/usr/bin/env python
"""Probe one shared multi-output ExtraTrees model for both metals.

The main pipeline keeps separate models because it selects Gold and Silver
independently. This script tests the alternative explicitly on the same folds,
so a shared model is rejected or retained based on validation rather than
because it is architecturally simpler.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit

from gold_silver.backtest import backtest_predictions
from gold_silver.config import load_config
from gold_silver.data import load_cached_market_data
from gold_silver.features import build_features, make_targets_for_assets
from gold_silver.validation import split_development_test


def main() -> None:
    config = load_config("configs/default.yaml")
    features = build_features(load_cached_market_data(config), config.features)
    X, targets = make_targets_for_assets(features)
    X_dev, X_test, y_dev, y_test = split_development_test(X, targets, config)
    splitter = TimeSeriesSplit(
        n_splits=config.search.n_splits,
        test_size=min(config.search.validation_window, len(X_dev) // (config.search.n_splits + 1)),
        gap=config.search.gap,
    )
    validation: dict[str, list[float]] = {"gold": [], "silver": []}
    for train_idx, valid_idx in splitter.split(X_dev):
        model = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=20,
            max_features=0.7,
            random_state=config.search.random_state,
            n_jobs=1,
        ).fit(X_dev.iloc[train_idx], y_dev.iloc[train_idx])
        predictions = model.predict(X_dev.iloc[valid_idx])
        for column, index in (("gold", 0), ("silver", 1)):
            validation[column].append(
                backtest_predictions(
                    pd.Series(predictions[:, index], index=X_dev.iloc[valid_idx].index),
                    y_dev.iloc[valid_idx][column],
                    config.backtest,
                ).metrics["sharpe"]
            )
    model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=20,
        max_features=0.7,
        random_state=config.search.random_state,
        n_jobs=1,
    ).fit(X_dev, y_dev)
    predictions = model.predict(X_test)
    rows = []
    for column, index in (("gold", 0), ("silver", 1)):
        report = backtest_predictions(
            pd.Series(predictions[:, index], index=X_test.index), y_test[column], config.backtest
        )
        rows.append(
            {
                "asset": column,
                "validation_mean_sharpe": float(np.mean(validation[column])),
                "validation_fold_sharpes": str(validation[column]),
                **report.metrics,
            }
        )
    output = Path(config.data.processed_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "joint_model_comparison.csv", index=False)


if __name__ == "__main__":
    main()
