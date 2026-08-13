#!/usr/bin/env python
"""Benchmark one shared two-output iTransformer-style model.

The production pipeline selects Gold and Silver independently.  This script
tests the alternative global model on exactly the same chronological folds,
one-day horizon and transaction costs, selecting one shared configuration from
the average of the two validation Sharpes.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from gold_silver.backtest import backtest_predictions
from gold_silver.config import load_config
from gold_silver.data import load_cached_market_data
from gold_silver.features import build_features, make_targets_for_assets
from gold_silver.models import GlobalITransformerRegressor
from gold_silver.validation import split_development_test


def _fold_splitter(X, config):
    test_size = min(config.search.validation_window, len(X) // (config.search.n_splits + 1))
    return TimeSeriesSplit(
        n_splits=config.search.n_splits,
        test_size=test_size,
        gap=config.search.gap,
    )


def main() -> None:
    config = load_config("configs/default.yaml")
    features = build_features(load_cached_market_data(config), config.features)
    X, targets = make_targets_for_assets(features)
    X_dev, X_test, y_dev, y_test = split_development_test(X, targets, config)

    candidates = [
        {"name": "global_itransformer_small", "lookback": 32, "hidden_dim": 32, "n_layers": 1, "epochs": 8, "patience": 3},
        {"name": "global_itransformer_long", "lookback": 64, "hidden_dim": 32, "n_layers": 1, "epochs": 8, "patience": 3},
    ]
    validation_rows = []
    for candidate in candidates:
        params = {key: value for key, value in candidate.items() if key != "name"}
        fold_scores = {"gold": [], "silver": []}
        for train_idx, valid_idx in _fold_splitter(X_dev, config).split(X_dev):
            model = GlobalITransformerRegressor(
                random_state=config.search.random_state,
                **params,
            ).fit(X_dev.iloc[train_idx], y_dev.iloc[train_idx])
            predictions = model.predict(X_dev.iloc[valid_idx])
            for asset, column in (("gold", "gold"), ("silver", "silver")):
                report = backtest_predictions(
                    pd.Series(
                        predictions[:, 0 if column == "gold" else 1],
                        index=X_dev.iloc[valid_idx].index,
                    ),
                    y_dev.iloc[valid_idx][column],
                    config.backtest,
                )
                fold_scores[asset].append(report.metrics["sharpe"])
        validation_rows.append(
            {
                "family": candidate["name"],
                "validation_gold_sharpe": float(np.mean(fold_scores["gold"])),
                "validation_silver_sharpe": float(np.mean(fold_scores["silver"])),
                "validation_joint_sharpe": float(
                    np.mean(fold_scores["gold"] + fold_scores["silver"])
                ),
                "gold_fold_sharpes": str(fold_scores["gold"]),
                "silver_fold_sharpes": str(fold_scores["silver"]),
                "params": str(params),
            }
        )

    validation = pd.DataFrame(validation_rows).sort_values("validation_joint_sharpe", ascending=False)
    selected = validation.iloc[0]
    selected_params = ast.literal_eval(selected["params"])
    model = GlobalITransformerRegressor(
        random_state=config.search.random_state,
        **selected_params,
    ).fit(X_dev, y_dev)
    test_predictions = model.predict(X_test)
    output = Path(config.data.processed_dir)
    output.mkdir(parents=True, exist_ok=True)
    validation.to_csv(output / "global_model_validation.csv", index=False)

    rows = []
    for asset, column_index in (("gold", 0), ("silver", 1)):
        predictions = pd.Series(test_predictions[:, column_index], index=X_test.index)
        report = backtest_predictions(predictions, y_test[asset], config.backtest)
        rows.append(
            {
                "asset": asset,
                "selected_family": selected["family"],
                "validation_joint_sharpe": float(selected["validation_joint_sharpe"]),
                "validation_asset_sharpe": float(selected[f"validation_{asset}_sharpe"]),
                **report.metrics,
            }
        )
        pd.DataFrame(
            {"realized_return": y_test[asset], "global_itransformer": predictions}
        ).to_csv(output / f"{asset}_global_predictions.csv")
    pd.DataFrame(rows).to_csv(output / "global_model_comparison.csv", index=False)


if __name__ == "__main__":
    main()
