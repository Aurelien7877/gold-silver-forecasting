#!/usr/bin/env python
"""Benchmark local Chronos and TimesFM on the locked protocol only.

This avoids retraining every local model when a user only wants to refresh the
optional foundation-model comparison. Checkpoints are passed as local paths so
the run is reproducible offline and no weights are committed to Git.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from gold_silver.backtest import backtest_predictions, bootstrap_sharpe_ci
from gold_silver.config import load_config
from gold_silver.data import load_cached_market_data
from gold_silver.features import build_features, make_targets
from gold_silver.validation import (
    evaluate_locked_test,
    run_walk_forward_search,
    split_development_test,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--chronos-path", required=True)
    parser.add_argument("--timesfm-path", required=True)
    args = parser.parse_args()
    os.environ["GOLD_SILVER_CHRONOS_PATH"] = args.chronos_path
    os.environ["GOLD_SILVER_TIMESFM_PATH"] = args.timesfm_path

    config = load_config(args.config)
    config.search.include_foundation_models = True
    raw = load_cached_market_data(config)
    features = build_features(raw, config.features)
    output = Path(config.data.processed_dir)
    output.mkdir(parents=True, exist_ok=True)

    for asset in ("gold", "silver"):
        X, y = make_targets(features, target=asset)
        X_dev, X_test, y_dev, y_test = split_development_test(X, y, config)
        rows = []
        prediction_frame = pd.DataFrame({"realized_return": y_test})
        for family in ("chronos", "timesfm"):
            selection = run_walk_forward_search(X_dev, y_dev, family, config, asset=asset)
            predictions, _ = evaluate_locked_test(
                selection, X_dev, y_dev, X_test, y_test, config
            )
            report = backtest_predictions(
                pd.Series(predictions, index=X_test.index), y_test, config.backtest
            )
            rows.append(
                {
                    "family": family,
                    "validation_sharpe": selection.best_score,
                    **report.metrics,
                    **bootstrap_sharpe_ci(report.equity["net_return"]),
                }
            )
            prediction_frame[family] = predictions
        pd.DataFrame(rows).to_csv(output / f"{asset}_foundation_comparison.csv", index=False)
        prediction_frame.to_csv(output / f"{asset}_foundation_predictions.csv")


if __name__ == "__main__":
    main()
