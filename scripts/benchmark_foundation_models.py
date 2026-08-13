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
    diebold_mariano_test,
    evaluate_locked_test,
    holm_bonferroni,
    paired_bootstrap_sharpe_difference,
    run_walk_forward_search,
    split_development_test,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--chronos-path", required=True)
    parser.add_argument(
        "--chronos2-covariates-path",
        default=None,
        help="Optional local Chronos-2 snapshot; providing it enables the covariate-aware track.",
    )
    parser.add_argument("--timesfm-path", required=True)
    parser.add_argument(
        "--timesfm-covariates-path",
        default=None,
        help="Optional local TimesFM 2.5 snapshot; enables the causal covariate track.",
    )
    args = parser.parse_args()
    os.environ["GOLD_SILVER_CHRONOS_PATH"] = args.chronos_path
    if args.chronos2_covariates_path:
        os.environ["GOLD_SILVER_CHRONOS2_COVARIATES_PATH"] = args.chronos2_covariates_path
    os.environ["GOLD_SILVER_TIMESFM_PATH"] = args.timesfm_path
    if args.timesfm_covariates_path:
        os.environ["GOLD_SILVER_TIMESFM_COVARIATES_PATH"] = args.timesfm_covariates_path

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
        families = ["chronos", "timesfm"]
        if args.chronos2_covariates_path:
            families.insert(1, "chronos2_covariates")
        if args.timesfm_covariates_path:
            families.append("timesfm_covariates")
        for family in families:
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

        # Compare every foundation forecast with the already selected local
        # winner on the same paired locked dates.  The local winner is read
        # from the cached main experiment; no model is selected using this
        # comparison table.
        local_oos = pd.read_csv(
            output / f"{asset}_oos_predictions.csv", index_col=0, parse_dates=True
        )
        local_comparison = pd.read_csv(output / f"{asset}_test_comparison.csv")
        local_winner = local_comparison.loc[local_comparison["selected"], "family"].iloc[0]
        local_prediction = local_oos[local_winner].reindex(y_test.index)
        local_returns = backtest_predictions(
            local_prediction, y_test, config.backtest
        ).equity["net_return"]
        statistical_rows = []
        for family in families:
            foundation_prediction = prediction_frame[family]
            foundation_returns = backtest_predictions(
                foundation_prediction, y_test, config.backtest
            ).equity["net_return"]
            dm = diebold_mariano_test(y_test, local_prediction, foundation_prediction)
            paired = paired_bootstrap_sharpe_difference(
                local_returns,
                foundation_returns,
                annualization=config.backtest.annualization,
            )
            statistical_rows.append({
                "local_winner": local_winner,
                "foundation_family": family,
                **dm,
                **paired,
                "local_lower_squared_error": dm["mean_loss_difference_a_minus_b"] < 0,
            })
        statistical = pd.DataFrame(statistical_rows)
        if not statistical.empty:
            statistical["p_value_holm"] = holm_bonferroni(
                statistical["p_value_two_sided"].to_numpy()
            )
            statistical["significant_5pct_holm"] = statistical["p_value_holm"] < 0.05
        statistical.to_csv(output / f"{asset}_foundation_statistical_tests.csv", index=False)


if __name__ == "__main__":
    main()
