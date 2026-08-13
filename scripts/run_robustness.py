#!/usr/bin/env python
"""Generate regime, rolling-origin and data-snooping diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.base import clone

from gold_silver.backtest import backtest_predictions
from gold_silver.config import load_config
from gold_silver.data import load_cached_market_data
from gold_silver.features import build_features, make_targets
from gold_silver.models import candidate_specs
from gold_silver.robustness import fixed_origin_stability, regime_performance, white_reality_check


def main() -> None:
    config = load_config("configs/default.yaml")
    root = Path(".")
    output = root / config.data.processed_dir
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((root / "reports/experiment_summary.json").read_text())
    features = build_features(load_cached_market_data(config), config.features)
    origins = ["2012-01-01", "2015-01-01", "2018-01-01", "2021-01-01", "2023-01-01", "2025-01-01"]

    for asset in ("gold", "silver"):
        X, y = make_targets(features, target=asset)
        predictions = pd.read_csv(
            output / f"{asset}_oos_predictions.csv", index_col=0, parse_dates=True
        )
        foundation_predictions_path = output / f"{asset}_foundation_predictions.csv"
        if foundation_predictions_path.exists():
            foundation_predictions = pd.read_csv(
                foundation_predictions_path, index_col=0, parse_dates=True
            )
            predictions = predictions.join(
                foundation_predictions.drop(columns=["realized_return"], errors="ignore"),
                how="inner",
            )
        comparison = pd.read_csv(output / f"{asset}_test_comparison.csv")
        winner = comparison.loc[comparison["selected"], "family"].iloc[0]
        selected_predictions = predictions[winner]
        regime_performance(selected_predictions, predictions["realized_return"], config.backtest).to_csv(
            output / f"{asset}_regime_performance.csv", index=False
        )

        strategy_returns = {}
        for family in [column for column in predictions.columns if column != "realized_return"]:
            strategy_returns[family] = backtest_predictions(
                predictions[family], predictions["realized_return"], config.backtest
            ).equity["net_return"]
        reality = white_reality_check(
            strategy_returns,
            annualization=config.backtest.annualization,
            n_bootstrap=1000,
            block_size=5,
            random_state=config.search.random_state,
        )
        pd.DataFrame([reality]).to_csv(output / f"{asset}_reality_check.csv", index=False)

        specs = candidate_specs(
            asset=asset,
            random_state=config.search.random_state,
            include_xgboost=config.search.include_xgboost,
            include_foundation_models=False,
        )
        params = summary["assets"][asset]["best_params"]
        estimator = clone(specs[winner][0]).set_params(**params)
        stability = fixed_origin_stability(
            X,
            y,
            estimator,
            origins,
            test_window=252,
            gap=config.search.gap,
            config=config.backtest,
        )
        stability.to_csv(output / f"{asset}_origin_stability.csv", index=False)


if __name__ == "__main__":
    main()
