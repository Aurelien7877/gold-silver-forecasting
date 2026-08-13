#!/usr/bin/env python
"""Compute paired locked-test statistics from cached foundation predictions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gold_silver.backtest import backtest_predictions, paired_bootstrap_sharpe_difference
from gold_silver.config import load_config
from gold_silver.data import load_cached_market_data
from gold_silver.features import build_features, make_targets
from gold_silver.validation import diebold_mariano_test, holm_bonferroni, split_development_test


def main() -> None:
    config = load_config("configs/default.yaml")
    output = Path(config.data.processed_dir)
    summary_path = Path("reports/experiment_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    features = build_features(load_cached_market_data(config), config.features)
    for asset in ("gold", "silver"):
        X, y = make_targets(features, target=asset)
        _, X_test, _, y_test = split_development_test(X, y, config)
        local_oos = pd.read_csv(output / f"{asset}_oos_predictions.csv", index_col=0, parse_dates=True)
        comparison = pd.read_csv(output / f"{asset}_test_comparison.csv")
        local_winner = comparison.loc[comparison["selected"], "family"].iloc[0]
        local_prediction = local_oos[local_winner].reindex(y_test.index)
        local_returns = backtest_predictions(local_prediction, y_test, config.backtest).equity["net_return"]
        foundation = pd.read_csv(
            output / f"{asset}_foundation_predictions.csv", index_col=0, parse_dates=True
        ).reindex(y_test.index)
        rows = []
        for family in foundation.columns:
            if family == "realized_return":
                continue
            prediction = foundation[family]
            returns = backtest_predictions(prediction, y_test, config.backtest).equity["net_return"]
            dm = diebold_mariano_test(y_test, local_prediction, prediction)
            paired = paired_bootstrap_sharpe_difference(
                local_returns, returns, annualization=config.backtest.annualization
            )
            rows.append({
                "local_winner": local_winner,
                "foundation_family": family,
                **dm,
                **paired,
                "local_lower_squared_error": dm["mean_loss_difference_a_minus_b"] < 0,
            })
        report = pd.DataFrame(rows)
        if not report.empty:
            report["p_value_holm"] = holm_bonferroni(report["p_value_two_sided"].to_numpy())
            report["significant_5pct_holm"] = report["p_value_holm"] < 0.05
        report.to_csv(output / f"{asset}_foundation_statistical_tests.csv", index=False)
        if summary is not None:
            comparison_rows = pd.read_csv(output / f"{asset}_foundation_comparison.csv").to_dict(orient="records")
            summary["assets"].setdefault(asset, {})["foundation_comparison"] = comparison_rows
            summary["assets"][asset]["foundation_statistical_tests"] = report.to_dict(orient="records")
        print(f"Wrote {asset}_foundation_statistical_tests.csv")
    if summary is not None:
        summary.setdefault("model_benchmark", {})["foundation_benchmark"] = {
            "comparison": "data/processed/{asset}_foundation_comparison.csv",
            "predictions": "data/processed/{asset}_foundation_predictions.csv",
            "statistics": "data/processed/{asset}_foundation_statistical_tests.csv",
            "protocol": "same locked dates, one-day horizon, 10 bps turnover cost",
        }
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
