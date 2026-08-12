"""End-to-end experiment orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.base import clone

from .analysis import correlation_report, summarize_correlations
from .artifacts import ModelBundle, fit_final_model
from .backtest import (
    backtest_predictions,
    bootstrap_sharpe_ci,
    cost_sensitivity,
    paired_bootstrap_sharpe_difference,
)
from .config import ProjectConfig
from .data import download_market_data, load_cached_market_data
from .features import build_features, make_targets
from .models import foundation_model_paths, foundation_model_status
from .validation import (
    diebold_mariano_test,
    evaluate_fixed_model_walk_forward,
    evaluate_locked_test,
    holm_bonferroni,
    select_best_family,
)


def run_experiment(config: ProjectConfig, use_cache: bool = True) -> dict:
    raw = load_cached_market_data(config) if use_cache else download_market_data(config)
    features = build_features(raw, config.features)
    corr = correlation_report(features)
    Path(config.data.processed_dir).mkdir(parents=True, exist_ok=True)
    corr.to_csv(Path(config.data.processed_dir) / "correlations.csv", index=False)
    (Path(config.data.processed_dir) / "correlations_summary.txt").write_text(summarize_correlations(corr), encoding="utf-8")

    summary: dict = {
        "correlations": corr.to_dict(orient="records"),
        "assets": {},
        "model_benchmark": {
            "horizon": "J+1 logarithmic return",
            "walk_forward": {
                "n_splits": config.search.n_splits,
                "validation_window": config.search.validation_window,
                "gap": config.search.gap,
                "test_fraction": config.search.test_fraction,
            },
            "transaction_cost_bps": config.backtest.transaction_cost_bps,
            "foundation_model_packages": foundation_model_status(),
            "foundation_model_checkpoints": foundation_model_paths(),
            "statistical_tests": {
                "forecast_loss": "one-step Diebold-Mariano, squared error, two-sided",
                "multiple_comparison_correction": "Holm-Bonferroni",
                "sharpe_uncertainty": "paired moving-block bootstrap, block_size=5, B=1000",
            },
        },
    }
    models: dict = {}
    feature_names: list[str] | None = None
    datasets: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]] = {}
    selections = {}
    for asset in ("gold", "silver"):
        X, y = make_targets(features, target=asset)
        X_dev, X_test, y_dev, y_test = _split(X, y, config.search.test_fraction)
        selection, leaderboard = select_best_family(X_dev, y_dev, config, asset=asset)
        selections[asset] = selection
        datasets[asset] = (X_dev, X_test, y_dev, y_test)
        predictions, test_metrics = evaluate_locked_test(selection, X_dev, y_dev, X_test, y_test, config)
        test_series = pd.Series(predictions, index=X_test.index)
        report = backtest_predictions(test_series, y_test, config.backtest)
        test_comparison = []
        prediction_map: dict[str, pd.Series] = {}
        return_map: dict[str, pd.Series] = {}
        all_results = getattr(selection, "candidate_results", {selection.family: selection})
        all_results = {"zero": None, **all_results}
        for family, family_result in all_results.items():
            if family == "zero":
                family_predictions = pd.Series(0.0, index=X_test.index)
                family_report = backtest_predictions(family_predictions, y_test, config.backtest)
            else:
                family_predictions, _family_metrics = evaluate_locked_test(
                    family_result, X_dev, y_dev, X_test, y_test, config
                )
                family_predictions = pd.Series(family_predictions, index=X_test.index)
                family_report = backtest_predictions(family_predictions, y_test, config.backtest)
            prediction_map[family] = family_predictions
            return_map[family] = family_report.equity["net_return"]
            test_comparison.append({
                "family": family,
                "selected": family == selection.family,
                **family_report.metrics,
                **bootstrap_sharpe_ci(
                    family_report.equity["net_return"],
                    annualization=config.backtest.annualization,
                ),
            })
        comparison = pd.DataFrame(test_comparison).sort_values("sharpe", ascending=False)
        comparison.to_csv(Path(config.data.processed_dir) / f"{asset}_test_comparison.csv", index=False)
        prediction_frame = pd.DataFrame({"realized_return": y_test, **prediction_map})
        prediction_frame.to_csv(Path(config.data.processed_dir) / f"{asset}_oos_predictions.csv")
        statistical_rows = []
        winner_predictions = prediction_map[selection.family]
        winner_returns = return_map[selection.family]
        for family, candidate_predictions in prediction_map.items():
            if family == selection.family:
                continue
            dm = diebold_mariano_test(y_test, winner_predictions, candidate_predictions)
            paired = paired_bootstrap_sharpe_difference(
                winner_returns,
                return_map[family],
                annualization=config.backtest.annualization,
            )
            statistical_rows.append({
                "winner": selection.family,
                "competitor": family,
                **dm,
                **paired,
                "winner_lower_squared_error": dm["mean_loss_difference_a_minus_b"] < 0,
            })
        statistical_tests = pd.DataFrame(statistical_rows)
        if not statistical_tests.empty:
            statistical_tests["p_value_holm"] = holm_bonferroni(
                statistical_tests["p_value_two_sided"].to_numpy()
            )
            statistical_tests["significant_5pct_holm"] = statistical_tests["p_value_holm"] < 0.05
        statistical_tests.to_csv(Path(config.data.processed_dir) / f"{asset}_statistical_tests.csv", index=False)
        sensitivity = cost_sensitivity(test_series, y_test, config.backtest)
        leaderboard.to_csv(Path(config.data.processed_dir) / f"{asset}_leaderboard.csv", index=False)
        sensitivity.to_csv(Path(config.data.processed_dir) / f"{asset}_cost_sensitivity.csv", index=False)
        report.equity.to_csv(Path(config.data.processed_dir) / f"{asset}_test_equity.csv")
        models[asset] = fit_final_model(X_dev, y_dev, selection)
        feature_names = list(X.columns)
        summary["assets"][asset] = {
            "winner": selection.family,
            "best_params": selection.best_params,
            "validation_score": selection.best_score,
            "test_metrics": test_metrics,
            "test_comparison": comparison.to_dict(orient="records"),
            "statistical_tests": statistical_tests.to_dict(orient="records"),
        }

    # If Gold has a promising predictive winner, explicitly test the same
    # family and hyperparameters on Silver as a transfer experiment.
    gold_selection = selections["gold"]
    silver_X_dev, silver_X_test, silver_y_dev, silver_y_test = datasets["silver"]
    transfer_validation_score = evaluate_fixed_model_walk_forward(
        gold_selection.best_estimator,
        silver_X_dev,
        silver_y_dev,
        config,
    )
    transfer_model = clone(gold_selection.best_estimator).fit(silver_X_dev, silver_y_dev)
    transfer_predictions = transfer_model.predict(silver_X_test)
    transfer_series = pd.Series(transfer_predictions, index=silver_X_test.index)
    transfer_report = backtest_predictions(transfer_series, silver_y_test, config.backtest)
    transfer_report.equity.to_csv(Path(config.data.processed_dir) / "silver_test_gold_winner_equity.csv")
    summary["cross_asset_tests"] = {
        "gold_winner_on_silver": {
            "family": gold_selection.family,
            "params": gold_selection.best_params,
            "validation_score": transfer_validation_score,
            "test_metrics": transfer_report.metrics,
        }
    }

    bundle = ModelBundle(models=models, feature_columns=feature_names or [], config=config.to_dict(), metrics=summary)
    bundle_path = bundle.save(Path("models") / "gold_silver_bundle.joblib")
    summary["bundle_path"] = str(bundle_path)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/experiment_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def _split(X: pd.DataFrame, y: pd.Series, test_fraction: float):
    split = max(1, int(len(X) * (1.0 - test_fraction)))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
