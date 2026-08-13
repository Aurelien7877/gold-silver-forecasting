import numpy as np
import pandas as pd

from gold_silver.backtest import backtest_predictions
from gold_silver.config import BacktestConfig


def test_backtest_charges_turnover_costs():
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    predictions = pd.Series([1.0, -1.0, 1.0, -1.0], index=index)
    realized = pd.Series([0.01, 0.01, 0.01, 0.01], index=index)
    free = backtest_predictions(predictions, realized, BacktestConfig(transaction_cost_bps=0))
    costly = backtest_predictions(predictions, realized, BacktestConfig(transaction_cost_bps=10))
    assert costly.metrics["cumulative_return"] < free.metrics["cumulative_return"]
    # Initial entry is 1 unit; every reversal is a 2-unit turnover.
    assert np.isclose(costly.equity["turnover"].sum(), 7.0)


def test_backtest_converts_log_return_to_simple_return():
    index = pd.date_range("2024-01-01", periods=1, freq="D")
    report = backtest_predictions(
        pd.Series([1.0], index=index),
        pd.Series([np.log(1.10)], index=index),
        BacktestConfig(transaction_cost_bps=0),
    )
    assert np.isclose(report.equity.iloc[0]["net_return"], 0.10)
