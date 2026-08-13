import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from gold_silver.config import BacktestConfig
from gold_silver.robustness import fixed_origin_stability, regime_performance, white_reality_check


def test_reality_check_accounts_for_candidate_selection():
    index = pd.date_range("2020-01-01", periods=80, freq="D")
    rng = np.random.default_rng(2)
    returns = {
        "candidate_a": pd.Series(rng.normal(0, 0.01, len(index)), index=index),
        "candidate_b": pd.Series(rng.normal(0, 0.01, len(index)), index=index),
    }
    result = white_reality_check(returns, n_bootstrap=100, block_size=4, random_state=3)
    assert result["n_candidates"] == 2.0
    assert 0.0 <= result["p_value_max_sharpe"] <= 1.0


def test_regime_report_is_descriptive_and_complete():
    index = pd.date_range("2020-01-01", periods=80, freq="D")
    predictions = pd.Series(np.where(np.arange(80) % 2, 0.01, -0.01), index=index)
    realized = pd.Series(np.where(np.arange(80) % 3, 0.005, -0.005), index=index)
    report = regime_performance(predictions, realized, BacktestConfig())
    assert set(report["regime_type"]) == {"direction", "volatility", "calendar_period"}
    assert report["n_observations"].sum() > len(index)


def test_fixed_origin_stability_respects_gap_and_window():
    index = pd.date_range("2020-01-01", periods=150, freq="D")
    X = pd.DataFrame({"x": np.arange(150, dtype=float)}, index=index)
    y = pd.Series(np.linspace(-0.01, 0.01, 150), index=index)
    report = fixed_origin_stability(
        X,
        y,
        Ridge(alpha=1.0),
        ["2020-05-01"],
        test_window=20,
        gap=1,
        config=BacktestConfig(transaction_cost_bps=0),
    )
    assert len(report) == 1
    assert report.iloc[0]["n_test"] == 20
