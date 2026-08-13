import numpy as np
import pandas as pd

from gold_silver.config import ProjectConfig, SearchConfig
from gold_silver.validation import diebold_mariano_test, holm_bonferroni, split_development_test


def test_split_is_chronological():
    X = pd.DataFrame({"x": np.arange(100)}, index=pd.date_range("2020-01-01", periods=100))
    y = pd.Series(np.arange(100), index=X.index)
    X_dev, X_test, y_dev, y_test = split_development_test(X, y, ProjectConfig(search=SearchConfig(test_fraction=0.2)))
    assert X_dev.index.max() < X_test.index.min()
    assert y_dev.index.max() < y_test.index.min()
    assert len(X_test) == 20


def test_split_gap_does_not_put_test_return_in_development_target():
    X = pd.DataFrame({"x": np.arange(100)}, index=pd.date_range("2020-01-01", periods=100))
    y = pd.Series(np.arange(100) + 1, index=X.index)
    config = ProjectConfig(search=SearchConfig(test_fraction=0.2, gap=1))
    X_dev, X_test, y_dev, y_test = split_development_test(X, y, config)
    assert X_dev.index.max() < X_test.index.min()
    assert y_dev.index.max() < X_test.index.min()
    assert y_dev.iloc[-1] != y_test.iloc[0]


def test_statistical_helpers_are_paired_and_corrected():
    rng = np.random.default_rng(3)
    actual = pd.Series(np.zeros(100))
    good = pd.Series(rng.normal(0, 0.01, 100))
    bad = pd.Series(rng.normal(0, 0.5, 100))
    result = diebold_mariano_test(actual, good, bad)
    assert result["mean_loss_difference_a_minus_b"] < 0
    assert result["p_value_two_sided"] < 0.05
    corrected = holm_bonferroni(np.array([0.01, 0.04, 0.8]))
    assert np.all(corrected >= np.array([0.01, 0.04, 0.8]))
    assert corrected[0] <= corrected[1]
