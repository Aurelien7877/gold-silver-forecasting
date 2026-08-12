import numpy as np
import pandas as pd

from gold_silver.config import FeatureConfig
from gold_silver.features import build_features, make_targets


def sample_market(n=100):
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "gold_close": np.linspace(1500, 1600, n),
        "gold_volume": np.arange(n) + 100,
        "silver_close": np.linspace(17, 20, n),
        "silver_volume": np.arange(n) + 200,
        "dxy_close": np.linspace(100, 95, n),
    }, index=index)


def test_targets_are_next_day_returns():
    features = build_features(sample_market(), FeatureConfig(min_history=1))
    X, y = make_targets(features, "gold")
    expected = features["gold_return"].shift(-1).rename(y.name).loc[y.index]
    pd.testing.assert_series_equal(y, expected, check_names=True)
    assert X.index.max() < features.index.max()


def test_features_do_not_use_future_rows():
    raw = sample_market()
    baseline = build_features(raw, FeatureConfig(min_history=1))
    changed = raw.copy()
    changed.iloc[-1, changed.columns.get_loc("gold_close")] = 99999
    altered = build_features(changed, FeatureConfig(min_history=1))
    common = baseline.index[:-1]
    pd.testing.assert_frame_equal(baseline.loc[common], altered.loc[common])
