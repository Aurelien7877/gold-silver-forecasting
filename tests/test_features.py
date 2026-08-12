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


def test_ohlc_features_are_causal_and_keep_available_rows():
    raw = sample_market()
    raw["gold_open"] = raw["gold_close"] * 0.99
    raw["gold_high"] = raw["gold_close"] * 1.01
    raw["gold_low"] = raw["gold_close"] * 0.98
    raw["silver_open"] = raw["silver_close"] * 0.99
    raw["silver_high"] = raw["silver_close"] * 1.01
    raw["silver_low"] = raw["silver_close"] * 0.98
    baseline = build_features(raw, FeatureConfig(min_history=1))
    altered = raw.copy()
    altered.iloc[-1, altered.columns.get_loc("gold_high")] = 99999.0
    changed = build_features(altered, FeatureConfig(min_history=1))
    common = baseline.index[:-1]
    pd.testing.assert_frame_equal(baseline.loc[common], changed.loc[common])
    assert baseline["gold_ohlc_available"].eq(1.0).all()
