import numpy as np
import pandas as pd

from gold_silver.models import GlobalITransformerRegressor, PatchTSTRegressor, TimeMixerRegressor


def test_sequence_models_do_not_use_future_rows_for_first_prediction():
    rng = np.random.default_rng(7)
    index = pd.date_range("2020-01-01", periods=90)
    X = pd.DataFrame(rng.normal(size=(90, 4)), index=index, columns=list("abcd"))
    y = pd.Series(rng.normal(size=90), index=index)
    X_train, X_test = X.iloc[:70], X.iloc[70:]
    y_train = y.iloc[:70]

    for estimator in (
        PatchTSTRegressor(lookback=12, hidden_dim=8, epochs=1, patience=1, random_state=1),
        TimeMixerRegressor(lookback=12, hidden_dim=8, epochs=1, patience=1, random_state=1),
    ):
        estimator.fit(X_train, y_train)
        original = estimator.predict(X_test)
        altered = X_test.copy()
        altered.iloc[1:, :] = altered.iloc[1:, :] + 1000.0
        changed = estimator.predict(altered)
        assert np.isfinite(original).all()
        assert original[0] == changed[0]


def test_global_itransformer_is_causal_and_has_two_outputs():
    rng = np.random.default_rng(8)
    index = pd.date_range("2020-01-01", periods=90)
    X = pd.DataFrame(rng.normal(size=(90, 4)), index=index, columns=list("abcd"))
    y = pd.DataFrame(rng.normal(size=(90, 2)), index=index, columns=["gold", "silver"])
    model = GlobalITransformerRegressor(
        lookback=12,
        hidden_dim=8,
        n_heads=2,
        epochs=1,
        patience=1,
        random_state=1,
    ).fit(X.iloc[:70], y.iloc[:70])
    original = model.predict(X.iloc[70:])
    altered = X.iloc[70:].copy()
    altered.iloc[1:, :] += 1000.0
    changed = model.predict(altered)
    assert original.shape == (20, 2)
    assert np.isfinite(original).all()
    np.testing.assert_allclose(original[0], changed[0])
