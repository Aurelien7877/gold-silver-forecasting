import numpy as np
import pandas as pd

from gold_silver.models import DirectionalLogisticRegressor


def test_directional_logistic_returns_centered_probability_score():
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("abc"))
    y = pd.Series(np.where(X["a"] > 0, 0.01, -0.01))
    model = DirectionalLogisticRegressor(C=0.1).fit(X, y)
    predictions = model.predict(X.iloc[:10])
    assert predictions.shape == (10,)
    assert np.isfinite(predictions).all()
    assert (predictions >= -0.5).all() and (predictions <= 0.5).all()
    assert np.corrcoef(predictions, X.iloc[:10]["a"])[0, 1] > 0


def test_directional_logistic_threshold_can_abstain():
    rng = np.random.default_rng(12)
    X = pd.DataFrame(rng.normal(size=(160, 3)), columns=list("abc"))
    y = pd.Series(np.where(X["a"] > 0, 0.01, -0.01))
    model = DirectionalLogisticRegressor(C=0.3, prediction_threshold=0.5).fit(X, y)
    predictions = model.predict(X)
    assert np.all(predictions == 0.0)
