import pandas as pd
import pytest

from gold_silver.models import Chronos2CovariateRegressor, ChronosRegressor


def test_incomplete_local_foundation_checkpoint_fails_with_actionable_error(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    model = ChronosRegressor(model_id=str(checkpoint))
    with pytest.raises(FileNotFoundError, match="no model weights"):
        model._validate_local_checkpoint()


def test_chronos2_context_uses_only_observed_features(monkeypatch):
    monkeypatch.setattr(Chronos2CovariateRegressor, "_load_model", lambda self: None)
    train = pd.DataFrame(
        {
            "gold_return_current": [0.01, -0.02],
            "silver_return_current": [0.03, 0.04],
            "gold_silver_log_ratio": [1.0, 1.1],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    future = pd.DataFrame(
        {
            "gold_return_current": [0.05],
            "silver_return_current": [0.06],
            "gold_silver_log_ratio": [1.2],
        },
        index=pd.date_range("2024-01-03", periods=1, freq="D"),
    )
    model = Chronos2CovariateRegressor(asset="gold", lookback=3).fit(
        train, pd.Series([999.0, 999.0], index=train.index)
    )
    context = model._build_context_dataframe(future)

    assert list(context["target"]) == [0.01, -0.02, 0.05]
    assert context.iloc[-1]["silver_return_current"] == 0.06
    assert 999.0 not in context["target"].to_numpy()
    assert context["timestamp"].is_monotonic_increasing
