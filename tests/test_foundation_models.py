import pytest

from gold_silver.models import ChronosRegressor


def test_incomplete_local_foundation_checkpoint_fails_with_actionable_error(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    model = ChronosRegressor(model_id=str(checkpoint))
    with pytest.raises(FileNotFoundError, match="no model weights"):
        model._validate_local_checkpoint()
