"""Portable model bundle and local inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


@dataclass
class ModelBundle:
    models: dict[str, Any]
    feature_columns: list[str]
    config: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "0.1.0"

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output)
        return output

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        return joblib.load(path)


def fit_final_model(X, y, selection_result):
    """Fit the selected estimator on all development data."""
    from sklearn.base import clone

    return clone(selection_result.best_estimator).fit(X, y)


def predict_next(features: pd.DataFrame, model_bundle: ModelBundle) -> dict[str, float]:
    row = features[model_bundle.feature_columns].tail(1)
    if row.empty:
        raise ValueError("No feature row available for inference.")
    return {asset: float(model.predict(row)[0]) for asset, model in model_bundle.models.items()}
