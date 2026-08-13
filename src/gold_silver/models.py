"""Candidate estimators for tabular, sequence and foundation-model forecasts.

The sequence estimators deliberately expose the scikit-learn estimator API.  At
each walk-forward fold their ``fit`` method only sees the past, and ``predict``
uses the tail of that fitted history as context for the validation/test rows.
This makes them usable by the same chronological search code as the tabular
models without constructing a future-looking global tensor.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BaselineRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, kind: str = "zero", asset: str = "gold"):
        self.kind = kind
        self.asset = asset

    def fit(self, X, y):
        self.mean_ = float(np.asarray(y).mean())
        return self

    def predict(self, X):
        if self.kind == "zero":
            return np.zeros(len(X))
        if self.kind == "mean":
            return np.full(len(X), self.mean_)
        if self.kind == "last":
            return np.asarray(X[f"{self.asset}_return_current"])
        if self.kind == "moving_average":
            column = f"{self.asset}_return_mean_5"
            return np.asarray(X[column])
        raise ValueError(f"Unknown baseline kind: {self.kind}")


class DirectionalLogisticRegressor(BaseEstimator, RegressorMixin):
    """Predict the next-day direction and expose a centered probability score.

    The deployed backtest only uses the sign of a forecast.  This estimator
    therefore optimizes the directional classification problem and returns
    ``P(up) - 0.5`` so the existing long/short interface can consume it.
    Class balancing is useful because daily positive/negative returns are not
    guaranteed to be exactly symmetric in a finite training window.
    """

    def __init__(
        self,
        C: float = 0.1,
        class_weight: str | None = "balanced",
        prediction_threshold: float = 0.0,
        max_iter: int = 2000,
    ):
        self.C = C
        self.class_weight = class_weight
        self.prediction_threshold = prediction_threshold
        self.max_iter = max_iter

    def fit(self, X, y):
        target = (np.asarray(y, dtype=float) > 0.0).astype(int)
        self.model_ = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=self.C,
                        class_weight=self.class_weight,
                        max_iter=self.max_iter,
                    ),
                ),
            ]
        ).fit(X, target)
        self.classes_ = self.model_.named_steps["model"].classes_
        return self

    def predict(self, X):
        probabilities = self.model_.predict_proba(X)
        if probabilities.shape[1] != 2:
            raise ValueError("DirectionalLogisticRegressor requires both return directions in training data.")
        score = probabilities[:, 1] - 0.5
        threshold = float(self.prediction_threshold)
        return np.where(np.abs(score) >= threshold, score, 0.0)


class TreeBlendRegressor(BaseEstimator, RegressorMixin):
    """Leakage-safe blend of two strong local tree learners.

    ExtraTrees averages many randomized trees and HistGradientBoosting builds
    trees sequentially. Their errors are not identical, so a fixed blend can
    reduce model-specific variance. The blend weight is selected chronologically
    by the outer walk-forward search; no locked-test value is used here.
    """

    def __init__(self, asset: str = "gold", extra_trees_weight: float = 0.75, random_state: int = 42):
        self.asset = asset
        self.extra_trees_weight = extra_trees_weight
        self.random_state = random_state

    def fit(self, X, y):
        if self.asset == "gold":
            extra_params = dict(max_depth=6, max_features=0.7, min_samples_leaf=10)
            gradient_params = dict(max_iter=100, learning_rate=0.03, max_leaf_nodes=7, l2_regularization=0.0)
        else:
            extra_params = dict(max_depth=4, max_features=1.0, min_samples_leaf=10)
            gradient_params = dict(max_iter=100, learning_rate=0.03, max_leaf_nodes=15, l2_regularization=0.0)
        self.extra_trees_ = ExtraTreesRegressor(
            n_estimators=300, random_state=self.random_state, n_jobs=1, **extra_params
        ).fit(X, y)
        self.gradient_boosting_ = HistGradientBoostingRegressor(
            random_state=self.random_state, **gradient_params
        ).fit(X, y)
        return self

    def predict(self, X):
        weight = float(self.extra_trees_weight)
        return weight * self.extra_trees_.predict(X) + (1.0 - weight) * self.gradient_boosting_.predict(X)


class TSMixerRegressor(BaseEstimator, RegressorMixin):
    """Small MLP-Mixer over the feature vector, designed for local CPU/MPS runs."""

    def __init__(self, hidden_dim: int = 32, n_blocks: int = 2, dropout: float = 0.1, learning_rate: float = 1e-3, epochs: int = 80, batch_size: int = 128, patience: int = 12, random_state: int = 42):
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

    def fit(self, X, y):
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise ImportError("TSMixerRegressor requires torch.") from exc
        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        values = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
        target = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.scaler_ = StandardScaler().fit(values)
        values = self.scaler_.transform(values).astype(np.float32)
        torch.manual_seed(self.random_state)
        self.device_ = "mps" if torch.backends.mps.is_available() else "cpu"
        n_features = values.shape[1]

        class MixerBlock(nn.Module):
            def __init__(self, tokens: int, hidden: int, dropout: float):
                super().__init__()
                self.norm1 = nn.LayerNorm(tokens)
                self.time = nn.Sequential(nn.Linear(tokens, tokens), nn.GELU(), nn.Dropout(dropout))
                self.norm2 = nn.LayerNorm(hidden)
                self.channel = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden))

            def forward(self, x):
                # x: batch x tokens x channels; for tabular features channels=hidden.
                time_mixed = self.time(self.norm1(x.transpose(1, 2))).transpose(1, 2)
                x = x + time_mixed
                x = x + self.channel(self.norm2(x))
                return x

        class Mixer(nn.Module):
            def __init__(self, tokens: int, hidden: int, blocks: int, dropout: float):
                super().__init__()
                self.projection = nn.Linear(1, hidden)
                self.blocks = nn.Sequential(*[MixerBlock(tokens, hidden, dropout) for _ in range(blocks)])
                self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Flatten(), nn.Linear(tokens * hidden, 1))

            def forward(self, x):
                return self.head(self.blocks(self.projection(x.unsqueeze(-1))))

        self.network_ = Mixer(n_features, self.hidden_dim, self.n_blocks, self.dropout).to(self.device_)
        optimizer = torch.optim.AdamW(self.network_.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        tensor_x = torch.tensor(values, device=self.device_)
        tensor_y = torch.tensor(target, device=self.device_)
        split = max(1, int(len(values) * 0.85))
        train_x, valid_x = tensor_x[:split], tensor_x[split:]
        train_y, valid_y = tensor_y[:split], tensor_y[split:]
        best_state, best_loss, stale = None, float("inf"), 0
        for _ in range(self.epochs):
            self.network_.train()
            permutation = torch.randperm(len(train_x), device=self.device_)
            for start in range(0, len(train_x), self.batch_size):
                idx = permutation[start:start + self.batch_size]
                optimizer.zero_grad()
                loss_fn(self.network_(train_x[idx]), train_y[idx]).backward()
                optimizer.step()
            self.network_.eval()
            with torch.no_grad():
                validation_loss = float(loss_fn(self.network_(valid_x), valid_y).item()) if len(valid_x) else float(loss_fn(self.network_(train_x), train_y).item())
            if validation_loss < best_loss:
                best_loss, stale = validation_loss, 0
                best_state = {key: value.detach().cpu().clone() for key, value in self.network_.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state:
            self.network_.load_state_dict(best_state)
        self.n_features_in_ = n_features
        return self

    def predict(self, X):
        import pandas as pd
        import torch

        values = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
        values = self.scaler_.transform(values).astype(np.float32)
        self.network_.eval()
        with torch.no_grad():
            output = self.network_(torch.tensor(values, device=self.device_)).detach().cpu().numpy().reshape(-1)
        return output


def _as_float_array(X) -> np.ndarray:
    """Convert a feature frame/array to a finite float32 matrix."""
    if hasattr(X, "to_numpy"):
        values = X.to_numpy(dtype=np.float32)
    else:
        values = np.asarray(X, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Sequence models require a finite 2D feature matrix.")
    return values


def _sequence_windows(values: np.ndarray, target: np.ndarray, lookback: int):
    """Create causal windows ending at each observed target date."""
    if len(values) < lookback:
        raise ValueError(f"Need at least {lookback} rows, got {len(values)}.")
    ends = np.arange(lookback - 1, len(values))
    windows = np.stack([values[end - lookback + 1:end + 1] for end in ends])
    return windows.astype(np.float32), target[ends].astype(np.float32)


class _SequenceRegressor(BaseEstimator, RegressorMixin):
    """Small torch sequence regressor with causal context handling."""

    def __init__(
        self,
        lookback: int = 32,
        hidden_dim: int = 32,
        n_layers: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 12,
        batch_size: int = 128,
        patience: int = 4,
        random_state: int = 42,
    ):
        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

    def _build_network(self, n_features: int, nn) -> Any:
        raise NotImplementedError

    def fit(self, X, y):
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise ImportError(f"{type(self).__name__} requires torch.") from exc
        from sklearn.preprocessing import StandardScaler

        values = _as_float_array(X)
        target = np.asarray(y, dtype=np.float32).reshape(-1)
        if len(values) != len(target):
            raise ValueError("X and y must have the same number of rows.")
        self.scaler_ = StandardScaler().fit(values)
        scaled = self.scaler_.transform(values).astype(np.float32)
        windows, targets = _sequence_windows(scaled, target, self.lookback)
        self.history_scaled_ = scaled
        self.history_index_ = getattr(X, "index", None)
        self.n_features_in_ = values.shape[1]
        torch.manual_seed(self.random_state)
        self.device_ = "mps" if torch.backends.mps.is_available() else "cpu"
        self.network_ = self._build_network(self.n_features_in_, nn).to(self.device_)
        optimizer = torch.optim.AdamW(
            self.network_.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        loss_fn = nn.MSELoss()
        tensor_x = torch.tensor(windows, device=self.device_)
        tensor_y = torch.tensor(targets.reshape(-1, 1), device=self.device_)
        split = min(max(1, int(len(tensor_x) * 0.85)), max(1, len(tensor_x) - 1))
        train_x, valid_x = tensor_x[:split], tensor_x[split:]
        train_y, valid_y = tensor_y[:split], tensor_y[split:]
        best_state, best_loss, stale = None, float("inf"), 0
        for _ in range(self.epochs):
            self.network_.train()
            permutation = torch.randperm(len(train_x), device=self.device_)
            for start in range(0, len(train_x), self.batch_size):
                idx = permutation[start:start + self.batch_size]
                optimizer.zero_grad()
                loss_fn(self.network_(train_x[idx]), train_y[idx]).backward()
                optimizer.step()
            self.network_.eval()
            with torch.no_grad():
                validation_loss = float(
                    loss_fn(self.network_(valid_x), valid_y).item()
                    if len(valid_x)
                    else loss_fn(self.network_(train_x), train_y).item()
                )
            if validation_loss < best_loss:
                best_loss, stale = validation_loss, 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.network_.state_dict().items()
                }
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.network_.load_state_dict(best_state)
        return self

    def _context_for_predict(self, X) -> tuple[np.ndarray, int]:
        values = _as_float_array(X)
        scaled = self.scaler_.transform(values).astype(np.float32)
        # GridSearchCV scores the training rows as well as validation rows.
        # Avoid duplicating the fitted history in that case.  For genuinely
        # later rows, prepend only the causal tail of the fitted history.
        if getattr(X, "index", None) is not None and self.history_index_ is not None:
            try:
                is_training_slice = X.index.isin(self.history_index_).all()
            except AttributeError:
                is_training_slice = False
        else:
            is_training_slice = False
        if is_training_slice:
            return scaled, 0
        tail = self.history_scaled_[-(self.lookback - 1):]
        return np.vstack([tail, scaled]), len(tail)

    def predict(self, X):
        import torch

        context, offset = self._context_for_predict(X)
        windows = np.stack([
            context[offset + i - self.lookback + 1:offset + i + 1]
            for i in range(len(context) - offset)
        ]).astype(np.float32)
        self.network_.eval()
        with torch.no_grad():
            output = self.network_(torch.tensor(windows, device=self.device_))
        return output.detach().cpu().numpy().reshape(-1)


class PatchTSTRegressor(_SequenceRegressor):
    """Compact PatchTST-style encoder for local CPU/MPS experiments.

    It uses channel-independent temporal patches followed by a small
    Transformer encoder and a one-step regression head.  It is intentionally
    small enough for a Mac and is not claimed to be the full original model.
    """

    def __init__(
        self,
        lookback: int = 32,
        hidden_dim: int = 32,
        n_layers: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 12,
        batch_size: int = 128,
        patience: int = 4,
        random_state: int = 42,
        patch_length: int = 8,
        patch_stride: int = 4,
    ):
        super().__init__(
            lookback=lookback, hidden_dim=hidden_dim, n_layers=n_layers,
            dropout=dropout, learning_rate=learning_rate, epochs=epochs,
            batch_size=batch_size, patience=patience, random_state=random_state,
        )
        self.patch_length = patch_length
        self.patch_stride = patch_stride

    def _build_network(self, n_features: int, nn):
        patch_length, stride = self.patch_length, self.patch_stride
        lookback, hidden, layers, dropout = self.lookback, self.hidden_dim, self.n_layers, self.dropout
        n_patches = 1 + max(0, (lookback - patch_length) // stride)
        n_heads = 4 if hidden % 4 == 0 else 2

        class PatchNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_projection = nn.Linear(patch_length, hidden)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 2,
                    dropout=dropout, batch_first=True, norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
                self.head = nn.Sequential(nn.LayerNorm(n_patches * n_features * hidden), nn.Linear(n_patches * n_features * hidden, 1))

            def forward(self, x):
                # B x L x C -> B x C x patches x patch_length
                x = x.transpose(1, 2).unfold(2, patch_length, stride)
                x = self.patch_projection(x)
                batch, channels, patches, width = x.shape
                x = x.reshape(batch * channels, patches, width)
                x = self.encoder(x).reshape(batch, channels * patches * width)
                return self.head(x)

        return PatchNet()


class TimeMixerRegressor(_SequenceRegressor):
    """Compact TimeMixer-style temporal/channel mixing network."""

    def _build_network(self, n_features: int, nn):
        lookback, hidden, layers, dropout = self.lookback, self.hidden_dim, self.n_layers, self.dropout

        class MixerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm_time = nn.LayerNorm(n_features)
                self.time = nn.Sequential(
                    nn.Linear(lookback, lookback), nn.GELU(), nn.Dropout(dropout)
                )
                self.norm_channel = nn.LayerNorm(n_features)
                self.channel = nn.Sequential(
                    nn.Linear(n_features, hidden), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(hidden, n_features),
                )

            def forward(self, x):
                # x: B x L x C; mix time independently for each channel.
                z = self.norm_time(x)
                z = self.time(z.transpose(1, 2)).transpose(1, 2)
                x = x + z
                return x + self.channel(self.norm_channel(x))

        class MixerNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.Sequential(*[MixerBlock() for _ in range(layers)])
                self.head = nn.Sequential(
                    nn.LayerNorm(lookback * n_features),
                    nn.Linear(lookback * n_features, hidden), nn.GELU(),
                    nn.Linear(hidden, 1),
                )

            def forward(self, x):
                return self.head(self.blocks(x).flatten(1))

        return MixerNet()


class GlobalITransformerRegressor(BaseEstimator, RegressorMixin):
    """Compact causal iTransformer-style model with Gold/Silver output heads.

    The input is a window of feature channels shaped ``batch x time x feature``.
    The model inverts this layout before attention: each feature becomes a token
    whose embedding is built from its temporal history, and attention mixes the
    feature tokens.  It is intentionally small and serves as a genuine shared
    two-target benchmark, not as a replacement for the separately selected
    production models.
    """

    def __init__(
        self,
        lookback: int = 32,
        hidden_dim: int = 32,
        n_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 12,
        batch_size: int = 128,
        patience: int = 4,
        random_state: int = 42,
    ):
        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

    def fit(self, X, y):
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise ImportError("GlobalITransformerRegressor requires torch.") from exc
        from sklearn.preprocessing import StandardScaler

        values = _as_float_array(X)
        target = np.asarray(y, dtype=np.float32)
        if target.ndim != 2 or target.shape[1] != 2:
            raise ValueError("GlobalITransformerRegressor requires two target columns: Gold and Silver.")
        if len(values) != len(target):
            raise ValueError("X and y must have the same number of rows.")
        self.scaler_ = StandardScaler().fit(values)
        self.target_scaler_ = StandardScaler().fit(target)
        scaled = self.scaler_.transform(values).astype(np.float32)
        scaled_target = self.target_scaler_.transform(target).astype(np.float32)
        if len(scaled) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} rows, got {len(scaled)}.")
        ends = np.arange(self.lookback - 1, len(scaled))
        windows = np.stack([scaled[end - self.lookback + 1:end + 1] for end in ends]).astype(np.float32)
        window_targets = scaled_target[ends]
        self.history_scaled_ = scaled
        self.history_index_ = getattr(X, "index", None)
        self.n_features_in_ = values.shape[1]
        torch.manual_seed(self.random_state)
        self.device_ = "mps" if torch.backends.mps.is_available() else "cpu"
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads.")
        lookback = self.lookback
        hidden_dim = self.hidden_dim
        n_heads = self.n_heads
        dropout = self.dropout
        n_layers = self.n_layers

        class InvertedNet(nn.Module):
            def __init__(self, n_features: int):
                super().__init__()
                self.temporal_embedding = nn.Linear(lookback, hidden_dim)
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=n_heads,
                    dim_feedforward=hidden_dim * 2,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
                self.head = nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, 2),
                )

            def forward(self, batch):
                # batch: B x lookback x features; invert to feature tokens.
                tokens = self.temporal_embedding(batch.transpose(1, 2))
                encoded = self.encoder(tokens)
                return self.head(encoded.mean(dim=1))

        self.network_ = InvertedNet(self.n_features_in_).to(self.device_)
        optimizer = torch.optim.AdamW(
            self.network_.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        loss_fn = nn.MSELoss()
        tensor_x = torch.tensor(windows, device=self.device_)
        tensor_y = torch.tensor(window_targets, device=self.device_)
        split = min(max(1, int(len(tensor_x) * 0.85)), max(1, len(tensor_x) - 1))
        train_x, valid_x = tensor_x[:split], tensor_x[split:]
        train_y, valid_y = tensor_y[:split], tensor_y[split:]
        best_state, best_loss, stale = None, float("inf"), 0
        for _ in range(self.epochs):
            self.network_.train()
            permutation = torch.randperm(len(train_x), device=self.device_)
            for start in range(0, len(train_x), self.batch_size):
                indices = permutation[start:start + self.batch_size]
                optimizer.zero_grad()
                loss_fn(self.network_(train_x[indices]), train_y[indices]).backward()
                optimizer.step()
            self.network_.eval()
            with torch.no_grad():
                validation_loss = float(
                    loss_fn(self.network_(valid_x), valid_y).item()
                    if len(valid_x)
                    else loss_fn(self.network_(train_x), train_y).item()
                )
            if validation_loss < best_loss:
                best_loss, stale = validation_loss, 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.network_.state_dict().items()
                }
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.network_.load_state_dict(best_state)
        return self

    def predict(self, X):
        import torch

        values = _as_float_array(X)
        scaled = self.scaler_.transform(values).astype(np.float32)
        if len(scaled) == 0:
            return np.empty((0, 2), dtype=float)
        tail = self.history_scaled_[-(self.lookback - 1):]
        context = np.vstack([tail, scaled])
        offset = len(tail)
        windows = np.stack([
            context[offset + i - self.lookback + 1:offset + i + 1]
            for i in range(len(scaled))
        ]).astype(np.float32)
        self.network_.eval()
        with torch.no_grad():
            output = self.network_(torch.tensor(windows, device=self.device_)).detach().cpu().numpy()
        return self.target_scaler_.inverse_transform(output)


class _FoundationRegressor(BaseEstimator, RegressorMixin):
    """Optional univariate adapter for a pretrained forecasting model.

    Foundation models are deliberately not used by default in the main
    selection unless their package is installed. They receive only the
    observed asset return history, while the learned local models receive the
    full feature matrix; this difference is recorded in the benchmark.
    """

    model_id = ""
    package_name = ""

    def __init__(self, asset: str = "gold", lookback: int = 256, model_id: str | None = None):
        self.asset = asset
        self.lookback = lookback
        self.model_id = model_id or self.model_id

    def fit(self, X, y):
        self.current_column_ = f"{self.asset}_return_current"
        if self.current_column_ not in X:
            raise ValueError(f"Missing {self.current_column_} for foundation adapter.")
        self.history_returns_ = np.asarray(X[self.current_column_], dtype=np.float32)
        self._load_model()
        return self

    def _validate_local_checkpoint(self):
        """Fail early when a local Hugging Face snapshot contains only config files."""
        path = Path(self.model_id)
        if not path.is_dir():
            return
        weight_patterns = ("*.safetensors", "*.bin", "*.pt", "*.pth")
        has_weights = any(any(path.glob(pattern)) for pattern in weight_patterns)
        if not has_weights:
            raise FileNotFoundError(
                f"Local checkpoint '{path}' has no model weights. Download the complete snapshot "
                "with `hf download <repo-id> --local-dir <path>` or pass a valid checkpoint path."
            )

    def _load_model(self):
        raise NotImplementedError

    def _forecast_one(self, context: np.ndarray) -> float:
        raise NotImplementedError

    def predict(self, X):
        return np.asarray(self._forecast_many(self._contexts_for_predict(X)), dtype=float)

    def _contexts_for_predict(self, X) -> list[np.ndarray]:
        current = np.asarray(X[self.current_column_], dtype=np.float32)
        context = self.history_returns_.tolist()
        contexts = []
        for value in current:
            context.append(float(value))
            contexts.append(np.asarray(context[-self.lookback:], dtype=np.float32))
        return contexts

    def _forecast_many(self, contexts: list[np.ndarray]) -> list[float]:
        return [self._forecast_one(context) for context in contexts]


class ChronosRegressor(_FoundationRegressor):
    package_name = "chronos"
    model_id = "amazon/chronos-2"
    _pipeline_cache: dict[str, Any] = {}

    def _load_model(self):
        try:
            import torch
            from chronos import BaseChronosPipeline, Chronos2Pipeline
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise ImportError("ChronosRegressor requires chronos-forecasting.") from exc
        self._validate_local_checkpoint()
        if self.model_id not in self._pipeline_cache:
            local_only = os.path.isdir(self.model_id)
            pipeline_class = Chronos2Pipeline if "chronos-2" in self.model_id else BaseChronosPipeline
            self._pipeline_cache[self.model_id] = pipeline_class.from_pretrained(
                self.model_id, device_map="cpu", torch_dtype=torch.float32,
                local_files_only=local_only,
            )
        self.pipeline_ = self._pipeline_cache[self.model_id]

    def _forecast_one(self, context: np.ndarray) -> float:
        import torch
        tensor = torch.tensor(context, dtype=torch.float32)
        if hasattr(self.pipeline_, "predict_quantiles"):
            quantiles, _ = self.pipeline_.predict_quantiles(
                [tensor], prediction_length=1, quantile_levels=[0.5]
            )
            if isinstance(quantiles, list):
                quantiles = quantiles[0]
            return float(np.asarray(quantiles).reshape(-1)[0])
        samples = self.pipeline_.predict([tensor], prediction_length=1)
        return float(samples[0].detach().cpu().numpy().reshape(-1).mean())

    def _forecast_many(self, contexts: list[np.ndarray]) -> list[float]:
        import torch

        if not contexts:
            return []
        quantiles, _ = self.pipeline_.predict_quantiles(
            [torch.tensor(context, dtype=torch.float32) for context in contexts],
            prediction_length=1,
            quantile_levels=[0.5],
        )
        if isinstance(quantiles, list):
            quantiles = np.asarray([np.asarray(item).reshape(-1)[0] for item in quantiles])
        else:
            quantiles = np.asarray(quantiles).reshape(len(contexts), -1)[:, 0]
        return quantiles.astype(float).tolist()


class Chronos2CovariateRegressor(_FoundationRegressor):
    """Chronos-2 adapter using past-only, economically related covariates.

    Chronos-2's dataframe interface expects regular timestamps.  Our market
    observations are trading-day rows with holidays removed, so each context
    receives a synthetic observation counter.  The order is unchanged and no
    calendar information is invented; the original calendar features remain
    available as explicit covariates when selected.

    The adapter deliberately uses a small related set rather than all
    engineered columns.  This keeps the experiment interpretable and avoids
    allowing unrelated external series to dominate a short financial context.
    """

    package_name = "chronos"
    model_id = "amazon/chronos-2"
    _pipeline_cache: dict[str, Any] = {}

    _RELATED_COLUMNS = (
        "gold_return_current",
        "silver_return_current",
        "gold_silver_log_ratio",
        "gold_silver_ratio_change",
        "gold_intraday_return",
        "silver_intraday_return",
        "gold_range_log",
        "silver_range_log",
        "gold_close_location",
        "silver_close_location",
        "gold_volatility_20",
        "silver_volatility_20",
        "dxy_adj_close_change",
        "vix_adj_close_change",
        "sp500_adj_close_change",
        "oil_adj_close_change",
        "copper_adj_close_change",
        "tnx_adj_close_change",
        "btc_adj_close_change",
    )

    def fit(self, X, y):
        self.current_column_ = f"{self.asset}_return_current"
        if self.current_column_ not in X:
            raise ValueError(f"Missing {self.current_column_} for Chronos-2 adapter.")
        self.covariate_columns_ = [
            column for column in self._RELATED_COLUMNS if column in X.columns
        ]
        if self.current_column_ not in self.covariate_columns_:
            self.covariate_columns_.insert(0, self.current_column_)
        self.history_frame_ = X[self.covariate_columns_].astype(float).copy()
        # ``y`` is intentionally not stored: it is the t+1 return and is not
        # available when the t row is forecast.
        self._load_model()
        return self

    def _load_model(self):
        try:
            import torch
            from chronos import Chronos2Pipeline
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise ImportError("Chronos2CovariateRegressor requires chronos-forecasting.") from exc
        self._validate_local_checkpoint()
        if self.model_id not in self._pipeline_cache:
            local_only = os.path.isdir(self.model_id)
            self._pipeline_cache[self.model_id] = Chronos2Pipeline.from_pretrained(
                self.model_id,
                device_map="cpu",
                torch_dtype=torch.float32,
                local_files_only=local_only,
            )
        self.pipeline_ = self._pipeline_cache[self.model_id]

    def _build_context_dataframe(self, X) -> pd.DataFrame:
        """Create one regular long-format Chronos context per forecast origin."""
        incoming = X[self.covariate_columns_].astype(float)
        observed = self.history_frame_.copy()
        frames = []
        for origin_number, (_, row) in enumerate(incoming.iterrows()):
            observed = pd.concat([observed, row.to_frame().T], axis=0)
            context = observed.tail(self.lookback).reset_index(drop=True)
            context.insert(0, "timestamp", pd.date_range("2000-01-01", periods=len(context), freq="D"))
            context.insert(0, "item_id", f"origin_{origin_number}")
            context = context.rename(columns={self.current_column_: "target"})
            frames.append(context)
        if not frames:
            return pd.DataFrame(columns=["item_id", "timestamp", "target", *self.covariate_columns_])
        return pd.concat(frames, ignore_index=True)

    def _forecast_many(self, contexts: list[np.ndarray]) -> list[float]:  # pragma: no cover - unused API path
        raise NotImplementedError("Chronos2CovariateRegressor forecasts from dataframe contexts.")

    def predict(self, X):
        context_df = self._build_context_dataframe(X)
        if context_df.empty:
            return np.asarray([], dtype=float)
        forecasts = self.pipeline_.predict_df(
            context_df,
            prediction_length=1,
            quantile_levels=[0.5],
            context_length=self.lookback,
            batch_size=64,
            cross_learning=False,
            freq="D",
        )
        by_item = forecasts.set_index("item_id")["predictions"]
        return np.asarray(
            [float(by_item.loc[f"origin_{number}"]) for number in range(len(X))],
            dtype=float,
        )


class TimesFMRegressor(_FoundationRegressor):
    package_name = "timesfm"
    model_id = "google/timesfm-2.5-200m-pytorch"
    _model_cache: dict[str, Any] = {}

    def _load_model(self):
        try:
            import timesfm
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise ImportError("TimesFMRegressor requires the timesfm package.") from exc
        self._validate_local_checkpoint()
        if self.model_id not in self._model_cache:
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                self.model_id, local_files_only=os.path.isdir(self.model_id)
            )
            model.compile(timesfm.ForecastConfig(max_context=self.lookback, max_horizon=1))
            self._model_cache[self.model_id] = model
        self.timesfm_ = self._model_cache[self.model_id]

    def _forecast_one(self, context: np.ndarray) -> float:
        point_forecast, _ = self.timesfm_.forecast(horizon=1, inputs=[context])
        return float(np.asarray(point_forecast).reshape(-1)[0])

    def _forecast_many(self, contexts: list[np.ndarray]) -> list[float]:
        if not contexts:
            return []
        point_forecast, _ = self.timesfm_.forecast(horizon=1, inputs=contexts)
        return np.asarray(point_forecast).reshape(len(contexts), -1)[:, 0].astype(float).tolist()


def foundation_model_status() -> dict[str, str]:
    """Return import availability without downloading model weights."""
    status = {}
    for name, package in (("chronos", "chronos"), ("timesfm", "timesfm")):
        try:
            __import__(package)
            status[name] = "available"
        except ImportError:
            status[name] = "not_installed"
    return status


def foundation_model_paths() -> dict[str, str]:
    """Optional local checkpoint paths, kept outside the repository."""
    return {
        "chronos": os.environ.get("GOLD_SILVER_CHRONOS_PATH") or "amazon/chronos-2",
        "chronos2_covariates": os.environ.get("GOLD_SILVER_CHRONOS2_COVARIATES_PATH") or "amazon/chronos-2",
        "timesfm": os.environ.get("GOLD_SILVER_TIMESFM_PATH") or "google/timesfm-2.5-200m-pytorch",
    }


def candidate_specs(
    asset: str = "gold",
    random_state: int = 42,
    include_xgboost: bool = True,
    include_foundation_models: bool = False,
) -> dict[str, tuple[BaseEstimator, dict[str, list[Any]]]]:
    specs: dict[str, tuple[BaseEstimator, dict[str, list[Any]]]] = {
        "zero": (BaselineRegressor(kind="zero", asset=asset), {}),
        "last_return": (BaselineRegressor(kind="last", asset=asset), {}),
        "moving_average": (BaselineRegressor(kind="moving_average", asset=asset), {}),
        "directional_logistic": (
            DirectionalLogisticRegressor(),
            [
                {"C": [0.01], "class_weight": ["balanced"], "prediction_threshold": [0.0]},
                {"C": [0.03], "class_weight": ["balanced"], "prediction_threshold": [0.0]},
                {"C": [0.1], "class_weight": ["balanced"], "prediction_threshold": [0.0]},
                {"C": [0.3], "class_weight": ["balanced"], "prediction_threshold": [0.0]},
                {"C": [1.0], "class_weight": ["balanced"], "prediction_threshold": [0.0]},
                {"C": [3.0], "class_weight": ["balanced"], "prediction_threshold": [0.0]},
                {"C": [10.0], "class_weight": ["balanced"], "prediction_threshold": [0.0]},
                {"C": [0.03], "class_weight": ["balanced"], "prediction_threshold": [0.02]},
                {"C": [0.1], "class_weight": ["balanced"], "prediction_threshold": [0.02]},
                {"C": [0.3], "class_weight": ["balanced"], "prediction_threshold": [0.02]},
                {"C": [1.0], "class_weight": ["balanced"], "prediction_threshold": [0.02]},
                {"C": [0.1], "class_weight": ["balanced"], "prediction_threshold": [0.05]},
                {"C": [0.3], "class_weight": ["balanced"], "prediction_threshold": [0.05]},
                {"C": [1.0], "class_weight": ["balanced"], "prediction_threshold": [0.05]},
                {"C": [0.3], "class_weight": ["balanced"], "prediction_threshold": [0.1]},
            ],
        ),
        "ridge": (Pipeline([("scale", StandardScaler()), ("model", Ridge())]), {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}),
        "elasticnet": (Pipeline([("scale", StandardScaler()), ("model", ElasticNet(max_iter=5000, random_state=random_state))]), {"model__alpha": [1e-4, 1e-3, 1e-2, 1e-1], "model__l1_ratio": [0.1, 0.5, 0.9]}),
        "hist_gradient_boosting": (HistGradientBoostingRegressor(random_state=random_state), {"max_iter": [100, 250], "learning_rate": [0.03, 0.08], "max_leaf_nodes": [7, 15], "l2_regularization": [0.0, 1.0]}),
        "extra_trees": (
            ExtraTreesRegressor(random_state=random_state, n_jobs=1),
            [
                {
                    "n_estimators": [200],
                    "max_depth": [4],
                    "min_samples_leaf": [20],
                    "max_features": [0.7],
                },
                {
                    "n_estimators": [300],
                    "max_depth": [4],
                    "min_samples_leaf": [10],
                    "max_features": [1.0],
                },
                {
                    "n_estimators": [300],
                    "max_depth": [6],
                    "min_samples_leaf": [10],
                    "max_features": [0.7],
                },
                {
                    "n_estimators": [300],
                    "max_depth": [8],
                    "min_samples_leaf": [20],
                    "max_features": [1.0],
                },
            ],
        ),
        "tree_blend": (
            TreeBlendRegressor(asset=asset, random_state=random_state),
            {"extra_trees_weight": [0.25, 0.5, 0.75, 1.0]},
        ),
        # One stable local configuration; repeated torch grids can segfault
        # with some macOS wheels, while tabular families use full grids.
        "tsmixer": (TSMixerRegressor(random_state=random_state, epochs=5, patience=2), {"hidden_dim": [16], "n_blocks": [1], "dropout": [0.0], "learning_rate": [3e-4], "epochs": [5], "patience": [2]}),
        "patch_tst": (
            PatchTSTRegressor(lookback=32, patch_length=8, patch_stride=4, hidden_dim=16, epochs=5, patience=2, random_state=random_state),
            {"lookback": [32], "patch_length": [8], "patch_stride": [4], "hidden_dim": [16], "n_layers": [1], "dropout": [0.1], "learning_rate": [1e-3], "epochs": [5], "patience": [2]},
        ),
        "time_mixer": (
            TimeMixerRegressor(lookback=32, hidden_dim=16, n_layers=1, epochs=5, patience=2, random_state=random_state),
            {"lookback": [32], "hidden_dim": [16], "n_layers": [1], "dropout": [0.1], "learning_rate": [1e-3], "epochs": [5], "patience": [2]},
        ),
    }
    if include_xgboost:
        try:
            from xgboost import XGBRegressor
            specs["xgboost"] = (XGBRegressor(objective="reg:squarederror", random_state=random_state, n_jobs=1), {"n_estimators": [100, 250], "max_depth": [2, 4], "learning_rate": [0.03, 0.08], "subsample": [0.8, 1.0]})
        except ImportError:
            pass
    if include_foundation_models:
        status = foundation_model_status()
        paths = foundation_model_paths()
        if status["chronos"] == "available":
            specs["chronos"] = (ChronosRegressor(asset=asset, model_id=paths["chronos"]), {})
            # Keep the heavier covariate track opt-in.  A normal Chronos-Bolt
            # benchmark must not accidentally try to download Chronos-2.
            chronos2_path = os.environ.get("GOLD_SILVER_CHRONOS2_COVARIATES_PATH")
            if chronos2_path:
                specs["chronos2_covariates"] = (
                    Chronos2CovariateRegressor(asset=asset, model_id=chronos2_path),
                    {},
                )
        if status["timesfm"] == "available":
            specs["timesfm"] = (TimesFMRegressor(asset=asset, model_id=paths["timesfm"]), {})
    return specs
