"""Typed configuration for the research pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    tickers: dict[str, str] = field(default_factory=lambda: {
        "gold": "GC=F",
        "silver": "SI=F",
        "dxy": "DX-Y.NYB",
        "vix": "^VIX",
        "sp500": "^GSPC",
        "oil": "CL=F",
        "copper": "HG=F",
        "tnx": "^TNX",
        "btc": "BTC-USD",
    })
    start: str | None = None
    end: str | None = None
    interval: str = "1d"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    use_fred: bool = False
    fred_series: dict[str, str] = field(default_factory=dict)


@dataclass
class FeatureConfig:
    lags: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20, 60])
    windows: list[int] = field(default_factory=lambda: [5, 10, 20, 60])
    min_history: int = 60
    include_current_returns: bool = True


@dataclass
class BacktestConfig:
    transaction_cost_bps: float = 10.0
    annualization: int = 252
    signal_threshold: float = 0.0


@dataclass
class SearchConfig:
    test_fraction: float = 0.2
    n_splits: int = 5
    validation_window: int = 126
    gap: int = 1
    random_state: int = 42
    n_jobs: int = 1
    max_trials: int = 4
    include_xgboost: bool = False
    include_foundation_models: bool = False


@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_config() -> ProjectConfig:
    return ProjectConfig()


def _merge_dataclass(cls: type, values: dict[str, Any] | None) -> Any:
    values = values or {}
    return cls(**{k: v for k, v in values.items() if k in cls.__dataclass_fields__})


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Load YAML configuration, falling back to typed defaults."""
    if path is None:
        return default_config()
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return ProjectConfig(
        data=_merge_dataclass(DataConfig, raw.get("data")),
        features=_merge_dataclass(FeatureConfig, raw.get("features")),
        backtest=_merge_dataclass(BacktestConfig, raw.get("backtest")),
        search=_merge_dataclass(SearchConfig, raw.get("search")),
    )
