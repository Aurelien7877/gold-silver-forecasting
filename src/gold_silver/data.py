"""Market-data download and normalization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ProjectConfig


class DataDownloadError(RuntimeError):
    """Raised when an external data source cannot be reached or parsed."""


def _column_name(alias: str, field: str) -> str:
    normalized_field = str(field).lower().replace(" ", "_")
    return f"{alias}_{normalized_field}"


def normalize_market_data(raw: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    """Convert yfinance's single- or multi-ticker output to stable flat columns."""
    if raw.empty:
        raise DataDownloadError("The downloaded market data is empty.")
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        flattened: dict[Any, str] = {}
        ticker_to_alias = {ticker: alias for alias, ticker in aliases.items()}
        for field, ticker in frame.columns:
            alias = ticker_to_alias.get(str(ticker), str(ticker).lower().replace("=", "_"))
            flattened[(field, ticker)] = _column_name(alias, field)
        frame.columns = [flattened[column] for column in frame.columns]
    else:
        alias = next(iter(aliases), "asset")
        frame.columns = [_column_name(alias, str(column)) for column in frame.columns]

    index = pd.to_datetime(frame.index, errors="coerce")
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    frame.index = index.normalize()
    frame = frame[~frame.index.isna()]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(axis=1, how="all")
    return frame


def _download_yfinance(config: ProjectConfig) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataDownloadError("Install project dependencies before downloading data.") from exc
    tickers = config.data.tickers
    try:
        kwargs = {
            "interval": config.data.interval,
            "auto_adjust": False,
            "group_by": "column",
            "progress": False,
            "threads": False,
        }
        if config.data.start or config.data.end:
            kwargs.update(start=config.data.start, end=config.data.end)
        else:
            kwargs["period"] = "max"
        raw = yf.download(list(tickers.values()), **kwargs)
    except Exception as exc:  # pragma: no cover - external service behavior
        raise DataDownloadError(f"Yahoo Finance download failed: {exc}") from exc
    return normalize_market_data(raw, tickers)


def _download_fred(config: ProjectConfig, frame: pd.DataFrame) -> pd.DataFrame:
    if not config.data.use_fred or not config.data.fred_series:
        return frame
    try:
        import os

        from fredapi import Fred
    except ImportError as exc:
        raise DataDownloadError("Install fredapi to use FRED data.") from exc
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise DataDownloadError("FRED_API_KEY is required when data.use_fred is true.")
    fred = Fred(api_key=api_key)
    result = frame.copy()
    for alias, series_id in config.data.fred_series.items():
        values = fred.get_series(series_id, observation_start=config.data.start, observation_end=config.data.end)
        series = pd.Series(values, name=f"fred_{alias}")
        series.index = pd.to_datetime(series.index).normalize()
        result = result.join(series, how="outer")
    return result.sort_index()


def download_market_data(config: ProjectConfig, save: bool = True) -> pd.DataFrame:
    """Download public market data and optionally cache it as parquet."""
    frame = _download_yfinance(config)
    frame = _download_fred(config, frame)
    if save:
        raw_dir = Path(config.data.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        output = raw_dir / "market_data.parquet"
        frame.to_parquet(output)
        manifest = {
            "path": str(output),
            "rows": len(frame),
            "columns": list(frame.columns),
            "start": frame.index.min().isoformat() if len(frame) else None,
            "end": frame.index.max().isoformat() if len(frame) else None,
            "sha256": sha256_file(output),
            "tickers": config.data.tickers,
        }
        (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return frame


def load_cached_market_data(config: ProjectConfig) -> pd.DataFrame:
    path = Path(config.data.raw_dir) / "market_data.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No cached data at {path}; run scripts/download_data.py first.")
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index).normalize()
    return frame.sort_index()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
