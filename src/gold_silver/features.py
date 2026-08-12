"""Leakage-safe feature engineering and targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig

ASSETS = ("gold", "silver")


def _find_column(frame: pd.DataFrame, alias: str, field: str) -> str | None:
    candidate = f"{alias}_{field}"
    return candidate if candidate in frame.columns else None


def build_features(raw_data: pd.DataFrame, config: FeatureConfig | object) -> pd.DataFrame:
    """Build features using data at or before each timestamp only."""
    feature_config = config if isinstance(config, FeatureConfig) else config.features
    frame = raw_data.copy().sort_index()
    frame.index = pd.to_datetime(frame.index).normalize()
    frame = frame[~frame.index.duplicated(keep="last")]
    output = pd.DataFrame(index=frame.index)

    for asset in ASSETS:
        close = _find_column(frame, asset, "close")
        if close is None:
            raise ValueError(f"Missing required column: {asset}_close")
        price_observed = pd.to_numeric(frame[close], errors="coerce").dropna()
        price = price_observed.reindex(frame.index)
        ret_observed = np.log(price_observed).diff().replace([np.inf, -np.inf], np.nan)
        ret = ret_observed.reindex(frame.index)
        output[f"{asset}_close"] = price
        output[f"{asset}_return"] = ret
        if feature_config.include_current_returns:
            output[f"{asset}_return_current"] = ret

        # End-of-day OHLC fields are known at the prediction timestamp. These
        # describe the completed session and never reference t+1.
        if feature_config.include_ohlc_features:
            open_col = _find_column(frame, asset, "open")
            high_col = _find_column(frame, asset, "high")
            low_col = _find_column(frame, asset, "low")
            if open_col and high_col and low_col:
                open_price = pd.to_numeric(frame[open_col], errors="coerce").reindex(frame.index)
                high_price = pd.to_numeric(frame[high_col], errors="coerce").reindex(frame.index)
                low_price = pd.to_numeric(frame[low_col], errors="coerce").reindex(frame.index)
                valid_range = (high_price > 0) & (low_price > 0) & (high_price >= low_price)
                output[f"{asset}_ohlc_available"] = valid_range.astype(float)
                output[f"{asset}_intraday_return"] = np.log(price / open_price.where(open_price > 0))
                output[f"{asset}_range_log"] = np.log(
                    (high_price / low_price.where(low_price > 0)).where(valid_range)
                )
                output[f"{asset}_close_location"] = (
                    ((price - low_price) / (high_price - low_price).replace(0, np.nan))
                    .clip(0.0, 1.0)
                    .where(valid_range)
                )
                output[f"{asset}_overnight_gap"] = np.log(
                    open_price.where(open_price > 0) / price_observed.shift(1).reindex(frame.index)
                )
                # A missing provider field is represented by a neutral value
                # plus the availability flag above. This keeps valid dates
                # instead of silently deleting whole historical periods.
                output[f"{asset}_intraday_return"] = output[f"{asset}_intraday_return"].fillna(0.0)
                output[f"{asset}_range_log"] = output[f"{asset}_range_log"].fillna(0.0)
                output[f"{asset}_close_location"] = output[f"{asset}_close_location"].fillna(0.5)
                output[f"{asset}_overnight_gap"] = output[f"{asset}_overnight_gap"].fillna(0.0)

        volume_col = _find_column(frame, asset, "volume")
        if volume_col:
            volume_observed = pd.to_numeric(frame[volume_col], errors="coerce").reindex(price_observed.index)
            volume = volume_observed.reindex(frame.index)
            output[f"{asset}_volume_log"] = np.log1p(volume.clip(lower=0))
            output[f"{asset}_volume_change"] = volume_observed.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).reindex(frame.index)
        for lag in feature_config.lags:
            output[f"{asset}_return_lag_{lag}"] = ret_observed.shift(lag).reindex(frame.index)
        for window in feature_config.windows:
            output[f"{asset}_momentum_{window}"] = price_observed.pct_change(window, fill_method=None).reindex(frame.index)
            output[f"{asset}_volatility_{window}"] = ret_observed.rolling(window).std().reindex(frame.index)
            output[f"{asset}_return_mean_{window}"] = ret_observed.rolling(window).mean().reindex(frame.index)
            rolling_max = price_observed.rolling(window).max()
            output[f"{asset}_drawdown_{window}"] = (price_observed / rolling_max - 1.0).reindex(frame.index)
            output[f"{asset}_ma_gap_{window}"] = (price_observed / price_observed.rolling(window).mean() - 1.0).reindex(frame.index)

    output["gold_silver_log_ratio"] = np.log(output["gold_close"] / output["silver_close"])
    output["gold_silver_ratio_change"] = output["gold_silver_log_ratio"].diff()
    for window in feature_config.windows:
        output[f"gold_silver_return_corr_{window}"] = output["gold_return"].rolling(window).corr(
            output["silver_return"]
        )
        gold_var = output["gold_return"].rolling(window).var()
        silver_var = output["silver_return"].rolling(window).var()
        covariance = output["gold_return"].rolling(window).cov(output["silver_return"])
        output[f"silver_beta_to_gold_{window}"] = (covariance / gold_var).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        output[f"gold_beta_to_silver_{window}"] = (covariance / silver_var).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)

    if feature_config.include_calendar_features:
        day_of_year = output.index.dayofyear.to_numpy()
        day_of_week = output.index.dayofweek.to_numpy()
        output["calendar_weekday_sin"] = np.sin(2.0 * np.pi * day_of_week / 5.0)
        output["calendar_weekday_cos"] = np.cos(2.0 * np.pi * day_of_week / 5.0)
        output["calendar_year_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
        output["calendar_year_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)

    # External market series are transformed into current changes and lags.
    external_features: dict[str, pd.Series] = {}
    protected = {"gold", "silver"}
    for column in frame.columns:
        if not (column.endswith("_close") or column.startswith("fred_")):
            continue
        if not any(column.startswith(f"{asset}_") for asset in protected):
            series = pd.to_numeric(frame[column], errors="coerce")
            safe = series.where(series > 0)
            transformed = np.log(safe).diff().replace([np.inf, -np.inf], np.nan)
            external_features[f"{column}_change"] = transformed
            for lag in (1, 5, 20):
                external_features[f"{column}_change_lag_{lag}"] = transformed.shift(lag)

    if external_features:
        output = pd.concat([output, pd.DataFrame(external_features, index=frame.index)], axis=1)

    # Model only common Gold/Silver trading dates. A missing external market
    # close means no new observation for that feature, represented as zero
    # change rather than as an unusable row.
    output = output.dropna(subset=["gold_close", "silver_close"])
    for window in feature_config.windows:
        output[f"gold_silver_return_corr_{window}"] = output["gold_return"].rolling(window).corr(
            output["silver_return"]
        )
    external_columns = [
        column
        for column in output.columns
        if column not in {"gold_close", "silver_close", "gold_return", "silver_return"}
        and not column.startswith(("gold_", "silver_"))
    ]
    if external_columns:
        output[external_columns] = output[external_columns].fillna(0.0)

    output = output.replace([np.inf, -np.inf], np.nan)
    return output.dropna(thresh=max(1, feature_config.min_history), axis=0)


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Return model inputs, excluding prices and realized returns."""
    excluded = {"gold_close", "silver_close", "gold_return", "silver_return"}
    return [column for column in features.columns if column not in excluded]


def make_targets(features: pd.DataFrame, target: str = "gold") -> tuple[pd.DataFrame, pd.Series]:
    """Create X/y where y[t] is the next close-to-close log return."""
    if target not in ASSETS:
        raise ValueError(f"target must be one of {ASSETS}")
    columns = feature_columns(features)
    X = features[columns].copy()
    y = features[f"{target}_return"].shift(-1).rename(f"target_{target}_return_next")
    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid], y.loc[valid]


def make_targets_for_assets(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = features[feature_columns(features)].copy()
    targets = pd.DataFrame({asset: features[f"{asset}_return"].shift(-1) for asset in ASSETS})
    valid = X.notna().all(axis=1) & targets.notna().all(axis=1)
    return X.loc[valid], targets.loc[valid]
