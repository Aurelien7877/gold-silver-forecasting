"""Correlation analysis and compact research summaries."""

from __future__ import annotations

import pandas as pd


def correlation_report(features: pd.DataFrame, windows: tuple[int, ...] = (20, 60, 252)) -> pd.DataFrame:
    """Return level, return, rolling and lead/lag Gold/Silver correlations."""
    gold_return = features["gold_return"]
    silver_return = features["silver_return"]
    rows: list[dict[str, float | str | int]] = []
    for kind, left, right in (
        ("levels", features["gold_close"], features["silver_close"]),
        ("returns", gold_return, silver_return),
    ):
        rows.append({"analysis": kind, "window": 0, "pearson": left.corr(right), "spearman": left.corr(right, method="spearman")})
    for window in windows:
        rolling = gold_return.rolling(window).corr(silver_return).dropna()
        rows.append({
            "analysis": "rolling_returns",
            "window": window,
            "pearson": rolling.mean(),
            "spearman": rolling.median(),
        })
    for lag in range(-5, 6):
        shifted = silver_return.shift(lag)
        rows.append({
            "analysis": "lead_lag_returns",
            "window": lag,
            "pearson": gold_return.corr(shifted),
            "spearman": gold_return.corr(shifted, method="spearman"),
        })
    return pd.DataFrame(rows)


def summarize_correlations(report: pd.DataFrame) -> str:
    """Generate a cautious, data-dependent text summary."""
    levels = report.query("analysis == 'levels'").iloc[0]
    returns = report.query("analysis == 'returns'").iloc[0]
    lead_lag = report.query("analysis == 'lead_lag_returns'").dropna(subset=["pearson"])
    if lead_lag.empty:
        lag_text = "aucun lag exploitable"
    else:
        best = lead_lag.iloc[lead_lag["pearson"].abs().argmax()]
        lag_text = f"lag {int(best['window'])} avec Pearson={best['pearson']:.3f}"
    return (
        f"Corrélation des niveaux: Pearson={levels['pearson']:.3f}. "
        f"Corrélation des rendements: Pearson={returns['pearson']:.3f}, "
        f"Spearman={returns['spearman']:.3f}. Meilleur lead/lag observé: {lag_text}. "
        "Ces statistiques décrivent une dépendance historique et ne démontrent pas une causalité."
    )
