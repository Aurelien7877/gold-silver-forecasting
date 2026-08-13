#!/usr/bin/env python
"""Prepare separate, private-by-default Hugging Face payloads for Gold/Silver."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import joblib

from gold_silver.artifacts import ModelBundle

DISPLAY_NAMES = {
    "chronos": "Chronos-2 (univariate)",
    "chronos2_covariates": "Chronos-2 (past-only covariates)",
    "timesfm": "TimesFM 2.5 (univariate)",
    "timesfm_covariates": "TimesFM 2.5 (causal covariates)",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path):
    import pandas as pd

    return pd.read_csv(path)


def _short_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _metric(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _card(asset: str, local: dict, validation: float, foundation, stats, reality, manifest: dict) -> str:
    title = "Gold" if asset == "gold" else "Silver"
    model_name = "ExtraTrees" if asset == "gold" else "Directional Logistic Regression"
    target_note = (
        "The estimator predicts the next-day Gold log return."
        if asset == "gold"
        else "The estimator predicts next-day Silver direction; its output is a centered probability score, not a calibrated return magnitude."
    )
    foundation_lines = []
    for row in foundation.to_dict(orient="records"):
        foundation_lines.append(
            f"| {DISPLAY_NAMES.get(row['family'], row['family'])} | {_metric(row['validation_sharpe'])} | {_metric(row['sharpe'])} | {_metric(row['max_drawdown'])} |"
        )
    stat_lines = []
    for row in stats.to_dict(orient="records"):
        p = row.get("p_value_holm")
        p_text = f"{float(p):.3g}" if p is not None else "n/a"
        stat_lines.append(
            f"| {DISPLAY_NAMES.get(row['foundation_family'], row['foundation_family'])} | {_metric(row['sharpe_difference'])} | {p_text} | {'yes' if row['local_lower_squared_error'] else 'no'} |"
        )
    loss_note = (
        "For Gold, the local winner has lower squared error than every foundation track, but none of the four differences survives Holm correction."
        if asset == "gold"
        else "For Silver, the foundation tracks have lower squared error after Holm correction, while the local directional rule has the better net trading result; forecast loss and signed strategy performance are different objectives."
    )
    test_return = local["cumulative_return"]
    return f'''---
library_name: scikit-learn
pipeline_tag: time-series-forecasting
tags:
- time-series-forecasting
- finance
- commodities
- {asset}
private: true
---

# {title} next-day forecasting model

This private model card contains the selected **{model_name}** estimator for one-day-ahead research on {title}. {target_note}

> Research and education only. This model is not financial advice, is not a trading recommendation and has no performance guarantee.

## What is included

- `estimator.joblib`: the fitted {model_name} estimator.
- `feature_schema.json`: exact input column order and target definition.
- `config.json`: data, feature, split and transaction-cost configuration.
- `metrics.json`: validation, locked-test, foundation-model and robustness results.

The payload contains no raw Yahoo Finance/FRED data, Hugging Face token or intermediate checkpoint. The estimator was fitted with code revision `{_short_commit()}`.

## Result under the repository protocol

| Metric | Value |
|---|---:|
| Validation net Sharpe | {_metric(validation)} |
| Locked-test net Sharpe | {_metric(local['sharpe'])} |
| Locked-test cumulative return | {_metric(test_return * 100, 1)}% |
| Locked-test maximum drawdown | {_metric(local['max_drawdown'] * 100, 1)}% |
| Turnover per observation | {_metric(local['turnover'])} |
| Information coefficient | {_metric(local['ic'])} |
| Locked-test observations | {_metric(local['n_observations'], 0)} |

The final 20% of common Gold/Silver dates is a locked test and was not used to select the model. The backtest maps the signal to `-1`, `0` or `+1`, applies it to the next realized return and charges 10 basis points per unit of turnover.

## Foundation-model audit

These external models were evaluated on the same dates, one-day horizon and costs. Their information representation differs: the local model receives engineered causal features, while the foundation tracks receive only their stated univariate or past-only covariate context.

| Model | Validation Sharpe | Locked-test Sharpe | Max drawdown |
|---|---:|---:|---:|
{chr(10).join(foundation_lines)}

The repository-level claim is **best local model found under the stated candidate set**, not universal SOTA. The external comparison is a reproducible audit, not a claim that this small experiment covers every published model, training recipe or market period.

{loss_note}

## Paired statistical comparison with the local winner

DM compares forecast loss on the same locked dates; `Holm p` corrects the four foundation comparisons. A “yes” means the local winner has lower squared error in that paired test.

| Foundation model | Local-minus-foundation Sharpe | Holm p | Local lower MSE |
|---|---:|---:|---:|
{chr(10).join(stat_lines)}

The candidate-aware White Reality Check for the full local-plus-foundation universe is `{_metric(reality['p_value_max_sharpe'], 3)}` for {title}. This is a conditional bootstrap p-value, not a probability that the model will be profitable.

## Data and reproducibility

- Source snapshot: Yahoo Finance tickers in `data/raw/manifest.json`.
- Snapshot dates: `{manifest.get('start', 'n/a')}` to `{manifest.get('end', 'n/a')}`.
- Target: `log(close[t+1] / close[t])`.
- Feature rule: information available at or before `t` only.
- Python: 3.11; see the source repository for installation and inference code.

Source repository: https://github.com/Aurelien7877/gold-silver-forecasting
'''


def export_asset(asset: str, bundle_path: Path, output_root: Path) -> Path:
    bundle = ModelBundle.load(bundle_path)
    summary = _read_json(Path("reports/experiment_summary.json"))
    manifest = _read_json(Path("data/raw/manifest.json"))
    local_table = _read_csv(Path(f"data/processed/{asset}_test_comparison.csv"))
    local = local_table.loc[local_table["selected"]].iloc[0].to_dict()
    foundation = _read_csv(Path(f"data/processed/{asset}_foundation_comparison.csv"))
    stats = _read_csv(Path(f"data/processed/{asset}_foundation_statistical_tests.csv"))
    reality = _read_csv(Path(f"data/processed/{asset}_reality_check.csv")).iloc[0].to_dict()
    asset_summary = summary["assets"][asset]

    output = output_root / asset
    output.mkdir(parents=True, exist_ok=True)
    model = bundle.models[asset]
    # Silver's public estimator is the fitted sklearn pipeline inside the
    # thin directional wrapper, so the payload can be loaded without the
    # private research package just to call predict_proba.
    estimator = getattr(model, "model_", model)
    joblib.dump(estimator, output / "estimator.joblib")
    (output / "feature_schema.json").write_text(
        json.dumps({
            "asset": asset,
            "feature_columns": bundle.feature_columns,
            "target": f"{asset}_return_next",
            "silver_signal": "P(up) - 0.5" if asset == "silver" else None,
        }, indent=2),
        encoding="utf-8",
    )
    (output / "config.json").write_text(json.dumps(bundle.config, indent=2), encoding="utf-8")
    metrics = {
        "asset": asset,
        "selected_model": asset_summary["winner"],
        "best_params": asset_summary["best_params"],
        "validation_score": asset_summary["validation_score"],
        "locked_test": local,
        "foundation_comparison": foundation.to_dict(orient="records"),
        "foundation_statistical_tests": stats.to_dict(orient="records"),
        "reality_check": reality,
        "source_manifest": manifest,
        "source_code_revision": _short_commit(),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (output / "README.md").write_text(
        _card(asset, local, float(asset_summary["validation_score"]), foundation, stats, reality, manifest),
        encoding="utf-8",
    )
    (output / "requirements.txt").write_text(
        "joblib>=1.3\nnumpy>=1.26\npandas>=2.2\nscikit-learn>=1.4\n",
        encoding="utf-8",
    )
    print(f"Prepared {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="models/gold_silver_bundle.joblib")
    parser.add_argument("--output", default="hf_export")
    parser.add_argument("--asset", choices=("gold", "silver", "all"), default="all")
    args = parser.parse_args()
    assets = ("gold", "silver") if args.asset == "all" else (args.asset,)
    for asset in assets:
        export_asset(asset, Path(args.bundle), Path(args.output))
    print("Upload privately with: hf upload <namespace>/<repo> hf_export/gold --private")
    print("and: hf upload <namespace>/<repo> hf_export/silver --private")


if __name__ == "__main__":
    main()
