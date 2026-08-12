# Gold/Silver Forecasting

Reproducible research for next-day (`J+1`) Gold and Silver log-return forecasting on macOS Apple Silicon. This repository is for research only and is not financial advice.

## What is included

- Strict chronological feature construction with leakage tests.
- Separate Gold and Silver models with a locked final test period.
- Tabular models: Ridge, ElasticNet, ExtraTrees and HistGradientBoosting.
- Lightweight causal TSMixer, PatchTST-style and TimeMixer-style models for local CPU/MPS experiments.
- Optional real foundation-model adapters for Chronos-Bolt Tiny and TimesFM 2.5.
- Walk-forward model selection, transaction-cost backtests, Diebold–Mariano tests, block bootstrap and Holm–Bonferroni correction.
- English notebooks with executable plots and a small set of tracked report figures.
- GitHub Actions tests on macOS 14 / Python 3.11.

## Installation

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Optional research dependencies:

```bash
python -m pip install -e '.[research]'
```

Optional foundation models:

```bash
python -m pip install -e '.[foundation]'
hf auth login
```

The base install intentionally excludes XGBoost, FRED, Optuna, Matplotlib, Seaborn and Jupyter. Install `.[research]` only when those workflows are needed.

## Reproducible data snapshot

A compressed research snapshot is included at `data/raw/market_data.parquet.zip`. Extract it before running the notebooks:

```bash
unzip -o data/raw/market_data.parquet.zip -d data/raw/
```

The snapshot contains the downloaded Yahoo Finance market data used in the current analysis. `data/raw/manifest.json` records sources, dates, columns and checksums. To download a fresh snapshot instead:

```bash
make download
```

Raw Parquet files, processed results, trained bundles, checkpoints and caches remain ignored by Git. Only the compressed snapshot and its manifest are tracked.

## Workflow

```bash
make analysis
make train
make predict
```

For the foundation-model benchmark using local checkpoints:

```bash
python scripts/train.py \
  --include-foundation-models \
  --chronos-path /path/to/chronos-bolt-tiny \
  --timesfm-path /path/to/timesfm-2.5-200m-pytorch
```

The checkpoint adapters follow the official [Chronos implementation](https://github.com/amazon-science/chronos-forecasting) and [TimesFM implementation](https://github.com/google-research/timesfm). Never commit Hugging Face tokens or model weights.

## Method

- Targets: `log(close[t+1] / close[t])` for each asset.
- Features: lagged returns, momentum, rolling volatility, moving averages, drawdown, volume, Gold/Silver ratio, rolling correlations, cross-asset returns and market covariates.
- Split: final 20% locked as test; first 80% searched with five expanding walk-forward folds and `gap=1`.
- Selection metric: mean annualized net Sharpe on validation, with turnover costs of 10 bps per unit turnover.
- Signals: `-1`, `0` or `+1`, applied to the next realized return.
- Statistical checks: one-step squared-error Diebold–Mariano tests, paired moving-block bootstrap for Sharpe differences and Holm–Bonferroni correction across competitors.

## Current benchmark

The current cache uses five walk-forward folds, a one-day horizon and 10 bps costs.

| Asset | Validation winner | Validation Sharpe | Locked-test Sharpe | Test cumulative return |
|---|---|---:|---:|---:|
| Gold | ExtraTrees | 0.485 | 0.779 | +65.9% |
| Silver | HistGradientBoosting | 0.165 | 0.025 | -24.2% |

Gold test comparison:

| Model | Net Sharpe |
|---|---:|
| ExtraTrees | 0.779 |
| HistGradientBoosting | 0.476 |
| TimesFM 2.5 | 0.275 |
| Chronos-Bolt Tiny | -0.409 |
| PatchTST-style | -0.872 |
| TimeMixer-style | -1.137 |

ExtraTrees is the best observed Gold model, but it is not statistically superior to HistGradientBoosting, TimesFM or Chronos after Holm–Bonferroni correction. The result is promising under this protocol, not proof of universal state of the art.

## Visual research summary

![Normalized Gold and Silver prices](docs/figures/normalized_prices.png)

![Model benchmark](docs/figures/model_benchmark.png)

![Gold robustness](docs/figures/gold_robustness.png)

## Notebooks

Run them from the repository root after extracting the snapshot:

1. `01_data_and_correlations.ipynb` — coverage, normalized prices, returns and Gold/Silver dependence.
2. `02_feature_quality.ipynb` — missingness, distributions, correlation heatmaps and leakage checks.
3. `03_model_benchmark.ipynb` — validation leaderboard, locked-test comparison and model plots.
4. `04_final_selection.ipynb` — equity curves, costs, bootstrap intervals and corrected statistical tests.

## Generated artifacts

- `data/processed/*_leaderboard.csv`: validation model search results.
- `data/processed/*_test_comparison.csv`: locked-test metrics for every evaluated family.
- `data/processed/*_statistical_tests.csv`: DM tests, Holm-adjusted p-values and paired Sharpe bootstrap intervals.
- `reports/experiment_summary.json`: complete run metadata and selected model bundle information.
- `models/gold_silver_bundle.joblib`: local final bundle, ignored by Git.

## Private publication

The GitHub repository is intended to remain private while the research is ongoing. Hugging Face publication is intentionally separate and should contain only a reviewed winner bundle and model card.

Yahoo Finance and FRED data may have usage and redistribution restrictions. The included snapshot is for this private research repository; verify the applicable terms before sharing it.
