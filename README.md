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

To refresh only the optional foundation comparison without rerunning every local model:

```bash
GOLD_SILVER_CHRONOS_PATH=/path/to/chronos-bolt-tiny \
GOLD_SILVER_TIMESFM_PATH=/path/to/timesfm-2.5-200m-pytorch \
make foundation-benchmark
```

## Method and plain-language glossary

- Targets: `log(close[t+1] / close[t])` for each asset.
- Features: lagged returns, momentum, rolling volatility, moving averages, drawdown, volume, Gold/Silver ratio, rolling correlations, cross-asset returns and market covariates.
- Split: final 20% locked as test; first 80% searched with five expanding walk-forward folds and `gap=1`.
- Selection metric: mean annualized net Sharpe on validation, with turnover costs of 10 bps per unit turnover.
- Signals: `-1`, `0` or `+1`, applied to the next realized return.
- Statistical checks: one-step squared-error Diebold–Mariano tests, paired moving-block bootstrap for Sharpe differences and Holm–Bonferroni correction across competitors.

Terms used in the reports:

- **Log return** — `log(close[t+1] / close[t])` measures the proportional price change and adds cleanly across days. The model sees the return realized after the feature date, never the future close itself.
- **OHLC features** — Open, High, Low and Close summarize the completed trading session. Intraday range, close location and overnight gap are available at the end of date `t`; missing historical fields receive a neutral value plus an availability flag.
- **Pearson correlation** — Linear co-movement between two series. A high value is descriptive and can arise from common trends, so it is not evidence that one metal causes the other.
- **Spearman correlation** — Rank-based co-movement, less sensitive to outliers and nonlinear scale. Agreement with Pearson suggests a stable monotonic relationship, not necessarily a trading edge.
- **Rolling correlation** — Correlation recomputed over a moving window, here 20, 60 or 252 observations. It shows regime changes rather than one number averaged over the whole history.
- **Lead/lag correlation** — Gold returns are compared with Silver returns shifted by several days. A peak at lag zero is contemporaneous association; a small nonzero peak is not sufficient evidence of tradable predictability.
- **Walk-forward validation** — Each fold trains on earlier dates and validates on later dates. This approximates deployment and prevents random shuffling from leaking future regimes into training.
- **Locked test** — The final 20% of dates is held untouched while features, models and hyperparameters are selected. It is the only period used for the final out-of-sample claim.
- **Sharpe ratio** — Average net daily strategy return divided by its volatility, annualized by `sqrt(252)`. It rewards return per unit of risk but is unstable in finite samples and is not a guarantee.
- **Turnover and costs** — Turnover is the absolute change in position; each unit costs 10 basis points in the reference backtest. This prevents a high-frequency signal from looking attractive only before implementation costs.
- **Bootstrap interval** — A moving-block resampling interval for the Sharpe difference. Blocks preserve short-run dependence, but the interval remains descriptive rather than a proof of future performance.
- **Diebold–Mariano test** — A paired test of forecast-loss differences on the same dates. The implementation uses a Newey–West variance estimate for short-memory dependence; it tests accuracy, not profitability by itself.
- **Holm–Bonferroni correction** — Adjusts p-values when many competitors are compared. It reduces false discoveries and makes a “significant winner” claim harder to obtain.
- **PatchTST** — A Transformer that converts temporal patches into tokens and models them with shared channel weights; the local implementation is intentionally compact for Apple Silicon.
- **TimeMixer** — A multiscale MLP architecture that mixes fine and coarse temporal patterns; the local implementation is a lightweight research approximation, not the full paper code.
- **Chronos and TimesFM** — Pretrained time-series foundation models used here as zero-shot univariate baselines. They are evaluated on the same dates and costs, but they do not receive the engineered cross-asset features.

## Current benchmark

The current cache uses five walk-forward folds, a one-day horizon and 10 bps costs.

| Asset | Validation winner | Validation Sharpe | Locked-test Sharpe | Test cumulative return |
|---|---|---:|---:|---:|
| Gold | ExtraTrees | 0.502 | 0.971 | +90.8% |
| Silver | Tree blend | 0.497 | 0.630 | +98.9% |

Gold test comparison:

| Model | Net Sharpe |
|---|---:|
| ExtraTrees | 0.971 |
| XGBoost | 0.475 |
| Tree blend | 0.150 |
| TimesFM 2.5 | 0.275 |
| Chronos-Bolt Tiny | -0.409 |
| PatchTST-style | -2.531 |
| TimeMixer-style | -1.045 |

ExtraTrees is the best observed Gold model and the tree blend is the selected Silver model under this run. However, after Newey–West Diebold–Mariano testing and Holm–Bonferroni correction, Gold is not statistically superior to the zero baseline, XGBoost or the strongest tabular alternatives; this is evidence of a promising local winner, not proof of universal SOTA.

Silver test comparison:

| Model | Net Sharpe |
|---|---:|
| Ridge | 1.130 |
| ExtraTrees | 0.756 |
| Tree blend (selected by validation) | 0.630 |
| XGBoost | 0.621 |
| HistGradientBoosting | 0.259 |
| Chronos-Bolt Tiny | -0.135 |
| TimesFM 2.5 | -0.238 |

The higher Silver Ridge test Sharpe is intentionally not selected after the fact: the model decision is made from validation only. This is a useful warning against choosing a winner by looking at the locked test.

We also probed one shared multi-output ExtraTrees model. It reached validation Sharpe 0.174 for Gold and −0.669 for Silver, versus 0.502 and 0.497 for the selected separate models, so the shared model is rejected by the same validation rule despite test-period Sharpe of 0.866 and 0.785.

The foundation-model comparison is deliberately separate because Chronos and TimesFM receive only the univariate return history, while local tabular models receive engineered OHLC and cross-asset features. Both foundation models are nevertheless evaluated on the same 970 locked-test dates, one-day horizon and 10 bps costs.

## Visual research summary

![Normalized Gold and Silver prices](docs/figures/normalized_prices.png)

This chart rescales both prices to 100 at the common start date, making co-movement visible despite different dollar prices. The return panel and rolling correlation panel show that similar long-run direction does not imply identical daily returns.

![Model benchmark](docs/figures/model_benchmark.png)

Validation bars show mean net Sharpe over expanding walk-forward folds; test bars are a separate locked-period diagnostic. A positive bar is not enough: stability across folds, costs and uncertainty must also be checked.

![Out-of-sample prediction diagnostics](docs/figures/prediction_diagnostics.png)

These panels compare 20-day rolling predicted and realized returns on dates never used for fitting. The scatter is clipped to the central 99% only for readability; the underlying metrics use every observation.

![Gold robustness](docs/figures/gold_robustness.png)

This figure shows the Gold winner against competitors after multiple-comparison correction. Confidence intervals crossing zero mean the observed advantage is compatible with sampling noise.

![Strategy equity and drawdown](docs/figures/strategy_performance.png)

The equity curve compounds net returns after turnover costs, while drawdown measures the fall from the previous equity high. A good model should have both acceptable growth and tolerable drawdowns.

## Notebooks

Run them from the repository root after extracting the snapshot:

1. `01_data_and_correlations.ipynb` — coverage, normalized prices, returns and Gold/Silver dependence.
2. `02_feature_quality.ipynb` — missingness, distributions, correlation heatmaps and leakage checks.
3. `03_model_benchmark.ipynb` — validation leaderboard, locked-test comparison and model plots.
4. `04_final_selection.ipynb` — equity curves, costs, bootstrap intervals and corrected statistical tests.

The optional `make joint-benchmark` command reproduces the shared-model probe; it is a diagnostic, not an after-the-fact replacement for the separately selected production candidates.

## Generated artifacts

- `data/processed/*_leaderboard.csv`: validation model search results.
- `data/processed/*_test_comparison.csv`: locked-test metrics for every evaluated family.
- `data/processed/*_statistical_tests.csv`: DM tests, Holm-adjusted p-values and paired Sharpe bootstrap intervals.
- `reports/experiment_summary.json`: complete run metadata and selected model bundle information.
- `models/gold_silver_bundle.joblib`: local final bundle, ignored by Git.

## Private publication

The GitHub repository is intended to remain private while the research is ongoing. Hugging Face publication is intentionally separate and should contain only a reviewed winner bundle and model card.

Yahoo Finance and FRED data may have usage and redistribution restrictions. The included snapshot is for this private research repository; verify the applicable terms before sharing it.
