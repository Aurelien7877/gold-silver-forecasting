# Gold/Silver next-day forecasting

Reproducible research on one-day-ahead Gold (`GC=F`) and Silver (`SI=F`) returns on Apple Silicon. The project is designed to answer a narrow question: do causal market features and modern time-series models produce a repeatable out-of-sample trading signal after costs?

This is research software, not investment advice. A result called “SOTA” below means the best model inside this repository’s pre-specified candidate set; it is not a claim of universal superiority.

## Current result

Two separate models are retained because the shared Gold/Silver iTransformer-style model was weaker on common validation folds.

| Asset | Selected model | Validation net Sharpe | Locked-test net Sharpe | Test return | Max drawdown |
|---|---|---:|---:|---:|---:|
| Gold | ExtraTrees | 0.618 | 1.187 | +125.0% | -22.6% |
| Silver | Directional Logistic Regression (`C=0.3`) | 1.451 | 1.315 | +545.6% | -44.4% |

The locked test contains 970 common dates and charges 10 bps per unit of turnover. Gold turnover is 0.331; Silver turnover is 0.931, so Silver’s larger return comes with materially higher implementation risk.

The candidate-aware White Reality Check is `p=0.106` for Gold and `p=0.042` for Silver across 18 candidates. These are conditional, block-bootstrap results—not proof that either signal will persist.

## Foundation-model comparison

Chronos-2 and TimesFM 2.5 were finally run locally, on the same locked dates, one-day horizon and 10 bps cost rule. “Covariates” means past-only related signals; their future values are supplied only when reconstructible from information already observed at date `t`.

| Asset | Chronos-2 | Chronos-2 + covariates | TimesFM 2.5 | TimesFM 2.5 + causal covariates | Local winner |
|---|---:|---:|---:|---:|---:|
| Gold test Sharpe | 0.916 | 0.159 | 0.551 | 0.279 | **1.187** |
| Silver test Sharpe | -0.255 | -0.504 | -0.038 | 0.409 | **1.315** |

Gold’s local winner has lower squared forecast error than every foundation track, although none of those four Gold differences survives Holm correction. Silver’s local winner has higher squared error than every foundation track after correction, but it produces the best signed trading result because it was selected for direction rather than return-magnitude accuracy. Forecast loss and tradable performance answer different questions.

## Essential figures

![Out-of-sample forecast story](docs/figures/forecast_story.png)

The upper panel gives price context and marks the untouched test period. The lower panels compare smoothed realized returns with the Gold forecast and Silver direction score; Silver is a sign signal, not a calibrated return forecast.

![Locked-test model comparison](docs/figures/model_benchmark.png)

Each bar is a net Sharpe on the same locked dates and costs. The local winners are highest; foundation models are useful external references, but they receive a different information representation than the feature-rich local models.

![Prediction diagnostics](docs/figures/prediction_diagnostics.png)

The time panels show whether forecasts move with realized returns; the scatter panels show the weak scale of daily prediction errors. Gold IC is 0.148 and Silver IC is 0.115, so information is present but noisy.

![Strategy robustness](docs/figures/robustness.png)

The left panel shows fixed-parameter performance at historical origins; sign changes reveal regime dependence. The right panel compares the observed best Sharpe with the maximum Sharpe expected under a candidate-aware no-skill bootstrap.

![Equity and drawdown](docs/figures/strategy_performance.png)

Equity compounds signed next-day positions after costs, while drawdown measures the distance below the previous peak. The curves make the Silver drawdown and turnover risk visible rather than hiding them behind cumulative return.

For one- or two-line explanations of every tracked graph and notebook panel, read [`docs/figure_guide.md`](docs/figure_guide.md). For complex terms such as leakage, Sharpe, IC, DM tests and SOTA, read [`docs/glossary.md`](docs/glossary.md).

## Method

- **Target:** `log(close[t+1] / close[t])`, separately for Gold and Silver.
- **Inputs:** lagged returns, OHLC-derived features, momentum, rolling volatility, moving-average gaps, drawdown, volume, Gold/Silver ratio, cross-asset statistics and market covariates.
- **Causality:** every feature uses data available at or before `t`; the target is shifted one observation forward. Tests check that changing the final raw row does not alter earlier features.
- **Protocol:** the final 20% is locked before model selection; the first 80% is searched with five expanding walk-forward folds and `gap=1`.
- **Trading rule:** forecast sign maps to `-1`, `0` or `+1`; the position earns the next realized return and pays 10 bps per turnover unit.
- **Evidence:** validation Sharpe selects models; the locked test is reported once; DM tests compare paired forecast loss; moving-block bootstrap estimates Sharpe uncertainty; Holm and White Reality Check address multiple comparisons.

The full literature boundary and tested/not-tested matrix are in [`docs/benchmark_review.md`](docs/benchmark_review.md). In particular, compact local PatchTST/TimeMixer/TSMixer probes are not full paper-scale reproductions, and no claim is made against every possible model, feature set or market period.

## Reproduce locally

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,research]'
unzip -o data/raw/market_data.parquet.zip -d data/raw/
make analysis
make train
make global-benchmark
make robustness
make figures
```

Foundation dependencies include CPU JAX because TimesFM’s official covariate/XReg interface uses it:

```bash
python -m pip install -e '.[foundation]'
python scripts/benchmark_foundation_models.py \
  --chronos-path /path/to/chronos-2 \
  --chronos2-covariates-path /path/to/chronos-2 \
  --timesfm-path /path/to/timesfm-2.5-200m-pytorch \
  --timesfm-covariates-path /path/to/timesfm-2.5-200m-pytorch
python scripts/compute_foundation_statistics.py
```

Weights, raw Parquet and processed CSVs stay local and are ignored by Git. The tracked compressed snapshot and [`data/raw/manifest.json`](data/raw/manifest.json) identify the data source, dates and checksum.

## Hugging Face payloads

The export script creates separate Gold and Silver model-card directories. Each contains only one selected model bundle, its feature schema, configuration and metrics; it does not contain Yahoo/FRED data or secrets.

```bash
python scripts/export_hf.py --asset all --output hf_export
```

The cards use “best local model under the stated protocol”, not universal SOTA, and clearly label the research-only limitation. Upload only after reviewing the generated cards; create each Hub repository as private.

## Repository map

- `src/gold_silver/`: data, causal features, models, validation, backtest and inference.
- `notebooks/`: data/correlation, feature quality, benchmark and final-selection analyses.
- `scripts/`: reproducible download, training, foundation benchmark, statistics, figures and export commands.
- `tests/`: temporal leakage, causal feature, model and statistical helper tests.
- `docs/`: glossary, figure explanations and literature/benchmark review.
