# Gold & Silver Forecasting

<p align="center">
  <img src="docs/figures/mascot_gold.png" alt="Aurum-1D Gold research mascot" width="25%" />
  <img src="docs/figures/mascot_silver.png" alt="Argent-1D Silver research mascot" width="25%" />
</p>

<p align="center"><em>Two models, two metals, one reproducible forecasting protocol.</em></p>

Research-grade, reproducible machine learning for one-day-ahead Gold (`GC=F`) and Silver (`SI=F`) futures. The repository compares causal feature models with Chronos-2 and TimesFM 2.5 under the same walk-forward splits, horizon and transaction costs.

> **Research only.** Historical backtests are not investment advice, a trading recommendation or a guarantee of future performance.

[![Tests](https://img.shields.io/badge/tests-18%20passed-2ea44f)](tests/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%20Apple%20Silicon-3776ab)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-8b5cf6)](LICENSE)
[![Hugging Face Aurum-1D](https://img.shields.io/badge/Hugging%20Face-Aurum--1D-f59e0b)](https://huggingface.co/AurelPx/Aurum-1D)
[![Hugging Face Argent-1D](https://img.shields.io/badge/Hugging%20Face-Argent--1D-cbd5e1)](https://huggingface.co/AurelPx/Argent-1D)

## Results at a glance

The final 20% of the common history is a locked test set: 970 daily observations, one-day horizon and 10 bps charged per unit of turnover. Models were selected only on the earlier expanding walk-forward validation folds.

| Asset | Selected model | Validation net Sharpe | Locked-test net Sharpe | Test return | Max drawdown |
|---|---|---:|---:|---:|---:|
| Gold | ExtraTrees | **0.618** | **1.187** | **+125.0%** | -22.6% |
| Silver | Directional Logistic Regression | **1.451** | **1.315** | **+545.6%** | -44.4% |

**What this means:** in the repository's identical trading backtest, the selected local models outperform every tested foundation track on net Sharpe for both metals. This is a strategy-performance result within this sample. It is not proof of universal SOTA or future profitability.

## Local models vs. foundation models

| Locked-test net Sharpe | Gold | Silver |
|---|---:|---:|
| Selected local model | **1.187** | **1.315** |
| Chronos-2, univariate | 0.916 | -0.255 |
| Chronos-2, past-only covariates | 0.159 | -0.504 |
| TimesFM 2.5, univariate | 0.551 | -0.038 |
| TimesFM 2.5, causal covariates | 0.279 | 0.409 |

### The important nuance

- **Trading objective:** our selected models win on net Sharpe after costs for Gold and Silver.
- **Forecast-loss objective:** Gold also has lower squared error than all four foundation tracks, but the paired differences do not survive Holm correction. Silver has lower foundation-model squared error, while the directional local model achieves the stronger signed trading result.
- **Statistical guardrail:** the candidate-aware White Reality Check gives `p=0.106` for Gold and `p=0.042` for Silver across 18 candidates. This adjusts for searching across many models; it is not a guarantee that the edge persists.

**Net Sharpe** is risk-adjusted return after transaction costs. **Squared error** measures numerical forecast accuracy. A model can be better at predicting small return magnitudes yet worse when its signal is converted into a long/short position.

![Locked-test model comparison](docs/figures/model_benchmark.png)

The bars use the same dates, horizon and cost rule. Local and foundation models are compared transparently, while the foundation models receive their own stated univariate or causal-covariate inputs.

## What the pipeline does

- Downloads and caches Yahoo Finance data for Gold and Silver, with optional macro and market covariates.
- Normalizes timestamps, checks missingness and common dates, and records a source manifest with checksums.
- Builds strictly causal features: lags, momentum, rolling volatility, moving-average gaps, drawdown, volume, Gold/Silver ratio, spreads, rolling correlations and cross-asset returns.
- Predicts `log(close[t+1] / close[t])` for Gold and Silver; Silver's final selected model is optimized for directional trading.
- Searches models with expanding walk-forward folds and `gap=1`; no random shuffle is used.
- Backtests positions in `{-1, 0, +1}` with explicit turnover costs.
- Runs robustness checks: bootstrap intervals, regime splits, Diebold-Mariano paired tests, Holm correction and White Reality Check.

![Out-of-sample forecast story](docs/figures/forecast_story.png)

The upper panel gives price context and marks the untouched test period. The lower panels show the Gold return forecast and the Silver directional score.

![Prediction diagnostics](docs/figures/prediction_diagnostics.png)

These diagnostics show why the result must be interpreted cautiously: daily return forecasts are noisy even when their signs produce a useful historical strategy.

![Strategy robustness](docs/figures/robustness.png)

Rolling-origin results and the candidate-aware bootstrap make regime dependence and model-selection uncertainty visible.

More concise explanations of the figures and technical terms are available in [`docs/figure_guide.md`](docs/figure_guide.md) and [`docs/glossary.md`](docs/glossary.md).

## Hugging Face model family

The selected estimators are published as the public **Aurum-1D / Argent-1D** model family. These are SOTA-era, foundation-model-benchmarked research artifacts with the fitted model weights included:

- [Aurum-1D, Gold](https://huggingface.co/AurelPx/Aurum-1D)
- [Argent-1D, Silver](https://huggingface.co/AurelPx/Argent-1D)

Each card starts with its mascot and contains `estimator.joblib`, the feature schema, configuration, metrics, inference requirements and the full Chronos-2 / TimesFM 2.5 audit. The payloads do not include raw Yahoo/FRED data or external foundation-model weights.

### Use the trained weights locally

No API or Space is required. Clone the public source repository, install the package, and refresh the local market-data cache:

```bash
git clone https://github.com/Aurelien7877/gold-silver-forecasting.git
cd gold-silver-forecasting
python -m pip install -e .
python scripts/download_data.py
```

Then download the already-trained weights from Hugging Face and run the causal feature builder locally:

```python
import json
import math
from pathlib import Path

import joblib
from huggingface_hub import hf_hub_download

from gold_silver.config import load_config
from gold_silver.data import load_cached_market_data
from gold_silver.features import build_features


def load_inputs(repo_id):
    model = joblib.load(hf_hub_download(repo_id, "model.joblib"))
    schema = json.loads(Path(
        hf_hub_download(repo_id, "feature_schema.json")
    ).read_text())
    config = load_config("configs/default.yaml")
    history = load_cached_market_data(config)
    features = build_features(history, config)
    X_next = features[schema["feature_columns"]].tail(1)
    return model, history, X_next


gold, history, X_gold = load_inputs("AurelPx/Aurum-1D")
gold_log_return = float(gold.predict(X_gold)[0])
last_gold_close = float(history["gold_close"].dropna().iloc[-1])
print("Gold next-day log return:", gold_log_return)
print("Gold implied next close:", last_gold_close * math.exp(gold_log_return))

silver, _, X_silver = load_inputs("AurelPx/Argent-1D")
silver_score = float(silver.predict_proba(X_silver)[0, 1] - 0.5)
silver_position = 1 if silver_score > 0 else -1 if silver_score < 0 else 0
print("Silver direction score:", silver_score)
print("Silver position:", silver_position)
```

Gold returns a next-day log-return estimate. Silver returns a centered probability score; its sign gives the long/short direction used by the research backtest. The feature builder remains in GitHub so preprocessing is local, causal and versioned.

## Reproduce

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
python scripts/export_hf.py --asset all --output hf_export
```

Foundation-model benchmarking is optional and requires the extra dependencies and local checkpoint paths described in [`docs/benchmark_review.md`](docs/benchmark_review.md). Large weights remain outside Git.

## Repository map

- `src/gold_silver/`: data, causal features, models, validation, backtesting and inference.
- `scripts/`: download, train, benchmark, statistical tests, figures and HF export.
- `notebooks/`: exploration, correlations, model benchmark and final selection.
- `tests/`: temporal leakage, causal-feature, model and statistical helper tests.
- `docs/`: benchmark boundary, glossary, figure guide and Hugging Face card patterns.

## Scope and limitations

This is a controlled research comparison, not a claim against every published architecture, feature set, market period or execution environment. Yahoo Finance data can change; transaction costs, slippage, liquidity and market impact are simplified. The repository reports both positive findings and uncertainty so the result can be reproduced and challenged.
