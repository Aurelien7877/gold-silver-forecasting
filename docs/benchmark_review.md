# Benchmark review: what is current, what is tested, and what is not claimed

This note keeps the model comparison honest. “State of the art” is used only for the best result inside the reproducible candidate set; it is not used as a claim about every published model.

## What the recent literature changes

- **Chronos-2** is the most important missing external benchmark for this project. Its technical report describes zero-shot univariate, multivariate and covariate-informed forecasting with group attention; the official release lists a 120M-parameter checkpoint. The repository now has a leakage-safe covariate adapter, but the cached experiment has not yet run because the complete checkpoint could not be downloaded in the current environment.
- **TimesFM 2.5** is a 200M-parameter decoder-only foundation model with a long context and probabilistic quantiles. The current cache evaluates its univariate point forecast on the same locked dates, while the production code does not yet use its optional covariate interface.
- A recent financial-return study compares foundation models with PatchTST, iTransformer and other train-from-scratch baselines under equalized contexts and rolling origins. It reports many ranking wins for foundation models but only sparse, statistically reliable gains over a random-walk benchmark; this is why this repository reports costs, folds and multiple-comparison tests instead of ranking models by one Sharpe.
- A recent Chronos-2 financial study finds that related multivariate context can improve forecasts, while mixing unrelated series can reduce accuracy. Gold and Silver should therefore be grouped only with economically related signals, not with every available market series by default.

Primary references:

- [Chronos-2 technical report](https://arxiv.org/abs/2510.15821) and [official Chronos implementation](https://github.com/amazon-science/chronos-forecasting).
- [TimesFM paper and official repository](https://github.com/google-research/timesfm).
- [Pretrained Time-Series Foundation Models for Financial Return Forecasting](https://arxiv.org/abs/2606.27100).
- [Multivariate Financial Forecasting using the Chronos Time Series Foundation Models](https://arxiv.org/abs/2605.21504).
- [PatchTST](https://arxiv.org/abs/2211.14730), [TimeMixer](https://arxiv.org/abs/2405.14616), and [iTransformer](https://arxiv.org/abs/2310.06625).

## What this repository actually tests

| Track | Information given to the model | Status | Interpretation |
|---|---|---|---|
| Gold local tabular | Causal OHLC, lags, rolling statistics, cross-asset and market features | Tested | ExtraTrees is the current local Gold winner: validation Sharpe `0.618`, locked-test Sharpe `1.187`. |
| Silver local tabular | Same causal feature design, with a directional objective | Tested | Directional Logistic Regression (`C=0.3`) is the current local Silver winner: validation Sharpe `1.451`, locked-test Sharpe `1.315`. |
| Compact PatchTST / TimeMixer / TSMixer | Causal feature windows, small CPU/MPS architectures | Tested | These are useful architecture probes, not full paper-scale reproductions; they underperform the selected local models in this snapshot. |
| Shared global iTransformer-style model | One compact two-output model over the feature matrix | Tested | It is rejected by the common validation rule; separate models are better for this dataset. |
| Chronos-Bolt Tiny / TimesFM 2.5 | Univariate return history only | Tested in cache | They are external zero-shot references, not apples-to-apples with the feature-rich local models. |
| Chronos-2 with covariates | Target plus past-only Gold/Silver-related covariates | Implemented, not yet run | This is the strongest pending external experiment. It needs a successful local checkpoint download and reuses the same dates, horizon, costs and tests through `scripts/benchmark_foundation_models.py`. |

## Current conclusion

The defensible statement is **“best local model found under this protocol”**, not universal SOTA. The Gold result is not significant against the candidate-aware Reality Check (`p=0.090`), while Silver clears this particular null (`p=0.021`); both still show negative historical origins, so neither is deployment-ready.

The next high-value experiment is not another unreported architecture name. It is the implemented Chronos-2 covariate benchmark with a complete local checkpoint, followed by a pre-registered comparison against the current Gold ExtraTrees and Silver directional-logistic winners. If Chronos-2 does not beat them under the same protocol, the local models remain the stronger practical choice for this snapshot.
