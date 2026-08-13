# Benchmark review: what is current, what is tested, and what is not claimed

This note keeps the model comparison honest. “State of the art” is used only for the best result inside the reproducible candidate set; it is not used as a claim about every published model.

## What the recent literature changes

- **Chronos-2** supports zero-shot univariate, multivariate and covariate-informed forecasting; the official release lists a 120M-parameter checkpoint. Both its univariate and past-only covariate tracks now run locally on this project’s locked protocol.
- **TimesFM 2.5** is a 200M-parameter decoder-only foundation model with a long context and probabilistic quantiles. Both its univariate point forecast and its official XReg/covariate interface are now evaluated; the latter uses only lagged covariates whose next value is known from the current row.
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
| Chronos-2 / TimesFM 2.5, univariate | Return history only | Tested | External zero-shot references on the same dates, horizon and costs; they do not receive the local engineered feature matrix. |
| Chronos-2, past-only covariates | Target plus related Gold/Silver and market covariates | Tested | The future context is assembled only from values observable at the forecast origin. |
| TimesFM 2.5, causal XReg covariates | Target plus lagged numerical covariates | Tested | Uses the official `forecast_with_covariates` interface with context-plus-horizon arrays and a fixed ridge penalty. |

## Current conclusion

The defensible statement is **“best local model found under this protocol”**, not universal SOTA. After adding the foundation candidates, the Reality Check is `p=0.106` for Gold and `p=0.042` for Silver; both assets still show negative historical origins, so neither is deployment-ready.

On locked-test net Sharpe, the local winners are Gold `1.187` and Silver `1.315`. The strongest external result is Chronos-2 univariate on Gold (`0.916`); TimesFM covariates are the strongest external Silver track (`0.409`). Neither beats the corresponding local winner. Gold’s local forecast loss is lower than every foundation track but not significantly after Holm correction; Silver’s local forecast loss is significantly higher than every foundation track, while its direction-oriented trading rule still wins on net Sharpe. These results support local-best claims for this snapshot, not a universal ranking.
