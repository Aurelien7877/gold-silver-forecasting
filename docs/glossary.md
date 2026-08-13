# Terms and metrics in plain English

Each entry gives the practical meaning used in this repository and the main caution when interpreting it.

## Data and prediction

- **Forecast horizon** — `J+1` means that features available at the close of date `t` are used to forecast the return realized on the next common trading date. A one-day horizon avoids pretending that intraday information is available before the session closes.
- **Log return** — `log(close[t+1] / close[t])` is an additive, percentage-like price change. The backtest converts it back to a simple return before compounding a strategy.
- **OHLC** — Open, High, Low and Close describe the completed session. Range, close location and overnight gap are valid only because the signal is formed after that session.
- **Feature** — A feature is an input variable computed from information known at the forecast origin. Lags, rolling statistics and cross-asset changes are inputs; the future target is never an input.
- **Target** — The target is the value the model is asked to predict: the next Gold or Silver log return. It is shifted one row after features are built, which is the key temporal alignment.
- **Covariate** — A covariate is an additional explanatory time series such as dollar, volatility or oil. A past-only covariate is safe; a future covariate is safe only when its future value would genuinely be known at prediction time.
- **Chronos-2 covariate track** — This is the Chronos-2 experiment where the target return is forecast jointly with a small set of past-only, economically related signals. The model sees no future target or future covariate values.
- **Synthetic regular timestamp** — Chronos-2's dataframe API expects evenly spaced timestamps, whereas market data omit weekends and holidays. The adapter uses a regular observation index while preserving row order; it does not invent weekend prices.
- **Lookback** — The lookback is the number of previous observations supplied to a sequence or foundation model. A longer lookback can capture slower regimes but increases computation and the risk of fitting stale relationships.
- **Signal score** — A score is the model output before the trading rule converts it into long, flat or short. For Silver, the selected logistic model outputs `P(up) - 0.5`, so it is a directional confidence score rather than a return estimate.
- **Signal threshold** — A threshold sets a no-trade zone around zero. It can reduce turnover, but it must be selected on chronological validation and never by inspecting the locked test.
- **Leakage** — Leakage occurs when information from the future enters a feature, fit, scaler or model decision. The tests and chronological split are designed to catch this, but passing a test is not a substitute for reviewing the data timestamps.
- **Target buffer / gap** — The one-row gap separates development targets from the locked test target. It prevents the final development label from being the return that starts the test period.
- **Out-of-sample (OOS)** — OOS means the prediction was generated without fitting on that date’s outcome. The locked test is the strictest OOS period in this project.
- **Walk-forward validation** — The model trains on earlier rows and validates on later rows, then the training window expands. This approximates repeated deployment and preserves temporal order.
- **Expanding window** — An expanding window keeps all earlier training observations while moving the validation origin forward. It is different from a rolling window, which discards old observations.
- **Locked test** — The final 20% is held untouched during feature decisions, model search and parameter selection. It should be read once as a final audit, not repeatedly used for tuning.

## Model families

- **Baseline** — A baseline is a simple reference such as zero return or the last return. It is not the final predictor; it tells us whether a complex model adds evidence beyond a trivial rule.
- **Ridge** — Ridge is linear regression with an L2 penalty that shrinks correlated coefficients. It is a useful low-variance benchmark when many engineered features overlap.
- **Elastic Net** — Elastic Net combines L1 sparsity and L2 shrinkage. It can remove weak variables while stabilizing groups of correlated features, but it still assumes a mostly linear relationship.
- **Logistic directional model** — Logistic regression estimates the probability that the next return is positive. The trading layer uses the sign of its centered probability, so good direction can matter even when return magnitude is poorly calibrated.
- **ExtraTrees** — ExtraTrees averages randomized decision trees and can capture nonlinear interactions without requiring feature scaling. It is strong on this snapshot for Gold, but a large test Sharpe alone does not establish generalization.
- **Gradient boosting** — Gradient boosting adds weak trees sequentially to correct previous errors. It can be powerful on tabular data but is sensitive to depth, learning rate and regime changes.
- **Transformer** — A Transformer uses attention to combine information across positions or variables. The compact local versions here are CPU/MPS experiments, not full-scale replicas of published training recipes.
- **PatchTST** — PatchTST groups adjacent observations into temporal patches before attention. The local implementation tests the idea under a small Apple-Silicon budget and is not a claim to reproduce the paper’s full benchmark.
- **TimeMixer / TSMixer** — These models mix information across time and channels with MLP-style blocks. Their poor result here is evidence against these compact configurations on this task, not against every possible configuration.
- **iTransformer** — iTransformer treats variables as tokens and attends across them after embedding their histories. Our global version tests shared Gold/Silver learning; it was rejected by the same validation rule as the separate models.
- **Foundation model** — A pretrained time-series model transfers patterns learned elsewhere instead of fitting all weights from scratch here. Transfer can help low-data tasks, but financial returns have weak persistence and structural breaks.
- **Zero-shot** — Zero-shot means the pretrained model is used without task-specific fitting. It is a fair external reference only when context, horizon, dates and costs are equalized.
- **Fine-tuning** — Fine-tuning updates a pretrained model on the target data. It can improve adaptation but increases compute, tuning degrees of freedom and leakage risk inside a small financial sample.
- **MPS** — MPS is Apple’s Metal backend for PyTorch. It can accelerate compatible neural operations, while CPU remains the safer reproducibility fallback.

## Trading and performance metrics

- **Position** — The strategy maps a forecast to `-1`, `0` or `+1`. The position is applied to the next realized return, not to the same-day return used to form the feature.
- **Turnover** — Turnover is the absolute change in position, including the initial entry. High turnover magnifies execution costs and can make a paper signal untradeable.
- **Basis point (bps)** — One basis point is `0.01%`; 10 bps is `0.001` in decimal return. Here the cost is charged per unit of turnover.
- **Volatility** — Volatility is the standard deviation of daily net strategy returns, annualized by `sqrt(252)`. It measures variability, not the probability of a loss.
- **Sharpe ratio** — Sharpe is mean daily net return divided by daily volatility, annualized. It is the primary selection metric here, but it is noisy and especially unstable when regimes differ.
- **Sortino ratio** — Sortino divides return by downside volatility only. It highlights downside behavior but can look extreme when the sample contains few negative observations.
- **Maximum drawdown** — Maximum drawdown is the largest peak-to-trough fall in cumulative equity. It describes path risk and recovery difficulty, not just the final return.
- **Calmar ratio** — Calmar is annualized return divided by absolute maximum drawdown. It rewards growth with smaller drawdowns but inherits the instability of both inputs.
- **Hit rate** — Hit rate is the fraction of net daily strategy returns above zero. It does not measure payoff asymmetry: a model can win often and still lose money on a few large moves.
- **Information coefficient (IC)** — IC is the correlation between forecast score and realized return. It measures directional information, not a guaranteed net trading edge.
- **MAE / RMSE** — MAE is average absolute forecast error; RMSE is the square root of average squared error and penalizes large misses more strongly. Neither metric includes positions or transaction costs.
- **Cost sensitivity** — Cost sensitivity recomputes the same locked predictions under 0, 5, 10 and 20 bps assumptions. A robust signal should not rely on a single optimistic friction assumption.

## Statistical evidence

- **Bootstrap interval** — A moving-block bootstrap resamples short contiguous blocks to estimate uncertainty while preserving some dependence. It is an uncertainty diagnostic, not a guarantee that the next sample will match the interval.
- **Confidence interval** — A 95% interval is a range produced by a specified resampling or sampling procedure. It does not mean there is a 95% probability that the fixed future performance lies inside it.
- **Diebold–Mariano (DM) test** — DM compares forecast losses on the same OOS dates. A small p-value supports different predictive accuracy under the test assumptions; it does not by itself prove higher profitability.
- **HAC / Newey–West variance** — HAC estimates uncertainty while allowing short-run autocorrelation in the loss difference. The choice of lag length matters and does not repair all forms of non-stationarity.
- **p-value** — A p-value is the probability of data at least this extreme under a specified null model. It is not the probability that a model is correct or will remain profitable.
- **Holm–Bonferroni correction** — Holm adjusts a family of p-values when many pairwise competitors are tested. It reduces false discoveries and makes a significance claim harder.
- **White Reality Check** — The Reality Check resamples all candidate strategy returns and asks how often the best Sharpe under a no-skill null is at least as large as the observed winner. It directly addresses selection among many candidates, but depends on the candidate universe and bootstrap design.
- **Regime analysis** — Regime tables split the locked test after the fact by direction, volatility or calendar period. They diagnose fragility; they must not become extra tuning labels.
- **Statistical power** — Power is the ability to detect a real effect with the available sample. Daily returns are noisy, so a non-significant result can mean either no edge or insufficient information.
- **Null hypothesis** — The null is the reference claim being tested, such as equal forecast loss or no candidate skill. A test result is only meaningful relative to that explicitly stated null.
- **SOTA** — SOTA means the strongest result inside a fully specified comparison set. This repository can support a reproducible local-best claim; it cannot prove superiority over every unpublished model or dataset.

## Reproducibility terms

- **Deterministic search** — The candidate list, folds, seed and scoring rule are fixed before evaluation. Determinism improves repeatability but does not remove statistical selection bias.
- **Feature preprocessing** — Scaling, missing-value handling and column order are part of the model pipeline. They must be fitted on training data only and stored with the final bundle.
- **Model bundle** — The bundle contains the fitted models, feature order, configuration and metrics needed for local inference. It is an artifact for research, not a promise that future data will behave similarly.
- **Benchmark parity** — Two models are comparable only when they receive the same dates, horizon, target definition, test split, cost rule and statistical evaluation. Different information sets should be labeled rather than hidden.
