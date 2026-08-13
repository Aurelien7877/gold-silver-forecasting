# Figure and panel guide

Every plot in the tracked figures and notebooks has a short interpretation below. A visual pattern is a diagnostic; the numerical tables and locked walk-forward protocol remain the evidence used for model selection.

## Tracked report figures

- **`forecast_story.png`** — The top panel connects normalized prices to the locked test window; the bottom panels show Gold's return forecast and Silver's direction score against realized returns. The current picture is visually consistent with useful signal, but the wide swings and different Silver scale warn against reading it as a guaranteed price path.
- **`normalized_prices.png`** — Both prices are rebased to 100 on the first common date, so co-movement can be seen without confusing dollar units. The level correlation is `0.926`, while the stronger visual trend does not establish daily predictive power.
- **`model_benchmark.png`** — Bars compare net locked-test Sharpe across local, shared-global and foundation families for Gold and Silver. Gold ExtraTrees reaches `1.187` and Silver directional logistic `1.315`; the foundation bars are same-window external references, not a universal ranking.
- **`prediction_diagnostics.png`** — The left panels smooth forecasts and realized returns over 20 days; the right panels show one-step forecast-versus-outcome pairs. The selected models have locked-test ICs of `0.148` for Gold and `0.115` for Silver, which indicates information but not calibrated return magnitude.
- **`gold_robustness.png`** — The left curve shows Gold equity after costs and the right curve shows Sharpe as transaction costs rise. Gold declines from `1.620` at zero cost to `0.751` at 20 bps, so it survives friction in this sample but is not cost-invariant.
- **`strategy_performance.png`** — The top panel compounds net strategy returns from one dollar; the bottom panel measures each fall from its previous equity high. Gold and Silver finish positively, but the visible drawdowns show why cumulative return must be read with path risk.
- **`robustness.png`** — The left panel evaluates fixed selected parameters on historical one-year origins; sign changes expose regime sensitivity. The right panel compares the observed best Sharpe with the candidate-aware block-bootstrap null: Gold p=`0.106` is inconclusive, while Silver p=`0.042` is conditional on the expanded 18-candidate universe.

## Notebook 01 — data and dependence

- **Normalized prices** — Rebase both closing-price series to 100 to compare relative paths. The common rise is consistent with strong level dependence (`Pearson=0.926`), but a level chart can be dominated by shared trends.
- **Daily log returns** — Show the close-to-close changes used as targets and inputs. Gold/Silver return Pearson correlation is `0.778`, so the assets co-move daily but still leave substantial asset-specific noise.
- **60-day rolling return correlation** — Recompute Pearson co-movement over the latest 60 observations. The full-sample 60-day average is about `0.781`, which looks stable descriptively but is not evidence that correlation forecasts the next return.
- **Gold/Silver price ratio** — Track the relative price level of Gold to Silver. It is useful for context and cross-asset features, but a ratio trend is not itself a causal or tradable signal.
- **Lead/lag bar chart** — Correlate Gold returns with Silver returns shifted from −5 to +5 days. The peak is contemporaneous at lag `0` (`0.778`); adjacent lags are near zero, so this analysis does not support a simple one-day leader claim.

## Notebook 02 — feature quality

- **Top feature missingness** — Each bar is the fraction of dates without a value before the pipeline’s explicit handling. A high bar signals provider coverage or initialization limits; it is not a reason to fill values from the future.
- **Smoothed feature examples** — Twenty-day averages make noisy returns and volatility features readable. Smoothing is only for visualization; the model receives the causal daily values, so a smooth line is not a smooth forecast.
- **Feature correlation heatmap** — Color shows pairwise linear redundancy among selected inputs. High correlation explains why shrinkage helps linear models, but the heatmap is not a predictive or causal feature-selection test.

## Notebook 03 — model benchmark

- **Validation Sharpe bars** — Each bar is the mean net Sharpe over expanding chronological folds, and the highlighted bar is selected without touching the locked test. The selected values are `0.618` for Gold and `1.451` for Silver, while negative folds still show regime fragility.
- **Locked-test Sharpe bars** — These bars evaluate every fitted candidate on the final untouched dates using identical horizon and costs. Gold reaches `1.187` and Silver `1.315`; these confirm that the selected rules remained positive here but cannot prove universal superiority.

## Notebook 04 — final selection and robustness

- **Equity curves** — Curves compound signed next-day positions after turnover costs; crossing or flattening shows when one selected strategy adds or loses value relative to the other. Silver’s larger ending equity comes with materially higher turnover and drawdown exposure.
- **Drawdown curves** — Drawdown is the percentage distance below the running equity peak. The locked-test maxima are about `−22.6%` for Gold and `−44.4%` for Silver, so final return alone is an incomplete risk description.
- **Transaction-cost sensitivity** — Net Sharpe is recalculated at 0, 5, 10 and 20 basis points per turnover unit. Both selected signals remain positive at 20 bps in this cache, but the decline quantifies implementation fragility.
- **Holm-adjusted DM bars** — The bars are `−log10(adjusted p-value)` from paired forecast-loss tests; taller bars provide stronger evidence against equal squared-error accuracy. This test concerns forecast loss, not profitability, and the Silver directional winner is not necessarily the lowest-MSE forecast.
- **Rolling-origin stability** — The selected fixed parameters are refit at several historical origins and tested for one year. Gold has four positive of six origins and Silver four of six, so both signals show regime dependence despite positive locked-test totals.
- **White Reality Check p-value** — The candidate-aware block bootstrap repeatedly selects the maximum Sharpe among the 18 available candidates under a no-skill null. The current p-values (`0.106` Gold, `0.042` Silver) are useful evidence, not a universal SOTA certificate.
