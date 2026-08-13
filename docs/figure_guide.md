# Figure and panel guide

Every plot in the tracked figures and notebooks has a short interpretation below. A visual pattern is a diagnostic; the numerical tables and locked walk-forward protocol remain the evidence used for model selection.

## Tracked report figures

- **`normalized_prices.png`** — Both prices are rebased to 100 on the first common date, so co-movement can be seen without confusing dollar units. Similar trends do not imply that daily returns are interchangeable.
- **`model_benchmark.png`** — Bars compare net locked-test Sharpe across the evaluated families, including the shared global iTransformer probe, for Gold and Silver. This is a diagnostic comparison after validation selection, not permission to choose the highest test bar retrospectively.
- **`prediction_diagnostics.png`** — The left panels smooth forecasts and realized returns over 20 days; the right panels show one-step forecast-versus-outcome pairs. The scatter is clipped visually at the central 99%, while reported metrics use all dates.
- **`gold_robustness.png`** — The left curve shows Gold equity after costs and the right curve shows Sharpe as transaction costs rise. A signal that collapses with modest costs is implementation-fragile.
- **`strategy_performance.png`** — The top panel compounds net strategy returns from one dollar; the bottom panel measures each fall from its previous equity high. Growth must be read together with drawdown depth and duration.
- **`robustness.png`** — The left panel evaluates fixed selected parameters on historical one-year origins; sign changes expose regime sensitivity. The right panel compares the observed best Sharpe with the candidate-aware block-bootstrap null, which now also includes the shared global-model candidate when its predictions exist.

## Notebook 01 — data and dependence

- **Normalized prices** — Rebase both closing-price series to 100 to compare relative paths. This is a level chart and can be dominated by common long-run trends.
- **Daily log returns** — Show close-to-close proportional changes used as targets and inputs. Spikes identify unusually large sessions, not necessarily predictable events.
- **60-day rolling return correlation** — Recompute Pearson co-movement over the latest 60 observations. Moving values show changing dependence rather than a single historical average.
- **Gold/Silver price ratio** — Track the relative price level of Gold to Silver. It is useful for context and cross-asset features, but a ratio trend is not itself a causal signal.
- **Lead/lag bar chart** — Correlate Gold returns with Silver returns shifted from −5 to +5 days. A nonzero bar indicates timing association in this sample, not proof that the relationship survives costs or publication timing.

## Notebook 02 — feature quality

- **Top feature missingness** — Each bar is the fraction of dates without a value before the pipeline’s explicit handling. Missingness must be explained and never filled with future observations.
- **Smoothed feature examples** — Twenty-day averages make noisy returns and volatility features readable. Smoothing is only for visualization; the model receives the causal daily feature values.
- **Feature correlation heatmap** — Color shows pairwise linear redundancy among selected inputs. High correlation may make coefficients unstable, but tree models can still use redundant information and the heatmap is not a feature-selection test.

## Notebook 03 — model benchmark

- **Validation Sharpe bars** — Each bar is the mean net Sharpe over expanding chronological folds, and the highlighted bar is selected without touching the locked test. Negative folds are important evidence of regime fragility.
- **Locked-test Sharpe bars** — These bars evaluate every fitted candidate on the final untouched dates using identical horizon and costs. They are useful for audit and comparison, but cannot be used to revise the selected model.

## Notebook 04 — final selection and robustness

- **Equity curves** — Curves compound signed next-day positions after turnover costs; crossing or flattening shows when one selected strategy adds or loses value relative to the other.
- **Drawdown curves** — Drawdown is the percentage distance below the running equity peak. It makes periods of capital stress visible even when the final cumulative return is positive.
- **Transaction-cost sensitivity** — Net Sharpe is recalculated at 0, 5, 10 and 20 basis points per turnover unit. A stable curve indicates less dependence on optimistic execution assumptions.
- **Holm-adjusted DM bars** — The bars are `−log10(adjusted p-value)` from paired forecast-loss tests; taller bars provide stronger evidence against equal squared-error accuracy. This test concerns forecast loss, not profitability.
- **Rolling-origin stability** — The selected fixed parameters are refit at several historical origins and tested for one year. Positive and negative points show whether the same rule survives different market regimes.
- **White Reality Check p-value** — The candidate-aware block bootstrap repeatedly selects the maximum Sharpe among all local candidates under a no-skill null. A low value is stronger evidence than a single high Sharpe, but it still depends on the candidate universe and bootstrap assumptions.
