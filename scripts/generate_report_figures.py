#!/usr/bin/env python
"""Generate small, repository-friendly PNG figures for the English README."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    output = ROOT / "docs/figures"
    output.mkdir(parents=True, exist_ok=True)
    processed = ROOT / "data/processed"

    raw = pd.read_parquet(ROOT / "data/raw/market_data.parquet")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = ["_".join(str(part) for part in column if str(part) != "") for column in raw.columns]
    close_columns = [column for column in raw.columns if str(column).lower().endswith("_close")]
    close = raw[close_columns].rename(columns=lambda column: str(column).lower().replace("_close", ""))
    close = close[[column for column in ["gold", "silver", "gc=f", "si=f"] if column in close]]
    close = close.rename(columns={"gc=f": "gold", "si=f": "silver"})
    close = close[[column for column in ["gold", "silver"] if column in close]].dropna()
    normalized = close.div(close.iloc[0]).mul(100)
    ax = normalized.plot(figsize=(10, 5), color=["#d49a00", "#777777"], title="Normalized Gold and Silver prices")
    ax.set_ylabel("Index (start = 100)")
    ax.figure.tight_layout()
    ax.figure.savefig(output / "normalized_prices.png", dpi=140)
    plt.close(ax.figure)

    gold = pd.read_csv(processed / "gold_test_comparison.csv")
    silver = pd.read_csv(processed / "silver_test_comparison.csv")
    for asset, table in (("gold", gold), ("silver", silver)):
        foundation = processed / f"{asset}_foundation_comparison.csv"
        if foundation.exists():
            extra = pd.read_csv(foundation).rename(columns={"validation_sharpe": "best_score"})
            extra["selected"] = False
            for column in table.columns:
                if column not in extra:
                    extra[column] = 0.0
            extra = extra[table.columns]
            if asset == "gold":
                gold = pd.concat([table, extra], ignore_index=True)
            else:
                silver = pd.concat([table, extra], ignore_index=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    positions = range(len(gold))
    ax.bar([position - 0.2 for position in positions], gold["sharpe"], width=0.4, label="Gold", color="#d49a00")
    ax.bar([position + 0.2 for position in positions], silver["sharpe"], width=0.4, label="Silver", color="#777777")
    ax.set_xticks(list(positions), gold["family"], rotation=35, ha="right")
    ax.set_ylabel("Net Sharpe")
    ax.set_title("Locked-test model comparison")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "model_benchmark.png", dpi=140)
    plt.close(fig)

    equity = pd.read_csv(processed / "gold_test_equity.csv", index_col=0, parse_dates=True)
    costs = pd.read_csv(processed / "gold_cost_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    equity["equity"].plot(ax=axes[0], color="#d49a00", title="Gold locked-test equity")
    axes[0].set_ylabel("Growth of $1")
    axes[1].plot(costs["transaction_cost_bps"], costs["sharpe"], marker="o", color="#244a7c")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(title="Gold cost sensitivity", xlabel="Cost (bps)", ylabel="Net Sharpe")
    fig.savefig(output / "gold_robustness.png", dpi=140)
    plt.close(fig)

    prediction_specs = {}
    for asset, color in (("Gold", "#d49a00"), ("Silver", "#777777")):
        frame = pd.read_csv(processed / f"{asset.lower()}_test_comparison.csv")
        winner = frame.loc[frame["selected"], "family"].iloc[0]
        prediction_specs[asset] = (f"{asset.lower()}_oos_predictions.csv", winner, color)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for row, (asset, (filename, model, color)) in enumerate(prediction_specs.items()):
        frame = pd.read_csv(processed / filename, index_col=0, parse_dates=True)
        actual = frame["realized_return"]
        prediction = frame[model]
        window = 20
        axes[row, 0].plot(actual.index, actual.rolling(window).mean(), color="#222222", linewidth=1.2, label="Realized")
        axes[row, 0].plot(prediction.index, prediction.rolling(window).mean(), color=color, linewidth=1.2, label=f"{model} prediction")
        axes[row, 0].axhline(0, color="black", linewidth=0.7)
        axes[row, 0].set_title(f"{asset}: 20-day rolling return — prediction vs realized")
        axes[row, 0].set_ylabel("Log return")
        axes[row, 0].legend(loc="upper left")
        combined = np.concatenate([prediction.to_numpy(), actual.to_numpy()])
        bound = float(np.nanquantile(np.abs(combined), 0.99))
        central = prediction.abs().le(bound) & actual.abs().le(bound)
        axes[row, 1].scatter(prediction[central], actual[central], s=8, alpha=0.25, color=color, edgecolors="none")
        axes[row, 1].plot([-bound, bound], [-bound, bound], linestyle="--", color="#555555", linewidth=0.8)
        axes[row, 1].axhline(0, color="black", linewidth=0.6)
        axes[row, 1].axvline(0, color="black", linewidth=0.6)
        axes[row, 1].set_title(f"{asset}: one-step prediction scatter (central 99%)")
        axes[row, 1].set_xlabel("Predicted log return")
        axes[row, 1].set_ylabel("Realized log return")
    fig.savefig(output / "prediction_diagnostics.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for asset, equity_file, color in [
        ("Gold", "gold_test_equity.csv", "#d49a00"),
        ("Silver", "silver_test_equity.csv", "#777777"),
    ]:
        equity = pd.read_csv(processed / equity_file, index_col=0, parse_dates=True)
        axes[0].plot(equity.index, equity["equity"], label=asset, color=color, linewidth=1.4)
        axes[1].plot(equity.index, equity["drawdown"], label=asset, color=color, linewidth=1.2)
    axes[0].set_title("Locked-test strategy equity")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend()
    axes[1].set_title("Locked-test drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].legend()
    fig.savefig(output / "strategy_performance.png", dpi=140)
    plt.close(fig)

    stability = {
        "Gold": pd.read_csv(processed / "gold_origin_stability.csv"),
        "Silver": pd.read_csv(processed / "silver_origin_stability.csv"),
    }
    reality = pd.concat(
        [
            pd.read_csv(processed / "gold_reality_check.csv").assign(asset="Gold"),
            pd.read_csv(processed / "silver_reality_check.csv").assign(asset="Silver"),
        ],
        ignore_index=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for asset, color in (("Gold", "#d49a00"), ("Silver", "#777777")):
        table = stability[asset]
        axes[0].plot(table["origin"], table["sharpe"], marker="o", label=asset, color=color)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Fixed-parameter rolling-origin stability")
    axes[0].set_ylabel("Net Sharpe")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend()
    positions = np.arange(len(reality))
    axes[1].bar(positions - 0.18, reality["observed_max_sharpe"], width=0.36, label="Observed max", color="#244a7c")
    axes[1].bar(positions + 0.18, reality["bootstrap_max_sharpe_95"], width=0.36, label="Bootstrap 95% max", color="#b7c9df")
    axes[1].set_xticks(positions, reality["asset"])
    axes[1].set_ylabel("Sharpe")
    axes[1].set_title("White Reality Check: winner vs data-snooping null")
    axes[1].legend(fontsize=8)
    for position, p_value in zip(positions, reality["p_value_max_sharpe"]):
        axes[1].text(position, max(reality["bootstrap_max_sharpe_95"]) * 1.02, f"p={p_value:.3f}", ha="center", fontsize=9)
    fig.savefig(output / "robustness.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
