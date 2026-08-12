#!/usr/bin/env python
"""Generate small, repository-friendly PNG figures for the English README."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    main()
