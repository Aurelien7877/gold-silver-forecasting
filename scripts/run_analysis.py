#!/usr/bin/env python
"""Build features and write correlation analysis from cached data."""

from __future__ import annotations

import argparse
from pathlib import Path

from gold_silver.analysis import correlation_report, summarize_correlations
from gold_silver.config import load_config
from gold_silver.data import load_cached_market_data
from gold_silver.features import build_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    raw = load_cached_market_data(config)
    features = build_features(raw, config.features)
    report = correlation_report(features)
    output = Path(config.data.processed_dir)
    output.mkdir(parents=True, exist_ok=True)
    report.to_csv(output / "correlations.csv", index=False)
    (output / "correlations_summary.txt").write_text(summarize_correlations(report), encoding="utf-8")
    print(summarize_correlations(report))


if __name__ == "__main__":
    main()
