#!/usr/bin/env python
"""Download and cache market data."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from gold_silver.config import load_config
from gold_silver.data import download_market_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    load_dotenv()
    config = load_config(args.config)
    frame = download_market_data(config, save=True)
    print(f"Downloaded {len(frame):,} rows and {len(frame.columns)} columns.")
    print(f"Range: {frame.index.min().date()} -> {frame.index.max().date()}")


if __name__ == "__main__":
    main()
