#!/usr/bin/env python
"""Predict the next Gold/Silver return from the latest cached data."""

from __future__ import annotations

import argparse
import json

from gold_silver.artifacts import ModelBundle, predict_next
from gold_silver.config import load_config
from gold_silver.data import load_cached_market_data
from gold_silver.features import build_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--bundle", default="models/gold_silver_bundle.joblib")
    args = parser.parse_args()
    config = load_config(args.config)
    features = build_features(load_cached_market_data(config), config.features)
    bundle = ModelBundle.load(args.bundle)
    print(json.dumps(predict_next(features, bundle), indent=2))


if __name__ == "__main__":
    main()
