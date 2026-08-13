#!/usr/bin/env python
"""Run model benchmark, selection and locked test evaluation."""

from __future__ import annotations

import argparse
import json

from gold_silver.config import load_config
from gold_silver.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--download", action="store_true", help="Download instead of using the local cache.")
    parser.add_argument(
        "--include-foundation-models",
        action="store_true",
        help="Use installed Chronos/TimesFM adapters; model weights may be downloaded.",
    )
    parser.add_argument("--chronos-path", default=None, help="Local Chronos checkpoint directory.")
    parser.add_argument(
        "--chronos2-covariates-path",
        default=None,
        help="Local Chronos-2 checkpoint for the covariate-aware adapter.",
    )
    parser.add_argument("--timesfm-path", default=None, help="Local TimesFM checkpoint directory.")
    parser.add_argument(
        "--timesfm-covariates-path",
        default=None,
        help="Local TimesFM checkpoint for the causal covariate-aware adapter.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.include_foundation_models:
        config.search.include_foundation_models = True
    if args.chronos_path:
        import os

        os.environ["GOLD_SILVER_CHRONOS_PATH"] = args.chronos_path
    if args.chronos2_covariates_path:
        import os

        os.environ["GOLD_SILVER_CHRONOS2_COVARIATES_PATH"] = args.chronos2_covariates_path
    if args.timesfm_path:
        import os

        os.environ["GOLD_SILVER_TIMESFM_PATH"] = args.timesfm_path
    if args.timesfm_covariates_path:
        import os

        os.environ["GOLD_SILVER_TIMESFM_COVARIATES_PATH"] = args.timesfm_covariates_path
    result = run_experiment(config, use_cache=not args.download)
    print(json.dumps(result.get("assets", {}), indent=2, default=str))


if __name__ == "__main__":
    main()
