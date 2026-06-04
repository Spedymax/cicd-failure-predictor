"""CLI to merge raw datasets and emit an engineered feature parquet.

Reads zero or more raw parquet files (real GHA + synthetic), concatenates
them with a `dataset_source` column, runs the full feature pipeline, then
writes the result.

Usage::

    python -m features.extract \\
        --inputs ../data/raw/synthetic.parquet ../data/raw/gha_runs.parquet \\
        --out ../data/processed/features.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from features.transformers import engineer_features, feature_columns

logger = logging.getLogger(__name__)


def _load_input(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df["dataset_source"] = "synthetic" if "synthetic" in path.stem else "real"
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Engineer features from raw datasets")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")

    frames = [_load_input(Path(p)) for p in args.inputs]
    raw = pd.concat(frames, ignore_index=True)
    logger.info("loaded %d raw rows from %d sources", len(raw), len(frames))

    engineered = engineer_features(raw)
    logger.info(
        "engineered %d feature columns: %s",
        len(feature_columns(engineered)),
        feature_columns(engineered)[:5] + ["..."],
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    engineered.to_parquet(out_path, index=False)
    logger.info("wrote %d rows × %d cols to %s", len(engineered), len(engineered.columns), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
