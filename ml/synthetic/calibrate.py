"""Fit per-class distribution parameters from real GHA data.

Reads the real workflow-run parquet (collected by ml/data_collection/hybrid/),
groups by failure_class (with conclusion='failure' rows lacking failure_class
treated as 'other_failure', non-failures as 'success'), and emits empirical
parameters for the columns shared between real and synthetic data:

  Counts (mean/std, modelled as truncated normal):
    files_changed, lines_added, lines_deleted, n_jobs

  Lognormal-shaped (log-median / log-sigma):
    duration_seconds, longest_job_seconds

  Bernoulli probabilities:
    has_dependency_change, has_dockerfile_change

The result is a single ``calibration.json`` dropped next to distributions.py.
distributions.py picks it up automatically and overrides the literature-based
priors for these columns. Synthetic-only columns (peak_memory_mb,
estimated_final_image_size_mb, new_dependencies_*, etc.) stay literature-based
because they have no real-world ground truth.

Usage::

    cd ml
    uv run python -m synthetic.calibrate \\
        --real ../data/raw/gha_runs_hybrid.parquet \\
        --out synthetic/calibration.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


CLASSES = (
    "success",
    "oom_killed",
    "test_timeout",
    "dependency_error",
    "docker_build_failed",
    "network_error",
    "other_failure",
)
COUNT_COLS = ("files_changed", "lines_added", "lines_deleted", "n_jobs")
LOGNORM_COLS = ("duration_seconds", "longest_job_seconds")
BOOL_COLS = ("has_dependency_change", "has_dockerfile_change")


def _label(df: pd.DataFrame) -> pd.Series:
    label = df["failure_class"].astype("object")
    failed = df["conclusion"] == "failure"
    label = label.where(~(label.isna() & failed), "other_failure")
    return label.fillna("success").astype(str)


def _fit_count(values: pd.Series) -> dict[str, float] | None:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return None
    return {
        "mean": float(v.mean()),
        "std":  float(max(v.std(ddof=0), 1.0)),
        "min":  int(max(v.min(), 0)),
        "max":  int(v.max()),
    }


def _fit_lognorm(values: pd.Series) -> dict[str, float] | None:
    v = pd.to_numeric(values, errors="coerce").dropna()
    v = v[v > 0]
    if len(v) < 5:
        return None
    log_v = np.log(v)
    return {
        "median": float(np.exp(log_v.mean())),
        "log_sigma": float(max(log_v.std(ddof=0), 0.1)),
    }


def _fit_bool(values: pd.Series) -> float | None:
    v = values.dropna()
    if v.empty:
        return None
    return float(v.astype(bool).mean())


def calibrate(real_path: Path) -> dict:
    df = pd.read_parquet(real_path)
    df["_label"] = _label(df)

    out: dict[str, dict] = {"per_class": {}}
    for cls in CLASSES:
        grp = df[df["_label"] == cls]
        if grp.empty:
            logger.warning("no real rows for class %r; skipping", cls)
            continue
        params: dict[str, dict] = {"n_samples": int(len(grp))}
        for col in COUNT_COLS:
            fit = _fit_count(grp[col]) if col in grp else None
            if fit:
                params[col] = fit
        for col in LOGNORM_COLS:
            fit = _fit_lognorm(grp[col]) if col in grp else None
            if fit:
                params[col] = fit
        for col in BOOL_COLS:
            fit = _fit_bool(grp[col]) if col in grp else None
            if fit is not None:
                params[col] = {"p": fit}
        out["per_class"][cls] = params

    out["_meta"] = {
        "source_parquet": str(real_path),
        "n_total_rows": int(len(df)),
        "class_distribution": df["_label"].value_counts().to_dict(),
        "calibrated_columns": {
            "counts": list(COUNT_COLS),
            "lognormal": list(LOGNORM_COLS),
            "bernoulli": list(BOOL_COLS),
        },
    }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--real", required=True, help="Path to real parquet")
    p.add_argument("--out", default="synthetic/calibration.json")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")

    cal = calibrate(Path(args.real))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cal, indent=2))
    logger.info("wrote %d-class calibration to %s", len(cal["per_class"]), out)
    for cls, params in cal["per_class"].items():
        logger.info("  %-20s n=%d cols=%d", cls, params["n_samples"], len(params) - 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
