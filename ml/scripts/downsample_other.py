"""Downsample the dominant `other_failure` class in the real-data parquet.

The classifier learns "everything → other_failure" because that class is
~46 % of the real dataset, dwarfing the specific failure classes (oom_killed
0.2 %, network_error 0.4 %, dependency_error 0.6 %). This script trims
``other_failure`` to a configurable fraction so the rare classes get a
fair seat at training time.

Usage::

    cd ml
    uv run python scripts/downsample_other.py \\
        --input ../data/raw/gha_runs_hybrid.parquet \\
        --output ../data/raw/gha_runs_balanced.parquet \\
        --other-fraction 0.15
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--other-fraction", type=float, default=0.15,
                   help="Keep this fraction of `other_failure` rows (default 0.15)")
    p.add_argument("--success-fraction", type=float, default=0.5,
                   help="Keep this fraction of `success` rows (default 0.5)")
    p.add_argument("--upsample-rare-to", type=int, default=0,
                   help="Upsample rare classes (oom/network/dep/timeout/docker) by "
                        "duplication until each reaches this count (0 = off)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    df = pd.read_parquet(args.input)
    df["_label"] = df["failure_class"].astype("object")
    failed = df["conclusion"] == "failure"
    df.loc[df["_label"].isna() & failed, "_label"] = "other_failure"
    df["_label"] = df["_label"].fillna("success")

    before = df["_label"].value_counts().to_dict()
    print("before:", before)

    other = df[df["_label"] == "other_failure"]
    success = df[df["_label"] == "success"]
    rare = df[~df["_label"].isin(["other_failure", "success"])]

    other_kept = other.sample(frac=args.other_fraction, random_state=args.seed) if len(other) else other
    success_kept = success.sample(frac=args.success_fraction, random_state=args.seed) if len(success) else success

    rare_parts = []
    if args.upsample_rare_to > 0:
        for cls, grp in rare.groupby("_label"):
            n = args.upsample_rare_to
            if len(grp) >= n:
                rare_parts.append(grp.sample(n=n, random_state=args.seed))
            else:
                rare_parts.append(grp.sample(n=n, replace=True, random_state=args.seed))
        rare_final = pd.concat(rare_parts, ignore_index=True) if rare_parts else rare
    else:
        rare_final = rare

    out = pd.concat([rare_final, success_kept, other_kept], ignore_index=True)
    out = out.drop(columns=["_label"])
    out.to_parquet(args.output, index=False)

    # recompute distribution post-downsample
    rel = out["failure_class"].astype("object")
    rel = rel.where(~(rel.isna() & (out["conclusion"] == "failure")), "other_failure")
    rel = rel.fillna("success")
    print("after :", rel.value_counts().to_dict())
    print(f"wrote {len(out)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
