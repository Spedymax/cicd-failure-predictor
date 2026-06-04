"""Evaluate the trained model on out-of-distribution holdout repos.

Pipeline:
  1. Collect a small fresh GHA dataset for repos that were NOT in
     training (``ml/data_collection/holdout_repos.txt``).
  2. Apply ``features.engineer_features`` to the collected rows.
  3. Predict each row with the active model (``ML_MODEL_DIR``).
  4. Compare ``predicted_class`` with the heuristic-derived
     ``failure_class``; for unlabelled failures the comparison is
     against the binary "any failure" target.
  5. Print confusion matrix + per-class precision/recall/F1.

The output is intentionally separate from the dashboard; this is a
**diagnostic** tool used in chapter 4 of the diploma.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for line in (REPO_ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
os.environ["ML_MODEL_DIR"] = str(REPO_ROOT / "data" / "artifacts" / "v4")

sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "ml"))

from data_collection.collect import (  # noqa: E402
    _process_repo,
    _records_to_dataframe,
)
from data_collection.github_client import GitHubClient  # noqa: E402
from features.transformers import engineer_features, feature_columns  # noqa: E402

from app.api.dependencies import get_inference_engine  # noqa: E402

logger = logging.getLogger(__name__)


async def _collect(token: str, repos: list[str], runs_per_repo: int, concurrency: int) -> pd.DataFrame:
    all_records = []
    sem = asyncio.Semaphore(concurrency)

    async with GitHubClient(token=token) as client:

        async def worker(name: str) -> None:
            async with sem:
                try:
                    recs = await _process_repo(
                        client, name, runs_per_repo=runs_per_repo, fetch_logs=True
                    )
                except Exception as exc:
                    logger.warning("skip %s: %s", name, exc)
                    recs = []
                all_records.extend(recs)

        await asyncio.gather(*(worker(r) for r in repos))

    return _records_to_dataframe(all_records)


def _load_repos(path: Path) -> list[str]:
    repos = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            repos.append(s)
    return repos


def _evaluate(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"error": "no rows collected"}
    df = engineer_features(df)
    feats = feature_columns(df)
    engine = get_inference_engine()
    expected = engine.feature_names

    def _row_features(row: pd.Series) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in expected:
            v = row.get(name)
            try:
                fv = float(v) if v is not None else 0.0
            except (TypeError, ValueError):
                fv = 0.0
            if fv != fv:  # NaN
                fv = 0.0
            out[name] = fv
        return out

    predicted = []
    confidences = []
    for _, row in df.iterrows():
        feats_dict = _row_features(row)
        res = engine.predict(feats_dict)
        predicted.append(res.predicted_class)
        confidences.append(res.confidence)

    df["predicted_class"] = predicted
    df["confidence"] = confidences
    def _derive_actual(r: pd.Series) -> str | None:
        if r["conclusion"] == "success":
            return "success"
        fc = r["failure_class"]
        if isinstance(fc, str) and fc:
            return fc
        if r["conclusion"] == "failure":
            # Unlabeled failure → other_failure (matches v4 training labels).
            return "other_failure"
        return None

    df["actual_class"] = df.apply(_derive_actual, axis=1)

    overall = df.dropna(subset=["actual_class"]).copy()
    n_total = len(overall)
    if n_total == 0:
        return {"summary": "no rows with derivable actual_class"}
    n_match = int((overall["predicted_class"] == overall["actual_class"]).sum())

    classes = sorted({*overall["predicted_class"], *overall["actual_class"]})
    confusion = {a: defaultdict(int) for a in classes}
    for _, r in overall.iterrows():
        confusion[r["actual_class"]][r["predicted_class"]] += 1

    per_class = {}
    for cls in classes:
        tp = confusion[cls][cls]
        n_actual = sum(confusion[cls].values())
        n_pred = sum(confusion[a][cls] for a in classes)
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_actual if n_actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[cls] = {
            "n_actual": n_actual,
            "tp": tp,
            "fp": n_pred - tp,
            "fn": n_actual - tp,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    binary = df.copy()
    binary["pred_failed"] = (binary["predicted_class"] != "success").astype(int)
    binary["actual_failed"] = (binary["conclusion"] == "failure").astype(int)
    bin_n = len(binary)
    bin_match = int((binary["pred_failed"] == binary["actual_failed"]).sum())

    return {
        "n_collected": int(len(df)),
        "n_with_actual_class": int(n_total),
        "accuracy": round(n_match / n_total, 3) if n_total else 0.0,
        "n_match": int(n_match),
        "per_class": per_class,
        "binary_failure_detection": {
            "n_total": bin_n,
            "n_match": bin_match,
            "accuracy": round(bin_match / bin_n, 3) if bin_n else 0.0,
        },
        "confusion_matrix": {a: dict(c) for a, c in confusion.items()},
        "per_repo_summary": (
            df.groupby("repo")
            .apply(
                lambda g: {
                    "n": int(len(g)),
                    "predicted_failure_rate": round(
                        float((g["predicted_class"] != "success").mean()), 3
                    ),
                    "actual_failure_rate": round(
                        float((g["conclusion"] == "failure").mean()), 3
                    ),
                },
                include_groups=False,
            )
            .to_dict()
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repos",
        default=str(REPO_ROOT / "ml" / "data_collection" / "holdout_repos.txt"),
    )
    parser.add_argument("--runs-per-repo", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "processed" / "holdout_eval.json"))
    args = parser.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")

    token = os.environ.get("GITHUB_API_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.error("GITHUB_API_TOKEN not set")
        return 2

    repos = _load_repos(Path(args.repos))
    logger.info("evaluating on %d holdout repos", len(repos))
    df = asyncio.run(_collect(token, repos, args.runs_per_repo, args.concurrency))
    logger.info("collected %d rows", len(df))
    if df.empty:
        return 1

    report = _evaluate(df)

    import json

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("wrote report to %s", out_path)

    print()
    print("=" * 60)
    print("HOLDOUT EVALUATION REPORT")
    print("=" * 60)
    print(f"Repositories:           {len(repos)}")
    print(f"Rows collected:         {report['n_collected']}")
    print(f"With derivable actual:  {report['n_with_actual_class']}")
    print(f"Multi-class accuracy:   {report['accuracy']:.1%} ({report['n_match']}/{report['n_with_actual_class']})")
    bf = report["binary_failure_detection"]
    print(f"Binary fail/pass:       {bf['accuracy']:.1%} ({bf['n_match']}/{bf['n_total']})")
    print()
    print("Per-class:")
    for cls, m in report["per_class"].items():
        print(f"  {cls:25s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} n={m['n_actual']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
