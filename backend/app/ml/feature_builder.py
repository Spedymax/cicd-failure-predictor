"""Build a feature vector from webhook payload + database history.

This is the *online* version of the feature extractor (the offline one
lives in ``ml/features/transformers.py`` and operates on whole
DataFrames). Online extraction works on a single commit and queries
the database for historical aggregates (author / repo).
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Commit, Prediction, Repository

DEPENDENCY_MARKERS = (
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "requirements.txt",
    "pyproject.toml",
    "poetry.lock",
    "uv.lock",
    "go.mod",
    "go.sum",
    "cargo.toml",
    "cargo.lock",
    "pom.xml",
    "build.gradle",
)


def _detect_dependency_change(filenames: list[str]) -> bool:
    return any(name.lower().endswith(m) or name.lower() == m for name in filenames for m in DEPENDENCY_MARKERS)


def _detect_dockerfile_change(filenames: list[str]) -> bool:
    return any("dockerfile" in n.lower() for n in filenames)


def _is_test_path(name: str) -> bool:
    n = name.lower()
    return (
        "/test_" in n
        or n.startswith("test_")
        or "/tests/" in n
        or n.startswith("tests/")
        or n.endswith(".test.js")
        or n.endswith(".test.ts")
        or n.endswith(".spec.js")
        or n.endswith(".spec.ts")
        or "_test.go" in n
    )


def _detect_test_only_changes(filenames: list[str]) -> bool:
    if not filenames:
        return False
    return all(_is_test_path(n) for n in filenames)


def _count_test_dir_changes(filenames: list[str]) -> int:
    return sum(1 for n in filenames if _is_test_path(n))


def _file_ext_count(filenames: list[str], ext: str) -> int:
    return sum(1 for n in filenames if n.lower().endswith(f".{ext}"))


def _author_history(db: Session, repo_id: int, author_email: str) -> dict[str, float]:
    rows = db.execute(
        select(Commit.id, Commit.committed_at)
        .where(Commit.author_email == author_email)
        .order_by(Commit.committed_at.desc())
        .limit(50)
    ).all()
    n_runs = len(rows)
    if n_runs == 0:
        return {"author_success_rate": 0.5, "author_n_runs_log": 0.0, "author_avg_duration_log": math.log1p(360)}

    # Pull successful conclusions via predictions.actual_outcome or fall
    # back to a 0.5 prior; in MVP we approximate using prediction risk.
    pred_rows = db.execute(
        select(Prediction.predicted_class, Prediction.actual_outcome)
        .join(Commit, Prediction.commit_id == Commit.id)
        .where(Commit.author_email == author_email)
        .order_by(Prediction.created_at.desc())
        .limit(50)
    ).all()
    if pred_rows:
        outcomes = [r[1] or r[0] for r in pred_rows]
        success_rate = sum(1 for o in outcomes if o is not None and str(o) == "success") / len(outcomes)
    else:
        success_rate = 0.5

    return {
        "author_success_rate": float(success_rate),
        "author_n_runs_log": float(math.log1p(n_runs)),
        "author_avg_duration_log": math.log1p(360),
    }


def _project_history(db: Session, repo_id: int) -> dict[str, float]:
    n_runs = db.scalar(select(func.count(Commit.id)).where(Commit.repository_id == repo_id)) or 0
    pred_count = db.scalar(select(func.count(Prediction.id)).where(Prediction.repository_id == repo_id)) or 0
    fail_count = (
        db.scalar(
            select(func.count(Prediction.id)).where(
                Prediction.repository_id == repo_id,
                Prediction.predicted_class != "success",
            )
        )
        or 0
    )
    failure_rate = (fail_count / pred_count) if pred_count else 0.3
    return {
        "project_failure_rate": float(failure_rate),
        "project_n_runs_log": float(math.log1p(n_runs)),
        "repo_failure_rate_global": float(failure_rate),
    }


def build_feature_vector(
    db: Session,
    repo: Repository,
    *,
    files: list[dict[str, Any]],
    lines_added: int,
    lines_deleted: int,
    has_dependency_change: bool | None = None,
    has_dockerfile_change: bool | None = None,
    new_dependencies_count: int = 0,
    new_dependencies_size_mb: float = 0.0,
    dockerfile_base_image_size_mb: float = 0.0,
    estimated_final_image_size_mb: float = 0.0,
    n_jobs: int = 1,
    longest_job_seconds: float = 0.0,
    run_attempt: int = 1,
    created_at: datetime,
    author_email: str,
) -> dict[str, float]:
    filenames = [f.get("filename", "") for f in files]
    files_changed = min(len(files), 1000)
    added = max(0, min(lines_added, 50_000))
    deleted = max(0, min(lines_deleted, 50_000))
    total_lines = added + deleted

    has_dep = (
        bool(has_dependency_change)
        if has_dependency_change is not None
        else _detect_dependency_change(filenames)
    )
    has_docker = (
        bool(has_dockerfile_change)
        if has_dockerfile_change is not None
        else _detect_dockerfile_change(filenames)
    )

    base_img = max(0.0, dockerfile_base_image_size_mb)
    final_img = max(0.0, estimated_final_image_size_mb)
    growth = (final_img / base_img) if base_img > 0 else 0.0
    growth = float(min(growth, 50.0))

    auth = _author_history(db, repo.id, author_email)
    proj = _project_history(db, repo.id)

    feats: dict[str, float] = {
        "feat_files_changed_log": float(np.log1p(files_changed)),
        "feat_lines_changed_log": float(np.log1p(total_lines)),
        "feat_lines_added_share": (added / total_lines) if total_lines else 0.0,
        "feat_avg_lines_per_file": (total_lines / files_changed) if files_changed else 0.0,
        "feat_has_dependency_change_int": int(has_dep),
        "feat_new_deps_count": float(min(new_dependencies_count, 100)),
        "feat_new_deps_size_mb_log": float(np.log1p(max(0.0, new_dependencies_size_mb))),
        "feat_has_dockerfile_change_int": int(has_docker),
        "feat_final_image_size_mb_log": float(np.log1p(final_img)),
        "feat_image_growth_ratio": growth,
        "feat_hour_of_day": int(created_at.hour),
        "feat_day_of_week": int(created_at.weekday()),
        "feat_is_weekend": int(created_at.weekday() in (5, 6)),
        "feat_is_business_hours": int(
            9 <= created_at.hour <= 18 and created_at.weekday() not in (5, 6)
        ),
        "feat_run_attempt": int(min(max(run_attempt, 1), 10)),
        "feat_run_attempt_gt1": int(run_attempt > 1),
        "feat_n_jobs_log": float(np.log1p(min(max(n_jobs, 0), 100))),
        "feat_longest_job_seconds_log": float(np.log1p(max(longest_job_seconds, 0.0))),
    }

    for ext in ("py", "js", "ts", "go", "rs", "java", "yml", "json"):
        feats[f"feat_ext_{ext}_count"] = float(_file_ext_count(filenames, ext))

    # Test-related signals (used by cause classifier to identify test_failure).
    test_dir_count = _count_test_dir_changes(filenames)
    test_only = _detect_test_only_changes(filenames) if filenames else False
    feats["feat_test_dir_changes"] = float(test_dir_count)
    feats["feat_test_only_changes_int"] = int(test_only)
    feats["feat_test_changes_share"] = (
        float(test_dir_count) / files_changed if files_changed else 0.0
    )
    feats["feat_has_lint_config_change_int"] = int(any(
        any(name in (f.get("filename") or "").lower() for name in (
            ".eslintrc", ".prettierrc", "ruff.toml", "pyproject.toml",
            ".flake8", "tslint", "stylelint", ".editorconfig",
        )) for f in files
    ))
    feats["feat_event_is_push_int"] = 1

    feats["feat_author_success_rate"] = auth["author_success_rate"]
    feats["feat_author_n_runs_log"] = auth["author_n_runs_log"]
    feats["feat_author_avg_duration_log"] = auth["author_avg_duration_log"]
    feats["feat_project_failure_rate"] = proj["project_failure_rate"]
    feats["feat_project_n_runs_log"] = proj["project_n_runs_log"]
    feats["feat_repo_failure_rate_global"] = proj["repo_failure_rate_global"]
    return feats
