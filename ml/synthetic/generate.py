"""Synthetic CI run generator with author and repo personas.

Personas (author skill + bias toward a failure class) are sampled once at
the start of generation. Each row then picks an author conditionally on
its failure class — success rows favour high-skill authors, while a
specific failure class favours authors biased toward that class. This
makes the expanding ``feat_author_success_rate`` and
``feat_project_failure_rate`` actually carry per-class signal once
computed by the feature extractor (a problem identified during EDA).

Usage::

    python -m synthetic.generate --n 10000 --out ../data/raw/synthetic.parquet --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic.distributions import CLASS_WEIGHTS, CLASSES, SAMPLERS

logger = logging.getLogger(__name__)


DEFAULT_BASE_TIME = datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc)
FAIL_CLASSES = tuple(c for c in CLASSES if c != "success")


def _sample_run_attempt(rng: np.random.Generator, failure_class: str) -> int:
    if failure_class == "network_error":
        if rng.random() < 0.60:
            return int(rng.integers(2, 5))
        return 1
    if rng.random() < 0.88:
        return 1
    return int(rng.integers(2, 4))


@dataclass
class AuthorPersona:
    aid: int
    skill: float
    fail_bias: str


@dataclass
class RepoPersona:
    rid: int
    name: str
    base_failure_rate: float


def _build_authors(rng: np.random.Generator, n_authors: int) -> list[AuthorPersona]:
    authors: list[AuthorPersona] = []
    for aid in range(n_authors):
        skill = float(np.clip(rng.beta(7, 3), 0.1, 0.99))
        fail_bias = str(rng.choice(FAIL_CLASSES))
        authors.append(AuthorPersona(aid=aid, skill=skill, fail_bias=fail_bias))
    return authors


def _build_repos(rng: np.random.Generator, n_repos: int) -> list[RepoPersona]:
    repos: list[RepoPersona] = []
    for rid in range(n_repos):
        repos.append(
            RepoPersona(
                rid=rid,
                name=f"synthetic/repo-{rid:03d}",
                base_failure_rate=float(np.clip(rng.beta(3, 7), 0.05, 0.7)),
            )
        )
    return repos


def _author_weights(authors: list[AuthorPersona], failure_class: str) -> np.ndarray:
    if failure_class == "success":
        weights = np.array([a.skill**1.5 for a in authors], dtype=float)
    else:
        weights = np.array(
            [
                (1.0 - a.skill) ** 1.2 * (3.0 if a.fail_bias == failure_class else 0.6)
                for a in authors
            ],
            dtype=float,
        )
    return weights / weights.sum()


def _repo_weights(repos: list[RepoPersona], failure_class: str) -> np.ndarray:
    if failure_class == "success":
        weights = np.array([1.0 - r.base_failure_rate for r in repos], dtype=float)
    else:
        weights = np.array([r.base_failure_rate + 0.05 for r in repos], dtype=float)
    return weights / weights.sum()


def _make_synthetic_row(
    rng: np.random.Generator,
    failure_class: str,
    *,
    author: AuthorPersona,
    repo: RepoPersona,
    base_time: datetime,
) -> dict:
    sample = SAMPLERS[failure_class](rng)
    is_success = failure_class == "success"
    delta_minutes = int(rng.integers(0, 60 * 24 * 90))
    created = base_time - timedelta(minutes=delta_minutes)
    updated = created + timedelta(seconds=int(sample.duration_seconds))
    file_extensions = {
        "py": int(round(sample.files_changed * sample.file_ext_py_share)),
        "js": int(round(sample.files_changed * sample.file_ext_js_share)),
        "yml": int(rng.integers(0, max(1, sample.files_changed // 4))),
    }
    # Class-conditional change-pattern signals (added in v4 for the OTHER_FAILURE class).
    test_changes_share = float(rng.beta(3, 7))
    test_dir_changes = int(round(sample.files_changed * test_changes_share))
    test_only_changes = test_dir_changes == sample.files_changed and sample.files_changed <= 4
    has_lint_config_change = bool(rng.random() < 0.04)
    event_is_push = bool(rng.random() < 0.85)
    return {
        "repo": repo.name,
        "run_id": int(rng.integers(10**9, 10**12)),
        "workflow_name": "synthetic",
        "head_sha": f"{rng.integers(0, 16**12):012x}",
        "head_branch": "main" if rng.random() < 0.7 else "feature/x",
        "event": "push",
        "status": "completed",
        "conclusion": "success" if is_success else "failure",
        "run_attempt": _sample_run_attempt(rng, failure_class),
        "created_at": created,
        "updated_at": updated,
        "duration_seconds": float(sample.duration_seconds),
        "actor_login": f"author{author.aid}",
        "actor_email": f"author{author.aid}@example.com",
        "n_jobs": sample.n_jobs,
        "longest_job_seconds": float(sample.longest_job_seconds),
        "failure_class": None if is_success else failure_class,
        "failure_evidence": None,
        "files_changed": sample.files_changed,
        "lines_added": sample.lines_added,
        "lines_deleted": sample.lines_deleted,
        "has_dockerfile_change": sample.has_dockerfile_change,
        "has_dependency_change": sample.has_dependency_change,
        "file_extensions": file_extensions,
        "test_dir_changes": test_dir_changes,
        "test_only_changes": test_only_changes,
        "has_lint_config_change": has_lint_config_change,
        "event_is_push": event_is_push,
        "new_dependencies_count": sample.new_dependencies_count,
        "new_dependencies_size_mb": float(sample.new_dependencies_size_mb),
        "dockerfile_base_image_size_mb": float(sample.dockerfile_base_image_size_mb),
        "estimated_final_image_size_mb": float(sample.estimated_final_image_size_mb),
        "peak_memory_mb": float(sample.peak_memory_mb),
        "author_success_rate": float(sample.author_success_rate),
        "project_failure_rate": float(sample.project_failure_rate),
        "synthetic": True,
    }


def generate(
    n: int,
    *,
    seed: int = 42,
    n_repos: int = 30,
    n_authors: int = 200,
    base_time: datetime | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    authors = _build_authors(rng, n_authors)
    repos = _build_repos(rng, n_repos)
    classes = list(CLASS_WEIGHTS.keys())
    weights = np.array([CLASS_WEIGHTS[c] for c in classes], dtype=float)
    weights = weights / weights.sum()
    labels = rng.choice(classes, size=n, p=weights)

    author_w_cache = {c: _author_weights(authors, c) for c in classes}
    repo_w_cache = {c: _repo_weights(repos, c) for c in classes}

    base_time = base_time or DEFAULT_BASE_TIME
    rows: list[dict] = []
    for i in range(n):
        cls = str(labels[i])
        author = authors[int(rng.choice(len(authors), p=author_w_cache[cls]))]
        repo = repos[int(rng.choice(len(repos), p=repo_w_cache[cls]))]
        rows.append(
            _make_synthetic_row(
                rng,
                failure_class=cls,
                author=author,
                repo=repo,
                base_time=base_time,
            )
        )
    return pd.DataFrame(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate synthetic CI run dataset")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-repos", type=int, default=30)
    p.add_argument("--n-authors", type=int, default=200)
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")
    df = generate(args.n, seed=args.seed, n_repos=args.n_repos, n_authors=args.n_authors)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    logger.info("wrote %d synthetic rows to %s", len(df), out_path)
    logger.info("class balance: %s", df["failure_class"].fillna("success").value_counts().to_dict())
    logger.info("conclusion balance: %s", df["conclusion"].value_counts().to_dict())
    logger.info("unique authors used: %d", df["actor_email"].nunique())
    logger.info("unique repos used: %d", df["repo"].nunique())
    assert set(df["conclusion"].unique()) <= {"success", "failure"}
    assert set(CLASSES) >= set(df["failure_class"].dropna().unique())
    return 0


if __name__ == "__main__":
    sys.exit(main())
