"""Collect GitHub Actions workflow runs into a parquet dataset.

Each row in the output corresponds to one workflow run and aggregates:
  - run-level metadata (status, conclusion, timing, attempts);
  - associated commit metadata (author, files changed, lines +/-);
  - failure classification heuristic over job logs (only for failed runs);
  - high-level resource hints (number of jobs, longest job duration).

Usage::

    python -m data_collection.collect \\
        --repos data_collection/repos.txt \\
        --token $GITHUB_TOKEN \\
        --runs-per-repo 100 \\
        --out ../data/raw/gha_runs.parquet
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.asyncio import tqdm

from data_collection.failure_heuristics import classify_with_evidence
from data_collection.github_client import GitHubClient

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    repo: str
    run_id: int
    workflow_name: str
    head_sha: str
    head_branch: str | None
    event: str
    status: str
    conclusion: str | None
    run_attempt: int
    created_at: datetime | None
    updated_at: datetime | None
    duration_seconds: float | None
    actor_login: str | None
    actor_email: str | None
    n_jobs: int = 0
    longest_job_seconds: float | None = None
    failure_class: str | None = None
    failure_evidence: str | None = None
    files_changed: int | None = None
    lines_added: int | None = None
    lines_deleted: int | None = None
    has_dockerfile_change: bool | None = None
    has_dependency_change: bool | None = None
    file_extensions: dict[str, int] = field(default_factory=dict)
    test_dir_changes: int = 0
    test_only_changes: bool = False
    has_lint_config_change: bool = False
    event_is_push: bool = True


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _detect_dependency_change(filenames: list[str]) -> bool:
    markers = (
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "requirements.txt",
        "pyproject.toml",
        "poetry.lock",
        "uv.lock",
        "pipfile",
        "go.mod",
        "go.sum",
        "cargo.toml",
        "cargo.lock",
        "pom.xml",
        "build.gradle",
    )
    return any(any(name.endswith(m) or name == m for m in markers) for name in filenames)


def _detect_dockerfile_change(filenames: list[str]) -> bool:
    return any(
        name.lower() == "dockerfile" or name.lower().endswith(".dockerfile")
        or "/dockerfile" in name.lower()
        for name in filenames
    )


_TEST_PATH_MARKERS = (
    "/tests/",
    "/test/",
    "/__tests__/",
    "/spec/",
    "/specs/",
)
_TEST_FILENAME_MARKERS = (
    "_test.go",
    "_test.py",
    ".test.ts",
    ".test.tsx",
    ".test.js",
    ".test.jsx",
    ".spec.ts",
    ".spec.js",
    ".spec.tsx",
    "test_",
    "_spec.rb",
)
_LINT_CONFIG_FILES = (
    ".eslintrc",
    ".eslintrc.js",
    ".eslintrc.json",
    ".eslintrc.yaml",
    ".eslintrc.yml",
    ".flake8",
    ".pylintrc",
    "mypy.ini",
    ".mypy.ini",
    "ruff.toml",
    ".ruff.toml",
    ".prettierrc",
    ".golangci.yml",
    ".golangci.yaml",
    "tslint.json",
    ".rubocop.yml",
)


def _is_test_path(name: str) -> bool:
    lower = name.lower()
    if any(m in lower for m in _TEST_PATH_MARKERS):
        return True
    base = lower.rsplit("/", 1)[-1]
    return any(base.endswith(m) or base.startswith(m) or m in base for m in _TEST_FILENAME_MARKERS)


def _detect_test_changes(filenames: list[str]) -> tuple[int, bool]:
    test_count = sum(1 for n in filenames if _is_test_path(n))
    test_only = bool(filenames) and test_count == len(filenames)
    return test_count, test_only


def _detect_lint_config_change(filenames: list[str]) -> bool:
    for n in filenames:
        base = n.rsplit("/", 1)[-1].lower()
        if base in _LINT_CONFIG_FILES or any(base.startswith(m) for m in _LINT_CONFIG_FILES):
            return True
    return False


def _file_ext_counts(filenames: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for n in filenames:
        ext = Path(n).suffix.lstrip(".").lower() or "noext"
        counts[ext] = counts.get(ext, 0) + 1
    return counts


async def _enrich_with_commit(client: GitHubClient, owner: str, repo: str, rec: RunRecord) -> None:
    commit = await client.get_commit(owner, repo, rec.head_sha)
    if not commit:
        return
    author = commit.get("commit", {}).get("author", {})
    rec.actor_email = author.get("email") or rec.actor_email
    files = commit.get("files") or []
    filenames = [f.get("filename", "") for f in files]
    rec.files_changed = len(files)
    rec.lines_added = sum(int(f.get("additions") or 0) for f in files)
    rec.lines_deleted = sum(int(f.get("deletions") or 0) for f in files)
    rec.has_dockerfile_change = _detect_dockerfile_change(filenames)
    rec.has_dependency_change = _detect_dependency_change(filenames)
    rec.file_extensions = _file_ext_counts(filenames)
    test_count, test_only = _detect_test_changes(filenames)
    rec.test_dir_changes = test_count
    rec.test_only_changes = test_only
    rec.has_lint_config_change = _detect_lint_config_change(filenames)


async def _enrich_with_jobs(client: GitHubClient, owner: str, repo: str, rec: RunRecord) -> None:
    jobs = await client.list_jobs(owner, repo, rec.run_id)
    rec.n_jobs = len(jobs)
    durations: list[float] = []
    failed_job: dict[str, Any] | None = None
    for j in jobs:
        started = _parse_iso(j.get("started_at"))
        completed = _parse_iso(j.get("completed_at"))
        if started and completed:
            durations.append((completed - started).total_seconds())
        if failed_job is None and j.get("conclusion") == "failure":
            failed_job = j
    if durations:
        rec.longest_job_seconds = max(durations)
    if rec.conclusion == "failure" and failed_job is not None:
        log = await client.get_job_logs(owner, repo, failed_job["id"])
        cls, evidence = classify_with_evidence(log)
        rec.failure_class = cls
        rec.failure_evidence = (evidence or "")[:200] if evidence else None


async def _process_repo(
    client: GitHubClient,
    full_name: str,
    *,
    runs_per_repo: int,
    fetch_logs: bool,
    failure_share: float = 0.5,
) -> list[RunRecord]:
    owner, _, repo = full_name.partition("/")
    if not repo:
        logger.warning("invalid repo entry: %s", full_name)
        return []

    n_failure = int(round(runs_per_repo * failure_share))
    n_success = runs_per_repo - n_failure
    runs: list[dict[str, Any]] = []
    try:
        if n_failure > 0:
            failure_runs = await client.list_workflow_runs(
                owner, repo, max_runs=n_failure, status="failure"
            )
            runs.extend(failure_runs)
        if n_success > 0:
            success_runs = await client.list_workflow_runs(
                owner, repo, max_runs=n_success, status="success"
            )
            runs.extend(success_runs)
    except httpx.HTTPStatusError as e:  # type: ignore[name-defined]
        logger.warning("failed to list runs for %s: %s", full_name, e)
        return []

    records: list[RunRecord] = []
    for run in runs:
        actor = run.get("actor") or {}
        head_commit = run.get("head_commit") or {}
        started = _parse_iso(run.get("run_started_at") or run.get("created_at"))
        updated = _parse_iso(run.get("updated_at"))
        rec = RunRecord(
            repo=full_name,
            run_id=int(run["id"]),
            workflow_name=run.get("name") or "",
            head_sha=run.get("head_sha") or "",
            head_branch=run.get("head_branch"),
            event=run.get("event") or "",
            status=run.get("status") or "",
            conclusion=run.get("conclusion"),
            run_attempt=int(run.get("run_attempt") or 1),
            created_at=started,
            updated_at=updated,
            duration_seconds=(
                (updated - started).total_seconds()
                if started and updated
                else None
            ),
            actor_login=actor.get("login"),
            actor_email=(head_commit.get("author") or {}).get("email"),
            event_is_push=(run.get("event") == "push"),
        )
        records.append(rec)

    sem = asyncio.Semaphore(8)

    async def enrich(rec: RunRecord) -> None:
        async with sem:
            try:
                await _enrich_with_commit(client, owner, repo, rec)
                if fetch_logs and rec.conclusion == "failure":
                    await _enrich_with_jobs(client, owner, repo, rec)
            except Exception as exc:
                logger.debug("enrich failure for %s/%s: %s", full_name, rec.run_id, exc)

    await asyncio.gather(*(enrich(r) for r in records))
    return records


def _load_repo_list(path: Path) -> list[str]:
    repos = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        repos.append(line)
    return repos


def _records_to_dataframe(records: list[RunRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append(
            {
                "repo": r.repo,
                "run_id": r.run_id,
                "workflow_name": r.workflow_name,
                "head_sha": r.head_sha,
                "head_branch": r.head_branch,
                "event": r.event,
                "status": r.status,
                "conclusion": r.conclusion,
                "run_attempt": r.run_attempt,
                "created_at": r.created_at,
                "updated_at": r.updated_at,
                "duration_seconds": r.duration_seconds,
                "actor_login": r.actor_login,
                "actor_email": r.actor_email,
                "n_jobs": r.n_jobs,
                "longest_job_seconds": r.longest_job_seconds,
                "failure_class": r.failure_class,
                "failure_evidence": r.failure_evidence,
                "files_changed": r.files_changed,
                "lines_added": r.lines_added,
                "lines_deleted": r.lines_deleted,
                "has_dockerfile_change": r.has_dockerfile_change,
                "has_dependency_change": r.has_dependency_change,
                "file_extensions": r.file_extensions,
                "test_dir_changes": r.test_dir_changes,
                "test_only_changes": r.test_only_changes,
                "has_lint_config_change": r.has_lint_config_change,
                "event_is_push": r.event_is_push,
            }
        )
    return pd.DataFrame(rows)


def _write(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)


async def main_async(args: argparse.Namespace) -> int:
    token = args.token or os.environ.get("GITHUB_API_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.error("GitHub token not provided (--token or GITHUB_API_TOKEN env)")
        return 2

    repos = _load_repo_list(Path(args.repos))
    logger.info(
        "collecting runs from %d repos, %d per repo, parallelism=%d",
        len(repos), args.runs_per_repo, args.concurrency,
    )

    out_path = Path(args.out)
    all_records: list[RunRecord] = []
    completed = 0

    async with GitHubClient(token=token) as client:
        sem = asyncio.Semaphore(max(1, args.concurrency))
        bar = tqdm(total=len(repos), desc="repos")

        async def worker(full_name: str) -> list[RunRecord]:
            nonlocal completed
            async with sem:
                bar.set_postfix_str(full_name)
                try:
                    recs = await _process_repo(
                        client,
                        full_name,
                        runs_per_repo=args.runs_per_repo,
                        fetch_logs=not args.no_logs,
                    )
                except Exception as exc:
                    logger.warning("skipping %s due to error: %s", full_name, exc)
                    recs = []
                completed += 1
                bar.update(1)
                return recs

        tasks = [asyncio.create_task(worker(name)) for name in repos]
        for fut in asyncio.as_completed(tasks):
            recs = await fut
            all_records.extend(recs)
            if completed % args.checkpoint_every == 0 and completed > 0:
                _write(_records_to_dataframe(all_records), out_path)
                logger.info(
                    "checkpoint: %d records from %d/%d repos",
                    len(all_records), completed, len(repos),
                )
        bar.close()

    df = _records_to_dataframe(all_records)
    _write(df, out_path)

    logger.info("wrote %d rows to %s", len(df), out_path)
    if not df.empty:
        summary = df["conclusion"].value_counts(dropna=False).to_dict()
        logger.info("conclusion distribution: %s", summary)
        fc_summary = df["failure_class"].value_counts(dropna=False).to_dict()
        logger.info("failure_class distribution: %s", fc_summary)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect GitHub Actions workflow runs")
    p.add_argument("--repos", required=True, help="Path to file with one owner/repo per line")
    p.add_argument(
        "--out", required=True, help="Output path (.parquet or .csv)"
    )
    p.add_argument(
        "--runs-per-repo", type=int, default=100, help="Max workflow runs to fetch per repository"
    )
    p.add_argument(
        "--no-logs",
        action="store_true",
        help="Skip downloading job logs (faster, no failure_class detection)",
    )
    p.add_argument("--token", help="GitHub PAT (defaults to $GITHUB_API_TOKEN or $GITHUB_TOKEN)")
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Flush partial results to disk every N repos (default: 5)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of repos to process in parallel (default: 4). "
             "Higher values trade GitHub rate-limit risk for speed.",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    global httpx  # imported lazily inside _process_repo
    import httpx as _httpx  # noqa: F401

    globals()["httpx"] = _httpx
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
