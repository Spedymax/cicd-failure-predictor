"""Stage 2: REST-enrich a balanced subset of BQ-fetched runs.

What REST gives us that GH Archive doesn't:
  - commit diff (files_changed, lines_added/deleted, file_extensions, ...)
  - job logs (for failure_class heuristic on conclusion='failure' runs)

We use a token pool so concurrency scales with PAT count instead of being
capped at 5000 req/hr.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pandas as pd
from tqdm.asyncio import tqdm

from data_collection.failure_heuristics import classify_with_evidence
from data_collection.hybrid.token_pool import TokenPool, TokenState

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 5


class PooledGitHubClient:
    """Same retry/backoff semantics as `data_collection.github_client` but
    rotates Authorization header per request via TokenPool."""

    def __init__(self, pool: TokenPool) -> None:
        self._pool = pool
        self._client = httpx.AsyncClient(
            base_url=GITHUB_API,
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "cicd-failure-predictor-hybrid/0.1",
            },
            timeout=DEFAULT_TIMEOUT,
            follow_redirects=True,
        )

    async def __aenter__(self) -> "PooledGitHubClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            state: TokenState = await self._pool.acquire()
            try:
                bearer = await state.source.get()
            except Exception as exc:
                logger.error("token source %s failed: %s", state.display, exc)
                await self._pool.release(state, remaining=None, reset_at=None, counted=False)
                raise
            headers = kwargs.pop("headers", {}) or {}
            headers["Authorization"] = f"Bearer {bearer}"
            try:
                resp = await self._client.request(method, path, headers=headers, **kwargs)
            except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                await self._pool.release(state, remaining=None, reset_at=None, counted=False)
                await asyncio.sleep(2**attempt)
                continue

            remaining_h = resp.headers.get("x-ratelimit-remaining")
            reset_h = resp.headers.get("x-ratelimit-reset")
            await self._pool.release(
                state,
                remaining=int(remaining_h) if remaining_h is not None else None,
                reset_at=float(reset_h) if reset_h is not None else None,
            )

            if resp.status_code in (200, 201, 202) or resp.status_code == 404:
                return resp
            if resp.status_code in (403, 429):
                retry_after = resp.headers.get("retry-after")
                wait = float(retry_after) if retry_after else 30.0
                logger.warning(
                    "secondary rate limit on %s (status %s) sleeping %.1fs",
                    state.display, resp.status_code, wait,
                )
                await asyncio.sleep(min(wait, 120.0))
                continue
            if 500 <= resp.status_code < 600:
                await asyncio.sleep(2**attempt)
                continue
            return resp
        if last_exc is not None:
            raise last_exc
        return resp

    async def get_commit(self, owner: str, repo: str, sha: str) -> dict[str, Any] | None:
        resp = await self._request("GET", f"/repos/{owner}/{repo}/commits/{sha}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    async def list_jobs(self, owner: str, repo: str, run_id: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        page = 1
        while True:
            resp = await self._request(
                "GET",
                f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs",
                params={"per_page": 100, "page": page},
            )
            if resp.status_code == 404:
                return out
            resp.raise_for_status()
            jobs = resp.json().get("jobs", [])
            out.extend(jobs)
            if len(jobs) < 100:
                return out
            page += 1

    async def get_job_logs(
        self, owner: str, repo: str, job_id: int, *, max_bytes: int = 200_000
    ) -> str | None:
        resp = await self._request(
            "GET", f"/repos/{owner}/{repo}/actions/jobs/{job_id}/logs"
        )
        if resp.status_code != 200:
            return None
        text = resp.text
        return text[-max_bytes:] if len(text) > max_bytes else text

    async def list_workflow_runs(
        self, owner: str, repo: str, *, max_runs: int = 100, status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Paginated list of workflow_runs, stopping at max_runs."""
        out: list[dict[str, Any]] = []
        page = 1
        per_page = 100
        while len(out) < max_runs:
            params: dict[str, Any] = {"per_page": per_page, "page": page}
            if status:
                params["status"] = status
            resp = await self._request(
                "GET", f"/repos/{owner}/{repo}/actions/runs", params=params
            )
            if resp.status_code == 404:
                return out
            resp.raise_for_status()
            runs = resp.json().get("workflow_runs", [])
            out.extend(runs)
            if len(runs) < per_page:
                return out[:max_runs]
            page += 1
        return out[:max_runs]


# ---------- feature derivation (mirrors collect.py helpers) ----------

_DEP_MARKERS = (
    "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "requirements.txt", "pyproject.toml", "poetry.lock", "uv.lock",
    "pipfile", "go.mod", "go.sum", "cargo.toml", "cargo.lock",
    "pom.xml", "build.gradle",
)
_TEST_PATH_MARKERS = ("/tests/", "/test/", "/__tests__/", "/spec/", "/specs/")
_TEST_FILENAME_MARKERS = (
    "_test.go", "_test.py", ".test.ts", ".test.tsx", ".test.js", ".test.jsx",
    ".spec.ts", ".spec.js", ".spec.tsx", "test_", "_spec.rb",
)
_LINT_CONFIG_FILES = (
    ".eslintrc", ".eslintrc.js", ".eslintrc.json", ".eslintrc.yaml", ".eslintrc.yml",
    ".flake8", ".pylintrc", "mypy.ini", ".mypy.ini", "ruff.toml", ".ruff.toml",
    ".prettierrc", ".golangci.yml", ".golangci.yaml", "tslint.json", ".rubocop.yml",
)


def _derive_commit_features(commit: dict[str, Any]) -> dict[str, Any]:
    files = commit.get("files") or []
    names = [f.get("filename", "") for f in files]
    lower = [n.lower() for n in names]

    def _is_test(n: str) -> bool:
        if any(m in n for m in _TEST_PATH_MARKERS):
            return True
        base = n.rsplit("/", 1)[-1]
        return any(base.endswith(m) or base.startswith(m) or m in base for m in _TEST_FILENAME_MARKERS)

    has_dockerfile = any(
        n == "dockerfile" or n.endswith(".dockerfile") or "/dockerfile" in n for n in lower
    )
    has_dependency = any(any(n.endswith(m) or n == m for m in _DEP_MARKERS) for n in lower)
    test_count = sum(1 for n in lower if _is_test(n))
    test_only = bool(lower) and test_count == len(lower)
    has_lint = any(
        (base := n.rsplit("/", 1)[-1]) in _LINT_CONFIG_FILES
        or any(base.startswith(m) for m in _LINT_CONFIG_FILES)
        for n in lower
    )
    ext_counts: dict[str, int] = {}
    for n in names:
        ext = n.rsplit(".", 1)[-1].lower() if "." in n.rsplit("/", 1)[-1] else "noext"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    return {
        "files_changed": len(files),
        "lines_added": sum(int(f.get("additions") or 0) for f in files),
        "lines_deleted": sum(int(f.get("deletions") or 0) for f in files),
        "has_dockerfile_change": has_dockerfile,
        "has_dependency_change": has_dependency,
        "file_extensions": ext_counts,
        "test_dir_changes": test_count,
        "test_only_changes": test_only,
        "has_lint_config_change": has_lint,
        "actor_email": (commit.get("commit", {}).get("author") or {}).get("email"),
    }


# ---------- run listing (replaces former BQ fetcher) ----------

async def list_runs_for_repos(
    pool: TokenPool,
    repos: list[str],
    *,
    max_per_repo: int,
    failure_share: float = 0.5,
    repo_concurrency: int | None = None,
) -> "pd.DataFrame":
    """Fan-out REST list_workflow_runs across all target repos via the pool.

    For each repo: fetch up to round(max_per_repo * failure_share) failed runs
    and the rest as successful runs. This uses GitHub's server-side
    ?status=failure / success filter so we don't waste requests on data we
    won't keep.
    """
    n_fail = int(round(max_per_repo * failure_share))
    n_succ = max_per_repo - n_fail
    sem = asyncio.Semaphore(repo_concurrency or len(pool) * 4)

    async def list_one(full_name: str, client: PooledGitHubClient) -> list[dict[str, Any]]:
        owner, _, repo = full_name.partition("/")
        if not repo:
            logger.warning("skipping invalid entry: %s", full_name)
            return []
        async with sem:
            rows: list[dict[str, Any]] = []
            for status, want in (("failure", n_fail), ("success", n_succ)):
                if want <= 0:
                    continue
                try:
                    runs = await client.list_workflow_runs(owner, repo, max_runs=want, status=status)
                except Exception as exc:
                    logger.warning("list_runs(%s, %s) failed: %s", full_name, status, exc)
                    runs = []
                for r in runs:
                    rows.append({
                        "repo": full_name,
                        "run_id": int(r["id"]),
                        "workflow_name": r.get("name") or "",
                        "head_sha": r.get("head_sha") or "",
                        "head_branch": r.get("head_branch"),
                        "event": r.get("event") or "",
                        "status": r.get("status") or "",
                        "conclusion": r.get("conclusion"),
                        "run_attempt": int(r.get("run_attempt") or 1),
                        "created_at": r.get("created_at") or r.get("created_at"),
                        "updated_at": r.get("updated_at"),
                        "actor_login": (r.get("actor") or {}).get("login"),
                        "event_is_push": r.get("event") == "push",
                    })
            return rows

    bar = tqdm(total=len(repos), desc="list runs")
    all_rows: list[dict[str, Any]] = []
    async with PooledGitHubClient(pool) as client:
        async def worker(full_name: str) -> None:
            rows = await list_one(full_name, client)
            all_rows.extend(rows)
            bar.update(1)
        await asyncio.gather(*(worker(r) for r in repos))
    bar.close()

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True, errors="coerce")
        df["duration_seconds"] = (
            df["updated_at"] - df["created_at"]
        ).dt.total_seconds()
        df = df.drop_duplicates(subset=["repo", "run_id", "run_attempt"], keep="last")
    logger.info(
        "listed %d runs across %d repos (failures=%d, successes=%d)",
        len(df), df["repo"].nunique() if not df.empty else 0,
        int((df["conclusion"] == "failure").sum()) if not df.empty else 0,
        int((df["conclusion"] == "success").sum()) if not df.empty else 0,
    )
    return df


# ---------- sampling + orchestration ----------

def balanced_sample(df: pd.DataFrame, *, max_per_repo: int, failure_share: float = 0.5,
                     seed: int = 42) -> pd.DataFrame:
    """Per repo: keep all failures up to budget, fill the rest with random successes."""
    out = []
    rng = pd.Series([0]).sample(frac=1, random_state=seed)  # init rng
    for repo, grp in df.groupby("repo"):
        fails = grp[grp["conclusion"] == "failure"]
        succs = grp[grp["conclusion"] == "success"]
        n_fail_target = int(round(max_per_repo * failure_share))
        n_succ_target = max_per_repo - n_fail_target
        picked_fails = fails.sample(n=min(len(fails), n_fail_target), random_state=seed) if len(fails) else fails
        picked_succs = succs.sample(n=min(len(succs), n_succ_target), random_state=seed) if len(succs) else succs
        out.append(pd.concat([picked_fails, picked_succs]))
    return pd.concat(out, ignore_index=True) if out else df.iloc[0:0]


_KEY_COLS = ("repo", "run_id", "run_attempt")


def _merge_extras(df: pd.DataFrame, extras: dict[int, dict[str, Any]]) -> pd.DataFrame:
    out = df.copy()
    feat_cols = {k for d in extras.values() for k in d}
    for col in feat_cols:
        out[col] = out.index.map(lambda i: extras.get(i, {}).get(col))
    return out


async def enrich_dataframe(
    df: pd.DataFrame,
    *,
    pool: TokenPool,
    fetch_logs: bool = True,
    concurrency_per_token: int = 8,
    checkpoint_path: "Path | None" = None,
    checkpoint_every: int = 500,
) -> pd.DataFrame:
    """Add commit-diff features + (optional) failure_class via REST.

    Resumability: if ``checkpoint_path`` exists, already-enriched rows (matched
    on ``repo, run_id, run_attempt``) are skipped. Periodic writes happen
    every ``checkpoint_every`` completed rows and on any exception, so a kill
    or rate-limit deadlock loses at most ``checkpoint_every`` rows of work.
    """
    from pathlib import Path

    if df.empty:
        return df

    df = df.reset_index(drop=True)
    done_df: pd.DataFrame | None = None
    if checkpoint_path is not None:
        cp = Path(checkpoint_path)
        if cp.exists():
            try:
                done_df = pd.read_parquet(cp)
                # only rows that actually got commit-enriched have files_changed set
                if "files_changed" in done_df.columns:
                    done_df = done_df[done_df["files_changed"].notna()].copy()
                logger.info("checkpoint loaded: %d already-enriched rows", len(done_df))
            except Exception as exc:
                logger.warning("could not load checkpoint %s: %s", cp, exc)
                done_df = None

    if done_df is not None and not done_df.empty:
        merge_key = list(_KEY_COLS)
        mask = df.merge(done_df[merge_key].assign(_seen=True), on=merge_key, how="left")["_seen"].fillna(False)
        todo = df[~mask].reset_index(drop=True)
        logger.info("enriching %d/%d rows (rest from checkpoint)", len(todo), len(df))
    else:
        todo = df

    if todo.empty:
        return done_df.reset_index(drop=True) if done_df is not None else df

    sem = asyncio.Semaphore(max(1, concurrency_per_token * len(pool)))
    extras: dict[int, dict[str, Any]] = {}
    extras_lock = asyncio.Lock()
    bar = tqdm(total=len(todo), desc="REST enrich")

    async def write_checkpoint() -> None:
        if checkpoint_path is None or not extras:
            return
        cp = Path(checkpoint_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        async with extras_lock:
            partial = _merge_extras(todo, dict(extras))
        if done_df is not None:
            combined = pd.concat([done_df, partial], ignore_index=True)
            combined = combined.drop_duplicates(subset=list(_KEY_COLS), keep="last")
        else:
            combined = partial
        tmp = cp.with_suffix(cp.suffix + ".tmp")
        combined.to_parquet(tmp, index=False)
        tmp.replace(cp)
        logger.info("checkpoint: wrote %d rows to %s", len(combined), cp)

    async with PooledGitHubClient(pool) as client:

        async def enrich_one(idx: int, row: pd.Series) -> None:
            async with sem:
                owner, _, repo = row["repo"].partition("/")
                features: dict[str, Any] = {}
                try:
                    commit = await client.get_commit(owner, repo, row["head_sha"])
                    if commit:
                        features.update(_derive_commit_features(commit))
                except Exception as exc:
                    logger.debug("commit fail %s: %s", row["head_sha"], exc)

                # Jobs metadata (n_jobs, longest_job_seconds) is fetched for
                # ALL runs -- success included -- otherwise those features
                # become a collection artifact (n_jobs=0 only ever on success).
                # Logs + failure_class are still failure-only.
                try:
                    jobs = await client.list_jobs(owner, repo, int(row["run_id"]))
                    features["n_jobs"] = len(jobs)
                    durations = []
                    failed_job = None
                    for j in jobs:
                        s = j.get("started_at")
                        c = j.get("completed_at")
                        if s and c:
                            durations.append(
                                (pd.Timestamp(c) - pd.Timestamp(s)).total_seconds()
                            )
                        if failed_job is None and j.get("conclusion") == "failure":
                            failed_job = j
                    if durations:
                        features["longest_job_seconds"] = max(durations)
                    if fetch_logs and row["conclusion"] == "failure" and failed_job is not None:
                        log = await client.get_job_logs(owner, repo, failed_job["id"])
                        cls, evidence = classify_with_evidence(log)
                        features["failure_class"] = cls
                        features["failure_evidence"] = (evidence or "")[:200] if evidence else None
                except Exception as exc:
                    logger.debug("jobs/logs fail %s/%s: %s", row["repo"], row["run_id"], exc)

                async with extras_lock:
                    extras[idx] = features
                    n_done = len(extras)
                bar.update(1)
                if n_done % checkpoint_every == 0:
                    await write_checkpoint()

        try:
            await asyncio.gather(*(enrich_one(i, r) for i, r in todo.iterrows()))
        except BaseException:
            logger.exception("enrich aborted; writing checkpoint before re-raise")
            await write_checkpoint()
            raise
    bar.close()

    enriched_part = _merge_extras(todo, extras)
    if done_df is not None and not done_df.empty:
        result = pd.concat([done_df, enriched_part], ignore_index=True)
        result = result.drop_duplicates(subset=list(_KEY_COLS), keep="last")
    else:
        result = enriched_part

    if checkpoint_path is not None:
        await write_checkpoint()
    return result.reset_index(drop=True)
