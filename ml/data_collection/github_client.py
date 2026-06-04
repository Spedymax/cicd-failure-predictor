"""Thin async wrapper around the GitHub REST API.

Handles authentication, exponential-backoff retries, secondary rate-limit
respect (X-RateLimit-Remaining + Retry-After), and a small in-memory LRU for
endpoints we hit repeatedly per repo (commits, contents).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 5


class GitHubClient:
    def __init__(self, token: str, *, per_page: int = 100) -> None:
        self._token = token
        self._per_page = per_page
        self._client = httpx.AsyncClient(
            base_url=GITHUB_API,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "cicd-failure-predictor/0.1",
            },
            timeout=DEFAULT_TIMEOUT,
            http2=False,
            follow_redirects=True,
        )
        self._commit_cache: dict[tuple[str, str, str], dict[str, Any] | None] = {}

    async def __aenter__(self) -> "GitHubClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = await self._client.request(method, path, **kwargs)
            except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                backoff = 2**attempt
                logger.warning("network error %s, backoff %ss", type(exc).__name__, backoff)
                await asyncio.sleep(backoff)
                continue
            if resp.status_code in (200, 201, 202):
                return resp
            if resp.status_code == 404:
                return resp
            if resp.status_code in (403, 429):
                remaining = resp.headers.get("x-ratelimit-remaining")
                retry_after = resp.headers.get("retry-after")
                reset = resp.headers.get("x-ratelimit-reset")
                wait = 60.0
                if retry_after:
                    wait = float(retry_after)
                elif remaining == "0" and reset:
                    wait = max(1.0, float(reset) - asyncio.get_event_loop().time())
                logger.warning("rate limited (%s) sleeping %.1fs", resp.status_code, wait)
                await asyncio.sleep(min(wait, 120.0))
                continue
            if 500 <= resp.status_code < 600:
                backoff = 2**attempt
                logger.warning("server error %s, backoff %ss", resp.status_code, backoff)
                await asyncio.sleep(backoff)
                continue
            return resp
        if last_exc is not None:
            raise last_exc
        return resp

    async def get_json(self, path: str, **params: Any) -> Any:
        resp = await self._request("GET", path, params=params)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    async def paginate(
        self, path: str, *, max_pages: int | None = None, **params: Any
    ) -> AsyncIterator[dict[str, Any]]:
        params.setdefault("per_page", self._per_page)
        page = 1
        while True:
            params["page"] = page
            resp = await self._request("GET", path, params=params)
            if resp.status_code == 404:
                return
            resp.raise_for_status()
            data = resp.json()
            items = data.get("workflow_runs") or data.get("jobs") or data
            if not isinstance(items, list):
                return
            for item in items:
                yield item
            if len(items) < params["per_page"]:
                return
            if max_pages is not None and page >= max_pages:
                return
            page += 1

    async def list_workflow_runs(
        self, owner: str, repo: str, *, max_runs: int = 200, status: str | None = None
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        runs: list[dict[str, Any]] = []
        async for run in self.paginate(
            f"/repos/{owner}/{repo}/actions/runs", **params
        ):
            runs.append(run)
            if len(runs) >= max_runs:
                break
        return runs

    async def list_jobs(self, owner: str, repo: str, run_id: int) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        async for job in self.paginate(
            f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs"
        ):
            jobs.append(job)
        return jobs

    async def get_commit(self, owner: str, repo: str, sha: str) -> dict[str, Any] | None:
        key = (owner, repo, sha)
        if key in self._commit_cache:
            return self._commit_cache[key]
        data = await self.get_json(f"/repos/{owner}/{repo}/commits/{sha}")
        self._commit_cache[key] = data
        return data

    async def get_job_logs(
        self, owner: str, repo: str, job_id: int, *, max_bytes: int = 200_000
    ) -> str | None:
        resp = await self._request(
            "GET", f"/repos/{owner}/{repo}/actions/jobs/{job_id}/logs"
        )
        if resp.status_code != 200:
            return None
        text = resp.text
        if len(text) > max_bytes:
            return text[-max_bytes:]
        return text
