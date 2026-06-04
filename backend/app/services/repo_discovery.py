"""On-create repository metadata discovery.

Probes the GitHub REST API for a single repo and returns:
  * language          — main language (from GET /repos/{owner}/{repo})
  * has_dockerfile    — True if a top-level Dockerfile exists
  * package_manager   — best guess from root manifest files

The call is cheap (one or two REST hits) and is invoked once when the
repository row is created. Failures are tolerated — the row is still
created with empty metadata, and a later webhook can refine it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Order matters: first match wins.
_PKG_HINTS: list[tuple[str, str]] = [
    ("pnpm-lock.yaml", "pnpm"),
    ("yarn.lock", "yarn"),
    ("package-lock.json", "npm"),
    ("package.json", "npm"),
    ("poetry.lock", "poetry"),
    ("uv.lock", "uv"),
    ("pyproject.toml", "pip"),
    ("requirements.txt", "pip"),
    ("Pipfile.lock", "pipenv"),
    ("go.sum", "go"),
    ("go.mod", "go"),
    ("Cargo.toml", "cargo"),
    ("pom.xml", "maven"),
    ("build.gradle", "gradle"),
    ("build.gradle.kts", "gradle"),
    ("Gemfile.lock", "bundler"),
    ("composer.json", "composer"),
]


@dataclass
class RepoMetadata:
    language: str | None = None
    has_dockerfile: bool = False
    package_manager: str | None = None


def _headers() -> dict[str, str]:
    settings = get_settings()
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {settings.github_api_token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "cicd-failure-predictor/0.1",
    }


def discover_repository_metadata(full_name: str, *, timeout: float = 8.0) -> RepoMetadata:
    """Best-effort metadata probe. Never raises — returns empty on failure."""
    owner, _, repo = full_name.partition("/")
    if not repo:
        return RepoMetadata()

    meta = RepoMetadata()
    try:
        with httpx.Client(timeout=timeout, headers=_headers()) as c:
            r_repo = c.get(f"https://api.github.com/repos/{owner}/{repo}")
            if r_repo.status_code == 200:
                meta.language = r_repo.json().get("language")
            else:
                logger.info("repo probe %s → %s; skipping", full_name, r_repo.status_code)

            r_root = c.get(f"https://api.github.com/repos/{owner}/{repo}/contents/")
            if r_root.status_code == 200:
                items = r_root.json() if isinstance(r_root.json(), list) else []
                names = {it.get("name", "") for it in items}
                meta.has_dockerfile = any(
                    n == "Dockerfile" or n.endswith(".dockerfile") for n in names
                )
                for hint_name, pm in _PKG_HINTS:
                    if hint_name in names:
                        meta.package_manager = pm
                        break
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("discovery failed for %s: %s", full_name, exc)

    return meta
