"""Send a handful of synthetic webhook events to populate the dashboard."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path

import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_env() -> str:
    env_path = REPO_ROOT / ".env"
    secret = os.environ.get("GITHUB_WEBHOOK_SECRET")
    if secret:
        return secret
    for line in env_path.read_text().splitlines():
        if line.startswith("GITHUB_WEBHOOK_SECRET="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("GITHUB_WEBHOOK_SECRET not set")


def _payload(repo: str, sha: str, author: str, *, files_modified: list[str], lines_added: int, lines_deleted: int, message: str) -> dict:
    return {
        "ref": "refs/heads/main",
        "after": sha,
        "repository": {
            "full_name": repo,
            "html_url": f"https://github.com/{repo}",
        },
        "head_commit": {
            "id": sha,
            "message": message,
            "author": {"email": author, "name": author.split("@")[0]},
            "added": [],
            "modified": files_modified,
            "removed": [],
        },
        "lines_added": lines_added,
        "lines_deleted": lines_deleted,
    }


SCENARIOS = [
    dict(
        repo="acme/api-gateway",
        sha="aa11" + "0" * 36,
        author="alice@acme.io",
        files_modified=["src/handler.py", "tests/test_handler.py"],
        lines_added=45,
        lines_deleted=12,
        message="fix: handle empty body in POST /events",
    ),
    dict(
        repo="acme/ml-pipeline",
        sha="bb22" + "0" * 36,
        author="bob@acme.io",
        files_modified=["pyproject.toml", "src/model.py", "Dockerfile"],
        lines_added=380,
        lines_deleted=20,
        message="feat: add tensorflow + new training pipeline",
    ),
    dict(
        repo="acme/web-frontend",
        sha="cc33" + "0" * 36,
        author="carol@acme.io",
        files_modified=["package.json", "package-lock.json", "src/App.tsx"],
        lines_added=210,
        lines_deleted=15,
        message="chore: bump react to 19, webpack to 6",
    ),
    dict(
        repo="acme/data-platform",
        sha="dd44" + "0" * 36,
        author="dan@acme.io",
        files_modified=["Dockerfile", "ops/Makefile"],
        lines_added=70,
        lines_deleted=10,
        message="ops: switch base image to ubuntu:22.04 + cuda",
    ),
    dict(
        repo="acme/monolith",
        sha="ee55" + "0" * 36,
        author="erin@acme.io",
        files_modified=["src/server.py", "tests/test_e2e.py"],
        lines_added=520,
        lines_deleted=120,
        message="refactor: extract worker pool into separate module",
    ),
]


def main() -> int:
    base = os.environ.get("APP_URL", "http://localhost:8000")
    secret = _read_env()
    for sc in SCENARIOS:
        body = json.dumps(_payload(**sc)).encode()
        sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        req = urllib.request.Request(
            f"{base}/api/v1/webhook/github",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-GitHub-Event": "push",
                "X-GitHub-Delivery": sc["sha"],
                "X-Hub-Signature-256": sig,
            },
        )
        with urllib.request.urlopen(req) as resp:
            print(f"  {sc['repo']:25s} sha={sc['sha'][:7]} → HTTP {resp.status}")
        time.sleep(0.4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
