"""Locust benchmark for the prediction pipeline (NFR-02, NFR-03).

Run:
    cd <repo> && set -a && . ./.env && set +a
    locust -f scripts/locustfile.py --host http://127.0.0.1:8000 \\
           --users 20 --spawn-rate 5 --headless -t 1m

Tasks:
- /api/v1/webhook/github with a synthetic push payload + valid HMAC signature
- /api/v1/predictions (list, paginated)

The webhook task measures end-to-end inference latency under load. The
list task exercises read paths so we get both p50/p95 numbers in one run.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
import uuid

from locust import HttpUser, between, task

_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET", "")


def _sign(body: bytes) -> str:
    digest = hmac.new(_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _payload() -> dict:
    sha = uuid.uuid4().hex + uuid.uuid4().hex[:8]
    return {
        "ref": "refs/heads/main",
        "before": "0" * 40,
        "after": sha,
        "repository": {
            "id": 1,
            "name": "cicd-predictor-demo",
            "full_name": "Spedymax/cicd-predictor-demo",
            "private": False,
            "owner": {"login": "Spedymax"},
            "default_branch": "main",
        },
        "pusher": {"name": "loadtest", "email": "loadtest@example.com"},
        "head_commit": {
            "id": sha,
            "message": "loadtest commit",
            "timestamp": "2026-05-26T00:00:00Z",
            "author": {"name": "loadtest", "email": "loadtest@example.com"},
            "added": ["README.md"] if random.random() < 0.5 else [],
            "modified": ["README.md"],
            "removed": [],
        },
        "commits": [],
    }


class PredictorUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(3)
    def push_webhook(self) -> None:
        body = json.dumps(_payload()).encode()
        headers = {
            "X-GitHub-Event": "push",
            "X-GitHub-Delivery": uuid.uuid4().hex,
            "X-Hub-Signature-256": _sign(body),
            "Content-Type": "application/json",
        }
        with self.client.post(
            "/api/v1/webhook/github", data=body, headers=headers,
            name="POST /webhook/github", catch_response=True,
        ) as resp:
            if resp.status_code not in (200, 202):
                resp.failure(f"unexpected {resp.status_code}")

    @task(1)
    def list_predictions(self) -> None:
        self.client.get("/api/v1/predictions?limit=20", name="GET /predictions")
