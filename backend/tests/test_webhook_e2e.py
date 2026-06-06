from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ARTIFACTS = Path(__file__).resolve().parents[2] / "data" / "artifacts" / "v26_5class"


@pytest.fixture(scope="module")
def client() -> TestClient:
    if not (ARTIFACTS / "risk_rf.joblib").exists():
        pytest.skip("trained artefacts not available")
    from app.main import app

    return TestClient(app)


def _push_payload(repo: str = "demo/cicd-failure-predictor", sha: str | None = None) -> dict:
    sha = sha or "abcdef0123456789abcdef0123456789abcdef01"
    return {
        "ref": "refs/heads/main",
        "before": "0" * 40,
        "after": sha,
        "repository": {
            "full_name": repo,
            "html_url": f"https://github.com/{repo}",
        },
        "head_commit": {
            "id": sha,
            "message": "feat: add tensorflow dependency",
            "author": {"email": "alice@example.com", "name": "Alice"},
            "added": ["src/model.py"],
            "modified": ["pyproject.toml", "Dockerfile"],
            "removed": [],
        },
        "lines_added": 320,
        "lines_deleted": 25,
    }


def _signed_post(client: TestClient, body: bytes) -> httpx.Response:  # noqa: F821
    import hashlib
    import hmac

    secret = os.environ["GITHUB_WEBHOOK_SECRET"]
    sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return client.post(
        "/api/v1/webhook/github",
        content=body,
        headers={
            "X-GitHub-Event": "push",
            "X-GitHub-Delivery": "test-delivery-1",
            "X-Hub-Signature-256": sig,
            "Content-Type": "application/json",
        },
    )


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ping_event(client: TestClient) -> None:
    body = json.dumps({"zen": "Approachable"}).encode()
    import hashlib
    import hmac

    secret = os.environ["GITHUB_WEBHOOK_SECRET"]
    sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    r = client.post(
        "/api/v1/webhook/github",
        content=body,
        headers={
            "X-GitHub-Event": "ping",
            "X-GitHub-Delivery": "ping-1",
            "X-Hub-Signature-256": sig,
            "Content-Type": "application/json",
        },
    )
    assert r.status_code == 202
    assert r.json()["accepted"] is True


def test_invalid_signature_rejected(client: TestClient) -> None:
    body = json.dumps(_push_payload()).encode()
    r = client.post(
        "/api/v1/webhook/github",
        content=body,
        headers={
            "X-GitHub-Event": "push",
            "X-Hub-Signature-256": "sha256=deadbeef",
            "Content-Type": "application/json",
        },
    )
    assert r.status_code == 403


def test_push_creates_prediction(client: TestClient) -> None:
    body = json.dumps(_push_payload(sha="11" + "a" * 38)).encode()
    r = _signed_post(client, body)
    assert r.status_code == 202, r.text
    data = r.json()
    assert data["accepted"] is True
    assert data["repository"] == "demo/cicd-failure-predictor"

    # Background task is executed inline by TestClient before returning.
    listing = client.get("/api/v1/predictions?limit=5").json()
    assert listing["total"] >= 1
    pred_id = listing["items"][0]["id"]
    detail = client.get(f"/api/v1/predictions/{pred_id}").json()
    assert detail["commit_short"]
    assert detail["predicted_class"] in (
        "success",
        "test_timeout",
        "test_failure",
        "dependency_error",
        "docker_build_failed",
        "other_failure",
    )
    assert "recommendations" in detail


def test_non_push_event_returns_unaccepted(client: TestClient) -> None:
    body = json.dumps({"action": "opened"}).encode()
    import hashlib
    import hmac

    secret = os.environ["GITHUB_WEBHOOK_SECRET"]
    sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    r = client.post(
        "/api/v1/webhook/github",
        content=body,
        headers={
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": sig,
            "Content-Type": "application/json",
        },
    )
    assert r.status_code == 202
    assert r.json()["accepted"] is False
