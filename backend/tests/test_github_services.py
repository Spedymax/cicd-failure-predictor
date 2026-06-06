"""Mock-based tests for github_status + github_actions helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.db.models import PredictionDecision
from app.services import github_actions, github_status


def _fake_client_ctx(routes):
    """Build a context-manager stub for httpx.Client whose get/post return
    a Response-like MagicMock matched by URL.
    """
    fake = MagicMock()

    def _match(url, **_kw):
        for u, resp in routes:
            if u in url:
                return resp
        return MagicMock(status_code=404, text="", json=lambda: {})

    fake.get.side_effect = _match
    fake.post.side_effect = _match
    cm = MagicMock()
    cm.__enter__.return_value = fake
    cm.__exit__.return_value = False
    return cm


# ------------- github_status.post_commit_status -------------


def test_post_commit_status_skips_synthetic_prefix() -> None:
    assert (
        github_status.post_commit_status(
            "acme/widget",
            "deadbeef",
            PredictionDecision.BLOCK,
            1,
        )
        is False
    )


def test_post_commit_status_skips_empty_or_malformed() -> None:
    assert (
        github_status.post_commit_status(
            "",
            "sha",
            PredictionDecision.BLOCK,
            1,
        )
        is False
    )
    assert (
        github_status.post_commit_status(
            "noslash",
            "sha",
            PredictionDecision.BLOCK,
            1,
        )
        is False
    )


def test_post_commit_status_success() -> None:
    ok = MagicMock(status_code=201, text="", json=lambda: {})
    with patch.object(
        github_status,
        "httpx",
        MagicMock(Client=MagicMock(return_value=_fake_client_ctx([("statuses", ok)]))),
    ):
        assert (
            github_status.post_commit_status(
                "Spedymax/cicd-predictor-demo",
                "abc" * 14,
                PredictionDecision.AUTO_APPROVE,
                7,
            )
            is True
        )


def test_post_commit_status_http_error_logged_no_raise() -> None:
    bad = MagicMock(status_code=500, text="boom", json=lambda: {})
    with patch.object(
        github_status,
        "httpx",
        MagicMock(Client=MagicMock(return_value=_fake_client_ctx([("statuses", bad)]))),
    ):
        assert (
            github_status.post_commit_status(
                "Spedymax/cicd-predictor-demo",
                "abc" * 14,
                PredictionDecision.BLOCK,
                9,
            )
            is False
        )


def test_post_commit_status_network_exception_swallowed() -> None:
    def boom(*_a, **_k):
        raise RuntimeError("connection refused")

    with patch.object(github_status, "httpx", MagicMock(Client=boom)):
        # Must not raise — best-effort by contract.
        assert (
            github_status.post_commit_status(
                "Spedymax/cicd-predictor-demo",
                "abc" * 14,
                PredictionDecision.WARN,
                1,
            )
            is False
        )


# ------------- github_actions.rerun_failed_runs_for_sha -------------


def test_rerun_skips_synthetic_and_malformed() -> None:
    assert github_actions.rerun_failed_runs_for_sha("acme/x", "sha") == 0
    assert github_actions.rerun_failed_runs_for_sha("", "sha") == 0
    assert github_actions.rerun_failed_runs_for_sha("noslash", "sha") == 0


def test_rerun_triggers_only_failed_completed_runs() -> None:
    listing = MagicMock(
        status_code=200,
        json=lambda: {
            "workflow_runs": [
                {"id": 1, "status": "completed", "conclusion": "failure"},
                {"id": 2, "status": "completed", "conclusion": "success"},  # skip
                {"id": 3, "status": "in_progress", "conclusion": None},  # skip
                {"id": 4, "status": "completed", "conclusion": "cancelled"},
            ]
        },
    )
    rerun_ok = MagicMock(status_code=201, text="", json=lambda: {})

    fake = MagicMock()
    fake.get.return_value = listing
    fake.post.return_value = rerun_ok
    cm = MagicMock()
    cm.__enter__.return_value = fake
    cm.__exit__.return_value = False

    with patch.object(github_actions, "httpx", MagicMock(Client=MagicMock(return_value=cm))):
        n = github_actions.rerun_failed_runs_for_sha("Spedymax/demo", "shaXYZ")

    assert n == 2  # runs 1 + 4
    # Two POSTs to rerun-failed-jobs
    assert fake.post.call_count == 2


def test_rerun_list_failure_short_circuits() -> None:
    listing = MagicMock(status_code=500, text="bad", json=lambda: {})
    fake = MagicMock()
    fake.get.return_value = listing
    cm = MagicMock()
    cm.__enter__.return_value = fake
    cm.__exit__.return_value = False
    with patch.object(github_actions, "httpx", MagicMock(Client=MagicMock(return_value=cm))):
        assert github_actions.rerun_failed_runs_for_sha("Spedymax/demo", "sha") == 0
    assert fake.post.call_count == 0
