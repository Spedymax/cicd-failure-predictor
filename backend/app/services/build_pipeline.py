"""Receive `workflow_run` events and close the feedback loop.

When GitHub completes a workflow it sends a workflow_run event. We use it to:
  1. Append a row to ``build_history`` for tracking past runs.
  2. UPDATE the matching ``predictions.actual_outcome`` so accuracy stats
     have ground-truth labels.

Failure classification is best-effort — we'd need job logs (REST GET) to run
``classify_with_evidence``. For now we record the run conclusion only;
detailed failure_class comes later via the offline retrain pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import (
    BuildHistory,
    BuildStatus,
    Commit,
    FailureClass,
    Prediction,
    Repository,
)

_CONCLUSION_TO_STATUS = {
    "success": BuildStatus.SUCCESS,
    "failure": BuildStatus.FAILURE,
    "cancelled": BuildStatus.CANCELLED,
    "timed_out": BuildStatus.FAILURE,
    "action_required": BuildStatus.FAILURE,
    "neutral": BuildStatus.SUCCESS,
    "skipped": BuildStatus.CANCELLED,
}

logger = logging.getLogger(__name__)


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def process_workflow_run_event(db: Session, payload: dict[str, Any]) -> dict[str, Any]:
    """Process a `workflow_run` webhook with action=completed.

    Returns a small dict describing what was updated (for logging / response).
    """
    action = payload.get("action")
    if action != "completed":
        return {"skipped": True, "reason": f"action={action}"}

    run = payload.get("workflow_run") or {}
    repo_name = (payload.get("repository") or {}).get("full_name")
    sha = run.get("head_sha")
    if not (repo_name and sha):
        return {"skipped": True, "reason": "missing repo or head_sha"}

    conclusion = run.get("conclusion")  # success, failure, cancelled, ...

    repo = db.scalar(select(Repository).where(Repository.full_name == repo_name))
    if repo is None:
        logger.info("workflow_run for unknown repo %s; skipping", repo_name)
        return {"skipped": True, "reason": "unknown repo"}

    commit = db.scalar(select(Commit).where(Commit.repository_id == repo.id, Commit.sha == sha))

    actual_outcome: FailureClass | None = None
    if conclusion == "success":
        actual_outcome = FailureClass.SUCCESS
    elif conclusion == "failure":
        # Mirror the offline relabeling priority (ml/training/data.py::_label_series).
        # Without it every failure collapses to OTHER_FAILURE and the feedback
        # loop teaches the model nothing about the infra-shape signal.
        prior_pred = None
        if commit is not None:
            prior_pred = db.scalar(
                select(Prediction)
                .where(Prediction.commit_id == commit.id)
                .order_by(Prediction.created_at.desc())
            )
        fv = (prior_pred.feature_vector or {}) if prior_pred else {}
        has_docker = bool(fv.get("feat_has_dockerfile_change_int", 0))
        has_deps = bool(fv.get("feat_has_dependency_change_int", 0))
        test_only = bool(fv.get("feat_test_only_changes_int", 0))
        if has_docker:
            actual_outcome = FailureClass.DOCKER_BUILD_FAILED
        elif has_deps:
            actual_outcome = FailureClass.DEPENDENCY_ERROR
        elif test_only:
            actual_outcome = FailureClass.TEST_FAILURE
        else:
            actual_outcome = FailureClass.OTHER_FAILURE

    # 1) build_history row (schema: external_run_id, status, failure_class,
    # started_at, completed_at, raw_payload)
    started = _parse_iso(run.get("run_started_at") or run.get("created_at"))
    completed = _parse_iso(run.get("updated_at"))
    # column is Integer; whole seconds match the DB type (and the regressor target unit)
    duration_seconds = int((completed - started).total_seconds()) if started and completed else None
    # GitHub's `status` is workflow-state ("completed"/"in_progress"); our
    # BuildStatus enum stores the *result*, so map from conclusion.
    if run.get("status") == "in_progress":
        bs = BuildStatus.IN_PROGRESS
    else:
        bs = _CONCLUSION_TO_STATUS.get(conclusion or "", BuildStatus.FAILURE)

    # Upsert: re-runs reuse the same external_run_id, so check for an
    # existing row before inserting to avoid the unique-constraint violation
    # on (repository_id, external_run_id).
    external_run_id = str(run.get("id") or "")
    raw_payload = {
        "workflow_name": run.get("name"),
        "conclusion": conclusion,
        "head_branch": run.get("head_branch"),
        "html_url": run.get("html_url"),
    }
    history = db.scalar(
        select(BuildHistory).where(
            BuildHistory.repository_id == repo.id,
            BuildHistory.external_run_id == external_run_id,
        )
    )
    if history is None:
        history = BuildHistory(
            repository_id=repo.id,
            commit_id=commit.id if commit else None,
            external_run_id=external_run_id,
            status=bs,
            failure_class=actual_outcome,
            duration_seconds=duration_seconds,
            started_at=started,
            completed_at=completed,
            raw_payload=raw_payload,
        )
        db.add(history)
    else:
        history.status = bs
        history.failure_class = actual_outcome
        history.duration_seconds = duration_seconds
        history.started_at = started or history.started_at
        history.completed_at = completed or history.completed_at
        history.raw_payload = raw_payload

    # 2) update matching predictions with ground truth
    n_updated = 0
    if commit is not None and actual_outcome is not None:
        preds = db.scalars(select(Prediction).where(Prediction.commit_id == commit.id)).all()
        for p in preds:
            p.actual_outcome = actual_outcome
            n_updated += 1

    db.commit()
    return {
        "repository": repo_name,
        "sha": sha,
        "conclusion": conclusion,
        "predictions_updated": n_updated,
        "build_history_id": history.id,
    }
