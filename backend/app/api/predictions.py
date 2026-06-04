from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.core.audit import record_audit
from app.core.auth import get_current_user
from app.db.models import FailureClass, Prediction, PredictionDecision, Repository, User
from app.db.session import get_db
from app.schemas.prediction import (
    PredictionListItem,
    PredictionListResponse,
    PredictionOut,
    RecommendationOut,
)


class OverrideIn(BaseModel):
    new_decision: PredictionDecision
    reason: str = Field(min_length=1, max_length=1000)

router = APIRouter(prefix="/predictions", tags=["predictions"])

DEMO_PREFIXES = ("Spedymax/cicd-predictor-demo",)

ALL_CLASSES: tuple[str, ...] = (
    "success",
    "oom_killed",
    "test_timeout",
    "dependency_error",
    "docker_build_failed",
    "network_error",
    "test_failure",
    "other_failure",
)


def _to_list_item(row: tuple[Prediction, Repository]) -> PredictionListItem:
    pred, repo = row
    sha = pred.commit.sha if pred.commit else ""
    return PredictionListItem(
        id=pred.id,
        repository_full_name=repo.full_name if repo else "",
        commit_short=sha[:7],
        author_email=pred.commit.author_email if pred.commit else "",
        predicted_class=pred.predicted_class,
        decision=pred.decision,
        risk_score=pred.risk_score,
        confidence=pred.confidence,
        created_at=pred.created_at,
    )


@router.get("", response_model=PredictionListResponse)
def list_predictions(
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    repository: str | None = Query(default=None),
    source: str = Query(default="all", pattern="^(all|demo|real)$"),
    predicted_class: str | None = Query(default=None),
) -> PredictionListResponse:
    stmt = (
        select(Prediction, Repository)
        .join(Repository, Prediction.repository_id == Repository.id)
        .order_by(Prediction.created_at.desc())
    )
    count_stmt = (
        select(func.count(Prediction.id))
        .select_from(Prediction)
        .join(Repository, Prediction.repository_id == Repository.id)
    )
    if repository:
        stmt = stmt.where(Repository.full_name == repository)
        count_stmt = count_stmt.where(Repository.full_name == repository)
    if source == "demo":
        demo_filter = or_(*[Repository.full_name.like(f"{p}%") for p in DEMO_PREFIXES])
        stmt = stmt.where(demo_filter)
        count_stmt = count_stmt.where(demo_filter)
    elif source == "real":
        for p in DEMO_PREFIXES:
            stmt = stmt.where(~Repository.full_name.like(f"{p}%"))
            count_stmt = count_stmt.where(~Repository.full_name.like(f"{p}%"))
    if predicted_class:
        try:
            cls = FailureClass(predicted_class)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"unknown predicted_class={predicted_class}") from exc
        stmt = stmt.where(Prediction.predicted_class == cls)
        count_stmt = count_stmt.where(Prediction.predicted_class == cls)
    rows = db.execute(stmt.limit(limit).offset(offset)).all()
    total = db.scalar(count_stmt) or 0
    items = [_to_list_item(r) for r in rows]
    return PredictionListResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/stats/accuracy")
def predictions_accuracy(
    db: Session = Depends(get_db),
    source: str = Query(default="real", pattern="^(all|demo|real)$"),
) -> dict:
    """Compare predicted_class with actual_outcome for rows that have both.

    Note: ``actual_outcome`` for real rows is derived from a regex
    heuristic over CI logs (see ``failure_heuristics.py``). It is a
    *noisy* ground truth and should be interpreted as such — agreement
    with the model means agreement with the heuristic, not necessarily
    with the true root cause.
    """
    stmt = (
        select(Prediction.predicted_class, Prediction.actual_outcome, Repository.full_name)
        .join(Repository, Prediction.repository_id == Repository.id)
        .where(Prediction.actual_outcome.is_not(None))
    )
    if source == "demo":
        stmt = stmt.where(or_(*[Repository.full_name.like(f"{p}%") for p in DEMO_PREFIXES]))
    elif source == "real":
        for p in DEMO_PREFIXES:
            stmt = stmt.where(~Repository.full_name.like(f"{p}%"))

    rows = db.execute(stmt).all()
    confusion: dict[str, dict[str, int]] = {
        a: {p: 0 for p in ALL_CLASSES} for a in ALL_CLASSES
    }
    n_total = 0
    n_match = 0
    for pred_cls, actual_cls, _full_name in rows:
        pv = pred_cls.value if hasattr(pred_cls, "value") else str(pred_cls)
        av = actual_cls.value if hasattr(actual_cls, "value") else str(actual_cls)
        if pv not in confusion or av not in confusion[av]:
            continue
        confusion[av][pv] += 1
        n_total += 1
        if pv == av:
            n_match += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for cls in ALL_CLASSES:
        row = confusion[cls]
        n_actual = sum(row.values())
        col_sum = sum(confusion[a][cls] for a in ALL_CLASSES)
        tp = row[cls]
        fp = col_sum - tp
        fn = n_actual - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[cls] = {
            "n_actual": n_actual,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    return {
        "source": source,
        "n_total": n_total,
        "n_match": n_match,
        "accuracy": round(n_match / n_total, 3) if n_total else 0.0,
        "confusion_matrix": confusion,
        "per_class": per_class,
        "ground_truth_note": (
            "actual_outcome for real rows is derived from a regex heuristic "
            "over CI logs and is noisy — agreement reflects model vs heuristic, "
            "not necessarily true root cause."
        ),
    }


@router.get("/by_sha/{sha}", response_model=PredictionOut)
def get_prediction_by_sha(sha: str, db: Session = Depends(get_db)) -> PredictionOut:
    """Look up the most recent prediction for a commit SHA.

    Used by CI workflows as a pre-build gate: the workflow polls this
    endpoint and aborts when ``decision == "block"``. Returns 404 if no
    prediction exists yet — caller is expected to poll/retry.
    """
    from app.db.models import Commit

    stmt = (
        select(Prediction)
        .join(Commit, Prediction.commit_id == Commit.id)
        .where(Commit.sha == sha)
        .order_by(Prediction.created_at.desc())
        .limit(1)
    )
    pred = db.scalar(stmt)
    if pred is None:
        raise HTTPException(status_code=404, detail="no prediction for sha")
    return get_prediction(pred.id, db)


@router.post("/{prediction_id}/override", response_model=PredictionOut)
def override_prediction(
    prediction_id: int,
    body: OverrideIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> PredictionOut:
    """Manual override (FR-16). Requires authenticated user; records user_id."""
    pred = db.get(Prediction, prediction_id)
    if pred is None:
        raise HTTPException(status_code=404, detail="prediction not found")
    if pred.decision == body.new_decision:
        raise HTTPException(status_code=400, detail="new_decision matches current")
    pred.decision = body.new_decision
    pred.override_reason = body.reason
    pred.overridden_at = datetime.now(tz=UTC)
    pred.overridden_by_user_id = user.id
    record_audit(
        db, user_id=user.id, action="prediction.override",
        entity_type="prediction", entity_id=pred.id,
        payload={"new_decision": body.new_decision.value, "reason": body.reason},
    )
    db.commit()
    db.refresh(pred)
    # Re-post commit status to GitHub so override unblocks (or re-blocks) the
    # merge gate accordingly. If the new decision is non-BLOCK, also kick off
    # a re-run of any failed workflow runs for the commit so the gate step
    # re-polls and downstream jobs (lint/test/docker-build) actually execute.
    try:
        from app.services.github_actions import rerun_failed_runs_for_sha
        from app.services.github_status import post_commit_status

        full = pred.commit.repository.full_name if pred.commit and pred.commit.repository else ""
        sha = pred.commit.sha if pred.commit else ""
        if full and sha:
            post_commit_status(full, sha, pred.decision, pred.id)
            if pred.decision != PredictionDecision.BLOCK:
                rerun_failed_runs_for_sha(full, sha)
    except Exception:  # noqa: BLE001
        pass
    return get_prediction(prediction_id, db)


@router.get("/{prediction_id}", response_model=PredictionOut)
def get_prediction(prediction_id: int, db: Session = Depends(get_db)) -> PredictionOut:
    pred = db.get(Prediction, prediction_id)
    if pred is None:
        raise HTTPException(status_code=404, detail="prediction not found")
    sha = pred.commit.sha if pred.commit else ""
    recs = [RecommendationOut(**r) for r in (pred.recommendations or [])]
    raw = (pred.commit.raw_metadata if pred.commit else {}) or {}
    workflow_name = raw.get("workflow_name") or None
    workflow_run_url = raw.get("external_run_html_url") or None
    repo_full_name = pred.commit.repository.full_name if pred.commit and pred.commit.repository else ""
    return PredictionOut(
        id=pred.id,
        repository_id=pred.repository_id,
        repository_full_name=repo_full_name,
        commit_sha=sha,
        commit_short=sha[:7],
        author_email=pred.commit.author_email if pred.commit else "",
        branch=pred.commit.branch if pred.commit else None,
        workflow_name=workflow_name,
        workflow_run_url=workflow_run_url,
        predicted_class=pred.predicted_class,
        decision=pred.decision,
        risk_score=pred.risk_score,
        confidence=pred.confidence,
        class_probabilities=pred.class_probabilities,
        feature_importance=pred.feature_importance,
        shap_explanation=pred.shap_explanation,
        predicted_memory_mb=pred.predicted_memory_mb,
        predicted_duration_min=pred.predicted_duration_min,
        recommendations=recs,
        inference_time_ms=pred.inference_time_ms,
        overridden_at=pred.overridden_at,
        actual_outcome=pred.actual_outcome,
        created_at=pred.created_at,
    )
