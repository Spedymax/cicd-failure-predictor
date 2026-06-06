"""Analytics endpoint: aggregate predictions for the dashboard."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Date, and_, case, cast, func, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.sql import ColumnElement

from app.api.predictions import DEMO_PREFIXES
from app.db.models import Prediction, PredictionDecision, Repository
from app.db.session import get_db

router = APIRouter(prefix="/stats", tags=["analytics"])


def _source_filter(source: str) -> ColumnElement[bool] | None:
    if source == "demo":
        return or_(*[Repository.full_name.like(f"{p}%") for p in DEMO_PREFIXES])
    if source == "real":
        return and_(*[~Repository.full_name.like(f"{p}%") for p in DEMO_PREFIXES])
    return None


@router.get("/trends")
def trends(
    db: Session = Depends(get_db),
    days: int = Query(30, ge=1, le=365),
    source: str = Query(default="all", pattern="^(all|demo|real)$"),
) -> dict:
    """Aggregate predictions over the last ``days``.

    Returns three series for the dashboard:
      * daily — stacked counts of auto_approve / warn / block per day
      * failure_class — total count by predicted_class
      * top_repos — repos with highest mean risk_score (min 3 predictions)
    """
    cutoff = datetime.now(tz=UTC) - timedelta(days=days)
    src_clause = _source_filter(source)

    day_col = cast(Prediction.created_at, Date).label("date")
    daily_stmt = (
        select(
            day_col,
            func.count(Prediction.id).label("total"),
            func.sum(
                case((Prediction.decision == PredictionDecision.AUTO_APPROVE, 1), else_=0)
            ).label("auto_approve"),
            func.sum(case((Prediction.decision == PredictionDecision.WARN, 1), else_=0)).label(
                "warn"
            ),
            func.sum(case((Prediction.decision == PredictionDecision.BLOCK, 1), else_=0)).label(
                "block"
            ),
        )
        .join(Repository, Prediction.repository_id == Repository.id)
        .where(Prediction.created_at >= cutoff)
        .group_by(day_col)
        .order_by(day_col)
    )
    if src_clause is not None:
        daily_stmt = daily_stmt.where(src_clause)
    daily = [
        {
            "date": r.date.isoformat(),
            "total": int(r.total),
            "auto_approve": int(r.auto_approve or 0),
            "warn": int(r.warn or 0),
            "block": int(r.block or 0),
        }
        for r in db.execute(daily_stmt).all()
    ]

    fc_stmt = (
        select(Prediction.predicted_class, func.count(Prediction.id).label("n"))
        .join(Repository, Prediction.repository_id == Repository.id)
        .where(Prediction.created_at >= cutoff)
        .group_by(Prediction.predicted_class)
    )
    if src_clause is not None:
        fc_stmt = fc_stmt.where(src_clause)
    failure_class = {
        (
            r.predicted_class.value
            if hasattr(r.predicted_class, "value")
            else str(r.predicted_class)
        ): int(r.n)
        for r in db.execute(fc_stmt).all()
    }

    top_stmt = (
        select(
            Repository.full_name,
            func.count(Prediction.id).label("n"),
            func.avg(Prediction.risk_score).label("avg_risk"),
        )
        .join(Prediction, Prediction.repository_id == Repository.id)
        .where(Prediction.created_at >= cutoff)
        .group_by(Repository.full_name)
        .having(func.count(Prediction.id) >= 3)
        .order_by(func.avg(Prediction.risk_score).desc())
        .limit(10)
    )
    if src_clause is not None:
        top_stmt = top_stmt.where(src_clause)
    top_repos = [
        {"repo": r.full_name, "n": int(r.n), "avg_risk": round(float(r.avg_risk), 3)}
        for r in db.execute(top_stmt).all()
    ]

    return {
        "window_days": days,
        "since": cutoff.isoformat(),
        "daily": daily,
        "failure_class": failure_class,
        "top_repos": top_repos,
        "totals": {
            "n_predictions": sum(d["total"] for d in daily),
            "n_block": sum(d["block"] for d in daily),
            "n_warn": sum(d["warn"] for d in daily),
            "n_auto": sum(d["auto_approve"] for d in daily),
        },
    }
