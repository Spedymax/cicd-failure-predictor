from __future__ import annotations

import json
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_inference_engine
from app.core.config import get_settings
from app.core.redis_client import claim_delivery
from app.core.security import SIGNATURE_HEADER, verify_signature
from app.db.session import SessionLocal, get_db
from app.ml.inference import InferenceEngine
from app.schemas.prediction import WebhookAck
from app.services.build_pipeline import process_workflow_run_event
from app.services.prediction_pipeline import process_push_event

logger = logging.getLogger(__name__)

router = APIRouter(tags=["webhooks"])


def _process_in_background(payload: dict, engine: InferenceEngine) -> None:
    db: Session = SessionLocal()
    try:
        process_push_event(db, engine, payload)
    except Exception:
        logger.exception("background prediction failed")
        db.rollback()
    finally:
        db.close()


def _process_workflow_run_in_background(payload: dict) -> None:
    db: Session = SessionLocal()
    try:
        result = process_workflow_run_event(db, payload)
        logger.info("workflow_run processed: %s", result)
    except Exception:
        logger.exception("workflow_run processing failed")
        db.rollback()
    finally:
        db.close()


@router.post("/webhook/github", response_model=WebhookAck, status_code=status.HTTP_202_ACCEPTED)
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_github_event: str | None = Header(default=None),
    x_github_delivery: str | None = Header(default=None),
    x_hub_signature_256: str | None = Header(default=None, alias=SIGNATURE_HEADER),
    engine: InferenceEngine = Depends(get_inference_engine),
    _: Session = Depends(get_db),
) -> WebhookAck:
    settings = get_settings()
    body = await request.body()
    if not verify_signature(settings.github_webhook_secret, body, x_hub_signature_256):
        raise HTTPException(status_code=403, detail="invalid webhook signature")

    if x_github_event == "ping":
        return WebhookAck(accepted=True, delivery_id=x_github_delivery)

    if x_github_event not in ("push", "workflow_run"):
        return WebhookAck(accepted=False, delivery_id=x_github_delivery)

    # Idempotency: drop duplicate deliveries with the same X-GitHub-Delivery
    # within an hour. claim_delivery returns False if already seen.
    if x_github_delivery and not claim_delivery(x_github_delivery):
        logger.info("duplicate delivery_id=%s ignored", x_github_delivery)
        return WebhookAck(accepted=False, delivery_id=x_github_delivery)

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="payload is not valid JSON") from exc

    payload["_delivery_id"] = x_github_delivery
    repo = (payload.get("repository") or {}).get("full_name")

    if x_github_event == "workflow_run":
        background_tasks.add_task(_process_workflow_run_in_background, payload)
        sha = (payload.get("workflow_run") or {}).get("head_sha")
        return WebhookAck(
            accepted=True, delivery_id=x_github_delivery,
            repository=repo, commit_sha=sha,
        )

    sha = (payload.get("head_commit") or {}).get("id") or payload.get("after")
    background_tasks.add_task(_process_in_background, payload, engine)
    return WebhookAck(
        accepted=True,
        delivery_id=x_github_delivery,
        repository=repo,
        commit_sha=sha,
    )
