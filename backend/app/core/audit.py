"""Audit log helpers (FR-15, NFR-11).

Writes structured records to the ``audit_logs`` table for sensitive
operations (login, override, policy CRUD) and emits a parallel
structlog event with email masking so the same trail is queryable in
both PostgreSQL and stdout log streams.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.db.models import AuditLog

_log = get_logger("audit")


def record_audit(
    db: Session,
    *,
    user_id: int | None,
    action: str,
    entity_type: str,
    entity_id: str | int | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    entry = AuditLog(
        user_id=user_id,
        action=action,
        entity_type=entity_type,
        entity_id=str(entity_id) if entity_id is not None else None,
        payload=payload or {},
        created_at=datetime.now(tz=UTC),
    )
    db.add(entry)
    # Caller owns the commit so audit and business write share the same tx.
    _log.info(
        "audit",
        action=action,
        entity_type=entity_type,
        entity_id=entry.entity_id,
        user_id=user_id,
        **(payload or {}),
    )
