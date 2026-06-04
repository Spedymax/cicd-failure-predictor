"""CRUD for decision policies (FR-13)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.audit import record_audit
from app.core.auth import require_roles
from app.db.models import Policy, UserRole
from app.db.session import get_db

router = APIRouter(prefix="/policies", tags=["policies"])

_require_policy_admin = require_roles(UserRole.ADMIN, UserRole.TEAM_LEAD)


class PolicyIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    auto_approve_threshold: float = Field(ge=0.0, le=1.0)
    warn_threshold: float = Field(ge=0.0, le=1.0)
    block_threshold: float = Field(ge=0.0, le=1.0)
    allow_override: bool = True
    specific_rules: dict[str, Any] = Field(default_factory=dict)
    is_default: bool = False

    @model_validator(mode="after")
    def _check_order(self) -> PolicyIn:
        if not (self.auto_approve_threshold <= self.warn_threshold <= self.block_threshold):
            raise ValueError(
                "thresholds must be ordered: auto_approve <= warn <= block"
            )
        return self


class PolicyOut(BaseModel):
    id: int
    name: str
    auto_approve_threshold: float
    warn_threshold: float
    block_threshold: float
    allow_override: bool
    specific_rules: dict[str, Any]
    is_default: bool

    @classmethod
    def from_orm_(cls, p: Policy) -> PolicyOut:
        return cls(
            id=p.id,
            name=p.name,
            auto_approve_threshold=p.auto_approve_threshold,
            warn_threshold=p.warn_threshold,
            block_threshold=p.block_threshold,
            allow_override=p.allow_override,
            specific_rules=p.specific_rules or {},
            is_default=p.is_default,
        )


def _unset_other_defaults(db: Session, except_id: int | None = None) -> None:
    others = db.scalars(select(Policy).where(Policy.is_default.is_(True))).all()
    for o in others:
        if except_id is None or o.id != except_id:
            o.is_default = False


@router.get("", response_model=list[PolicyOut])
def list_policies(db: Session = Depends(get_db)) -> list[PolicyOut]:
    rows = db.scalars(select(Policy).order_by(Policy.id)).all()
    return [PolicyOut.from_orm_(p) for p in rows]


@router.post("", response_model=PolicyOut, status_code=201)
def create_policy(
    body: PolicyIn,
    db: Session = Depends(get_db),
    actor=Depends(_require_policy_admin),
) -> PolicyOut:
    if body.is_default:
        _unset_other_defaults(db)
    p = Policy(**body.model_dump())
    db.add(p)
    db.flush()
    record_audit(
        db, user_id=actor.id, action="policy.create", entity_type="policy",
        entity_id=p.id, payload=body.model_dump(),
    )
    db.commit()
    db.refresh(p)
    return PolicyOut.from_orm_(p)


@router.put("/{policy_id}", response_model=PolicyOut)
def update_policy(
    policy_id: int,
    body: PolicyIn,
    db: Session = Depends(get_db),
    actor=Depends(_require_policy_admin),
) -> PolicyOut:
    p = db.get(Policy, policy_id)
    if p is None:
        raise HTTPException(status_code=404, detail="policy not found")
    if body.is_default:
        _unset_other_defaults(db, except_id=policy_id)
    for k, v in body.model_dump().items():
        setattr(p, k, v)
    record_audit(
        db, user_id=actor.id, action="policy.update", entity_type="policy",
        entity_id=p.id, payload=body.model_dump(),
    )
    db.commit()
    db.refresh(p)
    return PolicyOut.from_orm_(p)


@router.delete("/{policy_id}", status_code=204)
def delete_policy(
    policy_id: int,
    db: Session = Depends(get_db),
    actor=Depends(_require_policy_admin),
) -> None:
    p = db.get(Policy, policy_id)
    if p is None:
        raise HTTPException(status_code=404, detail="policy not found")
    if p.is_default:
        raise HTTPException(status_code=400, detail="cannot delete the default policy")
    record_audit(
        db, user_id=actor.id, action="policy.delete", entity_type="policy",
        entity_id=p.id, payload={"name": p.name},
    )
    db.delete(p)
    db.commit()
