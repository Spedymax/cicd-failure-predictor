"""JWT authentication endpoints (FR-09, NFR-09)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from app.core.audit import record_audit
from app.core.auth import (
    JWT_EXPIRE_HOURS,
    create_access_token,
    get_current_user,
    hash_password,
    require_roles,
    verify_password,
)
from app.db.models import User, UserRole
from app.db.session import get_db

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    name: str | None = Field(default=None, max_length=255)
    role: UserRole = UserRole.DEVELOPER


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = JWT_EXPIRE_HOURS * 3600


class UserOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    email: str
    name: str | None
    role: UserRole
    is_active: bool


@router.post("/register", response_model=UserOut, status_code=201)
def register(
    body: RegisterIn,
    db: Annotated[Session, Depends(get_db)],
    # Bootstrap: if no users exist, allow open registration (creates first admin).
    # Otherwise require admin token.
) -> User:
    has_users = db.query(User).first() is not None
    if has_users:
        # Lazy import to avoid circular dep; enforce admin role for subsequent registrations.
        from fastapi import Request  # noqa: F401

        raise HTTPException(
            status_code=403,
            detail="registration closed; use POST /auth/admin/users (admin only)",
        )
    user = User(
        email=body.email,
        name=body.name,
        role=UserRole.ADMIN,  # First user is admin by bootstrap.
        password_hash=hash_password(body.password),
        is_active=True,
    )
    db.add(user)
    db.flush()
    record_audit(
        db,
        user_id=user.id,
        action="user.register_bootstrap",
        entity_type="user",
        entity_id=user.id,
        payload={"email": user.email, "role": user.role.value},
    )
    db.commit()
    db.refresh(user)
    return user


@router.post("/admin/users", response_model=UserOut, status_code=201)
def admin_create_user(
    body: RegisterIn,
    db: Annotated[Session, Depends(get_db)],
    _admin: Annotated[User, Depends(require_roles(UserRole.ADMIN))],
) -> User:
    if db.query(User).filter(User.email == body.email).first() is not None:
        raise HTTPException(status_code=409, detail="email already registered")
    user = User(
        email=body.email,
        name=body.name,
        role=body.role,
        password_hash=hash_password(body.password),
        is_active=True,
    )
    db.add(user)
    db.flush()
    record_audit(
        db,
        user_id=_admin.id,
        action="user.create",
        entity_type="user",
        entity_id=user.id,
        payload={"email": user.email, "role": user.role.value},
    )
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenOut)
def login(body: LoginIn, db: Annotated[Session, Depends(get_db)]) -> TokenOut:
    user = db.query(User).filter(User.email == body.email).first()
    if user is None or not user.password_hash or not user.is_active:
        record_audit(
            db,
            user_id=None,
            action="user.login_failed",
            entity_type="user",
            entity_id=None,
            payload={"email": body.email},
        )
        db.commit()
        raise HTTPException(status_code=401, detail="invalid credentials")
    if not verify_password(body.password, user.password_hash):
        record_audit(
            db,
            user_id=user.id,
            action="user.login_failed",
            entity_type="user",
            entity_id=user.id,
            payload={"email": user.email},
        )
        db.commit()
        raise HTTPException(status_code=401, detail="invalid credentials")
    user.last_login_at = datetime.now(tz=UTC)
    record_audit(
        db,
        user_id=user.id,
        action="user.login",
        entity_type="user",
        entity_id=user.id,
        payload={"email": user.email},
    )
    db.commit()
    return TokenOut(access_token=create_access_token(user.email, user.role.value))


@router.get("/me", response_model=UserOut)
def me(user: Annotated[User, Depends(get_current_user)]) -> User:
    return user
