"""Unit tests for JWT auth + bcrypt helpers (Day 3 / NFR-09)."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.core.auth import (
    create_access_token,
    hash_password,
    require_roles,
    verify_password,
)
from app.db.models import User, UserRole


def test_bcrypt_roundtrip() -> None:
    h = hash_password("hunter2-very-strong")
    assert h != "hunter2-very-strong"
    assert verify_password("hunter2-very-strong", h)
    assert not verify_password("wrong", h)


def test_jwt_token_is_signed_and_contains_role() -> None:
    from jose import jwt

    from app.core.config import get_settings

    token = create_access_token("alice@example.com", "admin")
    payload = jwt.decode(token, get_settings().app_secret_key, algorithms=["HS256"])
    assert payload["sub"] == "alice@example.com"
    assert payload["role"] == "admin"
    assert payload["exp"] > payload["iat"]


def test_require_roles_allows_member() -> None:
    user = User(id=1, email="a@b.c", role=UserRole.ADMIN, is_active=True)
    dep = require_roles(UserRole.ADMIN, UserRole.TEAM_LEAD)
    assert dep(user) is user


def test_require_roles_blocks_outsider() -> None:
    user = User(id=2, email="d@e.f", role=UserRole.DEVELOPER, is_active=True)
    dep = require_roles(UserRole.ADMIN)
    with pytest.raises(HTTPException) as exc:
        dep(user)
    assert exc.value.status_code == 403
