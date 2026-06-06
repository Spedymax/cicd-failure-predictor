"""Tests for email masking processor (Day 10) and security headers (Day 12)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.logging_config import _mask_value, mask_email
from app.main import app


def test_mask_email_simple() -> None:
    assert mask_email("mso@anthillagency.com") == "m***@anthillagency.com"


def test_mask_email_in_sentence() -> None:
    s = "user mso@example.com logged in from admin@x.io"
    out = mask_email(s)
    assert "mso@example.com" not in out
    assert "admin@x.io" not in out
    assert "m***@example.com" in out
    assert "a***@x.io" in out


def test_mask_value_nested_dict_and_list() -> None:
    src = {"users": [{"email": "alice@example.com"}], "note": "ping bob@x.io"}
    masked = _mask_value(src)
    assert masked["users"][0]["email"] == "a***@example.com"
    assert "bob@x.io" not in masked["note"]


def test_security_headers_present_on_health() -> None:
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    h = r.headers
    assert h.get("x-content-type-options") == "nosniff"
    assert h.get("x-frame-options") == "DENY"
    assert h.get("referrer-policy") == "no-referrer"
    assert "content-security-policy" in {k.lower() for k in h}
    assert "permissions-policy" in {k.lower() for k in h}
