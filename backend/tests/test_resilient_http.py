"""Unit tests for the resilient HTTP layer (retry + circuit breaker, NFR-05)."""

from __future__ import annotations

import httpx
import pytest

from app.core import resilient_http as rh
from app.core.resilient_http import (
    CircuitBreaker,
    CircuitOpenError,
    call_with_resilience,
    get_breaker,
)


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    # Remove backoff sleeps so retry tests run instantly.
    monkeypatch.setattr(rh, "RETRY_WAIT_MULTIPLIER", 0.0)
    monkeypatch.setattr(rh, "RETRY_WAIT_MAX", 0.0)
    rh.reset_breakers()
    yield
    rh.reset_breakers()


def test_success_passes_through_and_keeps_breaker_closed() -> None:
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return "ok"

    assert call_with_resilience("svc", fn) == "ok"
    assert calls["n"] == 1
    assert get_breaker("svc").state == "closed"


def test_retries_transient_then_succeeds() -> None:
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ConnectError("boom")
        return "recovered"

    assert call_with_resilience("svc", flaky) == "recovered"
    assert attempts["n"] == 3  # failed twice, succeeded on the 3rd


def test_exhausts_retries_and_reraises() -> None:
    attempts = {"n": 0}

    def always_fail():
        attempts["n"] += 1
        raise httpx.ConnectTimeout("down")

    with pytest.raises(httpx.ConnectTimeout):
        call_with_resilience("svc", always_fail)
    assert attempts["n"] == rh.RETRY_ATTEMPTS  # exactly 3 attempts


def test_http_error_response_is_not_retried() -> None:
    attempts = {"n": 0}

    def server_error():
        attempts["n"] += 1
        return httpx.Response(500)

    resp = call_with_resilience("svc", server_error)
    assert resp.status_code == 500
    assert attempts["n"] == 1  # 5xx response returned, not retried


def test_breaker_opens_after_fail_max_and_short_circuits() -> None:
    breaker = CircuitBreaker("trip", fail_max=3, reset_timeout=60.0)

    for _ in range(3):
        breaker.record_failure()
    assert breaker.state == "open"
    assert breaker.allow() is False


def test_breaker_half_opens_after_timeout_and_recovers() -> None:
    breaker = CircuitBreaker("recover", fail_max=1, reset_timeout=0.0)
    breaker.record_failure()
    # reset_timeout=0 -> immediately half-open (a trial is allowed)
    assert breaker.state == "half_open"
    assert breaker.allow() is True
    breaker.record_success()
    assert breaker.state == "closed"


def test_open_breaker_raises_circuit_open_error() -> None:
    breaker = get_breaker("svc")
    breaker.fail_max = 1
    breaker.reset_timeout = 60.0
    breaker.record_failure()  # opens it

    def fn():
        raise AssertionError("should not be called when circuit is open")

    with pytest.raises(CircuitOpenError):
        call_with_resilience("svc", fn)
