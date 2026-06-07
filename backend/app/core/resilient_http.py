"""Resilient outbound HTTP for third-party APIs (NFR-05).

Wraps an HTTP call with:

* **Retry** — up to 3 attempts with exponential backoff, but only for
  *transient* transport/timeout errors. HTTP error *responses* (4xx/5xx) are
  returned to the caller unchanged — retrying a non-idempotent POST on a 5xx
  could double-trigger side effects (e.g. re-runs), so that is deliberately
  avoided.
* **Circuit breaker** — an in-process breaker per logical target (e.g.
  ``"github"``). After ``fail_max`` consecutive transport failures it *opens*
  and short-circuits further calls with :class:`CircuitOpenError` for
  ``reset_timeout`` seconds, then allows a single trial (half-open). A success
  closes it again. This stops a hammering a dead dependency.

Usage::

    def _send():
        with httpx.Client(timeout=8.0) as c:
            return c.post(url, headers=headers, json=body)

    resp = call_with_resilience("github", _send)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

import httpx
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Retry configuration (module-level so tests can shrink the backoff).
RETRY_ATTEMPTS = 3
RETRY_WAIT_MULTIPLIER = 0.3
RETRY_WAIT_MAX = 4.0

# Exceptions worth retrying — transient connectivity problems only.
_RETRYABLE_EXC = (httpx.TransportError, httpx.TimeoutException)


class CircuitOpenError(RuntimeError):
    """Raised when a call is short-circuited because the breaker is open."""


class CircuitBreaker:
    """Minimal in-process circuit breaker (CLOSED → OPEN → HALF_OPEN)."""

    def __init__(self, name: str, fail_max: int = 5, reset_timeout: float = 30.0) -> None:
        self.name = name
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if (time.monotonic() - self._opened_at) >= self.reset_timeout:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        """Whether a call may proceed right now."""
        return self.state != "open"

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.fail_max:
            if self._opened_at is None:
                logger.warning(
                    "circuit breaker '%s' opened after %d failures", self.name, self._failures
                )
            self._opened_at = time.monotonic()

    def reset(self) -> None:
        self._failures = 0
        self._opened_at = None


_breakers: dict[str, CircuitBreaker] = {}


def get_breaker(name: str) -> CircuitBreaker:
    breaker = _breakers.get(name)
    if breaker is None:
        breaker = CircuitBreaker(name)
        _breakers[name] = breaker
    return breaker


def reset_breakers() -> None:
    """Reset all breakers (used between tests)."""
    for breaker in _breakers.values():
        breaker.reset()


def call_with_resilience(name: str, fn: Callable[[], T]) -> T:
    """Run ``fn`` with retry + circuit breaker keyed by ``name``.

    Raises :class:`CircuitOpenError` immediately if the breaker is open, or the
    last transport exception after exhausting retries.
    """
    breaker = get_breaker(name)
    if not breaker.allow():
        raise CircuitOpenError(f"circuit '{name}' is open")

    retryer: Retrying = Retrying(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=RETRY_WAIT_MULTIPLIER, max=RETRY_WAIT_MAX),
        retry=retry_if_exception_type(_RETRYABLE_EXC),
        reraise=True,
    )
    try:
        result = retryer(fn)
    except _RETRYABLE_EXC:
        breaker.record_failure()
        raise
    breaker.record_success()
    return result
