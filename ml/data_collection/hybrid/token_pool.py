"""Token pool with pluggable token sources + request counter.

Each source produces a bearer token on demand. Two source kinds:

* StaticToken  — a fixed PAT.
* GitHubAppToken — derives a 1-hour installation token from a JWT signed
  with the App's private key, refreshing 5 minutes before expiry.

The pool round-robins across sources, prefers the one with the most
remaining rate-limit budget, and logs a status line every N requests so
you can see when the limit is approaching.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


# ----------------------------- sources -----------------------------------

class TokenSource(abc.ABC):
    """Anything that can produce a current GitHub bearer token."""

    @abc.abstractmethod
    async def get(self) -> str: ...

    @property
    @abc.abstractmethod
    def label(self) -> str: ...


class StaticToken(TokenSource):
    def __init__(self, token: str, label: str = "pat") -> None:
        self._token = token
        self._label = label

    async def get(self) -> str:
        return self._token

    @property
    def label(self) -> str:
        return self._label


class GitHubAppToken(TokenSource):
    """Installation access token from a GitHub App.

    Caches the token in memory; refreshes ~5 minutes before expiry. Token
    has its OWN 5000 req/hr quota, independent from any PAT.
    """

    def __init__(
        self,
        app_id: str,
        installation_id: str,
        private_key: str,
        *,
        label: str = "app",
    ) -> None:
        self._app_id = str(app_id)
        self._installation_id = str(installation_id)
        self._pk = private_key
        self._label = label
        self._cached: str | None = None
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()

    @classmethod
    def from_pem_file(
        cls,
        app_id: str,
        installation_id: str,
        pem_path: str | Path,
        *,
        label: str = "app",
    ) -> "GitHubAppToken":
        return cls(
            app_id=app_id,
            installation_id=installation_id,
            private_key=Path(pem_path).read_text(),
            label=label,
        )

    @property
    def label(self) -> str:
        return self._label

    async def get(self) -> str:
        async with self._lock:
            now = time.time()
            if self._cached and now < self._expires_at - 300:
                return self._cached
            return await self._refresh()

    async def _refresh(self) -> str:
        import jwt  # PyJWT[crypto]

        now = int(time.time())
        jwt_payload = {"iat": now - 60, "exp": now + 540, "iss": self._app_id}
        signed = jwt.encode(jwt_payload, self._pk, algorithm="RS256")

        url = (
            f"https://api.github.com/app/installations/"
            f"{self._installation_id}/access_tokens"
        )
        async with httpx.AsyncClient(timeout=15.0) as c:
            resp = await c.post(
                url,
                headers={
                    "Authorization": f"Bearer {signed}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        self._cached = data["token"]
        exp_iso = data["expires_at"].replace("Z", "+00:00")
        self._expires_at = datetime.fromisoformat(exp_iso).timestamp()
        logger.info(
            "%s: refreshed installation token, expires %s",
            self._label, data["expires_at"],
        )
        return self._cached


# ----------------------------- pool --------------------------------------

@dataclass
class TokenState:
    source: TokenSource
    remaining: int = 5000
    reset_at: float = 0.0
    in_flight: int = 0
    requests_made: int = 0

    @property
    def display(self) -> str:
        return self.source.label


class TokenPool:
    """Round-robin pool with per-source rate-limit tracking + request counter."""

    def __init__(self, sources: list[TokenSource], *, log_every: int = 1000) -> None:
        if not sources:
            raise ValueError("TokenPool requires at least one source")
        self._states: list[TokenState] = [TokenState(source=s) for s in sources]
        self._lock = asyncio.Lock()
        self._total_requests = 0
        self._log_every = max(1, log_every)

    @classmethod
    def from_strings(cls, tokens: list[str]) -> "TokenPool":
        sources = [StaticToken(t, label=f"pat{i}") for i, t in enumerate(tokens)]
        return cls(sources)

    def __len__(self) -> int:
        return len(self._states)

    @property
    def total_requests(self) -> int:
        return self._total_requests

    async def acquire(self) -> TokenState:
        while True:
            async with self._lock:
                # Optimistic auto-refill: if a source's reset window has
                # passed, GitHub has already restored its quota even though
                # our local `remaining` is stale (we only learn the true
                # number from response headers). Bump it back to 5000 so
                # we'll actually try a request -- the next response refreshes
                # the real value via release().
                now = time.time()
                for s in self._states:
                    if s.remaining <= 0 and s.reset_at and s.reset_at <= now:
                        logger.info(
                            "%s: reset window passed, optimistically refilling 0 -> 5000",
                            s.display,
                        )
                        s.remaining = 5000

                ready = [s for s in self._states if s.remaining > 0]
                if ready:
                    ready.sort(key=lambda s: (-s.remaining, s.in_flight))
                    chosen = ready[0]
                    chosen.in_flight += 1
                    return chosen
                soonest = min(self._states, key=lambda s: s.reset_at or (now + 60))
                wait = max(1.0, soonest.reset_at - now if soonest.reset_at else 60.0)
            logger.warning(
                "all %d sources exhausted; sleeping %.0fs until reset",
                len(self._states), wait,
            )
            await asyncio.sleep(min(wait, 60.0))

    async def release(
        self,
        state: TokenState,
        *,
        remaining: int | None,
        reset_at: float | None,
        counted: bool = True,
    ) -> None:
        async with self._lock:
            state.in_flight = max(0, state.in_flight - 1)
            if remaining is not None:
                state.remaining = remaining
            if reset_at is not None:
                state.reset_at = reset_at
            if counted:
                state.requests_made += 1
                self._total_requests += 1
                if self._total_requests % self._log_every == 0:
                    self._log_status_locked()

    def _log_status_locked(self) -> None:
        now = time.time()
        parts = []
        for s in self._states:
            reset_in = max(0, int(s.reset_at - now)) if s.reset_at else 0
            mins = reset_in // 60
            parts.append(
                f"{s.display}: {s.remaining}/5000 (-{s.requests_made}, reset {mins}m)"
            )
        logger.info("📊 %d requests done | %s", self._total_requests, " | ".join(parts))

    def snapshot(self) -> list[dict[str, float | int | str]]:
        now = time.time()
        return [
            {
                "label": s.display,
                "remaining": s.remaining,
                "reset_in_s": max(0, int(s.reset_at - now)) if s.reset_at else 0,
                "in_flight": s.in_flight,
                "requests_made": s.requests_made,
            }
            for s in self._states
        ]
