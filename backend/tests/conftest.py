from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "data" / "artifacts" / "v26_5class"

# Force test env BEFORE any app.* module is imported. Has to run at module
# top-level so it precedes pytest's collection-time imports — otherwise the
# real backend/.env (loaded by pydantic-settings via lru_cache) overrides
# our test values and signatures, JWT keys, etc. diverge.
os.environ["APP_SECRET_KEY"] = "x" * 32
os.environ["GITHUB_WEBHOOK_SECRET"] = "test-webhook-secret"
os.environ["GITHUB_API_TOKEN"] = "test-token"
os.environ["ML_MODEL_DIR"] = str(ARTIFACTS)
# Self-contained DB: in-memory SQLite (no external PostgreSQL/Docker needed).
os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"

# Clear cached settings in case anything imported it before this point.
try:
    from app.core.config import get_settings  # noqa: E402

    get_settings.cache_clear()
except Exception:
    pass


# Create the schema on the in-memory engine and seed deterministic users so the
# auth/RBAC integration tests resolve real rows (admin@example.com / dev@…).
def _bootstrap_db() -> None:
    from app.core.auth import hash_password
    from app.db import models  # noqa: F401  (register all tables)
    from app.db.base import Base
    from app.db.models import User, UserRole
    from app.db.session import SessionLocal, engine

    Base.metadata.create_all(engine)

    with SessionLocal() as session:
        existing = {u.email for u in session.query(User).all()}
        seed = [
            ("admin@example.com", "Admin", UserRole.ADMIN, "admin12345"),
            ("dev@example.com", "Dev", UserRole.DEVELOPER, "dev12345"),
        ]
        for email, name, role, pw in seed:
            if email not in existing:
                session.add(
                    User(
                        email=email,
                        name=name,
                        role=role,
                        password_hash=hash_password(pw),
                        is_active=True,
                    )
                )
        session.commit()


_bootstrap_db()


@pytest.fixture(autouse=True)
def _reset_circuit_breakers():
    """Isolate circuit-breaker state between tests so a failure in one test
    cannot open the shared 'github' breaker and trip the next."""
    from app.core.resilient_http import reset_breakers

    reset_breakers()
    yield
    reset_breakers()


def pytest_sessionstart(session):
    """Wipe webhook idempotency keys so hardcoded delivery_ids in tests don't
    collide with stale entries from a previous run. Redis is optional — the
    suite runs fine without it (claim_delivery degrades gracefully).
    """
    try:
        from app.core.redis_client import get_redis  # noqa: E402

        r = get_redis()
        for k in r.scan_iter("webhook:delivery:*"):
            r.delete(k)
    except Exception:
        pass
