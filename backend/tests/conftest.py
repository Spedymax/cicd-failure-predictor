from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "data" / "artifacts" / "v2"

# Force test env BEFORE any app.* module is imported. Has to run at module
# top-level so it precedes pytest's collection-time imports — otherwise the
# real backend/.env (loaded by pydantic-settings via lru_cache) overrides
# our test values and signatures, JWT keys, etc. diverge.
os.environ["APP_SECRET_KEY"] = "x" * 32
os.environ["POSTGRES_USER"] = "cicd_predictor"
os.environ["POSTGRES_PASSWORD"] = "changeme"
os.environ["POSTGRES_DB"] = "cicd_predictor"
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ["GITHUB_WEBHOOK_SECRET"] = "test-webhook-secret"
os.environ["GITHUB_API_TOKEN"] = "test-token"
os.environ["ML_MODEL_DIR"] = str(ARTIFACTS)

# Clear cached settings in case anything imported it before this point.
try:
    from app.core.config import get_settings  # noqa: E402

    get_settings.cache_clear()
except Exception:
    pass


def pytest_sessionstart(session):
    """Wipe webhook idempotency keys so hardcoded delivery_ids in tests don't
    collide with stale entries from a previous run.
    """
    try:
        from app.core.redis_client import get_redis  # noqa: E402

        r = get_redis()
        for k in r.scan_iter("webhook:delivery:*"):
            r.delete(k)
    except Exception:
        pass
