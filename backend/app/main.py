from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import FileResponse, Response

from app.api.routes import api_router
from app.core.config import get_settings
from app.core.logging_config import configure_logging

settings = get_settings()
configure_logging(settings.app_log_level)

app = FastAPI(
    title="CI/CD Failure Predictor",
    version="0.1.0",
    description="Превентивне прогнозування збоїв у CI/CD-конвеєрах",
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds defence-in-depth HTTP security headers (NFR-11)."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy", "geolocation=(), microphone=(), camera=()"
        )
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline';"
            " script-src 'self'; connect-src 'self'; frame-ancestors 'none'",
        )
        if settings.app_env != "development":
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response


app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


def _health_report() -> dict[str, object]:
    """Readiness probe (NFR-04): reports liveness plus component health.

    Redis is non-critical — webhook idempotency degrades gracefully when it is
    down — so only the database and the loaded model gate the ``ok`` status.
    """
    db_ok = False
    try:
        from sqlalchemy import text

        from app.db.session import SessionLocal

        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    redis_ok = False
    try:
        from app.core.redis_client import get_redis

        redis_ok = bool(get_redis().ping())
    except Exception:
        redis_ok = False

    model_loaded = False
    try:
        from app.api.dependencies import get_inference_engine

        model_loaded = bool(getattr(get_inference_engine(), "feature_names", None))
    except Exception:
        model_loaded = False

    status = "ok" if (db_ok and model_loaded) else "degraded"
    return {
        "status": status,
        "env": settings.app_env,
        "db_ok": db_ok,
        "redis_ok": redis_ok,
        "model_loaded": model_loaded,
    }


@app.get("/health")
def health() -> dict[str, object]:
    return _health_report()


@app.get("/api/v1/health")
def health_v1() -> dict[str, object]:
    return _health_report()


# ---------- Static SPA (built frontend) ----------
# In production the Docker build copies the Vite output to /app/static. We
# mount /assets/ directly so hashed bundles are served verbatim, and add a
# catch-all that returns index.html for client-side router paths
# (/predictions/123, /login, …). When the static dir is absent (local dev),
# both routes silently skip — the dev frontend is served by Vite on :3000.
_STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
if not _STATIC_DIR.exists():
    _STATIC_DIR = Path("/app/static")

if _STATIC_DIR.is_dir():
    _ASSETS_DIR = _STATIC_DIR / "assets"
    if _ASSETS_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="assets")

    _INDEX = _STATIC_DIR / "index.html"

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str) -> FileResponse:
        # Try a real file in /static first (favicon, robots.txt, etc.)
        candidate = _STATIC_DIR / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_INDEX)
