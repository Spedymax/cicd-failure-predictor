#!/usr/bin/env bash
set -euo pipefail

cd /app/backend

echo "[entrypoint] applying alembic migrations…"
alembic upgrade head

echo "[entrypoint] starting uvicorn on :8000"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips='*'
