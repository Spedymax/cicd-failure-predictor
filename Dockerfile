# syntax=docker/dockerfile:1.7

# ------- Frontend build stage -------
FROM node:20-slim AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json* /fe/
RUN npm install --no-audit --no-fund
COPY frontend /fe
RUN npm run build

# ------- Backend runtime stage -------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/backend/.venv

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# 1) deps layer (changes rarely)
COPY backend/pyproject.toml backend/uv.lock /app/backend/
RUN cd /app/backend && uv sync --frozen --no-dev

# 2) app code
COPY backend /app/backend

# 3) active ML artifacts (only v26_5class — keeps image small)
COPY data/artifacts/v26_5class /app/data/artifacts/v26_5class

# 4) training code + features parquet — needed by POST /admin/retrain
COPY ml /app/ml
COPY data/processed/features_real_only.parquet /app/data/processed/features_real_only.parquet

# 5) built frontend (served as static from FastAPI)
COPY --from=frontend /fe/dist /app/static

# 4) entrypoint
COPY infra/fly-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV ML_MODEL_DIR=/app/data/artifacts/v26_5class \
    PATH="/app/backend/.venv/bin:$PATH"

EXPOSE 8000
CMD ["/entrypoint.sh"]
