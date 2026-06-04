# CI/CD Failure Predictor

Дипломний проєкт «Превентивне прогнозування збоїв у CI/CD-конвеєрах через
ML на основі метаданих Git-комітів». Сервіс приймає GitHub webhooks
(`push`, `workflow_run`), будує ознаки коміту, виконує двостадійний
ML-інференс (P(failure) → P(class | failure)) і повертає рішення
**AUTO_APPROVE / WARN / BLOCK** разом із прогнозом ресурсів та
рекомендаціями.

Платформа CI/CD — **GitHub Actions** (Buildkite зарезервовано).

## Архітектура (коротко)

- **Backend (FastAPI)** — webhook receiver, авторизація (JWT/RBAC), CRUD
  політик і репозиторіїв, override, аналітика, аудит, retrain endpoint.
  Asynchrony через FastAPI `BackgroundTasks` + Redis (кеш агрегатів +
  idempotency).
- **ML inference engine** — двостадійна модель (`risk_rf+risk_lgb` →
  `cause_rf+cause_lgb`), активна версія `data/artifacts/v26_5class`.
  `safe_predict()` дає degraded fallback на будь-яке виключення.
- **Hybrid decision policy** — rule overlay над ML-вихідом (форма коміту:
  files/lines/Dockerfile/deps), пороги налаштовуються через `/policies`.
- **Frontend (React 18 + Vite + Tailwind)** — Login, Predictions list,
  Detail (SHAP + Override), Analytics (Chart.js), Policies CRUD,
  Repositories.
- **Storage** — PostgreSQL 15 (Alembic), Redis 7.

## Структура

```
backend/    FastAPI service, моделі, API, Alembic-міграції, pytest
frontend/   React 18 SPA, axios+JWT interceptor, AuthProvider
ml/         data collection (hybrid, 15k req/h), feature engineering,
            training (Optuna), regressors
data/       artifacts/v26_5class/ — активна prod-модель (у репозиторії);
            сирі та оброблені датасети — gitignored (відтворюються pipeline'ом)
scripts/    locustfile.py, seed_demo_predictions.py, etc.
.github/    workflows/retrain.yml (cron weekly + workflow_dispatch)
```

## Швидкий старт (dev)

```bash
cp .env.example .env  # відредагувати POSTGRES_*, GITHUB_*, APP_SECRET_KEY
docker compose up -d postgres redis

# Backend
cd backend
uv sync --extra dev
uv run alembic upgrade head
uv run uvicorn app.main:app --reload  # http://127.0.0.1:8000

# Frontend (інший термінал)
cd frontend
npm install
npm run dev  # http://127.0.0.1:3000
```

Перший вхід — реєстрація bootstrap-адміна:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"email":"admin@example.com","password":"changeme123","name":"Admin"}'
```

Далі — UI на `http://127.0.0.1:3000`, логін тим самим email/паролем.

## Тренування моделі

```bash
cd ml
# Класифікатор з Optuna tuning
uv run python -m training.train \
  --features ../data/processed/features_real_only.parquet \
  --out ../data/artifacts/vN --tune --n-trials 60
```

Дані можна оновити через гібридний pipeline (REST API + token pool,
~15 000 req/год):

```bash
cd ml
uv run python -m data_collection.hybrid.collect_hybrid \
  --repos data_collection/repos.txt --max-per-repo 100 \
  --out ../data/raw/gha_runs_hybrid.parquet
```

## Тести

```bash
cd backend && uv run pytest --cov=app --cov-report=term
```

25 тестів, поточне покриття 63 % (auth/JWT, bcrypt, RBAC, structlog
masking, security headers, webhook e2e, inference graceful degradation,
recommendation engine).

## Benchmark

```bash
locust -f scripts/locustfile.py --host http://127.0.0.1:8000 \
  --users 20 --spawn-rate 5 --headless -t 1m
```

Сценарії: підписані webhook'и + список прогнозів. Дає p50/p95 для NFR-02
(throughput) і NFR-03 (<10 s latency).

## Основні endpoints

- `POST /api/v1/auth/register` — bootstrap першого admin (далі заборонено)
- `POST /api/v1/auth/login` — JWT (HS256, 24 год)
- `POST /api/v1/auth/admin/users` — admin-only створення нових юзерів
- `POST /api/v1/webhook/github` — GitHub webhook (HMAC-SHA256, idempotency)
- `GET  /api/v1/predictions` — список з фільтрами + пагінація
- `POST /api/v1/predictions/{id}/override` — override (auth required)
- `GET  /api/v1/policies` / `POST /PUT /DELETE` — політики (admin/team_lead)
- `GET  /api/v1/repositories` / `POST` — реєстрація репо + GH discovery
- `GET  /api/v1/stats/trends?days=30` — analytics dashboard
- `POST /api/v1/admin/retrain` — admin-only, запускає тренування у
  `BackgroundTasks`, повертає `run_id`
- `GET  /api/v1/admin/retrain/{run_id}` — статус + tail логу

Повна Swagger-документація — `http://127.0.0.1:8000/docs`.

## Демо-сценарій (для захисту)

Окремий публічний репо `Spedymax/cicd-predictor-demo` з трьома комітами
у різних гілках:

| Гілка        | Зміни                    | Decision       |
|--------------|--------------------------|----------------|
| `demo-auto`  | README typo, 1 рядок     | AUTO_APPROVE   |
| `demo-warn`  | оновлення залежності     | WARN           |
| `demo-block` | переписаний Dockerfile   | BLOCK          |

Tunnel: `cloudflared tunnel run cicd-predictor` →
`cicd-predictor.spedymax.org` (named tunnel, постійний URL у GitHub
webhook config).

## Технологічний стек

- **Backend:** Python 3.11, FastAPI 0.115+, SQLAlchemy 2, Alembic,
  psycopg3, passlib[bcrypt], python-jose, structlog
- **ML:** scikit-learn 1.5+, LightGBM 4.5+, Optuna 4, SHAP 0.46+, joblib,
  pandas 2, pyarrow
- **Frontend:** React 18, Vite, Tailwind CSS, TanStack Query, axios,
  Chart.js, react-router-dom 6
- **Сховища:** PostgreSQL 15, Redis 7
- **Async:** FastAPI BackgroundTasks (без Celery)
- **Deploy:** Docker Compose локально, Fly.io / Railway production
- **Dev:** pytest, pytest-cov, ruff, mypy, pre-commit, locust

## Дипломна робота

- Студент: Соколов, ІП-з21
- Захист: червень 2026
