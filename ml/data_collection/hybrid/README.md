# Hybrid GH Archive + REST collector

Параллельный пайплайн рядом с `collect.py`. Основные отличия:

| | `collect.py` (v1) | `hybrid/collect_hybrid.py` |
|---|---|---|
| List workflow_runs | REST `/actions/runs` (2 calls/repo) | BigQuery `githubarchive.day.*` (1 SQL на все 180 репо) |
| Commit diff | REST `/commits/{sha}` для **всех** runs | REST только для **balanced-выборки** (~10% от v1) |
| Job logs | REST для failures | REST для failures (то же) |
| Rate limit на 1 PAT | ~9 часов на 180 репо | ~30–60 минут (даже на 1 PAT) |
| Многотокенность | ❌ | ✅ `TokenPool` round-robin |

## Setup (one-off)

1. Service-account JSON для BigQuery (виден через `gcloud auth application-default login` или явный файл):
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
   export BQ_PROJECT_ID=your-gcp-project
   ```
2. Пул токенов (от 1 до N):
   ```bash
   export GITHUB_TOKENS=ghp_aaa,ghp_bbb,ghp_ccc
   ```

## Запуск

```bash
cd ml

# Сначала прикинуть стоимость BQ-запроса (НЕ исполняет, бесплатно):
uv run python -m data_collection.hybrid.collect_hybrid \
    --repos data_collection/repos.txt --dry-run-bq

# Полный пайплайн:
uv run python -m data_collection.hybrid.collect_hybrid \
    --repos data_collection/repos.txt \
    --months 6 \
    --max-per-repo 100 \
    --out ../data/raw/gha_runs_hybrid.parquet

# Smoke-тест без REST (только BQ → parquet):
uv run python -m data_collection.hybrid.collect_hybrid \
    --repos data_collection/repos_smoke.txt \
    --no-enrich --months 1 \
    --out ../data/raw/smoke_hybrid.parquet
```

## Что делает каждый файл

- `sql/workflow_runs.sql` — параметризованный запрос к `githubarchive.day.*`.
- `bq_fetcher.py` — запуск запроса, дедуп по `(repo, run_id, run_attempt)`.
- `token_pool.py` — раздаёт токены с учётом `X-RateLimit-Remaining` каждого.
- `rest_enricher.py` — пулирует HTTP-клиент, добивает commit-diff + failure_class.
- `collect_hybrid.py` — orchestrator с CLI.

## Schema parity с v1

После enrichment колонки совместимы с `gha_runs.parquet`:
`repo, run_id, workflow_name, head_sha, head_branch, event, status, conclusion, run_attempt, created_at, updated_at, duration_seconds, actor_login, actor_email, n_jobs, longest_job_seconds, failure_class, failure_evidence, files_changed, lines_added, lines_deleted, has_dockerfile_change, has_dependency_change, file_extensions, test_dir_changes, test_only_changes, has_lint_config_change, event_is_push`.

> ⚠️ `event_is_push` пока не маппится — добавить при следующей итерации (есть в `event`).
