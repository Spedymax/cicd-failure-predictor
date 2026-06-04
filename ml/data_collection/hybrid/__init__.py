"""Hybrid GH Archive (BigQuery) + REST data collection pipeline.

Stage 1: pull workflow-run metadata for the seed repo list from
`githubarchive.day.*` via BigQuery -- no REST calls, no rate limit.

Stage 2: REST-enrich a balanced subset (all failures + sampled successes) with
commit diffs and (for failures) job logs to derive failure_class. Uses a
token pool with per-token rate-limit tracking to scale beyond 5000 req/hr.
"""
