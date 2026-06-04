"""End-to-end collector: REST list_runs (via token pool) -> REST enrich -> parquet.

Architecture:
  Stage 1: for each repo, fetch up to N failure + M success workflow_runs via
           REST `?status=failure|success`, distributed across the token pool
           (PATs + GitHub App installation tokens).
  Stage 2: REST-enrich each run with commit diff + (for failures) job logs
           to derive failure_class.

Why a "pool": GitHub rate-limit is per-user, so multiple PATs from one user
share one 5k/hr budget. But a GitHub App installation has its own 5k/hr
budget. Combining your PAT(s) + an App installation token multiplies the
effective rate limit, all from a single human's setup.

Counter: pool logs "📊 N requests done | <per-source breakdown>" every
`--log-every` requests so you can see when a token approaches its limit.

Usage::

    export GITHUB_API_TOKEN=ghp_xxx
    export GITHUB_API_TOKEN2=ghp_yyy            # second account, optional
    export GITHUB_APP_ID=Iv23li...              # client id or numeric app id
    export GITHUB_APP_INSTALLATION_ID=12345678
    export GITHUB_APP_PRIVATE_KEY_FILE=.secrets/github-app.pem

    cd ml
    uv run python -m data_collection.hybrid.collect_hybrid \\
        --repos data_collection/repos.txt \\
        --max-per-repo 100 \\
        --out ../data/raw/gha_runs_hybrid.parquet
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from data_collection.hybrid.rest_enricher import (
    enrich_dataframe,
    list_runs_for_repos,
)
from data_collection.hybrid.token_pool import (
    GitHubAppToken,
    StaticToken,
    TokenPool,
    TokenSource,
)

logger = logging.getLogger(__name__)


def _load_repos(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _load_pat_tokens(arg_value: str | None) -> list[str]:
    if arg_value:
        return [t.strip() for t in arg_value.split(",") if t.strip()]
    env = os.environ.get("GITHUB_TOKENS")
    if env:
        return [t.strip() for t in env.split(",") if t.strip()]
    out = []
    seen: set[str] = set()
    for key in ("GITHUB_API_TOKEN", "GITHUB_API_TOKEN2", "GITHUB_TOKEN"):
        v = os.environ.get(key)
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _build_sources(args: argparse.Namespace) -> list[TokenSource]:
    sources: list[TokenSource] = []
    for i, t in enumerate(_load_pat_tokens(args.tokens)):
        sources.append(StaticToken(t, label=f"pat{i}"))

    app_id = args.app_id or os.environ.get("GITHUB_APP_ID")
    inst_id = args.app_installation_id or os.environ.get("GITHUB_APP_INSTALLATION_ID")
    pem = args.app_private_key or os.environ.get("GITHUB_APP_PRIVATE_KEY_FILE")
    if app_id and inst_id and pem:
        sources.append(
            GitHubAppToken.from_pem_file(
                app_id=app_id, installation_id=inst_id, pem_path=pem, label="app",
            )
        )
        logger.info("loaded GitHub App source (app_id=%s, installation=%s)", app_id, inst_id)
    elif any([app_id, inst_id, pem]):
        logger.warning(
            "GitHub App partially configured (missing one of app_id/installation/pem); skipping"
        )

    return sources


def _write(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("wrote %d rows to %s", len(df), out_path)


async def _async_main(args: argparse.Namespace) -> int:
    repos = _load_repos(Path(args.repos))
    logger.info("loaded %d repos from %s", len(repos), args.repos)

    sources = _build_sources(args)
    if not sources:
        logger.error(
            "no token sources: provide --tokens / $GITHUB_TOKENS / "
            "$GITHUB_API_TOKEN[2], or a full GitHub App config"
        )
        return 2
    logger.info(
        "token pool: %d source(s) [%s] -> theoretical %d req/hr",
        len(sources), ", ".join(s.label for s in sources), len(sources) * 5000,
    )

    pool = TokenPool(sources, log_every=args.log_every)

    runs_df = await list_runs_for_repos(
        pool,
        repos,
        max_per_repo=args.max_per_repo,
        failure_share=args.failure_share,
    )
    if runs_df.empty:
        logger.warning("no runs collected; aborting")
        return 0

    if args.runs_cache:
        _write(runs_df, Path(args.runs_cache))

    if args.no_enrich:
        _write(runs_df, Path(args.out))
        logger.info("done (no-enrich). final pool state: %s", pool.snapshot())
        return 0

    cp_path = Path(args.checkpoint) if args.checkpoint else Path(args.out).with_suffix(".checkpoint.parquet")
    enriched = await enrich_dataframe(
        runs_df,
        pool=pool,
        fetch_logs=not args.no_logs,
        concurrency_per_token=args.concurrency_per_token,
        checkpoint_path=cp_path,
        checkpoint_every=args.checkpoint_every,
    )

    _write(enriched, Path(args.out))
    logger.info("done. final token pool state: %s", pool.snapshot())
    if "conclusion" in enriched:
        logger.info("conclusion: %s", enriched["conclusion"].value_counts(dropna=False).to_dict())
    if "failure_class" in enriched:
        logger.info("failure_class: %s", enriched["failure_class"].value_counts(dropna=False).to_dict())
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="REST-pool collector with GitHub App support")
    p.add_argument("--repos", required=True, help="Path to file with one owner/repo per line")
    p.add_argument("--out", default="../data/raw/gha_runs_hybrid.parquet")
    p.add_argument("--max-per-repo", type=int, default=100,
                   help="Max runs per repo (split by failure_share, default: 100)")
    p.add_argument("--failure-share", type=float, default=0.5,
                   help="Target share of failures (default: 0.5)")
    p.add_argument("--no-logs", action="store_true",
                   help="Skip job-log fetch (no failure_class derivation)")
    p.add_argument("--no-enrich", action="store_true",
                   help="Stop after listing runs; do not REST-enrich commits/logs")
    p.add_argument("--runs-cache", help="Optional path to dump listed runs before enrichment")
    p.add_argument("--tokens", help="Comma-separated PATs (defaults to $GITHUB_TOKENS / $GITHUB_API_TOKEN[2])")
    p.add_argument("--app-id", help="GitHub App ID or Client ID (defaults to $GITHUB_APP_ID)")
    p.add_argument("--app-installation-id",
                   help="Installation ID (defaults to $GITHUB_APP_INSTALLATION_ID)")
    p.add_argument("--app-private-key",
                   help="Path to App private-key .pem (defaults to $GITHUB_APP_PRIVATE_KEY_FILE)")
    p.add_argument("--concurrency-per-token", type=int, default=8,
                   help="Max concurrent in-flight requests per source (default: 8)")
    p.add_argument("--log-every", type=int, default=1000,
                   help="Log pool status every N requests (default: 1000)")
    p.add_argument("--checkpoint",
                   help="Path for incremental enrichment checkpoint "
                        "(default: <out>.checkpoint.parquet). Resumed automatically on restart.")
    p.add_argument("--checkpoint-every", type=int, default=500,
                   help="Write checkpoint every N enriched rows (default: 500)")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    sys.exit(main())
