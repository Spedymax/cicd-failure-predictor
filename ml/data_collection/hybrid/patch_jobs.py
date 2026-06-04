"""One-shot patch: backfill n_jobs / longest_job_seconds for success rows.

The original hybrid collection only called list_jobs for failed runs, so
every success row ended up with n_jobs=0 / longest_job_seconds=NaN -- a
collection artifact. rest_enricher.py is now fixed for future runs; this
script repairs the already-collected parquet in place.

Only rows where list_jobs is actually missing are fetched (success rows,
or any row with n_jobs==0). Writes a checkpoint every --checkpoint-every
rows so a crash/rate-limit loses at most that much work; rerun resumes.

Usage::

    cd ml
    set -a && . ../.env && set +a
    uv run python -m data_collection.hybrid.patch_jobs \\
        --parquet ../data/raw/gha_runs_hybrid.parquet
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm.asyncio import tqdm

from data_collection.hybrid.collect_hybrid import _build_sources
from data_collection.hybrid.rest_enricher import PooledGitHubClient
from data_collection.hybrid.token_pool import TokenPool

logger = logging.getLogger(__name__)


async def _patch(args: argparse.Namespace) -> int:
    path = Path(args.parquet)
    df = pd.read_parquet(path)
    logger.info("loaded %d rows from %s", len(df), path)

    # Rows needing a backfill: n_jobs missing or 0 (success rows were never
    # job-enriched). Failed rows already have real n_jobs, skip them.
    need = df[(df["n_jobs"].isna()) | (df["n_jobs"] == 0)].copy()
    logger.info("rows needing job backfill: %d", len(need))
    if need.empty:
        logger.info("nothing to patch")
        return 0

    sources = _build_sources(args)
    if not sources:
        logger.error("no token sources available")
        return 2
    pool = TokenPool(sources, log_every=args.log_every)
    logger.info("token pool: %d source(s) [%s]", len(sources), ", ".join(s.label for s in sources))

    updates: dict[int, dict[str, float]] = {}
    updates_lock = asyncio.Lock()
    sem = asyncio.Semaphore(max(1, args.concurrency_per_token * len(pool)))
    bar = tqdm(total=len(need), desc="patch jobs")

    def _flush() -> None:
        for idx, vals in updates.items():
            for col, v in vals.items():
                df.at[idx, col] = v
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)
        logger.info("checkpoint: flushed %d updates to %s", len(updates), path)

    async with PooledGitHubClient(pool) as client:

        async def patch_one(idx: int, row: pd.Series) -> None:
            async with sem:
                owner, _, repo = row["repo"].partition("/")
                if not repo:
                    bar.update(1)
                    return
                try:
                    jobs = await client.list_jobs(owner, repo, int(row["run_id"]))
                    durations = []
                    for j in jobs:
                        s = j.get("started_at")
                        c = j.get("completed_at")
                        if s and c:
                            durations.append(
                                (pd.Timestamp(c) - pd.Timestamp(s)).total_seconds()
                            )
                    vals: dict[str, float] = {"n_jobs": float(len(jobs))}
                    if durations:
                        vals["longest_job_seconds"] = max(durations)
                    async with updates_lock:
                        updates[idx] = vals
                        n = len(updates)
                except Exception as exc:
                    logger.debug("list_jobs fail %s/%s: %s", row["repo"], row["run_id"], exc)
                    n = len(updates)
                bar.update(1)
                if n and n % args.checkpoint_every == 0:
                    async with updates_lock:
                        _flush()

        await asyncio.gather(*(patch_one(i, r) for i, r in need.iterrows()))
    bar.close()

    async with updates_lock:
        _flush()
    logger.info("done: patched %d rows. final pool: %s", len(updates), pool.snapshot())

    # quick sanity
    succ = df[df["conclusion"] == "success"]
    logger.info(
        "success rows now: n_jobs mean=%.1f (zeros=%d), longest_job_seconds non-null=%d",
        succ["n_jobs"].mean(), int((succ["n_jobs"] == 0).sum()),
        int(succ["longest_job_seconds"].notna().sum()),
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backfill n_jobs for success rows")
    p.add_argument("--parquet", required=True, help="Path to gha_runs_hybrid.parquet to patch in place")
    p.add_argument("--tokens", help="Comma-separated PATs (defaults to env)")
    p.add_argument("--app-id", help="GitHub App ID (defaults to $GITHUB_APP_ID)")
    p.add_argument("--app-installation-id", help="defaults to $GITHUB_APP_INSTALLATION_ID")
    p.add_argument("--app-private-key", help="defaults to $GITHUB_APP_PRIVATE_KEY_FILE")
    p.add_argument("--concurrency-per-token", type=int, default=8)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return asyncio.run(_patch(args))


if __name__ == "__main__":
    sys.exit(main())
