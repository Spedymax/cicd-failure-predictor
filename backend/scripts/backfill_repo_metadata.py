"""One-shot: discover language/has_dockerfile/package_manager for existing repos.

Run from backend/ (so .env is picked up):

    cd backend
    uv run python scripts/backfill_repo_metadata.py

Only touches rows where language IS NULL (or --force to refresh everything).
Uses the same single-token discovery as POST /repositories — slow but safe;
GITHUB_API_TOKEN burns ~2 requests per repo.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Repository
from app.db.session import SessionLocal
from app.services.repo_discovery import discover_repository_metadata

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="Re-discover even if language is already set")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after N repos (default: all)")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    db: Session = SessionLocal()
    try:
        stmt = select(Repository).order_by(Repository.id)
        if not args.force:
            stmt = stmt.where(Repository.language.is_(None))
        if args.limit:
            stmt = stmt.limit(args.limit)
        repos = db.scalars(stmt).all()
        logger.info("backfilling %d repos (force=%s)", len(repos), args.force)

        for i, r in enumerate(repos, 1):
            meta = discover_repository_metadata(r.full_name)
            if meta.language or meta.has_dockerfile or meta.package_manager:
                r.language = meta.language or r.language
                r.has_dockerfile = meta.has_dockerfile or r.has_dockerfile
                r.package_manager = meta.package_manager or r.package_manager
                r.last_synced_at = datetime.now(tz=timezone.utc)
                logger.info(
                    "%3d/%d %s → lang=%s docker=%s pm=%s",
                    i, len(repos), r.full_name,
                    meta.language, meta.has_dockerfile, meta.package_manager,
                )
            else:
                logger.info("%3d/%d %s → no metadata (private? rate-limited?)",
                            i, len(repos), r.full_name)
            if i % 25 == 0:
                db.commit()
        db.commit()
        logger.info("done")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
