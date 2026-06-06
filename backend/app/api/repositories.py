"""Repository CRUD for the Add Repository wizard (UC-6, simplified)."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Repository
from app.db.session import get_db
from app.services.repo_discovery import discover_repository_metadata

router = APIRouter(prefix="/repositories", tags=["repositories"])


class RepositoryIn(BaseModel):
    full_name: str = Field(min_length=3, max_length=255, pattern=r"^[^/]+/[^/]+$")
    default_branch: str = Field(default="main", max_length=255)
    policy_id: int | None = None


class RepositoryOut(BaseModel):
    id: int
    provider: str
    full_name: str
    url: str
    default_branch: str
    ci_platform: str
    language: str | None
    package_manager: str | None
    has_dockerfile: bool
    webhook_secret: str | None  # only populated on creation
    policy_id: int | None
    last_synced_at: datetime | None
    is_active: bool

    @classmethod
    def from_orm_(cls, r: Repository, *, secret: str | None = None) -> RepositoryOut:
        return cls(
            id=r.id,
            provider=r.provider.value if hasattr(r.provider, "value") else str(r.provider),
            full_name=r.full_name,
            url=r.url,
            default_branch=r.default_branch,
            ci_platform=r.ci_platform.value
            if hasattr(r.ci_platform, "value")
            else str(r.ci_platform),
            language=r.language,
            package_manager=r.package_manager,
            has_dockerfile=r.has_dockerfile,
            webhook_secret=secret,
            policy_id=r.policy_id,
            last_synced_at=r.last_synced_at,
            is_active=r.is_active,
        )


@router.get("", response_model=list[RepositoryOut])
def list_repositories(db: Session = Depends(get_db)) -> list[RepositoryOut]:
    """Only return repos that were explicitly added via POST /repositories
    (i.e. have a webhook_secret_id set). Auto-created repos from the data
    collection pipeline are hidden — they don't have a webhook configured
    and pollute the UI."""
    rows = db.scalars(
        select(Repository)
        .where(
            Repository.webhook_secret_id.is_not(None),
            Repository.is_active.is_(True),
        )
        .order_by(Repository.id.desc())
    ).all()
    return [RepositoryOut.from_orm_(r) for r in rows]


@router.post("", response_model=RepositoryOut, status_code=201)
def create_repository(body: RepositoryIn, db: Session = Depends(get_db)) -> RepositoryOut:
    existing = db.scalar(select(Repository).where(Repository.full_name == body.full_name))
    if existing is not None:
        raise HTTPException(status_code=409, detail="repository already exists")
    secret = secrets.token_hex(32)
    metadata = discover_repository_metadata(body.full_name)
    repo = Repository(
        full_name=body.full_name,
        url=f"https://github.com/{body.full_name}",
        default_branch=body.default_branch,
        policy_id=body.policy_id,
        webhook_secret_id=secret,  # stored, returned once for copy-paste
        language=metadata.language,
        has_dockerfile=metadata.has_dockerfile,
        package_manager=metadata.package_manager,
        last_synced_at=datetime.now(tz=UTC),
    )
    db.add(repo)
    db.commit()
    db.refresh(repo)
    # secret is returned on creation only; never echoed back on subsequent GETs
    return RepositoryOut.from_orm_(repo, secret=secret)


@router.delete("/{repository_id}", status_code=204)
def deactivate_repository(repository_id: int, db: Session = Depends(get_db)) -> None:
    r = db.get(Repository, repository_id)
    if r is None:
        raise HTTPException(status_code=404, detail="repository not found")
    r.is_active = False
    db.commit()
