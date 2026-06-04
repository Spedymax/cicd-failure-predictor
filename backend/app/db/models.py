from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class UserRole(str, enum.Enum):
    DEVELOPER = "developer"
    DEVOPS = "devops"
    TEAM_LEAD = "team_lead"
    ADMIN = "admin"


class CIPlatform(str, enum.Enum):
    GITHUB_ACTIONS = "github_actions"
    BUILDKITE = "buildkite"


class GitProvider(str, enum.Enum):
    GITHUB = "github"
    GITLAB = "gitlab"


class FailureClass(str, enum.Enum):
    SUCCESS = "success"
    OOM_KILLED = "oom_killed"
    TEST_TIMEOUT = "test_timeout"
    DEPENDENCY_ERROR = "dependency_error"
    DOCKER_BUILD_FAILED = "docker_build_failed"
    NETWORK_ERROR = "network_error"
    TEST_FAILURE = "test_failure"
    OTHER_FAILURE = "other_failure"


class PredictionDecision(str, enum.Enum):
    AUTO_APPROVE = "auto_approve"
    WARN = "warn"
    BLOCK = "block"


class BuildStatus(str, enum.Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    IN_PROGRESS = "in_progress"


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(254), unique=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, name="user_role"), default=UserRole.DEVELOPER, nullable=False
    )
    password_hash: Mapped[str | None] = mapped_column(String(128))
    oauth_provider: Mapped[str | None] = mapped_column(String(32))
    oauth_subject: Mapped[str | None] = mapped_column(String(128))
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    overrides: Mapped[list["Prediction"]] = relationship(
        back_populates="overridden_by", foreign_keys="Prediction.overridden_by_user_id"
    )

    __table_args__ = (
        UniqueConstraint("oauth_provider", "oauth_subject", name="uq_users_oauth"),
    )


class Policy(Base, TimestampMixin):
    __tablename__ = "policies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    auto_approve_threshold: Mapped[float] = mapped_column(Float, default=0.20, nullable=False)
    warn_threshold: Mapped[float] = mapped_column(Float, default=0.60, nullable=False)
    block_threshold: Mapped[float] = mapped_column(Float, default=0.70, nullable=False)
    allow_override: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    specific_rules: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    repositories: Mapped[list["Repository"]] = relationship(back_populates="policy")


class Repository(Base, TimestampMixin):
    __tablename__ = "repositories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[GitProvider] = mapped_column(
        Enum(GitProvider, name="git_provider"), default=GitProvider.GITHUB, nullable=False
    )
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    url: Mapped[str] = mapped_column(String(512), nullable=False)
    default_branch: Mapped[str] = mapped_column(String(255), default="main", nullable=False)
    ci_platform: Mapped[CIPlatform] = mapped_column(
        Enum(CIPlatform, name="ci_platform"), default=CIPlatform.GITHUB_ACTIONS, nullable=False
    )
    language: Mapped[str | None] = mapped_column(String(64))
    package_manager: Mapped[str | None] = mapped_column(String(32))
    has_dockerfile: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    webhook_secret_id: Mapped[str | None] = mapped_column(String(128))
    policy_id: Mapped[int | None] = mapped_column(ForeignKey("policies.id", ondelete="SET NULL"))
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    policy: Mapped[Policy | None] = relationship(back_populates="repositories")
    commits: Mapped[list["Commit"]] = relationship(
        back_populates="repository", cascade="all, delete-orphan"
    )
    builds: Mapped[list["BuildHistory"]] = relationship(
        back_populates="repository", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("provider", "full_name", name="uq_repositories_provider_full_name"),
    )


class Commit(Base, TimestampMixin):
    __tablename__ = "commits"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False
    )
    sha: Mapped[str] = mapped_column(String(64), nullable=False)
    parent_sha: Mapped[str | None] = mapped_column(String(64))
    branch: Mapped[str | None] = mapped_column(String(255))
    author_email: Mapped[str] = mapped_column(String(254), nullable=False)
    author_name: Mapped[str | None] = mapped_column(String(255))
    committed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    message: Mapped[str | None] = mapped_column(Text)
    files_changed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    lines_added: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    lines_deleted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    raw_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    repository: Mapped[Repository] = relationship(back_populates="commits")
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="commit", cascade="all, delete-orphan"
    )
    builds: Mapped[list["BuildHistory"]] = relationship(back_populates="commit")

    __table_args__ = (
        UniqueConstraint("repository_id", "sha", name="uq_commits_repository_sha"),
        Index("ix_commits_author_email_committed_at", "author_email", "committed_at"),
    )


class BuildHistory(Base, TimestampMixin):
    __tablename__ = "build_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False
    )
    commit_id: Mapped[int | None] = mapped_column(
        ForeignKey("commits.id", ondelete="SET NULL")
    )
    external_run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[BuildStatus] = mapped_column(Enum(BuildStatus, name="build_status"))
    failure_class: Mapped[FailureClass | None] = mapped_column(
        Enum(FailureClass, name="failure_class")
    )
    duration_seconds: Mapped[int | None] = mapped_column(Integer)
    peak_memory_mb: Mapped[float | None] = mapped_column(Float)
    image_size_mb: Mapped[float | None] = mapped_column(Float)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    repository: Mapped[Repository] = relationship(back_populates="builds")
    commit: Mapped[Commit | None] = relationship(back_populates="builds")

    __table_args__ = (
        UniqueConstraint("repository_id", "external_run_id", name="uq_build_repo_run"),
        Index("ix_build_repository_completed_at", "repository_id", "completed_at"),
        Index("ix_build_failure_class", "failure_class"),
    )


class ModelVersion(Base, TimestampMixin):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    classifier_path: Mapped[str] = mapped_column(String(512), nullable=False)
    regressor_path: Mapped[str | None] = mapped_column(String(512))
    feature_pipeline_path: Mapped[str | None] = mapped_column(String(512))
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    training_dataset_size: Mapped[int] = mapped_column(Integer, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="model_version")


class Prediction(Base, TimestampMixin):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    commit_id: Mapped[int] = mapped_column(
        ForeignKey("commits.id", ondelete="CASCADE"), nullable=False
    )
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False
    )
    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=False
    )

    predicted_class: Mapped[FailureClass] = mapped_column(
        Enum(FailureClass, name="failure_class", create_type=False), nullable=False
    )
    class_probabilities: Mapped[dict[str, float]] = mapped_column(JSONB, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    decision: Mapped[PredictionDecision] = mapped_column(
        Enum(PredictionDecision, name="prediction_decision"), nullable=False
    )

    predicted_memory_mb: Mapped[float | None] = mapped_column(Float)
    predicted_duration_min: Mapped[float | None] = mapped_column(Float)

    feature_vector: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    feature_importance: Mapped[dict[str, float]] = mapped_column(JSONB, default=dict, nullable=False)
    shap_explanation: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    recommendations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    inference_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    overridden_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    overridden_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    override_reason: Mapped[str | None] = mapped_column(Text)
    actual_outcome: Mapped[FailureClass | None] = mapped_column(
        Enum(FailureClass, name="failure_class", create_type=False)
    )

    commit: Mapped[Commit] = relationship(back_populates="predictions")
    model_version: Mapped[ModelVersion] = relationship(back_populates="predictions")
    overridden_by: Mapped[User | None] = relationship(
        back_populates="overrides", foreign_keys=[overridden_by_user_id]
    )

    __table_args__ = (
        Index("ix_predictions_repo_created_at", "repository_id", "created_at"),
        Index("ix_predictions_commit_id", "commit_id"),
        Index("ix_predictions_decision", "decision"),
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_id: Mapped[str | None] = mapped_column(String(64))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    __table_args__ = (
        Index("ix_audit_entity", "entity_type", "entity_id"),
        Index("ix_audit_created_at", "created_at"),
    )
