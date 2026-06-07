"""Prediction pipeline: webhook payload → features → inference → DB.

Runs synchronously inside a FastAPI BackgroundTask.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import (
    Commit,
    FailureClass,
    ModelVersion,
    Prediction,
    PredictionDecision,
    Repository,
)
from app.ml.feature_builder import build_feature_vector
from app.ml.inference import InferenceEngine
from app.ml.recommendations import Recommendation, generate_recommendations

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = (0.50, 0.90)

DEPENDENCY_FILES = {
    "requirements.txt",
    "requirements-lock.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "poetry.lock",
    "uv.lock",
    "pipfile",
    "pipfile.lock",
}


def _fetch_raw_file(full_name: str, sha: str, path: str, token: str) -> str | None:
    """Best-effort fetch of a file's raw content at a given commit."""
    import httpx

    try:
        with httpx.Client(timeout=4.0) as c:
            r = c.get(
                f"https://raw.githubusercontent.com/{full_name}/{sha}/{path}",
                headers={"Authorization": f"Bearer {token}"} if token else {},
            )
            if r.status_code == 200:
                return r.text
    except Exception:  # noqa: BLE001
        pass
    return None


def _check_pypi_package(name_spec: str) -> bool:
    """Return True if package name exists on PyPI (ignoring version specifier)."""
    import re

    import httpx

    name = re.split(r"[<>=!~\s\[]", name_spec.strip(), maxsplit=1)[0]
    if not name or name.startswith("#") or name.startswith("-"):
        return True  # comments / pip flags → skip
    try:
        with httpx.Client(timeout=3.0) as c:
            r = c.get(f"https://pypi.org/pypi/{name}/json")
            return r.status_code == 200
    except Exception:  # noqa: BLE001
        return True  # network failure → don't penalize


def _validate_dependencies(content: str) -> bool:
    """Heuristic: every non-empty, non-comment requirement must resolve on PyPI."""
    lines = [
        ln.strip() for ln in content.splitlines() if ln.strip() and not ln.strip().startswith("#")
    ]
    if not lines:
        return True
    # cap to avoid slow webhooks
    return all(_check_pypi_package(ln) for ln in lines[:15])


def _validate_dockerfile(content: str) -> bool:
    """Cheap Dockerfile checks: parsable FROM line + base image exists on registry."""
    import re

    import httpx

    lines = [
        ln.strip() for ln in content.splitlines() if ln.strip() and not ln.strip().startswith("#")
    ]
    from_lines = [ln for ln in lines if ln.upper().startswith("FROM ")]
    if not from_lines:
        return False
    m = re.match(r"FROM\s+(\S+)", from_lines[0], re.IGNORECASE)
    if not m:
        return False
    image_spec = m.group(1).split("@")[0]  # strip digest
    if ":" in image_spec:
        image, tag = image_spec.rsplit(":", 1)
    else:
        image, tag = image_spec, "latest"
    if "/" not in image:
        image = f"library/{image}"
    valid_instructions = {
        "FROM",
        "RUN",
        "CMD",
        "ENTRYPOINT",
        "WORKDIR",
        "COPY",
        "ADD",
        "ENV",
        "ARG",
        "EXPOSE",
        "VOLUME",
        "USER",
        "LABEL",
        "ONBUILD",
        "STOPSIGNAL",
        "HEALTHCHECK",
        "SHELL",
        "MAINTAINER",
    }
    for ln in lines:
        first = ln.split(None, 1)[0].upper()
        if first not in valid_instructions:
            return False
    try:
        with httpx.Client(timeout=3.0) as c:
            r = c.get(f"https://hub.docker.com/v2/repositories/{image}/tags/{tag}/")
            if r.status_code == 200:
                return True
            if r.status_code == 404:
                return False
    except Exception:  # noqa: BLE001
        pass
    return True  # network failure → don't penalize


def _preflight_validate(
    full_name: str, sha: str, files: list[dict], token: str
) -> dict[str, bool | None]:
    """Fetch + validate changed dep manifests and Dockerfiles.

    Returns a dict with keys ``deps_valid``/``docker_valid``. Each value is:
      * True  — file fetched and validation passed
      * False — file fetched and validation failed (definitely broken)
      * None  — file not in the diff (no signal)
    """
    deps_valid: bool | None = None
    docker_valid: bool | None = None
    for f in files:
        path = f.get("filename") or ""
        base = path.rsplit("/", 1)[-1].lower()
        if base in DEPENDENCY_FILES and deps_valid is None:
            content = _fetch_raw_file(full_name, sha, path, token)
            if content is not None:
                deps_valid = _validate_dependencies(content)
        if "dockerfile" in base and docker_valid is None:
            content = _fetch_raw_file(full_name, sha, path, token)
            if content is not None:
                docker_valid = _validate_dockerfile(content)
        if deps_valid is not None and docker_valid is not None:
            break
    return {"deps_valid": deps_valid, "docker_valid": docker_valid}


def _resolve_thresholds(db: Session, repo: Repository | None) -> tuple[float, float]:
    """Pick (auto_approve_threshold, block_threshold) for this repo.

    Order: repo.policy → first is_default=True policy → DEFAULT_THRESHOLDS.
    """
    from app.db.models import Policy  # local import to avoid circular

    policy = None
    if repo is not None and repo.policy_id is not None:
        policy = db.get(Policy, repo.policy_id)
    if policy is None:
        policy = db.scalar(select(Policy).where(Policy.is_default.is_(True)))
    if policy is None:
        return DEFAULT_THRESHOLDS
    return policy.auto_approve_threshold, policy.block_threshold


def _resolve_predicted_class(
    top_class: str,
    class_probabilities: dict[str, float],
    features: dict,
) -> str:
    """Bridge the cause classifier's failure-only output back to `success`.

    The v17 cause classifier is trained only on failure rows so it cannot
    return `success` directly — it always picks the most likely *cause if it
    fails*. For trivial commits with no risky signal we override that with
    `success` so the UI does not show a phantom failure type for a no-op
    edit. The risk_score remains the ML output unchanged.

    Rules (first match wins):
      1. Empty diff → success.
      2. ≤2 files, <20 changed lines, no Dockerfile / no dependency manifest
         touched → success.
      3. Otherwise → trust the cause model (it knows which failure mode
         the change shape looks like).
    """
    files = int(features.get("files_changed", 0) or 0)
    lines_total = int(features.get("lines_added", 0) or 0) + int(
        features.get("lines_deleted", 0) or 0
    )
    has_docker = bool(features.get("has_dockerfile_change"))
    has_deps = bool(features.get("has_dependency_change"))
    has_test_only = bool(features.get("has_test_only_changes"))

    if files == 0 and lines_total == 0:
        return "success"
    # Trivial diff without any risky / test signal → success.
    if files <= 2 and lines_total < 20 and not has_docker and not has_deps and not has_test_only:
        return "success"
    return top_class


def _decide(
    risk_score: float,
    thresholds: tuple[float, float] = DEFAULT_THRESHOLDS,
    *,
    features: dict | None = None,
    predicted_class: str | None = None,
    confidence: float = 0.0,
    validation: dict[str, bool | None] | None = None,
) -> PredictionDecision:
    """Class-aware decision driven by v17 two-stage ML output.

    Decision order:
      1. Hard cap for truly massive PRs (>100 files OR >5000 lines) → BLOCK.
      2. Infrastructure-critical failure class predicted with confidence →
         BLOCK (docker_build_failed) or WARN (oom/timeout/network/deps).
      3. Generic class (other_failure / success) → fall through to risk
         thresholds from the active Policy.
    """
    auto, block = thresholds

    if features is not None:
        files = int(features.get("files_changed", 0) or 0)
        lines_total = int(features.get("lines_added", 0) or 0) + int(
            features.get("lines_deleted", 0) or 0
        )
        # Hard cap for genuinely massive PRs — reviewer cannot meaningfully
        # vouch for thousands of lines in one commit, regardless of risk.
        if files > 100 or lines_total > 5000:
            return PredictionDecision.BLOCK

    # When upstream resolution has labelled the commit as success (trivial diff
    # with no risky signal), trust that signal and skip the risk-based
    # threshold path entirely. The risk_score can stay high for repos with a
    # high prior failure rate, but a trivial diff there is still safe.
    if predicted_class == "success":
        return PredictionDecision.AUTO_APPROVE

    if predicted_class and confidence >= 0.55:
        deps_valid = (validation or {}).get("deps_valid")
        docker_valid = (validation or {}).get("docker_valid")
        # Risk-aware rule escalation: do not BLOCK on a rule alone when the
        # ML risk model itself considers success more likely than failure
        # (risk_score < 0.50). Without this guard the UI shows
        # ``P(success)=58%`` next to a BLOCK decision, which contradicts
        # defence narrative. With this guard the rule downgrades to WARN
        # when the model disagrees; BLOCK requires both signals to agree.
        ml_agrees_failure = risk_score >= 0.50
        if predicted_class == "docker_build_failed":
            if docker_valid is True:
                # Preflight: Dockerfile parses + base image exists → soften BLOCK to WARN.
                return PredictionDecision.WARN
            return PredictionDecision.BLOCK if ml_agrees_failure else PredictionDecision.WARN
        if predicted_class == "dependency_error":
            if deps_valid is True:
                # Preflight: all PyPI packages resolve → soften WARN to AUTO.
                return PredictionDecision.AUTO_APPROVE
            return PredictionDecision.WARN
        if predicted_class in {"oom_killed", "test_timeout", "network_error", "test_failure"}:
            return PredictionDecision.WARN

    if risk_score < auto:
        return PredictionDecision.AUTO_APPROVE
    if risk_score < block:
        return PredictionDecision.WARN
    return PredictionDecision.BLOCK


def _ensure_repository(db: Session, full_name: str, html_url: str) -> Repository:
    repo = db.scalar(select(Repository).where(Repository.full_name == full_name))
    if repo is not None:
        return repo
    repo = Repository(full_name=full_name, url=html_url)
    db.add(repo)
    db.flush()
    return repo


def _ensure_commit(
    db: Session,
    repo: Repository,
    *,
    sha: str,
    author_email: str,
    author_name: str | None,
    message: str | None,
    branch: str | None,
    committed_at: datetime,
    files_changed: int,
    lines_added: int,
    lines_deleted: int,
    raw_metadata: dict[str, Any],
) -> Commit:
    existing = db.scalar(select(Commit).where(Commit.repository_id == repo.id, Commit.sha == sha))
    if existing is not None:
        return existing
    commit = Commit(
        repository_id=repo.id,
        sha=sha,
        author_email=author_email,
        author_name=author_name,
        message=message,
        branch=branch,
        committed_at=committed_at,
        files_changed=files_changed,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        raw_metadata=raw_metadata,
    )
    db.add(commit)
    db.flush()
    return commit


def _ensure_active_model_version(db: Session, engine: InferenceEngine) -> ModelVersion:
    version_str = str(engine.version.get("trained_at", "unknown"))
    mv = db.scalar(select(ModelVersion).where(ModelVersion.version == version_str))
    if mv is not None:
        return mv
    metrics = engine.version.get("classes", [])
    mv = ModelVersion(
        version=version_str,
        classifier_path="classifier.joblib",
        regressor_path="regressor_memory.joblib,regressor_duration.joblib",
        feature_pipeline_path="feature_columns.json",
        trained_at=datetime.fromisoformat(version_str)
        if version_str != "unknown"
        else datetime.now(tz=UTC),
        training_dataset_size=int(engine.version.get("n_train", 0)),
        metrics={"classes": metrics},
        is_active=True,
    )
    db.add(mv)
    db.flush()
    return mv


def _serialise_recommendations(recs: list[Recommendation]) -> list[dict[str, Any]]:
    return [
        {
            "severity": r.severity,
            "category": r.category,
            "title": r.title,
            "description": r.description,
            "actions": list(r.actions),
            "estimated_impact": r.estimated_impact,
        }
        for r in recs
    ]


def process_push_event(
    db: Session,
    engine: InferenceEngine,
    payload: dict[str, Any],
) -> Prediction:
    repository = payload.get("repository") or {}
    full_name = str(repository.get("full_name") or "unknown/unknown")
    html_url = str(repository.get("html_url") or "")
    head_commit = payload.get("head_commit") or {}
    sha = str(head_commit.get("id") or payload.get("after") or "")
    if not sha:
        raise ValueError("payload missing head_commit.id / after")

    author = head_commit.get("author") or {}
    author_email = str(author.get("email") or "unknown@example.com")
    author_name = author.get("name")
    message = head_commit.get("message")
    branch = (payload.get("ref") or "").replace("refs/heads/", "") or None
    committed_at = datetime.now(tz=UTC)

    added_paths = head_commit.get("added") or []
    modified_paths = head_commit.get("modified") or []
    removed_paths = head_commit.get("removed") or []
    files = [{"filename": p} for p in [*added_paths, *modified_paths, *removed_paths]]

    # GitHub push webhooks do NOT include line-level stats — only filenames.
    # Pull the actual additions/deletions via the REST commits endpoint so the
    # model sees a meaningful feature vector. Best-effort; fall back to 0.
    lines_added = int(payload.get("lines_added") or 0)
    lines_deleted = int(payload.get("lines_deleted") or 0)
    if lines_added == 0 and lines_deleted == 0 and full_name and sha:
        try:
            import httpx

            from app.core.config import get_settings as _gs
            from app.core.resilient_http import call_with_resilience

            tok = _gs().github_api_token
            with httpx.Client(timeout=8.0) as c:
                r = call_with_resilience(
                    "github",
                    lambda: c.get(
                        f"https://api.github.com/repos/{full_name}/commits/{sha}",
                        headers={
                            "Authorization": f"Bearer {tok}",
                            "Accept": "application/vnd.github+json",
                        },
                    ),
                )
                if r.status_code == 200:
                    body = r.json()
                    stats = body.get("stats") or {}
                    lines_added = int(stats.get("additions") or 0)
                    lines_deleted = int(stats.get("deletions") or 0)
                    api_files = body.get("files") or []
                    if api_files:
                        files = [
                            {
                                "filename": f.get("filename", ""),
                                "additions": int(f.get("additions") or 0),
                                "deletions": int(f.get("deletions") or 0),
                            }
                            for f in api_files
                        ]
                    logger.info(
                        "enriched commit %s/%s: +%d/-%d over %d files",
                        full_name,
                        sha[:7],
                        lines_added,
                        lines_deleted,
                        len(files),
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("commit stats fetch failed for %s/%s: %s", full_name, sha[:7], exc)

    repo = _ensure_repository(db, full_name=full_name, html_url=html_url)
    commit = _ensure_commit(
        db,
        repo,
        sha=sha,
        author_email=author_email,
        author_name=author_name,
        message=message,
        branch=branch,
        committed_at=committed_at,
        files_changed=len(files),
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        raw_metadata={
            "added": added_paths,
            "modified": modified_paths,
            "removed": removed_paths,
            "delivery_id": payload.get("_delivery_id"),
        },
    )

    feats = build_feature_vector(
        db,
        repo,
        files=files,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        created_at=committed_at,
        author_email=author_email,
    )
    result = engine.safe_predict(feats)

    recs = generate_recommendations(
        result.predicted_class,
        risk_score=result.risk_score,
        class_probabilities=result.class_probabilities,
        predicted_memory_mb=result.predicted_memory_mb,
        predicted_duration_seconds=result.predicted_duration_seconds,
        features=feats,
    )

    mv = _ensure_active_model_version(db, engine)
    rule_features = {
        "files_changed": len(files),
        "lines_added": lines_added,
        "lines_deleted": lines_deleted,
        "has_dockerfile_change": any(
            (f.get("filename") or "").lower().endswith("dockerfile")
            or "/dockerfile" in (f.get("filename") or "").lower()
            or (f.get("filename") or "").lower() == "dockerfile"
            for f in files
        ),
        "has_dependency_change": any(
            (f.get("filename") or "").rsplit("/", 1)[-1].lower()
            in {
                "requirements.txt",
                "requirements-lock.txt",
                "package.json",
                "package-lock.json",
                "yarn.lock",
                "pnpm-lock.yaml",
                "pyproject.toml",
                "poetry.lock",
                "uv.lock",
                "pipfile",
                "go.mod",
                "go.sum",
                "cargo.toml",
                "cargo.lock",
                "pom.xml",
                "build.gradle",
                "gemfile.lock",
            }
            for f in files
        ),
        "has_test_only_changes": bool(feats.get("feat_test_only_changes_int", 0)),
    }
    resolved_class = _resolve_predicted_class(
        result.predicted_class,
        result.class_probabilities,
        rule_features,
    )
    validation: dict[str, bool | None] = {"deps_valid": None, "docker_valid": None}
    if rule_features["has_dependency_change"] or rule_features["has_dockerfile_change"]:
        try:
            from app.core.config import get_settings as _gs2

            validation = _preflight_validate(full_name, sha, files, _gs2().github_api_token)
            logger.info(
                "preflight: deps_valid=%s docker_valid=%s",
                validation["deps_valid"],
                validation["docker_valid"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("preflight validation failed: %s", exc)
    decision = _decide(
        result.risk_score,
        _resolve_thresholds(db, repo),
        features=rule_features,
        predicted_class=resolved_class,
        confidence=result.confidence,
        validation=validation,
    )

    prediction = Prediction(
        commit_id=commit.id,
        repository_id=repo.id,
        model_version_id=mv.id,
        predicted_class=FailureClass(resolved_class),
        class_probabilities=result.class_probabilities,
        risk_score=result.risk_score,
        confidence=result.confidence,
        decision=decision,
        predicted_memory_mb=result.predicted_memory_mb,
        predicted_duration_min=result.predicted_duration_seconds / 60,
        feature_vector=feats,
        feature_importance=result.feature_importance,
        shap_explanation=result.shap_explanation,
        recommendations=_serialise_recommendations(recs),
        inference_time_ms=result.inference_time_ms,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    logger.info(
        "prediction stored: id=%s sha=%s class=%s risk=%.2f decision=%s ms=%d",
        prediction.id,
        sha[:8],
        result.predicted_class,
        result.risk_score,
        decision.value,
        result.inference_time_ms,
    )
    # Post commit status back to GitHub so BLOCK actually gates merge via
    # branch protection. Best-effort: synthetic fixtures and disabled mode
    # are skipped inside the service.
    try:
        from app.services.github_status import post_commit_status

        post_commit_status(full_name, sha, decision, prediction.id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("github status post failed for %s@%s: %s", full_name, sha[:8], exc)
    return prediction
