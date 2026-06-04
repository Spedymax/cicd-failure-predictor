"""Seed predictions with realistic per-class feature vectors.

The plain webhook seeder produces all-zero engineered features (we don't
have the GitHub API enrichment in the demo flow), so every prediction
collapses to "success". This script bypasses the webhook and writes
predictions through the same InferenceEngine + persistence layer using
hand-crafted feature vectors that match the synthetic distributions.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# bootstrap env from .env
REPO_ROOT = Path(__file__).resolve().parents[1]
for line in (REPO_ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
os.environ.setdefault("ML_MODEL_DIR", str(REPO_ROOT / "data" / "artifacts" / "v2"))

sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.api.dependencies import get_inference_engine  # noqa: E402
from app.db.models import (  # noqa: E402
    Commit,
    FailureClass,
    ModelVersion,
    Prediction,
    PredictionDecision,
    Repository,
)
from app.db.session import SessionLocal  # noqa: E402
from app.ml.recommendations import generate_recommendations  # noqa: E402
from app.services.prediction_pipeline import (  # noqa: E402
    _decide,
    _ensure_active_model_version,
    _ensure_repository,
    _serialise_recommendations,
)


SCENARIOS: list[dict] = [
    dict(
        repo="acme/api-gateway",
        sha="aa110000000000000000000000000000000000a1",
        author="alice@acme.io",
        branch="main",
        message="fix: handle empty body in POST /events",
        files_changed=4,
        lines_added=45,
        lines_deleted=12,
        feats={
            "feat_files_changed_log": 1.6,
            "feat_lines_changed_log": 4.0,
            "feat_lines_added_share": 0.79,
            "feat_avg_lines_per_file": 14.0,
            "feat_has_dependency_change_int": 0,
            "feat_new_deps_count": 0,
            "feat_new_deps_size_mb_log": 0.0,
            "feat_has_dockerfile_change_int": 0,
            "feat_final_image_size_mb_log": 5.6,
            "feat_image_growth_ratio": 3.0,
            "feat_hour_of_day": 14,
            "feat_day_of_week": 1,
            "feat_is_weekend": 0,
            "feat_is_business_hours": 1,
            "feat_run_attempt": 1,
            "feat_run_attempt_gt1": 0,
            "feat_n_jobs_log": 1.4,
            "feat_longest_job_seconds_log": 5.5,
            "feat_ext_py_count": 2,
            "feat_ext_yml_count": 1,
            "feat_author_success_rate": 0.85,
            "feat_author_n_runs_log": 3.0,
            "feat_author_avg_duration_log": 5.7,
            "feat_project_failure_rate": 0.20,
            "feat_project_n_runs_log": 4.0,
            "feat_repo_failure_rate_global": 0.20,
        },
    ),
    dict(
        repo="acme/ml-pipeline",
        sha="bb220000000000000000000000000000000000b2",
        author="bob@acme.io",
        branch="feat/big-model",
        message="feat: add tensorflow + new training pipeline",
        files_changed=12,
        lines_added=380,
        lines_deleted=20,
        feats={
            "feat_files_changed_log": 2.6,
            "feat_lines_changed_log": 6.0,
            "feat_lines_added_share": 0.95,
            "feat_avg_lines_per_file": 33.0,
            "feat_has_dependency_change_int": 1,
            "feat_new_deps_count": 5,
            "feat_new_deps_size_mb_log": 5.5,
            "feat_has_dockerfile_change_int": 1,
            "feat_final_image_size_mb_log": 7.0,
            "feat_image_growth_ratio": 11.0,
            "feat_hour_of_day": 22,
            "feat_day_of_week": 5,
            "feat_is_weekend": 1,
            "feat_is_business_hours": 0,
            "feat_run_attempt": 1,
            "feat_run_attempt_gt1": 0,
            "feat_n_jobs_log": 1.6,
            "feat_longest_job_seconds_log": 6.5,
            "feat_ext_py_count": 8,
            "feat_ext_yml_count": 2,
            "feat_author_success_rate": 0.45,
            "feat_author_n_runs_log": 3.5,
            "feat_author_avg_duration_log": 6.6,
            "feat_project_failure_rate": 0.45,
            "feat_project_n_runs_log": 4.5,
            "feat_repo_failure_rate_global": 0.45,
        },
    ),
    dict(
        repo="acme/web-frontend",
        sha="cc330000000000000000000000000000000000c3",
        author="carol@acme.io",
        branch="chore/upgrade-react",
        message="chore: bump react to 19, webpack to 6",
        files_changed=5,
        lines_added=210,
        lines_deleted=15,
        feats={
            "feat_files_changed_log": 1.8,
            "feat_lines_changed_log": 5.4,
            "feat_lines_added_share": 0.93,
            "feat_avg_lines_per_file": 45.0,
            "feat_has_dependency_change_int": 1,
            "feat_new_deps_count": 9,
            "feat_new_deps_size_mb_log": 5.0,
            "feat_has_dockerfile_change_int": 0,
            "feat_final_image_size_mb_log": 5.6,
            "feat_image_growth_ratio": 3.0,
            "feat_hour_of_day": 11,
            "feat_day_of_week": 2,
            "feat_is_weekend": 0,
            "feat_is_business_hours": 1,
            "feat_run_attempt": 1,
            "feat_run_attempt_gt1": 0,
            "feat_n_jobs_log": 1.4,
            "feat_longest_job_seconds_log": 4.8,
            "feat_ext_js_count": 3,
            "feat_ext_json_count": 2,
            "feat_author_success_rate": 0.55,
            "feat_author_n_runs_log": 2.7,
            "feat_author_avg_duration_log": 5.2,
            "feat_project_failure_rate": 0.35,
            "feat_project_n_runs_log": 4.2,
            "feat_repo_failure_rate_global": 0.35,
        },
    ),
    dict(
        repo="acme/data-platform",
        sha="dd440000000000000000000000000000000000d4",
        author="dan@acme.io",
        branch="ops/cuda-base",
        message="ops: switch base image to ubuntu:22.04 + cuda",
        files_changed=2,
        lines_added=70,
        lines_deleted=10,
        feats={
            "feat_files_changed_log": 1.1,
            "feat_lines_changed_log": 4.4,
            "feat_lines_added_share": 0.88,
            "feat_avg_lines_per_file": 40.0,
            "feat_has_dependency_change_int": 0,
            "feat_new_deps_count": 0,
            "feat_new_deps_size_mb_log": 0.0,
            "feat_has_dockerfile_change_int": 1,
            "feat_final_image_size_mb_log": 7.9,
            "feat_image_growth_ratio": 6.0,
            "feat_hour_of_day": 10,
            "feat_day_of_week": 3,
            "feat_is_weekend": 0,
            "feat_is_business_hours": 1,
            "feat_run_attempt": 1,
            "feat_run_attempt_gt1": 0,
            "feat_n_jobs_log": 1.4,
            "feat_longest_job_seconds_log": 6.4,
            "feat_ext_yml_count": 1,
            "feat_author_success_rate": 0.6,
            "feat_author_n_runs_log": 3.2,
            "feat_author_avg_duration_log": 6.0,
            "feat_project_failure_rate": 0.35,
            "feat_project_n_runs_log": 4.1,
            "feat_repo_failure_rate_global": 0.35,
        },
    ),
    dict(
        repo="acme/external-sync",
        sha="ee550000000000000000000000000000000000e5",
        author="erin@acme.io",
        branch="hotfix/timeout",
        message="hotfix: retry on flaky API calls",
        files_changed=3,
        lines_added=80,
        lines_deleted=20,
        feats={
            "feat_files_changed_log": 1.4,
            "feat_lines_changed_log": 4.6,
            "feat_lines_added_share": 0.80,
            "feat_avg_lines_per_file": 33.0,
            "feat_has_dependency_change_int": 1,
            "feat_new_deps_count": 2,
            "feat_new_deps_size_mb_log": 3.0,
            "feat_has_dockerfile_change_int": 0,
            "feat_final_image_size_mb_log": 5.6,
            "feat_image_growth_ratio": 3.0,
            "feat_hour_of_day": 19,
            "feat_day_of_week": 4,
            "feat_is_weekend": 0,
            "feat_is_business_hours": 0,
            "feat_run_attempt": 3,
            "feat_run_attempt_gt1": 1,
            "feat_n_jobs_log": 2.2,
            "feat_longest_job_seconds_log": 6.7,
            "feat_ext_py_count": 2,
            "feat_ext_yml_count": 1,
            "feat_author_success_rate": 0.65,
            "feat_author_n_runs_log": 2.9,
            "feat_author_avg_duration_log": 5.8,
            "feat_project_failure_rate": 0.25,
            "feat_project_n_runs_log": 3.8,
            "feat_repo_failure_rate_global": 0.25,
        },
    ),
    dict(
        repo="acme/monolith",
        sha="ff660000000000000000000000000000000000f6",
        author="frank@acme.io",
        branch="refactor/long-tests",
        message="refactor: split worker pool, add e2e tests",
        files_changed=18,
        lines_added=520,
        lines_deleted=120,
        feats={
            "feat_files_changed_log": 2.9,
            "feat_lines_changed_log": 6.5,
            "feat_lines_added_share": 0.81,
            "feat_avg_lines_per_file": 36.0,
            "feat_has_dependency_change_int": 0,
            "feat_new_deps_count": 0,
            "feat_new_deps_size_mb_log": 0.0,
            "feat_has_dockerfile_change_int": 0,
            "feat_final_image_size_mb_log": 5.6,
            "feat_image_growth_ratio": 3.0,
            "feat_hour_of_day": 16,
            "feat_day_of_week": 2,
            "feat_is_weekend": 0,
            "feat_is_business_hours": 1,
            "feat_run_attempt": 1,
            "feat_run_attempt_gt1": 0,
            "feat_n_jobs_log": 2.5,
            "feat_longest_job_seconds_log": 7.7,
            "feat_ext_py_count": 12,
            "feat_ext_yml_count": 1,
            "feat_author_success_rate": 0.50,
            "feat_author_n_runs_log": 3.4,
            "feat_author_avg_duration_log": 7.0,
            "feat_project_failure_rate": 0.40,
            "feat_project_n_runs_log": 4.6,
            "feat_repo_failure_rate_global": 0.40,
        },
    ),
]


def main() -> int:
    engine = get_inference_engine()
    db = SessionLocal()
    try:
        mv = _ensure_active_model_version(db, engine)
        now = datetime.now(tz=timezone.utc)
        for i, sc in enumerate(SCENARIOS):
            feats_full = {n: sc["feats"].get(n, 0.0) for n in engine.feature_names}
            res = engine.predict(feats_full)

            repo = _ensure_repository(
                db, full_name=sc["repo"], html_url=f"https://github.com/{sc['repo']}"
            )
            committed_at = now - timedelta(minutes=15 * (len(SCENARIOS) - i))
            commit = Commit(
                repository_id=repo.id,
                sha=sc["sha"],
                author_email=sc["author"],
                branch=sc["branch"],
                committed_at=committed_at,
                message=sc["message"],
                files_changed=sc["files_changed"],
                lines_added=sc["lines_added"],
                lines_deleted=sc["lines_deleted"],
                raw_metadata={"demo": True},
            )
            db.add(commit)
            db.flush()

            recs = generate_recommendations(
                res.predicted_class,
                risk_score=res.risk_score,
                class_probabilities=res.class_probabilities,
                predicted_memory_mb=res.predicted_memory_mb,
                predicted_duration_seconds=res.predicted_duration_seconds,
                features=feats_full,
            )
            decision = _decide(res.risk_score)
            pred = Prediction(
                commit_id=commit.id,
                repository_id=repo.id,
                model_version_id=mv.id,
                predicted_class=FailureClass(res.predicted_class),
                class_probabilities=res.class_probabilities,
                risk_score=res.risk_score,
                confidence=res.confidence,
                decision=decision,
                predicted_memory_mb=res.predicted_memory_mb,
                predicted_duration_min=res.predicted_duration_seconds / 60,
                feature_vector=feats_full,
                feature_importance=res.feature_importance,
                recommendations=_serialise_recommendations(recs),
                inference_time_ms=res.inference_time_ms,
            )
            db.add(pred)
            db.flush()
            print(
                f"  {sc['repo']:25s} sha={sc['sha'][:7]} → {res.predicted_class:20s}"
                f" risk={res.risk_score:.2f} decision={decision.value:12s} recs={len(recs)}"
            )
        db.commit()
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
