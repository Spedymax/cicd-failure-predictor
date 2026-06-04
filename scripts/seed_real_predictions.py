"""Seed predictions from the collected real GHA dataset.

Reads ``data/raw/gha_runs.parquet``, runs the same feature engineering
pipeline as training, predicts each row with the active model, and
persists the result as a real ``Prediction`` row with a clickable
GitHub commit URL.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for line in (REPO_ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
os.environ.setdefault("ML_MODEL_DIR", str(REPO_ROOT / "data" / "artifacts" / "v26_5class"))

sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "ml"))

from features.transformers import engineer_features, feature_columns  # noqa: E402

from app.api.dependencies import get_inference_engine  # noqa: E402
from app.db.models import (  # noqa: E402
    Commit,
    FailureClass,
    Prediction,
    Repository,
)
from app.db.session import SessionLocal  # noqa: E402
from app.ml.recommendations import generate_recommendations  # noqa: E402
from app.services.prediction_pipeline import (  # noqa: E402
    _decide,
    _resolve_thresholds,
    _ensure_active_model_version,
    _ensure_repository,
    _serialise_recommendations,
)

MAX_PER_REPO = 30


def _purge_real(db) -> int:
    from sqlalchemy import select

    rows = (
        db.execute(
            select(Repository.id).where(
                ~Repository.full_name.like("acme/%"),
                ~Repository.full_name.like("demo/%"),
                Repository.full_name != "Spedymax/cicd-predictor-demo",
            )
        )
        .scalars()
        .all()
    )
    if not rows:
        return 0
    n_pred = (
        db.query(Prediction).filter(Prediction.repository_id.in_(rows)).delete(synchronize_session=False)
    )
    n_commit = db.query(Commit).filter(Commit.repository_id.in_(rows)).delete(synchronize_session=False)
    db.flush()
    print(f"purged {n_pred} predictions, {n_commit} commits across {len(rows)} real repos")
    return n_pred


def main() -> int:
    purge = "--purge" in sys.argv
    raw_path = REPO_ROOT / "data" / "raw" / "gha_runs.parquet"
    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found")
        return 2

    df = pd.read_parquet(raw_path)
    if df.empty:
        print("ERROR: empty dataset")
        return 2

    # Ranks: failed-with-known-class first, then other failures, then successes.
    df["_class_rank"] = df["failure_class"].notna().astype(int) * 2 + (df["conclusion"] == "failure").astype(int)
    df = df.sort_values(["_class_rank", "created_at"], ascending=[False, False])
    df = df.groupby("repo", group_keys=False).head(MAX_PER_REPO).reset_index(drop=True)
    df = df.drop(columns="_class_rank")
    print(f"selected {len(df)} runs from {df['repo'].nunique()} repos")
    print(f"  with known failure_class: {df['failure_class'].notna().sum()}")
    print(f"  unlabeled failures:       {((df['failure_class'].isna()) & (df['conclusion']=='failure')).sum()}")
    print(f"  successes:                {(df['conclusion']=='success').sum()}")

    engineered = engineer_features(df)
    feats = feature_columns(engineered)

    engine = get_inference_engine()
    db = SessionLocal()
    try:
        if purge:
            _purge_real(db)
            db.commit()
        mv = _ensure_active_model_version(db, engine)
        # Resolve thresholds once from the default Policy (one DB call vs per-row).
        thresholds = _resolve_thresholds(db, None)
        print(f"using thresholds (auto, block) = {thresholds}")
        n_written = 0
        for idx, row in engineered.iterrows():
            sha = str(row.get("head_sha") or "").strip()
            repo_full = str(row.get("repo") or "").strip()
            if not sha or not repo_full:
                continue

            committed_at = row.get("created_at")
            if pd.isna(committed_at):
                committed_at = datetime.now(tz=timezone.utc)
            elif isinstance(committed_at, pd.Timestamp):
                committed_at = committed_at.to_pydatetime()
            if committed_at.tzinfo is None:
                committed_at = committed_at.replace(tzinfo=timezone.utc)

            author_email = str(row.get("actor_email") or "unknown@github") or "unknown@github"
            branch = row.get("head_branch") or None

            existing = db.query(Commit).filter_by(sha=sha).first()
            if existing is not None:
                continue

            repo = _ensure_repository(
                db,
                full_name=repo_full,
                html_url=f"https://github.com/{repo_full}",
            )

            commit = Commit(
                repository_id=repo.id,
                sha=sha,
                author_email=author_email,
                author_name=row.get("actor_login"),
                message=None,
                branch=str(branch) if branch is not None else None,
                committed_at=committed_at,
                files_changed=int(row.get("files_changed") or 0),
                lines_added=int(row.get("lines_added") or 0),
                lines_deleted=int(row.get("lines_deleted") or 0),
                raw_metadata={
                    "source": "real_gha",
                    "run_id": int(row["run_id"]),
                    "workflow_name": str(row.get("workflow_name") or ""),
                    "external_run_html_url": (
                        f"https://github.com/{repo_full}/actions/runs/{int(row['run_id'])}"
                    ),
                },
            )
            db.add(commit)
            db.flush()

            def _clean(v: object) -> float:
                if v is None:
                    return 0.0
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    return 0.0
                return 0.0 if (fv != fv) else fv  # NaN check

            features_dict = {n: _clean(row.get(n)) for n in feats}
            features_dict = {n: features_dict.get(n, 0.0) for n in engine.feature_names}
            res = engine.predict(features_dict)
            cleaned_importance = {k: _clean(v) for k, v in res.feature_importance.items()}
            cleaned_probs = {k: _clean(v) for k, v in res.class_probabilities.items()}

            # The cause classifier is trained on failure-only rows so
            # ``res.predicted_class`` is always one of the failure types.
            # For rows where two-stage P(success) dominates the joint
            # distribution, surface ``success`` as the predicted class —
            # matching what ``_resolve_predicted_class`` does in the live
            # webhook flow (which we cannot reuse here because the offline
            # GHA dataset lacks raw diff fields).
            top_class = max(cleaned_probs, key=cleaned_probs.get) if cleaned_probs else res.predicted_class
            predicted_class_value = top_class

            recs = generate_recommendations(
                predicted_class_value,
                risk_score=res.risk_score,
                class_probabilities=cleaned_probs,
                predicted_memory_mb=res.predicted_memory_mb,
                predicted_duration_seconds=res.predicted_duration_seconds,
                features=features_dict,
            )
            decision = _decide(
                res.risk_score, thresholds,
                predicted_class=predicted_class_value,
                confidence=res.confidence,
            )
            actual = row.get("failure_class")
            if isinstance(actual, str) and actual:
                actual_outcome = FailureClass(str(actual))
            elif row.get("conclusion") == "success":
                actual_outcome = FailureClass.SUCCESS
            elif row.get("conclusion") == "failure":
                actual_outcome = FailureClass.OTHER_FAILURE
            else:
                actual_outcome = None

            pred = Prediction(
                commit_id=commit.id,
                repository_id=repo.id,
                model_version_id=mv.id,
                predicted_class=FailureClass(predicted_class_value),
                class_probabilities=cleaned_probs,
                risk_score=_clean(res.risk_score),
                confidence=_clean(res.confidence),
                decision=decision,
                predicted_memory_mb=_clean(res.predicted_memory_mb),
                predicted_duration_min=_clean(res.predicted_duration_seconds) / 60,
                feature_vector=features_dict,
                feature_importance=cleaned_importance,
                recommendations=_serialise_recommendations(recs),
                inference_time_ms=res.inference_time_ms,
                actual_outcome=actual_outcome,
            )
            db.add(pred)
            n_written += 1
            if n_written % 25 == 0:
                db.flush()
                print(f"  ... persisted {n_written}")
        db.commit()
        print(f"DONE: {n_written} real predictions stored")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
