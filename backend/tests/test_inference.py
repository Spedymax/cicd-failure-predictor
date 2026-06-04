from pathlib import Path

import pytest

from app.ml.inference import InferenceEngine

ARTIFACTS = Path(__file__).resolve().parents[2] / "data" / "artifacts" / "v2"


@pytest.fixture(scope="module")
def engine() -> InferenceEngine:
    if not (ARTIFACTS / "classifier_rf.joblib").exists():
        pytest.skip("trained artefacts not available")
    return InferenceEngine(ARTIFACTS)


def test_engine_loads(engine: InferenceEngine) -> None:
    assert len(engine.feature_names) >= 25
    assert "success" in engine.classes
    assert "oom_killed" in engine.classes


def test_zero_vector_predicts_low_risk(engine: InferenceEngine) -> None:
    res = engine.predict({n: 0.0 for n in engine.feature_names})
    assert res.predicted_class in engine.classes
    assert 0.0 <= res.risk_score <= 1.0
    assert res.inference_time_ms < 500


def test_oom_like_features_predict_oom(engine: InferenceEngine) -> None:
    feats = {n: 0.0 for n in engine.feature_names}
    feats.update(
        {
            "feat_lines_changed_log": 6.0,
            "feat_files_changed_log": 3.5,
            "feat_image_growth_ratio": 11.0,
            "feat_has_dependency_change_int": 1,
            "feat_new_deps_count": 4,
            "feat_new_deps_size_mb_log": 5.0,
            "feat_author_success_rate": 0.4,
            "feat_project_failure_rate": 0.5,
            "feat_n_jobs_log": 2.0,
            "feat_longest_job_seconds_log": 7.0,
        }
    )
    res = engine.predict(feats)
    assert res.predicted_class == "oom_killed"
    assert res.risk_score > 0.7
    assert res.predicted_memory_mb > 1500
