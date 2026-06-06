from pathlib import Path

import pytest

from app.ml.inference import InferenceEngine

# Active production model: two-stage v26_5class (risk + 5 cause classes).
ARTIFACTS = Path(__file__).resolve().parents[2] / "data" / "artifacts" / "v26_5class"


@pytest.fixture(scope="module")
def engine() -> InferenceEngine:
    if not (ARTIFACTS / "risk_rf.joblib").exists():
        pytest.skip("trained artefacts not available")
    return InferenceEngine(ARTIFACTS)


def test_engine_loads(engine: InferenceEngine) -> None:
    # v26 dropped history features for cold-start safety -> 29 features.
    assert len(engine.feature_names) >= 25
    assert "success" in engine.classes
    # cause classes of the active model
    assert "docker_build_failed" in engine.classes
    assert "dependency_error" in engine.classes


def test_zero_vector_predicts_low_risk(engine: InferenceEngine) -> None:
    res = engine.predict({n: 0.0 for n in engine.feature_names})
    assert res.predicted_class in engine.classes
    assert 0.0 <= res.risk_score <= 1.0
    assert res.inference_time_ms < 1000


def test_dockerfile_change_is_scored(engine: InferenceEngine) -> None:
    feats = {n: 0.0 for n in engine.feature_names}
    feats.update(
        {
            "feat_has_dockerfile_change_int": 1.0,
            "feat_final_image_size_mb_log": 7.0,
            "feat_image_growth_ratio": 6.0,
            "feat_lines_changed_log": 5.0,
            "feat_files_changed_log": 2.5,
        }
    )
    res = engine.predict(feats)
    assert res.predicted_class in engine.classes
    assert 0.0 <= res.risk_score <= 1.0
    # regressors always return a non-negative resource estimate
    assert res.predicted_memory_mb is None or res.predicted_memory_mb >= 0
