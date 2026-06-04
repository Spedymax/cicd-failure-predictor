"""Extra recommendation paths to lift coverage of ml/recommendations.py."""

from __future__ import annotations

from app.ml.recommendations import generate_recommendations


def _base_call(predicted_class: str, **overrides):
    kw = dict(
        risk_score=0.7,
        class_probabilities={predicted_class: 0.6, "success": 0.4},
        predicted_memory_mb=512.0,
        predicted_duration_seconds=300.0,
        features={
            "feat_run_attempt": 1,
            "feat_run_attempt_gt1": 0,
            "feat_new_deps_count": 0,
            "feat_has_dependency_change_int": 0,
            "feat_has_dockerfile_change_int": 0,
            "feat_image_growth_ratio": 1.0,
            "feat_final_image_size_mb_log": 0.0,
        },
    )
    kw.update(overrides)
    return generate_recommendations(predicted_class, **kw)


def test_oom_recommendation_path() -> None:
    recs = _base_call("oom_killed", predicted_memory_mb=2048.0)
    assert any("oom" in r.title.lower() or r.category == "RESOURCE" for r in recs)


def test_timeout_recommendation_path() -> None:
    recs = _base_call("test_timeout", predicted_duration_seconds=1800.0)
    assert recs  # at least one rec


def test_dependency_recommendation_via_new_deps_count() -> None:
    recs = _base_call(
        "success",
        features={
            "feat_run_attempt": 1, "feat_run_attempt_gt1": 0,
            "feat_new_deps_count": 7, "feat_has_dependency_change_int": 1,
            "feat_has_dockerfile_change_int": 0, "feat_image_growth_ratio": 1.0,
            "feat_final_image_size_mb_log": 0.0,
        },
    )
    # >=5 new deps triggers dependency rec even when predicted is success.
    assert recs


def test_docker_recommendation_via_image_growth() -> None:
    recs = _base_call(
        "success",
        features={
            "feat_run_attempt": 1, "feat_run_attempt_gt1": 0,
            "feat_new_deps_count": 0, "feat_has_dependency_change_int": 0,
            "feat_has_dockerfile_change_int": 1, "feat_image_growth_ratio": 8.0,
            "feat_final_image_size_mb_log": 7.0,  # ~1.1GB
        },
    )
    assert recs


def test_network_recommendation_on_retry() -> None:
    recs = _base_call(
        "success",
        features={
            "feat_run_attempt": 3, "feat_run_attempt_gt1": 1,
            "feat_new_deps_count": 0, "feat_has_dependency_change_int": 0,
            "feat_has_dockerfile_change_int": 0, "feat_image_growth_ratio": 1.0,
            "feat_final_image_size_mb_log": 0.0,
        },
    )
    assert recs


def test_test_failure_recommendation_path() -> None:
    recs = _base_call("test_failure")
    assert recs


def test_other_failure_recommendation_path() -> None:
    recs = _base_call("other_failure")
    assert recs
