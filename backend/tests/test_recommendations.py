from app.ml.recommendations import generate_recommendations


def test_oom_recommendation_emitted():
    feats = {"feat_image_growth_ratio": 12.0}
    recs = generate_recommendations(
        "oom_killed",
        risk_score=0.95,
        class_probabilities={"oom_killed": 0.95, "success": 0.05},
        predicted_memory_mb=4200,
        predicted_duration_seconds=900,
        features=feats,
    )
    titles = [r.title for r in recs]
    assert any("OOMKilled" in t for t in titles)


def test_dependency_recommendation_when_many_new_deps():
    feats = {"feat_new_deps_count": 8}
    recs = generate_recommendations(
        "success",
        risk_score=0.10,
        class_probabilities={"success": 0.9},
        predicted_memory_mb=1000,
        predicted_duration_seconds=300,
        features=feats,
    )
    assert any(r.category == "DEPENDENCY" for r in recs)


def test_no_recommendation_for_clean_low_risk():
    feats = {}
    recs = generate_recommendations(
        "success",
        risk_score=0.05,
        class_probabilities={"success": 0.95},
        predicted_memory_mb=800,
        predicted_duration_seconds=180,
        features=feats,
    )
    assert recs == []


def test_recommendations_sorted_by_severity():
    feats = {
        "feat_image_growth_ratio": 8.0,
        "feat_new_deps_count": 6,
        "feat_has_dockerfile_change_int": 1,
        "feat_run_attempt_gt1": 1,
        "feat_run_attempt": 2,
    }
    recs = generate_recommendations(
        "oom_killed",
        risk_score=0.85,
        class_probabilities={"oom_killed": 0.7, "test_timeout": 0.2, "success": 0.1},
        predicted_memory_mb=4200,
        predicted_duration_seconds=2400,
        features=feats,
    )
    severities = [r.severity for r in recs]
    severity_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    assert severities == sorted(severities, key=lambda s: severity_rank.get(s, 99))
