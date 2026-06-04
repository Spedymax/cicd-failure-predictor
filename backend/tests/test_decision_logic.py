"""Unit tests for the rule-overlay decision logic + class resolver."""

from __future__ import annotations

from app.db.models import PredictionDecision
from app.services.prediction_pipeline import (
    DEFAULT_THRESHOLDS,
    _decide,
    _resolve_predicted_class,
)


# ---------- _resolve_predicted_class ----------

def test_resolve_empty_diff_is_success() -> None:
    out = _resolve_predicted_class(
        "docker_build_failed",
        {"docker_build_failed": 0.9, "success": 0.1},
        {"files_changed": 0, "lines_added": 0, "lines_deleted": 0},
    )
    assert out == "success"


def test_resolve_trivial_diff_is_success() -> None:
    out = _resolve_predicted_class(
        "other_failure", {"other_failure": 0.7, "success": 0.3},
        {
            "files_changed": 1, "lines_added": 3, "lines_deleted": 1,
            "has_dockerfile_change": False, "has_dependency_change": False,
            "has_test_only_changes": False,
        },
    )
    assert out == "success"


def test_resolve_keeps_cause_when_diff_is_risky() -> None:
    out = _resolve_predicted_class(
        "docker_build_failed", {"docker_build_failed": 0.9},
        {
            "files_changed": 1, "lines_added": 50, "lines_deleted": 10,
            "has_dockerfile_change": True,
            "has_dependency_change": False,
            "has_test_only_changes": False,
        },
    )
    assert out == "docker_build_failed"


# ---------- _decide ----------

def test_decide_hard_cap_huge_pr_blocks() -> None:
    out = _decide(
        risk_score=0.0,  # even with zero risk
        features={"files_changed": 200, "lines_added": 0, "lines_deleted": 0},
        predicted_class="success", confidence=0.99,
    )
    assert out == PredictionDecision.BLOCK


def test_decide_success_class_is_auto_approve_even_high_risk() -> None:
    out = _decide(
        risk_score=0.95, predicted_class="success", confidence=0.99,
        features={"files_changed": 0, "lines_added": 0, "lines_deleted": 0},
    )
    assert out == PredictionDecision.AUTO_APPROVE


def test_decide_docker_class_blocks_when_ml_agrees() -> None:
    out = _decide(
        risk_score=0.8, predicted_class="docker_build_failed", confidence=0.8,
        features={"files_changed": 2, "lines_added": 30, "lines_deleted": 5},
    )
    assert out == PredictionDecision.BLOCK


def test_decide_docker_class_warns_when_ml_disagrees() -> None:
    # Risk < 0.50 → ML thinks success → rule alone shouldn't BLOCK.
    out = _decide(
        risk_score=0.10, predicted_class="docker_build_failed", confidence=0.7,
        features={"files_changed": 1, "lines_added": 5, "lines_deleted": 0},
    )
    assert out == PredictionDecision.WARN


def test_decide_docker_class_softened_to_warn_by_preflight() -> None:
    out = _decide(
        risk_score=0.9, predicted_class="docker_build_failed", confidence=0.8,
        features={"files_changed": 2, "lines_added": 10, "lines_deleted": 0},
        validation={"docker_valid": True},
    )
    assert out == PredictionDecision.WARN


def test_decide_deps_class_softened_to_auto_when_pypi_resolves() -> None:
    out = _decide(
        risk_score=0.9, predicted_class="dependency_error", confidence=0.8,
        features={"files_changed": 1, "lines_added": 1, "lines_deleted": 0},
        validation={"deps_valid": True},
    )
    assert out == PredictionDecision.AUTO_APPROVE


def test_decide_falls_back_to_thresholds_on_other_failure() -> None:
    auto, block = DEFAULT_THRESHOLDS
    out_low = _decide(
        risk_score=auto - 0.01, predicted_class="other_failure", confidence=0.0,
        features={"files_changed": 1, "lines_added": 1, "lines_deleted": 0},
    )
    out_mid = _decide(
        risk_score=(auto + block) / 2, predicted_class="other_failure", confidence=0.0,
        features={"files_changed": 1, "lines_added": 1, "lines_deleted": 0},
    )
    out_hi = _decide(
        risk_score=block + 0.01, predicted_class="other_failure", confidence=0.0,
        features={"files_changed": 1, "lines_added": 1, "lines_deleted": 0},
    )
    assert out_low == PredictionDecision.AUTO_APPROVE
    assert out_mid == PredictionDecision.WARN
    assert out_hi == PredictionDecision.BLOCK
