"""safe_predict() must never raise — failing model returns a degraded result (Day 12)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.api.dependencies import get_inference_engine


@pytest.fixture(scope="module")
def engine():
    return get_inference_engine()


def test_safe_predict_degraded_on_exception(engine) -> None:
    # Force predict() to blow up; safe_predict must catch and return a fallback.
    with patch.object(type(engine), "predict", side_effect=RuntimeError("model gone")):
        result = engine.safe_predict({})
    assert result.predicted_class == "other_failure"
    assert result.confidence == 0.0
    assert 0.0 <= result.risk_score <= 1.0
    assert result.shap_explanation is None
    # class_probabilities must still be a normalized-ish dict the API can serialize.
    assert isinstance(result.class_probabilities, dict)
    assert result.class_probabilities
