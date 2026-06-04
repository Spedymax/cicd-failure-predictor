"""Smoke tests for ``training.tune`` — verify Optuna study converges
on a tiny synthetic dataset and produces usable best params.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.tune import tune_classifier


@pytest.fixture
def tiny_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n_train = 200
    n_val = 60
    n_features = 8
    feature_names = [f"feat_{i}" for i in range(n_features)]

    def _build(n: int) -> tuple[pd.DataFrame, pd.Series]:
        X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=feature_names)
        # Label is a noisy sign of feat_0 + feat_1 — 3 classes for variety.
        score = X["feat_0"] + 0.5 * X["feat_1"] + rng.normal(scale=0.3, size=n)
        labels = np.where(score < -0.3, "fail_a", np.where(score < 0.3, "fail_b", "success"))
        return X, pd.Series(labels, name="failure_class")

    X_train, y_train = _build(n_train)
    X_val, y_val = _build(n_val)
    return X_train, y_train, X_val, y_val


def test_tune_classifier_returns_usable_best_params(tiny_dataset: tuple) -> None:
    X_train, y_train, X_val, y_val = tiny_dataset
    result = tune_classifier(
        X_train,
        y_train,
        X_val,
        y_val,
        n_trials=3,
        seed=7,
    )
    assert result.n_trials == 3
    assert result.n_completed >= 1
    assert 0.0 <= result.best_value <= 1.0
    assert "n_estimators" in result.best_rf_params
    assert "n_estimators" in result.best_lgb_params
    assert result.best_rf_params["random_state"] == 7
    assert result.best_lgb_params["num_class"] == y_train.nunique()


def test_tune_classifier_serialises_to_dict(tiny_dataset: tuple) -> None:
    X_train, y_train, X_val, y_val = tiny_dataset
    result = tune_classifier(X_train, y_train, X_val, y_val, n_trials=2, seed=11)
    payload = result.to_dict()
    assert {"best_val_f1_macro", "best_rf_params", "best_lgb_params", "trials"} <= payload.keys()
    assert len(payload["trials"]) == 2
    assert all("state" in t for t in payload["trials"])
