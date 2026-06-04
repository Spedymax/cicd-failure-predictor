"""Classifier training: RandomForest + LightGBM ensemble with class weights.

The two models are trained independently and combined at inference by
averaging their ``predict_proba`` outputs (soft-voting). Both use
``class_weight='balanced'`` so the rare classes (network_error,
docker_build_failed) contribute proportionally during loss computation.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder


@dataclass
class EnsembleClassifier:
    label_encoder: LabelEncoder
    rf: RandomForestClassifier
    lgb_model: lgb.LGBMClassifier
    feature_names: list[str]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        rf_probs = self.rf.predict_proba(X)
        lgb_probs = self.lgb_model.predict_proba(X)
        return (rf_probs + lgb_probs) / 2.0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = probs.argmax(axis=1)
        return self.label_encoder.inverse_transform(idx)

    @property
    def classes_(self) -> np.ndarray:
        return self.label_encoder.classes_


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    rf_params: dict | None = None,
    lgb_params: dict | None = None,
    seed: int = 42,
) -> EnsembleClassifier:
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)

    rf_defaults = {
        "n_estimators": 300,
        "max_depth": 18,
        "min_samples_leaf": 4,
        "class_weight": "balanced",
        "random_state": seed,
        "n_jobs": -1,
    }
    rf_defaults.update(rf_params or {})
    rf = RandomForestClassifier(**rf_defaults)
    rf.fit(X_train, y_train_enc)

    lgb_defaults = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "class_weight": "balanced",
        "objective": "multiclass",
        "num_class": len(le.classes_),
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }
    lgb_defaults.update(lgb_params or {})
    lgb_model = lgb.LGBMClassifier(**lgb_defaults)
    lgb_model.fit(
        X_train,
        y_train_enc,
        eval_set=[(X_val, y_val_enc)],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )

    return EnsembleClassifier(le, rf, lgb_model, list(X_train.columns))


def evaluate_classifier(
    model: EnsembleClassifier,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    probs = model.predict_proba(X)
    preds = model.predict(X)
    f1_macro = float(f1_score(y, preds, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y, preds, average="weighted", zero_division=0))
    report = classification_report(y, preds, zero_division=0, output_dict=True)
    rf_preds = model.label_encoder.inverse_transform(model.rf.predict(X))
    lgb_preds = model.label_encoder.inverse_transform(model.lgb_model.predict(X))
    return {
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_rf_macro": float(f1_score(y, rf_preds, average="macro", zero_division=0)),
        "f1_lgb_macro": float(f1_score(y, lgb_preds, average="macro", zero_division=0)),
        "per_class": report,
        "n_samples": int(len(y)),
    }
