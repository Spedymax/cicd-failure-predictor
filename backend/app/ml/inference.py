"""Inference engine — wraps the trained joblib artefacts.

Loaded once at worker start (NFR-03 says < 5 s load time, < 100 ms per
prediction). Thread-safe for ``predict()``: scikit-learn / lightgbm
predictions don't mutate state.

Artefact layout (produced by the offline training pipeline):
    classifier_rf.joblib            — sklearn RandomForestClassifier
    classifier_lgb.joblib           — lightgbm.LGBMClassifier
    classifier_label_encoder.joblib — sklearn LabelEncoder
    regressor_memory.joblib         — lightgbm.LGBMRegressor (log1p target)
    regressor_duration.joblib       — lightgbm.LGBMRegressor (log1p target)
    feature_columns.json            — ordered list of feature names
    version.json                    — metadata (trained_at, n_train, classes)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pandas as pd

try:
    import shap

    _SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    shap = None  # type: ignore[assignment]
    _SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceResult:
    predicted_class: str
    risk_score: float
    confidence: float
    class_probabilities: dict[str, float]
    feature_importance: dict[str, float]
    predicted_memory_mb: float
    predicted_duration_seconds: float
    inference_time_ms: int
    # Signed per-feature SHAP explanation for the *risk* model in two-stage
    # mode (binary failure probability) or the predicted class in single-stage
    # mode. ``contributions`` are top-K entries ordered by |shap_value| desc.
    # Each item: {feature, value, shap_value} — shap_value > 0 pushes the
    # target probability up, < 0 pulls it down. None when SHAP unavailable.
    shap_explanation: dict | None = None


class InferenceEngine:
    def __init__(self, model_dir: Path):
        self._dir = Path(model_dir)
        self._lock = Lock()
        # Detect model layout: two-stage (risk + cause) or legacy single classifier.
        mtype_path = self._dir / "model_type.json"
        mtype = json.loads(mtype_path.read_text()).get("type") if mtype_path.exists() else "single"
        self._two_stage = mtype == "two_stage"

        if self._two_stage:
            self._risk_rf = joblib.load(self._dir / "risk_rf.joblib")
            self._risk_lgb = joblib.load(self._dir / "risk_lgb.joblib")
            self._cause_rf = joblib.load(self._dir / "cause_rf.joblib")
            self._cause_lgb = joblib.load(self._dir / "cause_lgb.joblib")
            self._cause_le = joblib.load(self._dir / "cause_label_encoder.joblib")
            self._cause_classes: list[str] = list(self._cause_le.classes_)
            self._classes = ["success"] + [c for c in self._cause_classes if c != "success"]
            self._rf = self._cause_rf
            self._label_encoder = self._cause_le
        else:
            self._rf = joblib.load(self._dir / "classifier_rf.joblib")
            self._lgb = joblib.load(self._dir / "classifier_lgb.joblib")
            self._label_encoder = joblib.load(self._dir / "classifier_label_encoder.joblib")
            self._classes = list(self._label_encoder.classes_)

        self._reg_memory = joblib.load(self._dir / "regressor_memory.joblib")
        self._reg_duration = joblib.load(self._dir / "regressor_duration.joblib")
        with open(self._dir / "feature_columns.json") as f:
            self._feature_names: list[str] = json.load(f)["feature_names"]
        version_path = self._dir / "version.json"
        self._version: dict = json.loads(version_path.read_text()) if version_path.exists() else {}
        self._rf_importances = self._rf.feature_importances_

        self._explainer = None
        self._risk_explainer = None
        self._risk_failure_idx: int | None = None
        if _SHAP_AVAILABLE:
            try:
                self._explainer = shap.TreeExplainer(self._rf)
                logger.info(
                    "SHAP TreeExplainer initialised for cause/single RF (%d classes, two_stage=%s)",
                    len(self._classes), self._two_stage,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("SHAP cause TreeExplainer init failed: %s", exc)
                self._explainer = None
            if self._two_stage:
                try:
                    self._risk_explainer = shap.TreeExplainer(self._risk_rf)
                    # risk_rf classes are [0, 1] where 1 = success (sklearn lexicographic).
                    # We want explanations for the *failure* class (label 0).
                    risk_classes = list(self._risk_rf.classes_)
                    self._risk_failure_idx = risk_classes.index(0) if 0 in risk_classes else 0
                    logger.info("SHAP TreeExplainer initialised for risk RF (binary)")
                except Exception as exc:  # pragma: no cover
                    logger.warning("SHAP risk TreeExplainer init failed: %s", exc)
                    self._risk_explainer = None
        else:  # pragma: no cover
            logger.warning("shap package not installed — using pseudo-importance fallback")

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    @property
    def version(self) -> dict:
        return dict(self._version)

    def safe_predict(self, features: dict[str, float]) -> InferenceResult:
        """Wrap predict() so a model failure never breaks the webhook pipeline.

        On any exception we return a degraded result (WARN-territory risk_score,
        ``other_failure`` class, empty SHAP) and log the error. Webhook continues
        to record the prediction so the user sees a non-zero response instead of
        an HTTP 500 stuck commit gate.
        """
        try:
            return self.predict(features)
        except Exception as exc:  # noqa: BLE001 - we deliberately catch all
            import logging

            logging.getLogger(__name__).exception("inference failed, degrading: %s", exc)
            classes = self._classes or ["success", "other_failure"]
            uniform = {c: 1.0 / len(classes) for c in classes}
            return InferenceResult(
                predicted_class="other_failure",
                risk_score=0.5,
                confidence=0.0,
                class_probabilities=uniform,
                feature_importance={},
                predicted_memory_mb=0.0,
                predicted_duration_seconds=0.0,
                inference_time_ms=0,
                shap_explanation=None,
            )

    def predict(self, features: dict[str, float]) -> InferenceResult:
        ordered = [float(features.get(name, 0.0) or 0.0) for name in self._feature_names]
        X = pd.DataFrame([ordered], columns=self._feature_names)
        start = time.perf_counter()
        with self._lock:
            if self._two_stage:
                # Stage 1: P(failure) from binary risk model
                risk_probs = (self._risk_rf.predict_proba(X)[0] + self._risk_lgb.predict_proba(X)[0]) / 2.0
                # Risk model: class 1 = success, class 0 = failure (sklearn lexicographic order)
                p_success = float(risk_probs[list(self._risk_rf.classes_).index(1)])
                p_fail = 1.0 - p_success
                # Stage 2: P(class | failure) from cause model
                cause_probs = (self._cause_rf.predict_proba(X)[0] + self._cause_lgb.predict_proba(X)[0]) / 2.0
                # Build unified 7-class distribution: P(success) + P(failure)·P(class|failure)
                class_probs = {"success": p_success}
                for c, p in zip(self._cause_classes, cause_probs, strict=True):
                    class_probs[c] = float(p_fail * p)
                # Surface the most likely failure cause class so the UI always
                # has actionable info; risk_score (separate field) governs
                # AUTO/WARN/BLOCK in the decision layer. The downstream
                # pipeline rewrites the label to "success" for empty diffs.
                idx = int(np.argmax(cause_probs))
                predicted_class = self._cause_classes[idx]
                confidence = float(cause_probs[idx])
                risk_score = float(p_fail)
            else:
                probs = (self._rf.predict_proba(X)[0] + self._lgb.predict_proba(X)[0]) / 2.0
                class_probs = {c: float(p) for c, p in zip(self._classes, probs, strict=True)}
                idx = int(np.argmax(probs))
                predicted_class = self._classes[idx]
                risk_score = float(1.0 - class_probs.get("success", 0.0))
                confidence = float(probs[idx])
            log_mem = self._reg_memory.predict(X)[0]
            log_dur = self._reg_duration.predict(X)[0]
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        importance = self._compute_shap_importance(X, idx)
        if not importance:
            importance = self._compute_local_importance(ordered)
        shap_explanation = self._compute_shap_explanation(X, idx)
        return InferenceResult(
            predicted_class=predicted_class,
            risk_score=risk_score,
            confidence=confidence,
            class_probabilities=class_probs,
            feature_importance=importance,
            predicted_memory_mb=float(np.expm1(log_mem)),
            predicted_duration_seconds=float(np.expm1(log_dur)),
            inference_time_ms=elapsed_ms,
            shap_explanation=shap_explanation,
        )

    def _compute_shap_explanation(
        self, X: pd.DataFrame, cause_class_idx: int, top_k: int = 10
    ) -> dict | None:
        """Return signed SHAP explanation for the most defence-relevant target.

        Two-stage mode: explain *risk_rf* binary classifier on the failure
        class — this is what drives AUTO/WARN/BLOCK via ``risk_score``.
        Single-stage mode: explain *_rf* on the predicted class.

        Output shape::

            {
              "target": "risk_failure" | "class:<name>",
              "base_value": float,        # E[f(x)] of model on training set
              "predicted_value": float,   # base_value + sum(shap_values)
              "contributions": [
                {"feature": str, "value": float, "shap_value": float},
                ... top_k by |shap_value| desc
              ]
            }

        Returns None when SHAP is unavailable.
        """
        if self._two_stage and self._risk_explainer is not None:
            explainer = self._risk_explainer
            class_idx = self._risk_failure_idx or 0
            target = "risk_failure"
        elif self._explainer is not None:
            explainer = self._explainer
            class_idx = cause_class_idx
            target = f"class:{self._classes[cause_class_idx]}"
        else:
            return None

        try:
            with self._lock:
                shap_values = explainer.shap_values(X)
                expected = explainer.expected_value
        except Exception as exc:  # pragma: no cover
            logger.warning("SHAP explanation failed: %s", exc)
            return None

        # Normalise shap_values to a 1-D array of length n_features for the
        # chosen class. TreeExplainer returns:
        #   list[np.ndarray] of length n_classes  (older shap), each (n_samples, n_features)
        #   OR np.ndarray of shape (n_samples, n_features, n_classes)
        #   OR np.ndarray of shape (n_samples, n_features)  (binary, output_margin)
        if isinstance(shap_values, list):
            class_shap = np.asarray(shap_values[class_idx])[0]
            base = float(np.asarray(expected)[class_idx]) if hasattr(expected, "__len__") else float(expected)
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                class_shap = arr[0, :, class_idx]
                base = float(np.asarray(expected)[class_idx]) if hasattr(expected, "__len__") else float(expected)
            elif arr.ndim == 2:
                class_shap = arr[0]
                base = float(np.asarray(expected).item()) if hasattr(expected, "item") else float(expected)
            else:
                return None

        class_shap = class_shap.astype(float)
        order = np.argsort(np.abs(class_shap))[::-1][:top_k]
        feature_values = X.iloc[0].to_dict()
        contributions = [
            {
                "feature": self._feature_names[i],
                "value": float(feature_values.get(self._feature_names[i], 0.0)),
                "shap_value": float(class_shap[i]),
            }
            for i in order
            if abs(float(class_shap[i])) > 0.0
        ]
        predicted_value = base + float(class_shap.sum())
        return {
            "target": target,
            "base_value": base,
            "predicted_value": predicted_value,
            "contributions": contributions,
        }

    def _compute_shap_importance(
        self, X: pd.DataFrame, class_idx: int, top_k: int = 8
    ) -> dict[str, float]:
        if self._explainer is None:
            return {}
        try:
            with self._lock:
                shap_values = self._explainer.shap_values(X)
        except Exception as exc:  # pragma: no cover
            logger.warning("SHAP shap_values failed: %s", exc)
            return {}

        if isinstance(shap_values, list):
            class_shap = np.asarray(shap_values[class_idx])[0]
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                class_shap = arr[0, :, class_idx]
            elif arr.ndim == 2:
                class_shap = arr[0]
            else:
                return {}

        abs_shap = np.abs(class_shap.astype(float))
        total = float(abs_shap.sum())
        if total == 0.0:
            return {}
        order = np.argsort(abs_shap)[::-1][:top_k]
        return {
            self._feature_names[i]: float(abs_shap[i] / total)
            for i in order
            if abs_shap[i] > 0
        }

    def _compute_local_importance(
        self, values: list[float], top_k: int = 8
    ) -> dict[str, float]:
        weighted: list[tuple[str, float]] = []
        total = 0.0
        for name, val, imp in zip(self._feature_names, values, self._rf_importances, strict=True):
            score = float(abs(val) * imp)
            weighted.append((name, score))
            total += score
        if total == 0:
            return {}
        weighted.sort(key=lambda kv: kv[1], reverse=True)
        return {name: float(score / total) for name, score in weighted[:top_k]}
