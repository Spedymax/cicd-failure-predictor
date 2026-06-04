"""Regression heads: predict peak memory and run duration in MB / seconds.

Both targets have long right tails (multi-hour builds, multi-GB peak
memory) — squared-error fitting on log1p alone is fooled by extreme
outliers. We mitigate it with two techniques:

1. **Winsorization** — clip the target above the 99-th percentile of the
   training set before applying log1p. The clip is computed only on
   ``y_train`` and applied to val/test prior to fitting (val/test labels
   used for evaluation stay raw — we report MAE / R² on the original
   scale).
2. **Huber objective** — LightGBM with ``objective='huber'`` is robust
   to outliers in the remaining tail.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

DEFAULT_WINSORIZE_UPPER_PCT: float = 99.0


@dataclass
class RegressionHead:
    target: str
    model: lgb.LGBMRegressor
    feature_names: list[str]
    winsorize_upper: float | None = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


def _winsorize_upper(y: pd.Series, cap: float | None) -> pd.Series:
    if cap is None:
        return y.clip(lower=0)
    return y.clip(lower=0, upper=cap)


def train_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    target_name: str,
    seed: int = 42,
    winsorize_upper_pct: float | None = DEFAULT_WINSORIZE_UPPER_PCT,
    objective: str = "huber",
    huber_alpha: float = 0.9,
) -> RegressionHead:
    cap = (
        float(np.percentile(y_train.dropna(), winsorize_upper_pct))
        if winsorize_upper_pct is not None and len(y_train.dropna()) > 0
        else None
    )
    log_y_train = np.log1p(_winsorize_upper(y_train, cap))
    log_y_val = np.log1p(_winsorize_upper(y_val, cap))

    params: dict = {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
        "objective": objective,
    }
    if objective == "huber":
        params["alpha"] = huber_alpha

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        log_y_train,
        eval_set=[(X_val, log_y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )
    return RegressionHead(
        target=target_name,
        model=model,
        feature_names=list(X_train.columns),
        winsorize_upper=cap,
    )


def evaluate_regressor(reg: RegressionHead, X: pd.DataFrame, y: pd.Series) -> dict:
    log_pred = reg.predict(X)
    pred = np.expm1(log_pred)
    return {
        "target": reg.target,
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "median_pred": float(np.median(pred)),
        "median_actual": float(np.median(y)),
        "n_samples": int(len(y)),
        "winsorize_upper": reg.winsorize_upper,
    }
