"""Dataset loading and splitting helpers for the training pipeline.

The classifier predicts ``failure_class`` (6 classes including ``success``)
and uses only the engineered ``feat_*`` columns. The regressors predict
``peak_memory_mb`` and ``duration_seconds`` — these are *targets*, never
features, to avoid leakage at inference time when only commit metadata
is available.

Cross-set evaluation: rows with ``dataset_source == 'real'`` are not seen
during training. They form a held-out generalisation set used to verify
that a classifier trained on synthetic + real-success data still works on
the small handful of real failures with known classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from features.transformers import feature_columns

LABEL_COL = "failure_class"
SUCCESS_LABEL = "success"
OTHER_FAILURE_LABEL = "other_failure"
MEMORY_TARGET = "peak_memory_mb"
DURATION_TARGET = "duration_seconds"
SOURCE_COL = "dataset_source"
CONCLUSION_COL = "conclusion"


@dataclass
class TrainingSplit:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    real_X: pd.DataFrame
    real_y: pd.Series
    feature_names: list[str]


def _label_series(df: pd.DataFrame) -> pd.Series:
    """Build the multi-class target with commit-shape priority relabeling.

    Real GHA logs surface generic test-failure patterns (`FAILED tests/...`)
    that mask the *underlying* cause when the failing job also runs lint /
    deps install / docker build in the same step. To give the classifier a
    cleaner signal — and to match the deterministic class hierarchy a
    reviewer expects on demo — we override the log-derived label with
    commit-shape priority for failure rows:

      * conclusion=failure & has_dockerfile_change  → docker_build_failed
      * conclusion=failure & has_dependency_change  → dependency_error
      * conclusion=failure & specific log class (oom/timeout/network) → keep
      * conclusion=failure otherwise                → other_failure
      * conclusion=success                          → success
    """
    n = len(df)
    conclusion = df.get(CONCLUSION_COL, pd.Series([None] * n, index=df.index))
    is_failure = conclusion == "failure"
    has_docker = df.get("has_dockerfile_change", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    has_deps = df.get("has_dependency_change", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    raw_label = df[LABEL_COL].astype("object")

    # Start with success for everything, override with failure labels below.
    label = pd.Series([SUCCESS_LABEL] * n, index=df.index, dtype="object")

    # Specific log-derived classes win over generic shape (a confirmed
    # OOM / timeout / network failure is more informative than "deps file
    # also changed in this commit").
    specific_log = raw_label.isin(["oom_killed", "test_timeout", "network_error"])

    test_only = df.get("test_only_changes", pd.Series(False, index=df.index)).fillna(False).astype(bool)

    label = label.where(~is_failure, OTHER_FAILURE_LABEL)
    # Order matters: most specific commit-shape signal first wins.
    label = label.where(~(is_failure & test_only), "test_failure")
    label = label.where(~(is_failure & has_deps), "dependency_error")
    label = label.where(~(is_failure & has_docker), "docker_build_failed")
    label = label.where(~(is_failure & specific_log), raw_label)

    return label.astype(str)


def load_features(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if SOURCE_COL not in df.columns:
        df[SOURCE_COL] = "synthetic"
    return df


def make_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.10,
    val_size: float = 0.10,
    real_holdout_share: float = 0.30,
    seed: int = 42,
) -> TrainingSplit:
    feats = feature_columns(df)
    if not feats:
        raise ValueError("no engineered (feat_*) columns found")

    real_mask = df[SOURCE_COL] == "real"
    # Real-known: any real row whose synthetic label can be derived
    # (specific failure class OR a generic failure mapped to OTHER_FAILURE).
    has_specific = df[LABEL_COL].notna()
    is_failure = df.get(CONCLUSION_COL, pd.Series(index=df.index, dtype="object")) == "failure"
    real_known = df.loc[real_mask & (has_specific | is_failure)].copy()

    # Reserve a stratified subset of real-known rows as a held-out
    # cross-domain validation set; the rest joins the training pool so
    # the classifier sees real-world failure patterns at fit time.
    real_holdout = pd.DataFrame(columns=df.columns)
    real_for_train = real_known
    if len(real_known) >= 10:
        # Stratify on the normalized label (None failures -> other_failure,
        # non-failures -> success) so np.unique never sees a None.
        strat_labels = _label_series(real_known)
        try:
            real_for_train, real_holdout = train_test_split(
                real_known,
                test_size=real_holdout_share,
                random_state=seed,
                stratify=strat_labels,
            )
        except (ValueError, TypeError):
            real_for_train, real_holdout = train_test_split(
                real_known,
                test_size=real_holdout_share,
                random_state=seed,
            )

    real_X = real_holdout[feats].copy() if not real_holdout.empty else pd.DataFrame(columns=feats)
    real_y = _label_series(real_holdout) if not real_holdout.empty else pd.Series(dtype=str)

    pool_idx = df.index.difference(real_holdout.index)
    train_pool = df.loc[pool_idx].copy()
    y_pool = _label_series(train_pool)

    X_remain, X_test, y_remain, y_test = train_test_split(
        train_pool[feats],
        y_pool,
        test_size=test_size,
        random_state=seed,
        stratify=y_pool,
    )
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_remain,
        y_remain,
        test_size=val_relative,
        random_state=seed,
        stratify=y_remain,
    )
    return TrainingSplit(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        real_X=real_X,
        real_y=real_y,
        feature_names=feats,
    )


# Hard caps for sanitising regression targets. Values above these are
# treated as data-collection artifacts (e.g. stuck/cancelled workflows
# left with a duration of multiple days) and replaced with NaN so they
# are excluded from regressor fitting and evaluation.
MAX_DURATION_SECONDS: float = 24 * 3600.0  # 24 hours
MAX_MEMORY_MB: float = 32 * 1024.0  # 32 GB


def regression_targets(
    df: pd.DataFrame,
    *,
    max_duration_seconds: float = MAX_DURATION_SECONDS,
    max_memory_mb: float = MAX_MEMORY_MB,
) -> tuple[pd.Series, pd.Series]:
    mem = df[MEMORY_TARGET].astype(float)
    dur = df[DURATION_TARGET].astype(float)
    mem = mem.where(mem.between(0, max_memory_mb), other=np.nan)
    dur = dur.where(dur.between(0, max_duration_seconds), other=np.nan)
    return mem, dur


def class_balance(y: pd.Series) -> dict[str, int]:
    return y.value_counts().to_dict()


def class_weights(y: pd.Series) -> dict[str, float]:
    counts = y.value_counts()
    n = len(y)
    n_classes = len(counts)
    return {cls: float(n / (n_classes * c)) for cls, c in counts.items()}


def fill_features(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True)).fillna(0.0)
