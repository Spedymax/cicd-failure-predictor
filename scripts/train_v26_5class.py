#!/usr/bin/env python3
"""Train v26_5class — two-stage with reduced cause-class set.

Drops oom_killed and network_error (each had <15 holdout samples and F1
<0.15 on v20). Those labels are absorbed into other_failure unless they
match the dockerfile / dependency / test-only commit-shape rules.

Cause classes after merge (5):
  dependency_error, docker_build_failed, test_failure, test_timeout, other_failure
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

from features.transformers import feature_columns  # noqa: E402

DROP_FEATS = {
    "feat_author_success_rate", "feat_author_n_runs_log", "feat_author_avg_duration_log",
    "feat_project_failure_rate", "feat_project_n_runs_log", "feat_repo_failure_rate_global",
    "feat_n_jobs_log", "feat_longest_job_seconds_log",
}

CAUSE_KEEP = {"dependency_error", "docker_build_failed", "test_failure", "test_timeout"}
# oom_killed, network_error -> other_failure


def labels_5class(df: pd.DataFrame) -> pd.Series:
    """Same logic as training.data._label_series but with the
    ``specific_log`` set reduced to {test_timeout} — oom_killed and
    network_error are no longer surfaced as their own class."""
    n = len(df)
    conclusion = df.get("conclusion", pd.Series([None] * n, index=df.index))
    is_failure = conclusion == "failure"
    has_docker = df.get("has_dockerfile_change", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    has_deps = df.get("has_dependency_change", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    test_only = df.get("test_only_changes", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    raw_label = df["failure_class"].astype("object") if "failure_class" in df.columns else pd.Series(["other_failure"] * n, index=df.index)
    specific_log = raw_label.isin(["test_timeout"])  # only keep test_timeout

    label = pd.Series(["success"] * n, index=df.index, dtype="object")
    label = label.where(~is_failure, "other_failure")
    label = label.where(~(is_failure & test_only), "test_failure")
    label = label.where(~(is_failure & has_deps), "dependency_error")
    label = label.where(~(is_failure & has_docker), "docker_build_failed")
    label = label.where(~(is_failure & specific_log), raw_label)
    return label.astype(str)


def main() -> int:
    out = ROOT / "data/artifacts/v26_5class"
    out.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(ROOT / "data/processed/features_real_only.parquet")
    feats = [c for c in feature_columns(base) if c not in DROP_FEATS]
    print(f"feature_count={len(feats)} rows={len(base)}")

    labels = labels_5class(base)
    print(f"label dist:\n{labels.value_counts().to_string()}")

    X = base.reindex(columns=feats).fillna(0.0).replace([np.inf, -np.inf], 0.0)

    # --- risk model (binary) ---
    y_bin = (labels == "success").astype(int)
    X_tr, X_te, yb_tr, yb_te = train_test_split(
        X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )
    rf_r = RandomForestClassifier(
        n_estimators=400, max_depth=20, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ).fit(X_tr, yb_tr)
    lgb_r = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
    ).fit(X_tr, yb_tr)
    pred_r = ((rf_r.predict_proba(X_te) + lgb_r.predict_proba(X_te)) / 2).argmax(axis=1)
    p, r, f, _ = precision_recall_fscore_support(yb_te, pred_r, average="binary", pos_label=0, zero_division=0)
    print(f"\n== RISK MODEL ==\n  accuracy={(pred_r == yb_te).mean():.4f} F1(fail)={f:.4f} P={p:.4f} R={r:.4f}")

    # --- cause model ---
    fail_mask = labels != "success"
    Xc = X.loc[fail_mask]
    yc = labels.loc[fail_mask]
    print(f"\ncause split (failures only): n={len(Xc)} dist={yc.value_counts().to_dict()}")
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        Xc, yc, test_size=0.2, random_state=42, stratify=yc
    )
    le = LabelEncoder()
    yc_tr_enc = le.fit_transform(yc_tr)
    yc_te_enc = le.transform(yc_te)
    rf_c = RandomForestClassifier(
        n_estimators=400, max_depth=20, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ).fit(Xc_tr, yc_tr_enc)
    lgb_c = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        class_weight="balanced", objective="multiclass",
        num_class=len(le.classes_), random_state=42, n_jobs=-1, verbose=-1,
    ).fit(Xc_tr, yc_tr_enc)
    probs = (rf_c.predict_proba(Xc_te) + lgb_c.predict_proba(Xc_te)) / 2
    pred_c = probs.argmax(axis=1)
    print(f"\n== CAUSE MODEL (5 classes) ==")
    print(f"  F1-macro    = {f1_score(yc_te_enc, pred_c, average='macro'):.4f}")
    print(f"  F1-weighted = {f1_score(yc_te_enc, pred_c, average='weighted'):.4f}")
    print(f"  accuracy    = {(pred_c == yc_te_enc).mean():.4f}")
    print(classification_report(yc_te_enc, pred_c, target_names=list(le.classes_), zero_division=0, digits=3))

    joblib.dump(rf_r, out / "risk_rf.joblib")
    joblib.dump(lgb_r, out / "risk_lgb.joblib")
    joblib.dump(rf_c, out / "cause_rf.joblib")
    joblib.dump(lgb_c, out / "cause_lgb.joblib")
    joblib.dump(le, out / "cause_label_encoder.joblib")
    (out / "feature_columns.json").write_text(json.dumps({"feature_names": feats}, indent=2))
    (out / "model_type.json").write_text(json.dumps({"type": "two_stage", "version": "v26_5class"}))
    (out / "version.json").write_text(json.dumps({
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "type": "two_stage",
        "feature_count": len(feats),
        "base_dataset": "data/processed/features_real_only.parquet",
        "cause_classes": list(le.classes_),
        "merged_into_other_failure": ["oom_killed", "network_error"],
    }, indent=2))
    # Copy regressors from v20 (same 29-feature set)
    import shutil
    v20 = ROOT / "data/artifacts/v20_no_posthoc"
    for f in ("regressor_memory.joblib", "regressor_duration.joblib"):
        if (v20 / f).exists():
            shutil.copy(v20 / f, out / f)
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
