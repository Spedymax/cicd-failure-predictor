#!/usr/bin/env python3
"""Active learning retrain: pull predictions with actual_outcome from the
production DB, merge with the original GHA training set, retrain v21+.

Usage:
    python scripts/retrain_from_feedback.py --out data/artifacts/v21_feedback
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "backend"))

from features.transformers import feature_columns  # noqa: E402

DROP_FEATS = {
    "feat_author_success_rate",
    "feat_author_n_runs_log",
    "feat_author_avg_duration_log",
    "feat_project_failure_rate",
    "feat_project_n_runs_log",
    "feat_repo_failure_rate_global",
    "feat_n_jobs_log",
    "feat_longest_job_seconds_log",
}


def pull_feedback_rows() -> pd.DataFrame:
    """Pull (feature_vector, actual_outcome) rows from production DB."""
    os.environ.setdefault("APP_SECRET_KEY", "x" * 32)
    from app.db.session import SessionLocal  # noqa: E402
    from app.db.models import Prediction  # noqa: E402
    from sqlalchemy import select

    db = SessionLocal()
    rows = db.execute(
        select(Prediction.feature_vector, Prediction.actual_outcome)
        .where(Prediction.actual_outcome.is_not(None))
        .where(Prediction.feature_vector.is_not(None))
    ).all()
    db.close()
    records = [
        {**(r[0] or {}), "actual_outcome": r[1].value if r[1] else None}
        for r in rows
        if r[0]
    ]
    return pd.DataFrame.from_records(records)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/artifacts/v21_feedback")
    ap.add_argument("--base-features", default="data/processed/features_real_only.parquet")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    base = pd.read_parquet(ROOT / args.base_features)
    feats = [c for c in feature_columns(base) if c not in DROP_FEATS]
    print(f"feature set: {len(feats)} columns")

    fb = pull_feedback_rows()
    print(f"feedback rows from DB: {len(fb)}")
    if len(fb) == 0:
        print("no feedback rows available — nothing to retrain on top of v20")
        return 0

    # Reuse the same label series the offline pipeline would compute.
    from training.data import _label_series  # noqa: E402

    base_labels = _label_series(base)
    base_X = base[feats].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    base_df = base_X.copy()
    base_df["__label"] = base_labels.values

    fb_X = fb.reindex(columns=feats).fillna(0.0).replace([np.inf, -np.inf], 0.0)
    fb_df = fb_X.copy()
    fb_df["__label"] = fb["actual_outcome"].fillna("other_failure").values
    fb_df["__weight"] = 3.0  # upweight live feedback: 1 real run > 1 backfilled run
    base_df["__weight"] = 1.0

    full = pd.concat([base_df, fb_df], ignore_index=True)
    print(
        f"merged: base={len(base_df)} + feedback={len(fb_df)} = {len(full)}; "
        f"label dist={full['__label'].value_counts().to_dict()}"
    )

    X = full[feats]
    y_binary = (full["__label"] == "success").astype(int)
    sample_w = full["__weight"].values

    X_tr, X_te, yb_tr, yb_te, w_tr, _ = train_test_split(
        X, y_binary, sample_w, test_size=0.2, random_state=42, stratify=y_binary
    )

    rf_r = RandomForestClassifier(
        n_estimators=400, max_depth=20, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ).fit(X_tr, yb_tr, sample_weight=w_tr)
    lgb_r = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
    ).fit(X_tr, yb_tr, sample_weight=w_tr)
    preds = ((rf_r.predict_proba(X_te) + lgb_r.predict_proba(X_te)) / 2).argmax(axis=1)
    print(f"risk F1 = {f1_score(yb_te, preds):.3f}")

    fail_mask = full["__label"] != "success"
    Xc = X.loc[fail_mask]
    yc = full.loc[fail_mask, "__label"]
    wc = full.loc[fail_mask, "__weight"]
    Xc_tr, Xc_te, yc_tr, yc_te, wc_tr, _ = train_test_split(
        Xc, yc, wc, test_size=0.2, random_state=42, stratify=yc
    )

    le = LabelEncoder()
    yc_tr_enc = le.fit_transform(yc_tr)
    yc_te_enc = le.transform(yc_te)
    rf_c = RandomForestClassifier(
        n_estimators=400, max_depth=20, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ).fit(Xc_tr, yc_tr_enc, sample_weight=wc_tr)
    lgb_c = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        class_weight="balanced", objective="multiclass",
        num_class=len(le.classes_), random_state=42, n_jobs=-1, verbose=-1,
    ).fit(Xc_tr, yc_tr_enc, sample_weight=wc_tr)
    probs = (rf_c.predict_proba(Xc_te) + lgb_c.predict_proba(Xc_te)) / 2
    cause_preds = probs.argmax(axis=1)
    print(f"cause F1-macro = {f1_score(yc_te_enc, cause_preds, average='macro'):.3f}")
    print(
        classification_report(
            yc_te_enc, cause_preds, target_names=list(le.classes_),
            zero_division=0, digits=2,
        )
    )

    joblib.dump(rf_r, out / "risk_rf.joblib")
    joblib.dump(lgb_r, out / "risk_lgb.joblib")
    joblib.dump(rf_c, out / "cause_rf.joblib")
    joblib.dump(lgb_c, out / "cause_lgb.joblib")
    joblib.dump(le, out / "cause_label_encoder.joblib")
    (out / "feature_columns.json").write_text(
        json.dumps({"feature_names": feats}, indent=2)
    )
    (out / "model_type.json").write_text(
        json.dumps({"type": "two_stage", "version": out.name})
    )
    (out / "version.json").write_text(json.dumps({
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "type": "two_stage",
        "feature_count": len(feats),
        "base_dataset": str(args.base_features),
        "feedback_rows": int(len(fb)),
        "feedback_weight": 3.0,
    }, indent=2))

    # Copy regressors from v20 (still valid on the same 29-feature set).
    v20 = ROOT / "data/artifacts/v20_no_posthoc"
    for f in ("regressor_memory.joblib", "regressor_duration.joblib"):
        if (v20 / f).exists():
            shutil.copy(v20 / f, out / f)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
