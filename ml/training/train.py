"""End-to-end training CLI for the CI failure predictor.

Loads engineered features, splits into train/val/test (and a held-out
real-data set), trains the RF+LightGBM classifier ensemble, two
regression heads (memory + duration), evaluates on every split, and
serialises everything under ``--out``.

Usage::

    python -m training.train \
        --features ../data/processed/features.parquet \
        --out artifacts/v1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

from training.classifier import evaluate_classifier, train_classifier
from training.data import (
    DURATION_TARGET,
    MEMORY_TARGET,
    class_balance,
    class_weights,
    fill_features,
    load_features,
    make_split,
    regression_targets,
)
from training.regressor import evaluate_regressor, train_regressor
from training.tune import tune_classifier

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter tuning before final training (slower).",
    )
    parser.add_argument(
        "--no-regressors",
        action="store_true",
        help="Train only the classifier; skip memory/duration regressors. "
             "Use for real-only datasets that lack peak_memory_mb ground truth.",
    )
    parser.add_argument(
        "--drop-features",
        type=str,
        default=None,
        help="Comma-separated feat_* column names to exclude from the "
             "classifier (e.g. synthetic-only features absent in real data).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials when --tune is enabled (default: 50).",
    )
    parser.add_argument(
        "--tune-timeout",
        type=float,
        default=None,
        help="Stop tuning after N seconds even if --n-trials not reached.",
    )
    parser.add_argument(
        "--tuning-from",
        type=str,
        default=None,
        help=(
            "Path to a previous tuning.json. If provided (and --tune not set), "
            "re-uses best_rf_params/best_lgb_params from that file without running a new study."
        ),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")

    df = load_features(args.features)
    logger.info("loaded %d rows from %s", len(df), args.features)

    split = make_split(df, seed=args.seed)

    if args.drop_features:
        drop = {f.strip() for f in args.drop_features.split(",") if f.strip()}
        kept = [c for c in split.feature_names if c not in drop]
        missing = drop - set(split.feature_names)
        if missing:
            logger.warning("--drop-features: not found, ignored: %s", sorted(missing))
        logger.info("dropping %d features: %s", len(drop) - len(missing), sorted(drop - missing))
        split.feature_names = kept
        split.X_train = split.X_train[kept]
        split.X_val = split.X_val[kept]
        split.X_test = split.X_test[kept]
        if not split.real_X.empty:
            split.real_X = split.real_X[kept]

    logger.info(
        "split sizes: train=%d val=%d test=%d real=%d (features=%d)",
        len(split.X_train), len(split.X_val), len(split.X_test), len(split.real_X),
        len(split.feature_names),
    )
    logger.info("train class balance: %s", class_balance(split.y_train))
    logger.info("class weights: %s", {k: round(v, 3) for k, v in class_weights(split.y_train).items()})

    X_train = fill_features(split.X_train)
    X_val = fill_features(split.X_val)
    X_test = fill_features(split.X_test)
    X_real = fill_features(split.real_X) if not split.real_X.empty else split.real_X

    tuning_meta: dict | None = None
    rf_overrides: dict | None = None
    lgb_overrides: dict | None = None
    if args.tune:
        logger.info(
            "running Optuna tuning: n_trials=%d, timeout=%s",
            args.n_trials, args.tune_timeout,
        )
        tuning = tune_classifier(
            X_train,
            split.y_train,
            X_val,
            split.y_val,
            n_trials=args.n_trials,
            timeout=args.tune_timeout,
            seed=args.seed,
        )
        logger.info(
            "tuning done: best_val_f1_macro=%.4f (%d completed / %d pruned / %d total)",
            tuning.best_value, tuning.n_completed, tuning.n_pruned, tuning.n_trials,
        )
        rf_overrides = tuning.best_rf_params
        lgb_overrides = tuning.best_lgb_params
        tuning_meta = tuning.to_dict()
    elif args.tuning_from:
        tuning_path = Path(args.tuning_from)
        logger.info("re-using tuning result from %s (no new study)", tuning_path)
        cached = json.loads(tuning_path.read_text())
        rf_overrides = dict(cached["best_rf_params"])
        lgb_overrides = dict(cached["best_lgb_params"])
        # Pin random_state to the current run's seed so downstream
        # reproducibility is consistent with --seed even if the cached
        # study used a different value.
        rf_overrides["random_state"] = args.seed
        lgb_overrides["random_state"] = args.seed
        tuning_meta = {
            **cached,
            "reused_from": str(tuning_path),
        }

    logger.info("training ensemble classifier")
    clf = train_classifier(
        X_train,
        split.y_train,
        X_val,
        split.y_val,
        rf_params=rf_overrides,
        lgb_params=lgb_overrides,
        seed=args.seed,
    )
    metrics = {
        "classifier": {
            "train": evaluate_classifier(clf, X_train, split.y_train),
            "val": evaluate_classifier(clf, X_val, split.y_val),
            "test": evaluate_classifier(clf, X_test, split.y_test),
            "real_holdout": (
                evaluate_classifier(clf, X_real, split.real_y) if len(split.real_X) else None
            ),
        }
    }
    if tuning_meta is not None:
        metrics["tuning"] = tuning_meta
    logger.info(
        "classifier F1: train_macro=%.3f val_macro=%.3f test_macro=%.3f",
        metrics["classifier"]["train"]["f1_macro"],
        metrics["classifier"]["val"]["f1_macro"],
        metrics["classifier"]["test"]["f1_macro"],
    )

    mem_reg = dur_reg = None
    if not args.no_regressors:
        logger.info("training memory regressor")
        mem_train, dur_train = regression_targets(df.loc[split.X_train.index])
        mem_val, dur_val = regression_targets(df.loc[split.X_val.index])
        mem_test, dur_test = regression_targets(df.loc[split.X_test.index])

        # Real rows lack peak_memory_mb / duration_seconds (those are
        # synthetic-only ground-truth fields). Filter them out before fitting
        # the regressors — predicting on them is still possible because the
        # input features have already been imputed.
        mem_train_mask = mem_train.notna()
        mem_val_mask = mem_val.notna()
        mem_test_mask = mem_test.notna()
        dur_train_mask = dur_train.notna()
        dur_val_mask = dur_val.notna()
        dur_test_mask = dur_test.notna()
        logger.info(
            "regressor train sizes: memory=%d/%d duration=%d/%d",
            int(mem_train_mask.sum()), len(mem_train),
            int(dur_train_mask.sum()), len(dur_train),
        )

        mem_reg = train_regressor(
            X_train.loc[mem_train_mask], mem_train.loc[mem_train_mask],
            X_val.loc[mem_val_mask], mem_val.loc[mem_val_mask],
            target_name=MEMORY_TARGET, seed=args.seed,
        )
        dur_reg = train_regressor(
            X_train.loc[dur_train_mask], dur_train.loc[dur_train_mask],
            X_val.loc[dur_val_mask], dur_val.loc[dur_val_mask],
            target_name=DURATION_TARGET, seed=args.seed,
        )
        metrics["regressors"] = {
            MEMORY_TARGET: {
                "train": evaluate_regressor(mem_reg, X_train.loc[mem_train_mask], mem_train.loc[mem_train_mask]),
                "val": evaluate_regressor(mem_reg, X_val.loc[mem_val_mask], mem_val.loc[mem_val_mask]),
                "test": evaluate_regressor(mem_reg, X_test.loc[mem_test_mask], mem_test.loc[mem_test_mask]),
            },
            DURATION_TARGET: {
                "train": evaluate_regressor(dur_reg, X_train.loc[dur_train_mask], dur_train.loc[dur_train_mask]),
                "val": evaluate_regressor(dur_reg, X_val.loc[dur_val_mask], dur_val.loc[dur_val_mask]),
                "test": evaluate_regressor(dur_reg, X_test.loc[dur_test_mask], dur_test.loc[dur_test_mask]),
            },
        }
    else:
        logger.info("--no-regressors: skipping memory/duration regressors")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save the inner sklearn / lightgbm objects separately so the backend
    # can reconstruct an EnsembleClassifier without importing the training
    # package (avoids ModuleNotFoundError when unpickling).
    joblib.dump(clf.rf, out_dir / "classifier_rf.joblib")
    joblib.dump(clf.lgb_model, out_dir / "classifier_lgb.joblib")
    joblib.dump(clf.label_encoder, out_dir / "classifier_label_encoder.joblib")
    if mem_reg is not None:
        joblib.dump(mem_reg.model, out_dir / "regressor_memory.joblib")
    if dur_reg is not None:
        joblib.dump(dur_reg.model, out_dir / "regressor_duration.joblib")
    (out_dir / "feature_columns.json").write_text(
        json.dumps({"feature_names": split.feature_names}, indent=2)
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    if tuning_meta is not None:
        (out_dir / "tuning.json").write_text(json.dumps(tuning_meta, indent=2, default=str))
    (out_dir / "version.json").write_text(
        json.dumps(
            {
                "trained_at": datetime.now(tz=timezone.utc).isoformat(),
                "seed": args.seed,
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_test": len(X_test),
                "n_real_holdout": len(X_real),
                "feature_count": len(split.feature_names),
                "classes": list(clf.classes_),
                "tuned": tuning_meta is not None,
                "tuning_best_val_f1_macro": (
                    tuning_meta["best_val_f1_macro"] if tuning_meta else None
                ),
                "tuning_n_trials": tuning_meta["n_trials"] if tuning_meta else None,
            },
            indent=2,
        )
    )
    logger.info("wrote artifacts to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
