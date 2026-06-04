"""Hyperparameter tuning for the ensemble classifier via Optuna.

Optimises RandomForest + LightGBM hyperparameters jointly by maximising
val F1-macro of the soft-voted ensemble. Uses TPE sampler with a
MedianPruner so unpromising trials are stopped early.

Search space is deliberately narrow around the existing defaults to keep
runtime reasonable on a laptop (~50 trials in 5-15 min on the current
10k-row dataset).

Usage as standalone CLI::

    python -m training.tune \\
        --features ../data/processed/features.parquet \\
        --out artifacts/tuning.json \\
        --n-trials 60

Or invoked from train.py via ``--tune``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from training.data import (
    class_balance,
    fill_features,
    load_features,
    make_split,
)

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """Outcome of an Optuna study over the ensemble classifier."""

    best_value: float
    best_rf_params: dict[str, Any]
    best_lgb_params: dict[str, Any]
    n_trials: int
    n_completed: int
    n_pruned: int
    trials: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_val_f1_macro": self.best_value,
            "best_rf_params": self.best_rf_params,
            "best_lgb_params": self.best_lgb_params,
            "n_trials": self.n_trials,
            "n_completed": self.n_completed,
            "n_pruned": self.n_pruned,
            "trials": self.trials,
        }


def _suggest_rf_params(trial: optuna.Trial, seed: int) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("rf_n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("rf_max_depth", 6, 24),
        "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "rf_max_features", ["sqrt", "log2", 0.5]
        ),
        "class_weight": "balanced",
        "random_state": seed,
        "n_jobs": -1,
    }


def _suggest_lgb_params(trial: optuna.Trial, n_classes: int, seed: int) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("lgb_n_estimators", 200, 800, step=50),
        "learning_rate": trial.suggest_float("lgb_learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("lgb_num_leaves", 15, 127),
        "max_depth": trial.suggest_int("lgb_max_depth", -1, 24),
        "min_child_samples": trial.suggest_int("lgb_min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("lgb_reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("lgb_reg_lambda", 1e-4, 10.0, log=True),
        "class_weight": "balanced",
        "objective": "multiclass",
        "num_class": n_classes,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def _ensemble_val_f1(
    rf_params: dict[str, Any],
    lgb_params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train_enc: np.ndarray,
    X_val: pd.DataFrame,
    y_val_enc: np.ndarray,
) -> float:
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train_enc)
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_train,
        y_train_enc,
        eval_set=[(X_val, y_val_enc)],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )
    probs = (rf.predict_proba(X_val) + lgb_model.predict_proba(X_val)) / 2.0
    preds = probs.argmax(axis=1)
    return float(f1_score(y_val_enc, preds, average="macro", zero_division=0))


def tune_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    n_trials: int = 50,
    timeout: float | None = None,
    seed: int = 42,
    study_name: str | None = None,
) -> TuningResult:
    """Run Optuna study over RF+LightGBM ensemble hyperparameters.

    Returns the best parameter sets ready to be plugged into
    ``train_classifier`` via ``rf_params`` / ``lgb_params`` overrides.
    """
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    n_classes = len(le.classes_)

    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name or f"ensemble_f1_macro_seed{seed}",
    )

    def objective(trial: optuna.Trial) -> float:
        rf_params = _suggest_rf_params(trial, seed)
        lgb_params = _suggest_lgb_params(trial, n_classes, seed)
        score = _ensemble_val_f1(
            rf_params, lgb_params, X_train, y_train_enc, X_val, y_val_enc
        )
        trial.set_user_attr("rf_params", rf_params)
        trial.set_user_attr("lgb_params", lgb_params)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    best = study.best_trial
    return TuningResult(
        best_value=float(best.value),
        best_rf_params=best.user_attrs["rf_params"],
        best_lgb_params=best.user_attrs["lgb_params"],
        n_trials=len(study.trials),
        n_completed=len(completed),
        n_pruned=len(pruned),
        trials=[
            {
                "number": t.number,
                "value": float(t.value) if t.value is not None else None,
                "state": t.state.name,
                "params": dict(t.params),
            }
            for t in study.trials
        ],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune ensemble classifier with Optuna")
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True, help="Path to tuning_result.json")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=None, help="seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")

    df = load_features(args.features)
    split = make_split(df, seed=args.seed)
    X_train = fill_features(split.X_train)
    X_val = fill_features(split.X_val)
    logger.info(
        "tuning on %d train / %d val rows, class balance: %s",
        len(X_train), len(X_val), class_balance(split.y_train),
    )

    result = tune_classifier(
        X_train,
        split.y_train,
        X_val,
        split.y_val,
        n_trials=args.n_trials,
        timeout=args.timeout,
        seed=args.seed,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.to_dict(), indent=2))
    logger.info(
        "best val f1_macro=%.4f after %d trials (%d completed, %d pruned)",
        result.best_value, result.n_trials, result.n_completed, result.n_pruned,
    )
    logger.info("wrote tuning result to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
