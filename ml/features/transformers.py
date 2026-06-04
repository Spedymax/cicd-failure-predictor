"""Feature engineering transformers for the CI failure predictor.

The transformers operate on a pandas DataFrame produced either by the GHA
collector or the synthetic generator (or a concatenation of both). Each
function returns a new DataFrame with additional columns; original columns
are preserved so the notebook can keep raw values for EDA.

Six feature groups (mirroring FR-04 from chapter 2):

  1. Code complexity   — derived from files/lines metadata (with hard caps
                          on extreme outliers from auto-merge commits)
  2. Dependencies      — derived from has_dependency_change, new_deps_*
  3. Docker            — derived from has_dockerfile_change, image sizes
  4. Time              — created_at -> hour_of_day, day_of_week, is_weekend
  5. Historical        — expanding aggregates per author / per repo
                          (no leakage: shift(1) before expanding)
  6. Repo / file ext   — repo-level aggregates and language counts

Post-EDA changes (v2):
  * Hard outlier caps for files_changed (≤1000) and lines (≤50000) — real
    GHA collected commits had a 175k-line tail from auto-merged branches
    that don't exist in synthetic.
  * Redundant features dropped: total_lines_changed (kept log version),
    repo_total_runs (kept project_n_runs), project_active_contributors
    (correlated >0.95 with project_n_runs), base_image_size (kept final +
    growth_ratio).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

ENGINEERED_PREFIX = "feat_"

# Hard caps to suppress auto-merge / vendored-deps outliers in real data.
# These are intentionally generous — typical commits sit far below.
MAX_FILES_CHANGED = 1000
MAX_LINES_PER_SIDE = 50_000


def _clip_series(s: pd.Series, hi: float) -> pd.Series:
    return s.fillna(0).clip(lower=0, upper=hi)


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    files = _clip_series(out["files_changed"], MAX_FILES_CHANGED)
    added = _clip_series(out["lines_added"], MAX_LINES_PER_SIDE)
    deleted = _clip_series(out["lines_deleted"], MAX_LINES_PER_SIDE)
    total = added + deleted
    out[f"{ENGINEERED_PREFIX}files_changed_log"] = np.log1p(files)
    out[f"{ENGINEERED_PREFIX}lines_changed_log"] = np.log1p(total)
    out[f"{ENGINEERED_PREFIX}lines_added_share"] = added / total.replace(0, np.nan)
    out[f"{ENGINEERED_PREFIX}avg_lines_per_file"] = total / files.replace(0, np.nan)
    return out


def add_dependency_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[f"{ENGINEERED_PREFIX}has_dependency_change_int"] = (
        out["has_dependency_change"].fillna(False).astype(int)
    )
    out[f"{ENGINEERED_PREFIX}new_deps_count"] = out.get(
        "new_dependencies_count", pd.Series(0, index=out.index)
    ).fillna(0).clip(lower=0, upper=100)
    deps_size = out.get("new_dependencies_size_mb", pd.Series(0, index=out.index)).fillna(0)
    out[f"{ENGINEERED_PREFIX}new_deps_size_mb_log"] = np.log1p(deps_size.clip(lower=0))
    return out


def add_docker_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[f"{ENGINEERED_PREFIX}has_dockerfile_change_int"] = (
        out["has_dockerfile_change"].fillna(False).astype(int)
    )
    base = out.get("dockerfile_base_image_size_mb", pd.Series(0, index=out.index)).fillna(0)
    final = out.get("estimated_final_image_size_mb", pd.Series(0, index=out.index)).fillna(0)
    out[f"{ENGINEERED_PREFIX}final_image_size_mb_log"] = np.log1p(final.clip(lower=0))
    growth = (final / base.replace(0, np.nan)).fillna(0)
    out[f"{ENGINEERED_PREFIX}image_growth_ratio"] = growth.clip(lower=0, upper=50)
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["created_at"], utc=True, errors="coerce")
    out[f"{ENGINEERED_PREFIX}hour_of_day"] = ts.dt.hour.fillna(-1).astype(int)
    out[f"{ENGINEERED_PREFIX}day_of_week"] = ts.dt.dayofweek.fillna(-1).astype(int)
    out[f"{ENGINEERED_PREFIX}is_weekend"] = ts.dt.dayofweek.isin([5, 6]).fillna(False).astype(int)
    out[f"{ENGINEERED_PREFIX}is_business_hours"] = (
        ts.dt.hour.between(9, 18, inclusive="both") & ~ts.dt.dayofweek.isin([5, 6])
    ).fillna(False).astype(int)
    return out


def add_run_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[f"{ENGINEERED_PREFIX}run_attempt"] = (
        out.get("run_attempt", pd.Series(1, index=out.index)).fillna(1).astype(int).clip(1, 10)
    )
    out[f"{ENGINEERED_PREFIX}run_attempt_gt1"] = (
        out[f"{ENGINEERED_PREFIX}run_attempt"] > 1
    ).astype(int)
    out[f"{ENGINEERED_PREFIX}n_jobs_log"] = np.log1p(
        out.get("n_jobs", pd.Series(0, index=out.index)).fillna(0).clip(lower=0, upper=100)
    )
    out[f"{ENGINEERED_PREFIX}longest_job_seconds_log"] = np.log1p(
        out.get("longest_job_seconds", pd.Series(0, index=out.index)).fillna(0).clip(lower=0)
    )
    return out


def add_change_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-specific change-pattern signals (v4 additions for OTHER_FAILURE)."""
    out = df.copy()
    out[f"{ENGINEERED_PREFIX}test_dir_changes"] = (
        out.get("test_dir_changes", pd.Series(0, index=out.index)).fillna(0).clip(lower=0, upper=1000).astype(int)
    )
    out[f"{ENGINEERED_PREFIX}test_only_changes_int"] = (
        out.get("test_only_changes", pd.Series(False, index=out.index)).fillna(False).astype(int)
    )
    out[f"{ENGINEERED_PREFIX}has_lint_config_change_int"] = (
        out.get("has_lint_config_change", pd.Series(False, index=out.index)).fillna(False).astype(int)
    )
    out[f"{ENGINEERED_PREFIX}event_is_push_int"] = (
        out.get("event_is_push", pd.Series(True, index=out.index)).fillna(True).astype(int)
    )
    files = out.get("files_changed", pd.Series(0, index=out.index)).fillna(0).clip(lower=1)
    out[f"{ENGINEERED_PREFIX}test_changes_share"] = (
        out[f"{ENGINEERED_PREFIX}test_dir_changes"] / files
    ).clip(lower=0, upper=1)
    return out


def _expanding_share(group: pd.Series, target_value: Any) -> pd.Series:
    is_target = (group == target_value).astype(float)
    return is_target.shift(1).expanding().mean().fillna(0.5)


def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["created_at"] = pd.to_datetime(out["created_at"], utc=True, errors="coerce")

    out = out.sort_values(["actor_email", "created_at"]).reset_index(drop=True)
    out[f"{ENGINEERED_PREFIX}author_success_rate"] = (
        out.groupby("actor_email", group_keys=False)["conclusion"]
        .apply(lambda s: _expanding_share(s, "success"))
    )
    out[f"{ENGINEERED_PREFIX}author_n_runs_log"] = np.log1p(
        out.groupby("actor_email").cumcount()
    )
    median_dur = out["duration_seconds"].median()
    out[f"{ENGINEERED_PREFIX}author_avg_duration_log"] = np.log1p(
        out.groupby("actor_email", group_keys=False)["duration_seconds"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .fillna(median_dur)
        .clip(lower=0)
    )

    out = out.sort_values(["repo", "created_at"]).reset_index(drop=True)
    out[f"{ENGINEERED_PREFIX}project_failure_rate"] = (
        out.groupby("repo", group_keys=False)["conclusion"]
        .apply(lambda s: _expanding_share(s, "failure"))
    )
    out[f"{ENGINEERED_PREFIX}project_n_runs_log"] = np.log1p(
        out.groupby("repo").cumcount()
    )
    return out.sort_index()


def add_repo_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    repo_stats = out.groupby("repo").agg(
        repo_failure_rate_global=("conclusion", lambda s: (s == "failure").mean()),
    )
    out = out.merge(repo_stats, left_on="repo", right_index=True, how="left")
    out = out.rename(
        columns={"repo_failure_rate_global": f"{ENGINEERED_PREFIX}repo_failure_rate_global"}
    )
    return out


def add_file_ext_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _ext_count(row: dict | None, key: str) -> int:
        if not isinstance(row, dict):
            return 0
        return int(row.get(key, 0) or 0)

    for ext in ("py", "js", "ts", "go", "rs", "java", "yml", "json"):
        out[f"{ENGINEERED_PREFIX}ext_{ext}_count"] = out["file_extensions"].apply(
            lambda r, e=ext: _ext_count(r, e)
        )
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = add_ratio_features(out)
    out = add_dependency_features(out)
    out = add_docker_features(out)
    out = add_time_features(out)
    out = add_run_features(out)
    out = add_change_pattern_features(out)
    out = add_file_ext_features(out)
    out = add_historical_features(out)
    out = add_repo_features(out)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(ENGINEERED_PREFIX)]
