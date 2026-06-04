import numpy as np
import pandas as pd

from features.transformers import (
    ENGINEERED_PREFIX,
    add_dependency_features,
    add_docker_features,
    add_file_ext_features,
    add_historical_features,
    add_ratio_features,
    add_repo_features,
    add_time_features,
    engineer_features,
    feature_columns,
)
from synthetic.generate import generate


def test_ratio_features_added():
    df = pd.DataFrame({"lines_added": [10, 0], "lines_deleted": [5, 0], "files_changed": [3, 1]})
    out = add_ratio_features(df)
    assert f"{ENGINEERED_PREFIX}lines_changed_log" in out
    assert f"{ENGINEERED_PREFIX}files_changed_log" in out
    assert f"{ENGINEERED_PREFIX}lines_added_share" in out
    assert out[f"{ENGINEERED_PREFIX}lines_changed_log"].iloc[0] == np.log1p(15)
    assert pd.isna(out[f"{ENGINEERED_PREFIX}lines_added_share"].iloc[1])


def test_ratio_features_clip_outliers():
    df = pd.DataFrame(
        {"lines_added": [200_000, 100], "lines_deleted": [50_000, 0], "files_changed": [5_000, 2]}
    )
    out = add_ratio_features(df)
    assert out[f"{ENGINEERED_PREFIX}files_changed_log"].iloc[0] == np.log1p(1000)
    assert out[f"{ENGINEERED_PREFIX}lines_changed_log"].iloc[0] == np.log1p(100_000)


def test_time_features_handle_iso_strings():
    df = pd.DataFrame({"created_at": ["2026-05-09T10:30:00Z", "2026-05-10T22:00:00+00:00"]})
    out = add_time_features(df)
    assert out[f"{ENGINEERED_PREFIX}hour_of_day"].tolist() == [10, 22]
    assert out[f"{ENGINEERED_PREFIX}is_weekend"].tolist() == [1, 1]


def test_dependency_features_default_to_zero_when_missing():
    df = pd.DataFrame({"has_dependency_change": [True, False]})
    out = add_dependency_features(df)
    assert (out[f"{ENGINEERED_PREFIX}new_deps_count"] == 0).all()
    assert out[f"{ENGINEERED_PREFIX}has_dependency_change_int"].tolist() == [1, 0]


def test_docker_features_image_growth_ratio():
    df = pd.DataFrame(
        {
            "has_dockerfile_change": [True, False],
            "dockerfile_base_image_size_mb": [100.0, 0.0],
            "estimated_final_image_size_mb": [400.0, 0.0],
        }
    )
    out = add_docker_features(df)
    assert out[f"{ENGINEERED_PREFIX}image_growth_ratio"].iloc[0] == 4.0
    assert out[f"{ENGINEERED_PREFIX}image_growth_ratio"].iloc[1] == 0.0


def test_docker_features_growth_ratio_clipped():
    df = pd.DataFrame(
        {
            "has_dockerfile_change": [True],
            "dockerfile_base_image_size_mb": [10.0],
            "estimated_final_image_size_mb": [10000.0],
        }
    )
    out = add_docker_features(df)
    assert out[f"{ENGINEERED_PREFIX}image_growth_ratio"].iloc[0] == 50.0


def test_historical_features_no_leakage_first_row():
    df = pd.DataFrame(
        {
            "actor_email": ["a@x", "a@x", "a@x"],
            "repo": ["r1", "r1", "r1"],
            "conclusion": ["failure", "success", "success"],
            "created_at": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03"], utc=True
            ),
            "duration_seconds": [100.0, 200.0, 300.0],
        }
    )
    out = add_historical_features(df).sort_values("created_at").reset_index(drop=True)
    sr = out[f"{ENGINEERED_PREFIX}author_success_rate"]
    assert sr.iloc[0] == 0.5
    assert sr.iloc[1] == 0.0
    assert sr.iloc[2] == 0.5


def test_file_ext_features():
    df = pd.DataFrame({"file_extensions": [{"py": 5, "js": 2}, {"go": 3}, None]})
    out = add_file_ext_features(df)
    assert out[f"{ENGINEERED_PREFIX}ext_py_count"].tolist() == [5, 0, 0]
    assert out[f"{ENGINEERED_PREFIX}ext_go_count"].tolist() == [0, 3, 0]


def test_repo_features():
    df = pd.DataFrame(
        {
            "repo": ["a", "a", "b"],
            "run_id": [1, 2, 3],
            "conclusion": ["failure", "success", "success"],
        }
    )
    out = add_repo_features(df)
    repo_a = out[out["repo"] == "a"].iloc[0]
    assert repo_a[f"{ENGINEERED_PREFIX}repo_failure_rate_global"] == 0.5
    repo_b = out[out["repo"] == "b"].iloc[0]
    assert repo_b[f"{ENGINEERED_PREFIX}repo_failure_rate_global"] == 0.0


def test_engineer_features_on_synthetic():
    df = generate(500, seed=1)
    out = engineer_features(df)
    cols = feature_columns(out)
    assert len(cols) >= 20
    assert len(out) == len(df)
    assert not out[cols].isna().all().any()


def test_engineered_columns_have_no_constant_nan():
    df = generate(1000, seed=2)
    out = engineer_features(df)
    for col in feature_columns(out):
        assert out[col].notna().sum() > 0, col
