"""Pure unit tests for the online feature builder."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.db.models import Repository
from app.ml import feature_builder as fb


# ---------- helpers ----------

def test_detect_dependency_change_npm_and_pip() -> None:
    assert fb._detect_dependency_change(["package.json"])
    assert fb._detect_dependency_change(["src/requirements.txt"])
    assert fb._detect_dependency_change(["go.sum", "main.go"])
    assert not fb._detect_dependency_change(["README.md"])


def test_detect_dockerfile_change_case_insensitive() -> None:
    assert fb._detect_dockerfile_change(["Dockerfile"])
    assert fb._detect_dockerfile_change(["build/Dockerfile.prod"])
    assert not fb._detect_dockerfile_change(["docker-compose.yml"])


def test_is_test_path_variants() -> None:
    assert fb._is_test_path("tests/test_foo.py")
    assert fb._is_test_path("src/api.test.ts")
    assert fb._is_test_path("internal/handler_test.go")
    assert not fb._is_test_path("src/api.py")


def test_detect_test_only_changes() -> None:
    assert fb._detect_test_only_changes(["tests/test_x.py", "tests/test_y.py"])
    assert not fb._detect_test_only_changes(["tests/test_x.py", "src/x.py"])
    assert not fb._detect_test_only_changes([])  # empty diff = no test-only


def test_count_test_dir_changes() -> None:
    assert fb._count_test_dir_changes(
        ["tests/a.py", "tests/b.py", "src/x.py"]
    ) == 2


def test_file_ext_count() -> None:
    files = ["a.py", "b.py", "c.js", "Dockerfile"]
    assert fb._file_ext_count(files, "py") == 2
    assert fb._file_ext_count(files, "js") == 1
    assert fb._file_ext_count(files, "go") == 0


# ---------- build_feature_vector ----------

@pytest.fixture
def stub_repo() -> Repository:
    r = Repository(
        full_name="acme/widget",
        url="https://github.com/acme/widget",
        default_branch="main",
        provider="github",
        ci_platform="github_actions",
        is_active=True,
    )
    r.id = 42
    return r


def test_build_feature_vector_returns_expected_keys(monkeypatch, stub_repo) -> None:
    monkeypatch.setattr(
        fb, "_author_history",
        lambda db, rid, email: {"author_success_rate": 0.8, "author_n_runs_log": 2.3, "author_avg_duration_log": 5.9},
    )
    monkeypatch.setattr(
        fb, "_project_history",
        lambda db, rid: {"project_failure_rate": 0.2, "project_n_runs_log": 4.0, "repo_failure_rate_global": 0.2},
    )

    feats = fb.build_feature_vector(
        db=None,  # type: ignore[arg-type] — history mocked above
        repo=stub_repo,
        files=[{"filename": "package.json"}, {"filename": "src/x.ts"}],
        lines_added=120,
        lines_deleted=10,
        created_at=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),  # Tuesday 14:00
        author_email="dev@example.com",
        n_jobs=4,
        longest_job_seconds=180,
        run_attempt=2,
        new_dependencies_count=3,
        dockerfile_base_image_size_mb=100,
        estimated_final_image_size_mb=250,
    )

    # spot-check structure
    assert feats["feat_has_dependency_change_int"] == 1  # package.json
    assert feats["feat_has_dockerfile_change_int"] == 0  # no Dockerfile
    assert feats["feat_run_attempt_gt1"] == 1
    assert feats["feat_is_business_hours"] == 1
    assert feats["feat_is_weekend"] == 0
    assert feats["feat_image_growth_ratio"] == pytest.approx(2.5, rel=1e-3)
    assert feats["feat_author_success_rate"] == 0.8
    assert feats["feat_project_failure_rate"] == 0.2
    # Author/project hist mocked; expected keys present:
    assert {"feat_author_n_runs_log", "feat_project_n_runs_log"} <= feats.keys()


def test_build_feature_vector_clamps_extreme_values(monkeypatch, stub_repo) -> None:
    monkeypatch.setattr(fb, "_author_history", lambda *_a, **_k: {
        "author_success_rate": 0.5, "author_n_runs_log": 0.0, "author_avg_duration_log": 5.9,
    })
    monkeypatch.setattr(fb, "_project_history", lambda *_a, **_k: {
        "project_failure_rate": 0.3, "project_n_runs_log": 0.0, "repo_failure_rate_global": 0.3,
    })

    # Outrageous inputs must not produce inf / nan.
    feats = fb.build_feature_vector(
        db=None,  # type: ignore[arg-type]
        repo=stub_repo,
        files=[{"filename": f"f{i}.py"} for i in range(5000)],
        lines_added=10_000_000,
        lines_deleted=10_000_000,
        created_at=datetime(2026, 1, 3, 23, 30, tzinfo=timezone.utc),  # Saturday
        author_email="x@y.z",
        n_jobs=10_000,
        longest_job_seconds=10_000_000,
        run_attempt=999,
        dockerfile_base_image_size_mb=0.0,  # growth must guard against /0
        estimated_final_image_size_mb=999.0,
    )
    assert feats["feat_is_weekend"] == 1
    assert feats["feat_image_growth_ratio"] == 0.0  # base==0 path
    assert feats["feat_run_attempt"] == 10  # clamped
    # No inf, no nan in any value.
    for v in feats.values():
        assert v == v  # NaN check
        assert v not in (float("inf"), float("-inf"))
