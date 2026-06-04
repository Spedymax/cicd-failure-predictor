import pandas as pd

from synthetic.distributions import CLASS_WEIGHTS, CLASSES, SAMPLERS
from synthetic.generate import generate


def test_class_weights_sum_to_one():
    assert abs(sum(CLASS_WEIGHTS.values()) - 1.0) < 1e-6


def test_samplers_cover_all_classes():
    assert set(SAMPLERS.keys()) == set(CLASSES)


def test_generate_reproducible_with_seed():
    a = generate(500, seed=123)
    b = generate(500, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_generate_class_distribution_within_5pp():
    df = generate(5000, seed=42)
    actual = df["failure_class"].fillna("success").value_counts(normalize=True).to_dict()
    for cls, expected in CLASS_WEIGHTS.items():
        observed = actual.get(cls, 0.0)
        assert abs(observed - expected) < 0.05, (cls, expected, observed)


def test_failure_rows_have_failure_conclusion():
    df = generate(500, seed=7)
    failed = df[df["failure_class"].notna()]
    assert (failed["conclusion"] == "failure").all()
    successful = df[df["failure_class"].isna()]
    assert (successful["conclusion"] == "success").all()


def test_oom_class_has_higher_memory_than_success():
    df = generate(2000, seed=99)
    oom_mem = df[df["failure_class"] == "oom_killed"]["peak_memory_mb"].median()
    succ_mem = df[df["failure_class"].isna()]["peak_memory_mb"].median()
    assert oom_mem > succ_mem * 1.5


def test_timeout_class_has_higher_duration_than_success():
    df = generate(2000, seed=11)
    timeout_dur = df[df["failure_class"] == "test_timeout"]["duration_seconds"].median()
    succ_dur = df[df["failure_class"].isna()]["duration_seconds"].median()
    assert timeout_dur > succ_dur * 1.5


def test_dependency_class_always_changes_dependencies():
    df = generate(1000, seed=21)
    deps = df[df["failure_class"] == "dependency_error"]
    assert deps["has_dependency_change"].all()


def test_docker_class_always_changes_dockerfile():
    df = generate(1000, seed=22)
    docker = df[df["failure_class"] == "docker_build_failed"]
    assert docker["has_dockerfile_change"].all()
