"""Per-class distributions used by the synthetic generator.

The numbers below are derived from the failure-share survey reported in
the introductory chapter of the diploma (resource 35-40%, timeout 25-30%,
dependency 20-25%, docker 10-15%, network 5-10%) combined with industry
SRE post-mortems summarised in the literature review.

Design notes for v2 (post-EDA review):

  * Per-class historical priors. ``author_success_rate`` and
    ``project_failure_rate`` are class-conditional — an author who
    repeatedly causes OOM or timeout has a lower historical success
    rate, while network errors are nearly random and don't reflect on
    the author. This was the most impactful change after EDA.
  * Stronger separation on resource axes. Means for ``peak_memory_mb``,
    ``duration_seconds`` and ``estimated_final_image_size_mb`` were
    pushed apart between success and the corresponding failure class
    so the classifier has a clearer signal.
  * Wider tails. Sigma for log-normal distributions of resource
    metrics is increased so the right tail of the success class
    overlaps with real-world distributions observed in the GHA
    collector dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---- empirical calibration from real GHA data (optional) ----
# If synthetic/calibration.json exists, sampler helpers below use those
# per-class params for the 8 shared columns instead of the literature
# constants. Run `python -m synthetic.calibrate` to regenerate.
_CAL_PATH = Path(__file__).parent / "calibration.json"
try:
    _CAL = json.loads(_CAL_PATH.read_text()).get("per_class", {}) if _CAL_PATH.exists() else {}
except (json.JSONDecodeError, OSError):
    _CAL = {}


def _cal(cls: str, col: str) -> dict | None:
    """Return calibrated params for (class, column), or None if absent."""
    return _CAL.get(cls, {}).get(col)


def _cal_count(rng: np.random.Generator, cls: str, col: str, fallback: int) -> int | None:
    p = _cal(cls, col)
    if not p:
        return None
    v = rng.normal(p["mean"], p["std"])
    return int(np.clip(round(v), p.get("min", 0), p.get("max", 10**6)))


def _cal_lognorm(rng: np.random.Generator, cls: str, col: str) -> float | None:
    p = _cal(cls, col)
    if not p:
        return None
    return float(rng.lognormal(mean=np.log(max(p["median"], 1e-3)), sigma=p["log_sigma"]))


def _cal_bool(rng: np.random.Generator, cls: str, col: str) -> bool | None:
    p = _cal(cls, col)
    if not p:
        return None
    return bool(rng.random() < p["p"])

CLASSES = (
    "success",
    "oom_killed",
    "test_timeout",
    "dependency_error",
    "docker_build_failed",
    "network_error",
)

CLASS_WEIGHTS: dict[str, float] = {
    "success": 0.65,
    "oom_killed": 0.12,
    "test_timeout": 0.09,
    "dependency_error": 0.07,
    "docker_build_failed": 0.04,
    "network_error": 0.03,
}


@dataclass
class FeatureSample:
    files_changed: int
    lines_added: int
    lines_deleted: int
    has_dockerfile_change: bool
    has_dependency_change: bool
    new_dependencies_count: int
    new_dependencies_size_mb: float
    dockerfile_base_image_size_mb: float
    estimated_final_image_size_mb: float
    peak_memory_mb: float
    duration_seconds: float
    n_jobs: int
    longest_job_seconds: float
    author_success_rate: float
    project_failure_rate: float
    file_ext_py_share: float
    file_ext_js_share: float
    hour_of_day: int
    day_of_week: int


def _clip_int(rng: np.random.Generator, mean: float, sigma: float, lo: int, hi: int) -> int:
    return int(np.clip(round(rng.normal(mean, sigma)), lo, hi))


def _lognormal(rng: np.random.Generator, mean: float, sigma: float) -> float:
    return float(rng.lognormal(mean=np.log(max(mean, 1e-3)), sigma=sigma))


def _bool(rng: np.random.Generator, p: float) -> bool:
    return bool(rng.random() < p)


def _beta(rng: np.random.Generator, a: float, b: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(rng.beta(a, b), lo, hi))


def _maybe(value: int | float | bool | None, fallback):
    """Return value if not None else fallback. Keeps sampler bodies readable."""
    return fallback if value is None else value


def sample_success(rng: np.random.Generator) -> FeatureSample:
    cls = "success"
    return FeatureSample(
        files_changed=_maybe(_cal_count(rng, cls, "files_changed", 8), _clip_int(rng, 8, 6, 1, 80)),
        lines_added=_maybe(_cal_count(rng, cls, "lines_added", 90), _clip_int(rng, 90, 80, 1, 800)),
        lines_deleted=_maybe(_cal_count(rng, cls, "lines_deleted", 60), _clip_int(rng, 60, 60, 0, 700)),
        has_dockerfile_change=_maybe(_cal_bool(rng, cls, "has_dockerfile_change"), _bool(rng, 0.05)),
        has_dependency_change=_maybe(_cal_bool(rng, cls, "has_dependency_change"), _bool(rng, 0.20)),
        new_dependencies_count=_clip_int(rng, 0.3, 1.0, 0, 6),
        new_dependencies_size_mb=_lognormal(rng, 1.0, 0.8),
        dockerfile_base_image_size_mb=_lognormal(rng, 90, 0.5),
        estimated_final_image_size_mb=_lognormal(rng, 280, 0.5),
        peak_memory_mb=_lognormal(rng, 1100, 0.45),
        duration_seconds=_maybe(_cal_lognorm(rng, cls, "duration_seconds"), _lognormal(rng, 360, 0.6)),
        n_jobs=_maybe(_cal_count(rng, cls, "n_jobs", 4), _clip_int(rng, 4, 2, 1, 16)),
        longest_job_seconds=_maybe(_cal_lognorm(rng, cls, "longest_job_seconds"), _lognormal(rng, 280, 0.6)),
        author_success_rate=_beta(rng, 9, 1.5, 0.5, 0.99),
        project_failure_rate=_beta(rng, 2, 9, 0.02, 0.4),
        file_ext_py_share=_beta(rng, 5, 5),
        file_ext_js_share=_beta(rng, 2, 8),
        hour_of_day=int(rng.integers(0, 24)),
        day_of_week=int(rng.integers(0, 7)),
    )


def _apply_calibration(base: FeatureSample, rng: np.random.Generator, cls: str) -> FeatureSample:
    """Overwrite the 8 calibrated columns on ``base`` if real-data params exist."""
    for col in ("files_changed", "lines_added", "lines_deleted", "n_jobs"):
        v = _cal_count(rng, cls, col, 0)
        if v is not None:
            setattr(base, col, v)
    for col in ("duration_seconds", "longest_job_seconds"):
        v = _cal_lognorm(rng, cls, col)
        if v is not None:
            setattr(base, col, v)
    for col in ("has_dependency_change", "has_dockerfile_change"):
        v = _cal_bool(rng, cls, col)
        if v is not None:
            setattr(base, col, v)
    return base


def sample_oom(rng: np.random.Generator) -> FeatureSample:
    base = sample_success(rng)
    base.lines_added = _clip_int(rng, 350, 250, 50, 4000)
    base.files_changed = _clip_int(rng, 30, 20, 5, 250)
    base.has_dependency_change = _bool(rng, 0.65)
    base.new_dependencies_count = _clip_int(rng, 4, 2, 0, 14)
    base.new_dependencies_size_mb = _lognormal(rng, 150, 1.1)
    base.peak_memory_mb = _lognormal(rng, 6000, 0.30)
    base.duration_seconds = _lognormal(rng, 1100, 0.45)
    base.longest_job_seconds = _lognormal(rng, 900, 0.45)
    base.estimated_final_image_size_mb = _lognormal(rng, 950, 0.55)
    base.author_success_rate = _beta(rng, 4, 5, 0.05, 0.95)
    base.project_failure_rate = _beta(rng, 4, 6, 0.1, 0.7)
    return _apply_calibration(base, rng, "oom_killed")


def sample_timeout(rng: np.random.Generator) -> FeatureSample:
    base = sample_success(rng)
    base.lines_added = _clip_int(rng, 220, 180, 30, 2500)
    base.files_changed = _clip_int(rng, 25, 18, 5, 200)
    base.duration_seconds = _lognormal(rng, 2400, 0.30)
    base.longest_job_seconds = _lognormal(rng, 2200, 0.30)
    base.n_jobs = _clip_int(rng, 7, 4, 1, 24)
    base.peak_memory_mb = _lognormal(rng, 2400, 0.45)
    base.author_success_rate = _beta(rng, 5, 4, 0.1, 0.95)
    base.project_failure_rate = _beta(rng, 4, 6, 0.1, 0.7)
    return _apply_calibration(base, rng, "test_timeout")


def sample_dependency(rng: np.random.Generator) -> FeatureSample:
    base = sample_success(rng)
    base.has_dependency_change = True
    base.new_dependencies_count = _clip_int(rng, 8, 4, 1, 30)
    base.new_dependencies_size_mb = _lognormal(rng, 120, 1.0)
    base.lines_added = _clip_int(rng, 120, 100, 10, 1500)
    base.files_changed = _clip_int(rng, 12, 10, 1, 100)
    base.duration_seconds = _lognormal(rng, 200, 0.7)
    base.longest_job_seconds = _lognormal(rng, 150, 0.7)
    base.author_success_rate = _beta(rng, 6, 4, 0.2, 0.95)
    base.project_failure_rate = _beta(rng, 3, 7, 0.05, 0.55)
    return _apply_calibration(base, rng, "dependency_error")


def sample_docker(rng: np.random.Generator) -> FeatureSample:
    base = sample_success(rng)
    base.has_dockerfile_change = True
    base.dockerfile_base_image_size_mb = _lognormal(rng, 650, 0.55)
    base.estimated_final_image_size_mb = _lognormal(rng, 2800, 0.55)
    base.lines_added = _clip_int(rng, 80, 70, 5, 800)
    base.files_changed = _clip_int(rng, 8, 7, 1, 60)
    base.duration_seconds = _lognormal(rng, 700, 0.5)
    base.author_success_rate = _beta(rng, 6, 4, 0.2, 0.95)
    base.project_failure_rate = _beta(rng, 4, 6, 0.1, 0.7)
    return _apply_calibration(base, rng, "docker_build_failed")


def sample_network(rng: np.random.Generator) -> FeatureSample:
    base = sample_success(rng)
    base.duration_seconds = _lognormal(rng, 600, 0.55)
    base.longest_job_seconds = _lognormal(rng, 850, 0.55)
    base.n_jobs = _clip_int(rng, 9, 4, 2, 24)
    base.has_dependency_change = _bool(rng, 0.55)
    base.new_dependencies_count = _clip_int(rng, 2, 1, 0, 6)
    base.new_dependencies_size_mb = _lognormal(rng, 25, 1.0)
    base.author_success_rate = _beta(rng, 7, 3, 0.3, 0.97)
    base.project_failure_rate = _beta(rng, 2, 8, 0.05, 0.4)
    return _apply_calibration(base, rng, "network_error")


SAMPLERS = {
    "success": sample_success,
    "oom_killed": sample_oom,
    "test_timeout": sample_timeout,
    "dependency_error": sample_dependency,
    "docker_build_failed": sample_docker,
    "network_error": sample_network,
}
