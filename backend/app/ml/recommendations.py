"""Rule-based recommendation engine (FR-08).

Takes the predicted failure class, predicted resource usage and the raw
commit features and produces a list of actionable recommendations
sorted by severity. The rules are intentionally declarative so they
can be edited without retraining the model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Recommendation:
    severity: str
    category: str
    title: str
    description: str
    actions: list[str]
    estimated_impact: str | None = None


SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def _oom_recommendation(predicted_memory_mb: float, image_growth: float) -> Recommendation:
    actions = [
        "Збільшити memory_limit у конфігурації CI (наприклад, до 6GB у .github/workflows/...)",
        "Профілювати тести/збірку локально для виявлення витоків (memray, valgrind)",
        "Розпаралелити роботу на менші воркери замість одного з великим обсягом",
    ]
    if image_growth > 5:
        actions.append(
            f"Розмір образу зростає у {image_growth:.1f}× порівняно з base — спробуйте slim base image"  # noqa: E501  (user-facing message, not worth splitting)
        )
    return Recommendation(
        severity="HIGH",
        category="RESOURCE",
        title="Високий ризик OOMKilled",
        description=(
            f"Прогнозоване використання пам'яті ~{predicted_memory_mb:.0f} МБ — близько "
            "до типового ліміту 4 ГБ для GitHub-hosted runners. Перевиконання призведе "
            "до примусового завершення процесу (exit code 137)."
        ),
        actions=actions,
        estimated_impact="Зменшує ймовірність OOM приблизно з 70% до 15%",
    )


def _timeout_recommendation(predicted_duration_seconds: float) -> Recommendation:
    minutes = predicted_duration_seconds / 60
    return Recommendation(
        severity="HIGH",
        category="RESOURCE",
        title="Високий ризик перевищення часу виконання",
        description=(
            f"Прогнозована тривалість збірки ~{minutes:.1f} хв близька до типового "
            "ліміту job-у в 360 хв або вашого workflow timeout-minutes."
        ),
        actions=[
            "Розпаралелити тести (matrix strategy у workflow.yml)",
            "Кешувати залежності через actions/cache@v4",
            "Винести довгі інтеграційні тести в окремий nightly workflow",
        ],
        estimated_impact="Зменшує ймовірність timeout приблизно на 50%",
    )


def _dependency_recommendation(new_deps_count: int, has_dependency_change: bool) -> Recommendation:
    if new_deps_count > 0:
        description = (
            f"Коміт додає {new_deps_count} нових залежностей. Великі версійні зрушення "
            "часто призводять до peer-conflict (npm ERESOLVE) або runtime ImportError."
        )
    elif has_dependency_change:
        description = (
            "Коміт змінює файли менеджера пакетів (lockfile / pyproject / package.json). "
            "Версійні зрушення в існуючих залежностях типово викликають peer-conflict "
            "або runtime ImportError навіть без додавання нових пакетів."
        )
    else:
        description = (
            "Профіль коміту нагадує сценарій конфлікту залежностей (за поведінкою "
            "автора та проєкту), хоча самі lockfile-и наразі не зачеплено. "
            "Перевірте, чи не оновилися транзитивні залежності."
        )
    return Recommendation(
        severity="HIGH",
        category="DEPENDENCY",
        title="Можливий конфлікт залежностей",
        description=description,
        actions=[
            "Запустити npm ls / pip check / cargo tree локально перед push",
            "Зафіксувати точні версії в lockfile (npm ci, uv lock)",
            "Перевірити CHANGELOG великих оновлень на breaking changes",
        ],
        estimated_impact="Зменшує ймовірність dependency_error приблизно на 60%",
    )


def _docker_recommendation(image_size_mb: float, base_size_mb: float) -> Recommendation:
    # Size estimate is only meaningful when the Dockerfile was actually
    # fetched + parsed (yields a non-trivial estimate). Otherwise both args
    # default to 0 and rendering "~0 МБ" looks like a bug to the reviewer.
    if image_size_mb >= 1.0:
        size_sentence = (
            f"Прогнозований розмір остаточного образу ~{image_size_mb:.0f} МБ "
            f"(base {base_size_mb:.0f} МБ). Це може перевищити доступне місце на runner-і "
            "або спричинити timeout під час push до registry."
        )
    else:
        size_sentence = (
            "Зміни у Dockerfile підвищують ризик помилок збірки — "
            "перевірте base image, шари кешу та доступність залежностей."
        )
    return Recommendation(
        severity="HIGH",
        category="DOCKER",
        title="Високий ризик невдалої збірки Docker-образу",
        description=size_sentence,
        actions=[
            "Перейти на slim/alpine base image (наприклад python:3.11-slim замість 3.11)",
            "Очищати apt/pip кеш в одному RUN: rm -rf /var/lib/apt/lists/*",
            "Використовувати multi-stage build для відокремлення build-toolchain",
            "Додати .dockerignore для виключення тестів та документації",
        ],
        estimated_impact="Зменшує розмір образу типово на 40-70%",
    )


def _network_recommendation(run_attempt: int) -> Recommendation:
    return Recommendation(
        severity="MEDIUM",
        category="NETWORK",
        title="Підвищений ризик мережевих помилок",
        description=(
            "Профіль коміту відповідає сценарію, де часто виникають мережеві помилки "
            f"(retries-pattern, поточна спроба #{run_attempt})."
        ),
        actions=[
            "Додати retry-логіку для зовнішніх запитів (curl --retry 3, requests з urllib3 retry)",
            "Закешувати непостійні зовнішні артефакти (Docker pull through cache, npm proxy)",
            "Винести залежність від нестабільних зовнішніх API в окремий job з continue-on-error",
        ],
        estimated_impact="Зменшує network-related failures приблизно на 40%",
    )


def _test_failure_recommendation(features: dict[str, float]) -> Recommendation:
    test_count = int(features.get("feat_test_dir_changes", 0))
    py = int(features.get("feat_ext_py_count", 0))
    js_ts = int(features.get("feat_ext_js_count", 0)) + int(features.get("feat_ext_ts_count", 0))
    go = int(features.get("feat_ext_go_count", 0))

    actions = [
        "Запустіть повний test-suite локально перед push: переконайтеся що нові тести "
        "проходять і не ламають існуючі",
        "Перевірте assertion-логіку — найчастіша причина test-failure це невірне "
        "очікування у `assert` (`==` замість `is`, забутий `.strip()`, etc.)",
        "Перегляньте fixtures та mocks: чи вони відображають актуальний production-код "
        "після останніх рефакторингів",
    ]
    if py >= 1:
        actions.append(
            "Python: `pytest -xvs tests/test_<file>.py::<name>` для швидкої локальної перевірки"
        )
    if js_ts >= 1:
        actions.append(
            "JS/TS: `vitest run tests/<name>.test.ts` або `jest --testPathPattern=<name>`"
        )
    if go >= 1:
        actions.append("Go: `go test ./... -run TestName -v`")

    return Recommendation(
        severity="MEDIUM",
        category="TESTS",
        title="Імовірний test-failure",
        description=(
            f"Коміт містить зміни лише у test-файлах ({test_count} test-шляхів). "
            "Найімовірніша причина потенційного збою — нова assertion або зламана "
            "fixture, що падає під час pytest/vitest/go test."
        ),
        actions=actions,
        estimated_impact="Локальний запуск тестів виявляє 95%+ test-failures до push",
    )


def _other_failure_recommendation(features: dict[str, float]) -> Recommendation:
    test_only = features.get("feat_test_only_changes_int", 0) >= 1
    has_lint = features.get("feat_has_lint_config_change_int", 0) >= 1
    title = "Ймовірна помилка тестів або лінтера"
    description = (
        "Профіль коміту відповідає звичайному failure-сценарію (test/lint/typecheck), "
        "що не підпадає під специфічні категорії OOM/timeout/Docker/dependency."
    )
    actions = [
        "Запустити повний test-suite та лінтер локально перед push",
        "Перевірити останні зміни на assertion-логіку та граничні випадки",
    ]
    if test_only:
        actions.append(
            "Зміни торкаються лише тестів — перевірте, чи fixtures та mocks "
            "узгоджені з production-кодом"
        )
    if has_lint:
        actions.append(
            "Змінено конфігурацію лінтера — перевірте чи всі попередні правила "
            "ще задовольняються (npm run lint, ruff check, mypy)"
        )
    return Recommendation(
        severity="MEDIUM",
        category="TESTS",
        title=title,
        description=description,
        actions=actions,
        estimated_impact="Зменшує ймовірність тестових збоїв приблизно на 30-40%",
    )


def generate_recommendations(
    predicted_class: str,
    *,
    risk_score: float,
    class_probabilities: dict[str, float],
    predicted_memory_mb: float,
    predicted_duration_seconds: float,
    features: dict[str, float],
) -> list[Recommendation]:
    recs: list[Recommendation] = []

    if predicted_class == "oom_killed" or class_probabilities.get("oom_killed", 0) > 0.25:
        recs.append(
            _oom_recommendation(
                predicted_memory_mb=predicted_memory_mb,
                image_growth=features.get("feat_image_growth_ratio", 1.0),
            )
        )
    if predicted_class == "test_timeout" or class_probabilities.get("test_timeout", 0) > 0.25:
        recs.append(_timeout_recommendation(predicted_duration_seconds))

    if predicted_class == "dependency_error" or features.get("feat_new_deps_count", 0) >= 5:
        recs.append(
            _dependency_recommendation(
                int(features.get("feat_new_deps_count", 0)),
                bool(features.get("feat_has_dependency_change_int", 0)),
            )
        )

    if predicted_class == "docker_build_failed" or (
        features.get("feat_has_dockerfile_change_int", 0)
        and features.get("feat_image_growth_ratio", 0) > 5
    ):
        recs.append(
            _docker_recommendation(
                image_size_mb=float(
                    pow(2.71828, features.get("feat_final_image_size_mb_log", 0)) - 1
                ),
                base_size_mb=0.0,
            )
        )

    if predicted_class == "network_error" or features.get("feat_run_attempt_gt1", 0):
        recs.append(_network_recommendation(int(features.get("feat_run_attempt", 1))))

    if predicted_class == "test_failure" or class_probabilities.get("test_failure", 0) > 0.25:
        recs.append(_test_failure_recommendation(features))

    if predicted_class == "other_failure":
        recs.append(_other_failure_recommendation(features))

    if not recs and risk_score > 0.5:
        recs.append(
            Recommendation(
                severity="MEDIUM",
                category="GENERAL",
                title="Помірний ризик невдачі без явної категорії",
                description=(
                    f"Сукупний ризик невдачі {risk_score * 100:.0f}% без домінантного "
                    "класу. Рекомендується ручний перегляд коміту перед merge."
                ),
                actions=[
                    "Запустити локально пайплайн через act / docker compose",
                    "Перевірити останні зміни залежностей",
                ],
            )
        )

    recs.sort(key=lambda r: SEVERITY_ORDER.get(r.severity, 99))
    return recs
