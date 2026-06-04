"""Heuristic classifier mapping GitHub Actions job logs to failure_class.

Logs from the public Actions API rarely include the structured exit reason
(e.g. OOMKilled), so we rely on regex patterns over stdout/stderr emitted by
common runners and tools. The classifier is intentionally conservative: when
no pattern matches we return None and the row is later treated as
"unclassified failure" — those rows are kept out of the training set.
"""

from __future__ import annotations

import re

OOM_PATTERNS = [
    re.compile(r"\boomkilled\b", re.IGNORECASE),
    re.compile(r"out of memory", re.IGNORECASE),
    re.compile(r"cannot allocate memory", re.IGNORECASE),
    re.compile(r"exit code 137"),
    re.compile(r"killed.*signal 9", re.IGNORECASE),
    re.compile(r"MemoryError"),
    re.compile(r"std::bad_alloc"),
    re.compile(r"fatal error: runtime: out of memory"),
    # JVM heap exhaustion (Maven, Gradle, sbt, Kotlin/JS)
    re.compile(r"java\.lang\.OutOfMemoryError", re.IGNORECASE),
    re.compile(r"GC overhead limit exceeded", re.IGNORECASE),
    re.compile(r"Java heap space", re.IGNORECASE),
    # Node.js V8 heap (very common on large monorepos with ts-loader / webpack)
    re.compile(r"FATAL ERROR:.*allocation failed.*heap out of memory", re.IGNORECASE),
    re.compile(r"JavaScript heap out of memory", re.IGNORECASE),
    re.compile(r"v8::internal::Heap::FatalProcessOutOfMemory", re.IGNORECASE),
    # Linux kernel OOM-killer in dmesg-style output
    re.compile(r"\bOom-?killer\b.*invoked", re.IGNORECASE),
    re.compile(r"Out of memory:.*Kill process", re.IGNORECASE),
    re.compile(r"Memory cgroup out of memory", re.IGNORECASE),
    # Go runtime OOM
    re.compile(r"runtime: out of memory:\s*cannot allocate", re.IGNORECASE),
    # Rust / cargo
    re.compile(r"memory allocation of \d+ bytes failed", re.IGNORECASE),
    # Generic 'allocator' patterns
    re.compile(r"\bENOMEM\b"),
    re.compile(r"could not reserve enough space for", re.IGNORECASE),
]

TIMEOUT_PATTERNS = [
    re.compile(r"the job .*has exceeded the maximum execution time", re.IGNORECASE),
    re.compile(r"timeout-minutes", re.IGNORECASE),
    re.compile(r"test timed out after", re.IGNORECASE),
    re.compile(r"jest .*exceeded.*timeout", re.IGNORECASE),
    re.compile(r"pytest.*Timeout"),
    re.compile(r"deadline exceeded", re.IGNORECASE),
    re.compile(r"step .*has timed out", re.IGNORECASE),
    # GitHub Actions specific cancellation by timeout
    re.compile(r"Error:.*The action has timed out", re.IGNORECASE),
    re.compile(r"step .*was cancelled because the step has timed out", re.IGNORECASE),
    re.compile(r"##\[error\].*timed out", re.IGNORECASE),
    # Go context deadline
    re.compile(r"context deadline exceeded", re.IGNORECASE),
    # E2E test runners
    re.compile(r"Cypress .*timed out", re.IGNORECASE),
    re.compile(r"playwright .*Timeout exceeded", re.IGNORECASE),
    re.compile(r"Test timeout of \d+ms exceeded", re.IGNORECASE),
    re.compile(r"TimeoutError:\s*page\.", re.IGNORECASE),
    # Selenium / Selenium-WebDriver
    re.compile(r"selenium\.common\.exceptions\.TimeoutException", re.IGNORECASE),
    re.compile(r"Wait timed out after", re.IGNORECASE),
    # JUnit / Maven Surefire
    re.compile(r"test timed out after \d+ \w+", re.IGNORECASE),
    re.compile(r"org\.junit.*timed out", re.IGNORECASE),
    # Gradle / Kotlin
    re.compile(r"Timeout has been exceeded", re.IGNORECASE),
    # Pytest-timeout specific
    re.compile(r"Failed: Timeout >\d+\.\d+s", re.IGNORECASE),
    # General curl / wget
    re.compile(r"Operation timed out after \d+", re.IGNORECASE),
    # Mocha
    re.compile(r"Error: Timeout of \d+ms exceeded", re.IGNORECASE),
    # Generic 'job exceeded' / 'maximum runtime'
    re.compile(r"maximum (?:run)?time .*exceeded", re.IGNORECASE),
]

DEPENDENCY_PATTERNS = [
    # npm / yarn / pnpm
    re.compile(r"ERESOLVE"),
    re.compile(r"unable to resolve dependency tree", re.IGNORECASE),
    re.compile(r"npm ERR! peer dep", re.IGNORECASE),
    re.compile(r"npm ERR! code E\w+"),
    re.compile(r"npm ERR! 404"),
    re.compile(r"yarn (?:install )?error.*depend", re.IGNORECASE),
    re.compile(r"ERR_PNPM_PEER_DEP_ISSUES", re.IGNORECASE),
    re.compile(r"ERR_PNPM_NO_MATCHING_VERSION", re.IGNORECASE),
    re.compile(r"workspace dependency .* could not be resolved", re.IGNORECASE),
    # pip / poetry / uv
    re.compile(r"could not find a version that satisfies", re.IGNORECASE),
    re.compile(r"no matching distribution found", re.IGNORECASE),
    re.compile(r"ResolutionImpossible", re.IGNORECASE),
    re.compile(r"version solving failed", re.IGNORECASE),
    re.compile(r"pip._vendor.resolvelib.*ResolutionImpossible", re.IGNORECASE),
    re.compile(r"Because.*depends on.*which depends on", re.IGNORECASE),
    re.compile(r"\bSolverProblemError\b"),
    # uv / pip-tools
    re.compile(r"No solution found when resolving dependencies", re.IGNORECASE),
    re.compile(r"resolver-undetermined", re.IGNORECASE),
    # JVM (Maven, Gradle)
    re.compile(r"Could not resolve dependencies for project", re.IGNORECASE),
    re.compile(r"Plugin .* or one of its dependencies could not be resolved", re.IGNORECASE),
    re.compile(r"Could not resolve all files for configuration", re.IGNORECASE),
    re.compile(r"Could not find\s+\S+:\s*\S+:\s*[\d\.]+"),
    # Cargo (Rust)
    re.compile(r"error: failed to select a version", re.IGNORECASE),
    re.compile(r"package .*collides with a previously imported package", re.IGNORECASE),
    re.compile(r"error: no matching package named", re.IGNORECASE),
    re.compile(r"failed to resolve patches", re.IGNORECASE),
    # Go modules
    re.compile(r"cannot find module providing package", re.IGNORECASE),
    re.compile(r"module .* found, but does not contain package", re.IGNORECASE),
    re.compile(r"go: errors parsing go\.(mod|sum)", re.IGNORECASE),
    re.compile(r"unknown revision", re.IGNORECASE),
    # NuGet / .NET
    re.compile(r"Unable to resolve dependencies for project", re.IGNORECASE),
    re.compile(r"NU1\d{3}:", re.IGNORECASE),  # NuGet error codes
    # Composer / PHP (kept for future PHP repos)
    re.compile(r"Your requirements could not be resolved", re.IGNORECASE),
    # Generic
    re.compile(r"VersionConflict"),
    re.compile(r"package .*has no installation candidate", re.IGNORECASE),
    re.compile(r"dependency .*not satisfied", re.IGNORECASE),
]

DOCKER_PATTERNS = [
    re.compile(r"error response from daemon", re.IGNORECASE),
    re.compile(r"no space left on device", re.IGNORECASE),
    re.compile(r"failed to solve", re.IGNORECASE),
    re.compile(r"failed to build .* dockerfile", re.IGNORECASE),
    re.compile(r"buildx failed"),
    re.compile(r"manifest unknown", re.IGNORECASE),
    re.compile(r"unable to prepare context", re.IGNORECASE),
    # Registry / auth issues
    re.compile(r"denied:\s*requested access to the resource is denied", re.IGNORECASE),
    re.compile(r"unauthorized: authentication required", re.IGNORECASE),
    re.compile(r"failed to fetch oauth token", re.IGNORECASE),
    re.compile(r"pull access denied", re.IGNORECASE),
    re.compile(r"toomanyrequests:\s*you have reached", re.IGNORECASE),
    re.compile(r"docker login.*failed", re.IGNORECASE),
    # BuildKit / buildx exit codes
    re.compile(r"BuildKit.*exited with code [^0]"),
    re.compile(r"docker buildx .*error", re.IGNORECASE),
    re.compile(r"failed to copy files", re.IGNORECASE),
    re.compile(r"failed to compute cache key", re.IGNORECASE),
    re.compile(r"docker:\s*invalid reference format", re.IGNORECASE),
    re.compile(r"the command .* returned a non-zero code", re.IGNORECASE),
    # Dockerfile syntax
    re.compile(r"Dockerfile.*\bUnknown instruction\b", re.IGNORECASE),
    re.compile(r"parse error on line \d+", re.IGNORECASE),
    re.compile(r"ARG \w+ requires.*value", re.IGNORECASE),
    # Disk / device errors common in CI runners
    re.compile(r"write\s.*:\s*no space left on device", re.IGNORECASE),
    re.compile(r"could not stat .*: device or resource busy", re.IGNORECASE),
    # Docker compose
    re.compile(r"ERROR:.*docker-compose", re.IGNORECASE),
]

NETWORK_PATTERNS = [
    # POSIX errno
    re.compile(r"ECONNREFUSED"),
    re.compile(r"ECONNRESET"),
    re.compile(r"ENETUNREACH"),
    re.compile(r"EHOSTUNREACH"),
    re.compile(r"\bETIMEDOUT\b"),
    re.compile(r"\bEAI_AGAIN\b"),
    re.compile(r"ENOTFOUND"),
    # DNS / connection
    re.compile(r"could not resolve host", re.IGNORECASE),
    re.compile(r"name or service not known", re.IGNORECASE),
    re.compile(r"connection .*timed out", re.IGNORECASE),
    re.compile(r"temporary failure in name resolution", re.IGNORECASE),
    re.compile(r"could not connect to .*server", re.IGNORECASE),
    re.compile(r"getaddrinfo (?:failed|ENOTFOUND|EAI_AGAIN)", re.IGNORECASE),
    # TLS / SSL
    re.compile(r"ssl.*handshake failed", re.IGNORECASE),
    re.compile(r"certificate verify failed", re.IGNORECASE),
    re.compile(r"unable to get local issuer certificate", re.IGNORECASE),
    re.compile(r"SSLEOFError|SSLCertVerificationError", re.IGNORECASE),
    # HTTP client tooling
    re.compile(r"npm ERR! network", re.IGNORECASE),
    re.compile(r"yarn install.*network", re.IGNORECASE),
    re.compile(r"curl:\s*\(\d+\)\s*(?:Could not|Failed)", re.IGNORECASE),
    re.compile(r"wget:\s*unable to resolve", re.IGNORECASE),
    re.compile(r"HTTPS?ConnectionPool.*Max retries exceeded", re.IGNORECASE),
    re.compile(r"requests\.exceptions\.Connection(?:Error|Timeout)", re.IGNORECASE),
    re.compile(r"httpx\.Connect(?:Error|Timeout)", re.IGNORECASE),
    re.compile(r"urlopen error", re.IGNORECASE),
    # apt / yum mirror failures (Debian/Alpine in Docker)
    re.compile(r"failed to fetch.*apt\.", re.IGNORECASE),
    re.compile(r"could not retrieve mirror list", re.IGNORECASE),
    re.compile(r"403\s+Forbidden.*archive\.ubuntu", re.IGNORECASE),
    # Go HTTP
    re.compile(r"dial tcp.*i/o timeout", re.IGNORECASE),
    re.compile(r"net/http:\s*TLS handshake timeout", re.IGNORECASE),
    # GitHub Actions specific
    re.compile(r"Error downloading.*from\s+https?://", re.IGNORECASE),
    re.compile(r"failed to make request:.*connect", re.IGNORECASE),
]

# Test/lint/typecheck failures — the most common open-source CI failure
# class (~70-90% of all failures). Mapped to ``other_failure`` to give the
# classifier a target instead of dropping these rows.
OTHER_FAILURE_PATTERNS = [
    # pytest
    re.compile(r"FAILED\s+tests?[/\\].+::", re.IGNORECASE),
    re.compile(r"^={3,}\s*FAILURES\s*={3,}", re.MULTILINE),
    re.compile(r"\b\d+ failed,?\s+\d+ passed\b", re.IGNORECASE),
    re.compile(r"\b\d+ tests? failed\b", re.IGNORECASE),
    re.compile(r"^\s*FAIL\b", re.MULTILINE),
    re.compile(r"\bAssertionError\b"),
    # jest / vitest / mocha / jasmine / ava
    re.compile(r"^\s*Tests:\s*.*\d+ failed", re.MULTILINE | re.IGNORECASE),
    re.compile(r"Test Suites:\s*\d+ failed", re.IGNORECASE),
    re.compile(r"^\s*✗\s", re.MULTILINE),
    re.compile(r"^\s*●\s", re.MULTILINE),
    re.compile(r"jest:.*failing", re.IGNORECASE),
    re.compile(r"^\s*\d+ failing\b", re.MULTILINE | re.IGNORECASE),
    re.compile(r"Vitest .*FAIL", re.IGNORECASE),
    re.compile(r"\bExpect(?:ed)?(?:ation)? failed\b", re.IGNORECASE),
    re.compile(r"expect\(.*\)\.to.*FAIL", re.IGNORECASE),
    # ruby rspec / minitest
    re.compile(r"\d+ examples?,\s*[1-9]\d* failures?", re.IGNORECASE),
    re.compile(r"\d+ runs?, \d+ assertions?, [1-9]\d* failures?", re.IGNORECASE),
    # go test
    re.compile(r"^--- FAIL:\s+\w+", re.MULTILINE),
    re.compile(r"^FAIL\s+\S+\s+\d+\.\d+s$", re.MULTILINE),
    re.compile(r"\bgo test\b.*FAIL", re.IGNORECASE),
    re.compile(r"\bgolangci-lint\b.*errors?", re.IGNORECASE),
    # rust cargo test
    re.compile(r"\bcargo test\b.*FAILED", re.IGNORECASE),
    re.compile(r"^test result:\s*FAILED", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^failures:\s*$", re.MULTILINE | re.IGNORECASE),
    # maven surefire / failsafe
    re.compile(r"BUILD FAILURE", re.IGNORECASE),
    re.compile(r"Tests run:\s*\d+,\s*Failures:\s*[1-9]\d*", re.IGNORECASE),
    re.compile(r"There (?:were|are) test failures", re.IGNORECASE),
    # gradle
    re.compile(r"BUILD FAILED", re.IGNORECASE),
    re.compile(r"Task\s*'[^']+'\s*FAILED", re.IGNORECASE),
    re.compile(r"There were failing tests", re.IGNORECASE),
    # .NET / xunit / nunit
    re.compile(r"Total tests:.*Failed:\s*[1-9]", re.IGNORECASE),
    re.compile(r"Test Run Failed", re.IGNORECASE),
    re.compile(r"^\s*Failed\s+\S+\.\S+", re.MULTILINE),
    # Linters & type checkers
    re.compile(r"mypy.*error:", re.IGNORECASE),
    re.compile(r"\bFound \d+ errors? in \d+ files?\b", re.IGNORECASE),
    re.compile(r"\bruff\b.*\bfailed\b", re.IGNORECASE),
    re.compile(r"Found \d+ error", re.IGNORECASE),  # ruff / mypy general
    re.compile(r"\beslint\b.*\bfailed\b", re.IGNORECASE),
    re.compile(r"\d+ problems?\s*\(\d+ errors?,\s*\d+ warnings?\)", re.IGNORECASE),
    re.compile(r"flake8 .*errors?", re.IGNORECASE),
    re.compile(r"^.*error TS\d+:", re.MULTILINE),  # tsc
    re.compile(r"^Error:.*?error TS\d+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"stylelint.*\d+ problems", re.IGNORECASE),
    re.compile(r"prettier.*Code style issues found", re.IGNORECASE),
    re.compile(r"^\s*black.*would reformat", re.MULTILINE | re.IGNORECASE),
    re.compile(r"isort.*Imports are incorrectly sorted", re.IGNORECASE),
    # CI script-level
    re.compile(r"##\[error\].*Process completed with exit code [^0]", re.IGNORECASE),
    re.compile(r"Error:\s+Process completed with exit code [^0]", re.IGNORECASE),
]


CATEGORY_PATTERNS: list[tuple[str, list[re.Pattern[str]]]] = [
    # Specific causes are checked first because they often coincide
    # with generic test-failure noise in the same log.
    ("oom_killed", OOM_PATTERNS),
    ("test_timeout", TIMEOUT_PATTERNS),
    ("dependency_error", DEPENDENCY_PATTERNS),
    ("docker_build_failed", DOCKER_PATTERNS),
    ("network_error", NETWORK_PATTERNS),
    ("other_failure", OTHER_FAILURE_PATTERNS),
]


# Context filters reject matches that look like they came from test code,
# code comments, markdown documentation or pretty-printed kubectl output —
# none of those reflect the *actual* runner-level failure cause.
_REJECT_LINE_PATTERNS = [
    re.compile(r"^\s*[#/]"),                # # or // or /* comments
    re.compile(r"^\s*\*"),                  # * comment-block continuation
    re.compile(r"^\s*<!--"),                # html/markdown comments
    re.compile(r"^\s*\|"),                  # markdown / kubectl table row
    re.compile(r"^\s*[-=]{3,}\s*$"),        # markdown rule / table separator (whole line only)
    re.compile(r"\bdef\s+test_", re.IGNORECASE),
    re.compile(r"\bclass\s+\w*Test", re.IGNORECASE),
    re.compile(r"\b(it|describe|test)\(", re.IGNORECASE),
    re.compile(r"\bassertraises", re.IGNORECASE),
    re.compile(r"\bmock", re.IGNORECASE),
    re.compile(r"docstring|example|illustrat", re.IGNORECASE),
]


def _line_around(text: str, position: int) -> str:
    """Return the full line containing ``position`` in ``text``."""
    start = text.rfind("\n", 0, position)
    end = text.find("\n", position)
    if start < 0:
        start = 0
    else:
        start += 1
    if end < 0:
        end = len(text)
    return text[start:end]


def _is_legit_match(text: str, match: re.Match[str]) -> bool:
    line = _line_around(text, match.start())
    return not any(p.search(line) for p in _REJECT_LINE_PATTERNS)


def classify(log_text: str | None) -> str | None:
    """Return the most specific failure class detected in the log, or None."""
    label, _ = classify_with_evidence(log_text)
    return label


def classify_with_evidence(log_text: str | None) -> tuple[str | None, str | None]:
    """Same as classify(), but also returns the matched substring.

    The matched substring is rejected when its surrounding line looks like
    test source / code comment / markdown / pretty-printed kubectl output —
    those are common false-positive sources, especially for tokens like
    ``OOMKilled`` that appear verbatim in test fixtures and docs.
    """
    if not log_text:
        return None, None
    for label, patterns in CATEGORY_PATTERNS:
        for pat in patterns:
            for m in pat.finditer(log_text):
                if _is_legit_match(log_text, m):
                    return label, m.group(0)
    return None, None
