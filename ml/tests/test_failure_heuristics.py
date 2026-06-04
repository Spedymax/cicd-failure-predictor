from data_collection.failure_heuristics import classify, classify_with_evidence


def test_returns_none_for_empty_log():
    assert classify(None) is None
    assert classify("") is None


def test_oom_patterns():
    samples = [
        "process killed (exit code 137)",
        "OOMKilled\nstderr",
        "MemoryError: Unable to allocate 5.0 GiB",
        "fatal error: runtime: out of memory",
        "Cannot allocate memory",
    ]
    for s in samples:
        assert classify(s) == "oom_killed", s


def test_timeout_patterns():
    samples = [
        "Error: The job has exceeded the maximum execution time of 360 minutes",
        "Test timed out after 30000ms",
        "pytest.Timeout: deadline exceeded",
    ]
    for s in samples:
        assert classify(s) == "test_timeout", s


def test_dependency_patterns():
    samples = [
        "npm ERR! ERESOLVE unable to resolve dependency tree",
        "ERROR: Could not find a version that satisfies the requirement",
        "version solving failed: requires foo >=2 and bar <2",
    ]
    for s in samples:
        assert classify(s) == "dependency_error", s


def test_docker_patterns():
    samples = [
        "Error response from daemon: dockerfile parse error",
        "ERROR: failed to solve: process \"/bin/sh -c\" exited",
        "no space left on device",
    ]
    for s in samples:
        assert classify(s) == "docker_build_failed", s


def test_network_patterns():
    samples = [
        "curl: (6) Could not resolve host: registry.npmjs.org",
        "ECONNREFUSED 127.0.0.1:5432",
        "Temporary failure in name resolution",
        "Connection timed out after 30s",
    ]
    for s in samples:
        assert classify(s) == "network_error", s


def test_classify_with_evidence_returns_match():
    label, evidence = classify_with_evidence("npm ERR! ERESOLVE issue here")
    assert label == "dependency_error"
    assert evidence == "ERESOLVE"


def test_priority_oom_over_other_noise():
    log = "Error: ERESOLVE peer dep ... but actually killed (exit code 137)"
    assert classify(log) == "oom_killed"


def test_skips_match_inside_test_code():
    # OOM-related token is only inside test source — must NOT be reported
    # as oom_killed. The FAILED tests/... line still maps to other_failure
    # which is the correct signal for an assertion-style failure.
    log = (
        "Running tests...\n"
        "    def test_handles_oomkilled_gracefully(self):\n"
        "        # ensure scheduler retries when pod was OOMKilled\n"
        "        ...\n"
        "FAILED tests/test_scheduler.py::test_oom - AssertionError\n"
    )
    assert classify(log) == "other_failure"


def test_skips_match_inside_comment():
    log = "# OOMKilled is the most common cause we observe\nLine without the marker\n"
    assert classify(log) is None


def test_keeps_match_in_actual_runner_output():
    log = (
        "step ran for 200ms\n"
        "process killed (exit code 137)\n"
        "build failed: container terminated\n"
    )
    assert classify(log) == "oom_killed"


def test_skips_kubectl_table_output():
    log = (
        "NAME                READY   STATUS      RESTARTS   AGE\n"
        "| my-pod          | 0/1   | OOMKilled  | 3         | 5m  |\n"
    )
    assert classify(log) is None


def test_skips_markdown_docstring_mention():
    log = "Documentation reference example: process exit code 137 (OOMKilled)\nactual log line\n"
    # Single line containing 'example' should be rejected; classifier must look
    # for an unambiguous match elsewhere — there isn't one, so None.
    assert classify(log) is None


def test_other_failure_pytest_assertion():
    log = "Running pytest...\nFAILED tests/test_user.py::test_login - AssertionError\n"
    assert classify(log) == "other_failure"


def test_other_failure_jest_summary():
    log = "Test Suites: 3 failed, 12 passed\nTests: 5 failed, 102 passed\n"
    assert classify(log) == "other_failure"


def test_other_failure_eslint():
    log = "running lint\nESLint failed with 4 errors\n"
    assert classify(log) == "other_failure"


def test_other_failure_go_test():
    log = "running go test\nFAIL    github.com/foo/bar/pkg/baz   0.123s\n"
    assert classify(log) == "other_failure"


def test_specific_class_takes_priority_over_other_failure():
    log = "FAILED tests/test_x.py::test_y\nprocess killed (exit code 137)\n"
    assert classify(log) == "oom_killed"
