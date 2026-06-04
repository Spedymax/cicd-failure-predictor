"""Mock-based tests for repo_discovery."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from app.services import repo_discovery


def _ctx(routes):
    fake = MagicMock()
    def _get(url, **_kw):
        for u, resp in routes:
            if u in url:
                return resp
        return MagicMock(status_code=404, json=lambda: {})
    fake.get.side_effect = _get
    cm = MagicMock()
    cm.__enter__.return_value = fake
    cm.__exit__.return_value = False
    return cm


def test_discovery_malformed_returns_empty() -> None:
    m = repo_discovery.discover_repository_metadata("noslash")
    assert m.language is None
    assert m.has_dockerfile is False
    assert m.package_manager is None


def test_discovery_extracts_language_and_dockerfile_and_pkgmgr() -> None:
    repo_resp = MagicMock(status_code=200, json=lambda: {"language": "Python"})
    contents_resp = MagicMock(
        status_code=200,
        json=lambda: [
            {"name": "Dockerfile"},
            {"name": "pyproject.toml"},
            {"name": "README.md"},
        ],
    )
    with patch.object(
        repo_discovery, "httpx",
        MagicMock(
            Client=MagicMock(return_value=_ctx([
                ("/contents/", contents_resp),
                ("/repos/", repo_resp),
            ])),
            HTTPError=httpx.HTTPError,
        ),
    ):
        m = repo_discovery.discover_repository_metadata("acme/widget")

    assert m.language == "Python"
    assert m.has_dockerfile is True
    # pyproject.toml is a Python pkg manager hint — accept whatever string the
    # module decides; just check it surfaced *something*.
    assert m.package_manager is not None


def test_discovery_swallows_http_errors() -> None:
    def raiser(*_a, **_kw):
        raise httpx.HTTPError("boom")
    fake = MagicMock()
    fake.get.side_effect = raiser
    cm = MagicMock()
    cm.__enter__.return_value = fake
    cm.__exit__.return_value = False
    with patch.object(
        repo_discovery, "httpx",
        MagicMock(
            Client=MagicMock(return_value=cm),
            HTTPError=httpx.HTTPError,
        ),
    ):
        m = repo_discovery.discover_repository_metadata("acme/x")
    # Never raises, returns defaults.
    assert m.language is None
    assert m.has_dockerfile is False
