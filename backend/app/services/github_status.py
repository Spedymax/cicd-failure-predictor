"""Post commit status checks back to GitHub so BLOCK actually gates merge.

The system stores predictions advisorily, but to **block merges** we need to
post a Commit Status (legacy Statuses API). When the repo's branch protection
requires the ``cicd-failure-predictor`` context, a ``failure`` status will
block merging the PR until the predictor's verdict changes (e.g. via override).

Mapping::

    AUTO_APPROVE → success    (green check; merge allowed)
    WARN         → success    (green check + "review recommended" description;
                               *does not block merge* — risk signal lives in
                               the description and the dashboard, branch
                               protection lets the PR through)
    BLOCK        → failure    (red X; branch protection blocks merge)

Only BLOCK is intended to gate merges via branch protection. WARN is an
informational signal — pending was tried first but GitHub treats pending as
"not yet passing" and that also blocks merge, which is too aggressive.

Endpoint: POST /repos/{owner}/{repo}/statuses/{sha}
Docs:     https://docs.github.com/en/rest/commits/statuses

The call is best-effort — failures are logged but do not raise. Only fires
when ``settings.github_post_status`` is true and the repository belongs to
a real GitHub owner (``acme/`` / ``demo/`` fixtures are skipped).
"""

from __future__ import annotations

import logging

import httpx

from app.core.config import get_settings
from app.core.resilient_http import call_with_resilience
from app.db.models import PredictionDecision

logger = logging.getLogger(__name__)

# Repo prefixes that are local fixtures / synthetic seed data — never post
# real status checks for these (the repos don't exist on GitHub).
_SYNTHETIC_PREFIXES = ("acme/", "demo/")

# Branch-protection context name. Add this in GitHub repo settings ->
# Branches -> Branch protection rule -> "Require status checks to pass".
STATUS_CONTEXT = "cicd-failure-predictor"

_DECISION_TO_STATE: dict[PredictionDecision, str] = {
    PredictionDecision.AUTO_APPROVE: "success",
    PredictionDecision.WARN: "success",
    PredictionDecision.BLOCK: "failure",
}

_DECISION_TO_DESCRIPTION: dict[PredictionDecision, str] = {
    PredictionDecision.AUTO_APPROVE: "Low CI failure risk — merge allowed",
    PredictionDecision.WARN: "Elevated CI failure risk — review recommended",
    PredictionDecision.BLOCK: "High CI failure risk — merge blocked",
}


def _should_post(full_name: str) -> bool:
    settings = get_settings()
    if not settings.github_post_status:
        return False
    if not full_name or "/" not in full_name:
        return False
    return not any(full_name.startswith(p) for p in _SYNTHETIC_PREFIXES)


def post_commit_status(
    full_name: str,
    sha: str,
    decision: PredictionDecision,
    prediction_id: int,
) -> bool:
    """Post a commit status to GitHub. Returns True on HTTP 2xx.

    Safe to call from request handlers — never raises; on network/HTTP error
    just logs a warning and returns False.
    """
    if not _should_post(full_name):
        logger.debug("github_status: skip %s (synthetic or disabled)", full_name)
        return False

    settings = get_settings()
    url = f"https://api.github.com/repos/{full_name}/statuses/{sha}"
    target_url = f"{settings.app_public_url.rstrip('/')}/predictions/{prediction_id}"
    state = _DECISION_TO_STATE.get(decision, "pending")
    description = _DECISION_TO_DESCRIPTION.get(decision, decision.value)
    body = {
        "state": state,
        "target_url": target_url,
        "description": description,
        "context": STATUS_CONTEXT,
    }
    headers = {
        "Authorization": f"Bearer {settings.github_api_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        with httpx.Client(timeout=8.0) as c:
            r = call_with_resilience("github", lambda: c.post(url, headers=headers, json=body))
        if 200 <= r.status_code < 300:
            logger.info(
                "github_status: posted state=%s for %s@%s pred=%d",
                state,
                full_name,
                sha[:8],
                prediction_id,
            )
            return True
        logger.warning(
            "github_status: failed %s@%s status=%d body=%s",
            full_name,
            sha[:8],
            r.status_code,
            r.text[:200],
        )
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("github_status: exception for %s@%s — %s", full_name, sha[:8], exc)
        return False
