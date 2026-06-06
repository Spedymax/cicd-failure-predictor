"""Trigger GitHub Actions workflow re-runs after a manual override.

When the predictor returns BLOCK, our workflow gate fires ``exit 1`` and
downstream jobs are skipped. A subsequent override (BLOCK → AUTO/WARN)
re-posts a passing status, but **GitHub Actions does not automatically
re-evaluate completed runs** — they stay failed forever unless someone
clicks "Re-run jobs".

This service finds the failed workflow runs for a commit SHA and asks
GitHub to re-run only the failed/skipped jobs. The next poll inside the
gate step will see the new verdict (now AUTO/WARN) and pass through,
allowing downstream lint/test/build to actually execute.

Best-effort: any HTTP/network error is logged and ignored.
"""

from __future__ import annotations

import logging

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Same prefixes skipped in github_status — synthetic repos don't exist.
_SYNTHETIC_PREFIXES = ("acme/", "demo/")


def rerun_failed_runs_for_sha(full_name: str, sha: str) -> int:
    """Re-run all failed/cancelled workflow runs at the given commit SHA.

    Returns the number of re-runs successfully triggered. Never raises.
    """
    if not full_name or "/" not in full_name:
        return 0
    if any(full_name.startswith(p) for p in _SYNTHETIC_PREFIXES):
        return 0
    settings = get_settings()
    headers = {
        "Authorization": f"Bearer {settings.github_api_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    triggered = 0
    try:
        with httpx.Client(timeout=8.0) as c:
            list_url = (
                f"https://api.github.com/repos/{full_name}/actions/runs?head_sha={sha}&per_page=20"
            )
            r = c.get(list_url, headers=headers)
            if r.status_code != 200:
                logger.warning(
                    "github_actions: list runs failed for %s@%s status=%d",
                    full_name,
                    sha[:8],
                    r.status_code,
                )
                return 0
            runs = r.json().get("workflow_runs", []) or []
            for run in runs:
                # Re-run only runs that actually failed at the gate — leave
                # successful runs untouched (no point) and in-progress alone.
                conclusion = run.get("conclusion")
                status = run.get("status")
                if status != "completed":
                    continue
                if conclusion not in ("failure", "cancelled", "timed_out"):
                    continue
                run_id = run.get("id")
                rerun_url = (
                    f"https://api.github.com/repos/{full_name}"
                    f"/actions/runs/{run_id}/rerun-failed-jobs"
                )
                rr = c.post(rerun_url, headers=headers)
                if 200 <= rr.status_code < 300:
                    triggered += 1
                    logger.info(
                        "github_actions: rerun triggered for %s run_id=%s",
                        full_name,
                        run_id,
                    )
                else:
                    logger.warning(
                        "github_actions: rerun failed for run_id=%s status=%d body=%s",
                        run_id,
                        rr.status_code,
                        rr.text[:200],
                    )
    except Exception as exc:  # noqa: BLE001
        logger.warning("github_actions: exception for %s@%s — %s", full_name, sha[:8], exc)
    return triggered
