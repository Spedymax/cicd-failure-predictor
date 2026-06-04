#!/usr/bin/env python3
"""Run the live defence demo: 4 commits, 4 ML predictions, 4 CI verdicts.

Usage: ./scripts/demo_run.py [--fast]
  --fast   skip the CI-verdict wait (use when network is slow)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo-repo"
BACKEND = "http://127.0.0.1:8000"
REPO = "Spedymax/cicd-predictor-demo"

C_OK = "\033[32m"
C_WARN = "\033[33m"
C_BAD = "\033[31m"
C_DIM = "\033[2m"
C_BOLD = "\033[1m"
C_RST = "\033[0m"

BASELINE_REQ = """pytest>=8.0
httpx>=0.27
pydantic>=2.10
"""
BASELINE_DOCKER = """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ src/
CMD ["python", "-m", "src.app"]
"""


def color_decision(d: str) -> str:
    palette = {"auto_approve": C_OK, "warn": C_WARN, "block": C_BAD}
    return f"{palette.get(d, '')}{d.upper()}{C_RST}"


def color_ci(c: str | None) -> str:
    if c == "success":
        return f"{C_OK}✓ pass{C_RST}"
    if c == "failure":
        return f"{C_BAD}✗ fail{C_RST}"
    return f"{C_DIM}…{C_RST}"


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=DEMO, capture_output=True, text=True)


def write(rel: str, body: str) -> None:
    (DEMO / rel).write_text(body)


def reset_baseline() -> None:
    write("requirements.txt", BASELINE_REQ)
    write("Dockerfile", BASELINE_DOCKER)
    # Strip any leftover demo test file.
    for stale in ("tests/test_bad.py", "tests/test_demo_extra.py"):
        p = DEMO / stale
        if p.exists():
            p.unlink()


def push(label: str, msg: str) -> str:
    git("add", "-A")
    if git("diff", "--cached", "--quiet").returncode == 0:
        return ""
    git("commit", "-m", msg, "-q")
    sha = git("rev-parse", "--short", "HEAD").stdout.strip()
    git("push", "origin", "main")
    return sha


def fetch_prediction(sha: str, retries: int = 8) -> dict | None:
    for _ in range(retries):
        time.sleep(2)
        try:
            data = json.loads(
                urllib.request.urlopen(
                    f"{BACKEND}/api/v1/predictions?limit=5&source=demo", timeout=5
                ).read()
            )
        except Exception:
            continue
        for p in data.get("items", []):
            if p["commit_short"].startswith(sha[:7]):
                return p
    return None


def fetch_ci(sha: str, retries: int = 30) -> str:
    # Returns 'success' | 'failure' | '?'
    for _ in range(retries):
        out = subprocess.run(
            ["gh", "run", "list", "--repo", REPO, "--limit", "10",
             "--json", "headSha,status,conclusion"],
            capture_output=True, text=True,
        )
        try:
            rows = json.loads(out.stdout)
        except Exception:
            rows = []
        for r in rows:
            if r["headSha"].startswith(sha):
                if r["status"] == "completed":
                    return r.get("conclusion") or "?"
                break
        time.sleep(15)
    return "?"


def banner(title: str) -> None:
    print()
    print(f"{C_BOLD}━━━ {title} ━━━{C_RST}")


def step(label: str, msg: str, mutate, *, wait_ci: bool) -> dict:
    banner(label)
    print(f"{C_DIM}commit:{C_RST} {msg}")
    mutate()
    sha = push(label, msg)
    if not sha:
        print(f"{C_DIM}(no diff, skipped){C_RST}")
        return {"label": label}
    print(f"{C_DIM}pushed:{C_RST} {sha}  ", end="", flush=True)
    pred = fetch_prediction(sha)
    if pred is None:
        print(f"{C_BAD}prediction not received in time{C_RST}")
        return {"label": label, "sha": sha}
    print(
        f"ML → class={C_BOLD}{pred['predicted_class']}{C_RST} "
        f"decision={color_decision(pred['decision'])} "
        f"risk={pred['risk_score']:.2f} conf={pred['confidence']:.2f}"
    )
    ci = "?"
    if wait_ci:
        print(f"{C_DIM}awaiting CI…{C_RST}", end=" ", flush=True)
        ci = fetch_ci(sha)
        print(color_ci(ci))
    return {"label": label, "sha": sha, "pred": pred, "ci": ci}


def summary(rows: list[dict]) -> None:
    banner("SUMMARY")
    print(f"{'commit':32s} | {'class':22s} | {'decision':12s} | CI")
    print("-" * 90)
    for r in rows:
        if "pred" not in r:
            continue
        p = r["pred"]
        print(
            f"{r['label']:32s} | {p['predicted_class']:22s} | "
            f"{color_decision(p['decision']):20s} | {color_ci(r.get('ci'))}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip CI-verdict wait")
    args = ap.parse_args()
    wait_ci = not args.fast

    if not DEMO.is_dir():
        print(f"demo repo not found at {DEMO}", file=sys.stderr)
        return 1

    print(f"{C_BOLD}CI/CD Failure Predictor — live demo{C_RST}")
    print(f"backend: {BACKEND}  |  demo repo: {REPO}")
    print(f"fast mode: {'on (no CI wait)' if args.fast else 'off (waits for CI)'}")

    reset_baseline()
    # Bring main up to a known-green baseline first (silent commit).
    git("add", "-A")
    if git("diff", "--cached", "--quiet").returncode != 0:
        git("commit", "-m", "chore: reset baseline before demo", "-q")
        git("push", "origin", "main")
        time.sleep(2)

    results: list[dict] = []

    # 1. Trivial change → success / AUTO_APPROVE
    results.append(step(
        "1 README touch (trivial)",
        "docs: README touch",
        lambda: (DEMO / "README.md").open("a").write("\nminor update\n"),
        wait_ci=wait_ci,
    ))

    # 2. Valid deps bump — preflight should soften to AUTO
    results.append(step(
        "2 add rich (valid dep)",
        "deps: add rich",
        lambda: write("requirements.txt", BASELINE_REQ + "rich>=13.0\n"),
        wait_ci=wait_ci,
    ))

    # 3. Broken dep → WARN
    results.append(step(
        "3 fake package (broken dep)",
        "deps: bump to nonexistent pkg",
        lambda: write(
            "requirements.txt",
            BASELINE_REQ + "this-package-does-not-exist-zzz>=999.0\n",
        ),
        wait_ci=wait_ci,
    ))

    # Restore deps silently (so the next test isn't poisoned).
    write("requirements.txt", BASELINE_REQ)
    git("add", "-A")
    if git("diff", "--cached", "--quiet").returncode != 0:
        git("commit", "-m", "chore: restore deps", "-q")
        git("push", "origin", "main")
        time.sleep(2)

    # 4. Broken Dockerfile → BLOCK
    results.append(step(
        "4 broken Dockerfile (BLOCK)",
        "docker: invalid base + bogus instruction",
        lambda: write("Dockerfile", "FROM python:99.99-slim\nBOGUS_CMD doomsday\n"),
        wait_ci=wait_ci,
    ))

    # Restore on the way out, ready for the next demo run.
    reset_baseline()
    git("add", "-A")
    if git("diff", "--cached", "--quiet").returncode != 0:
        git("commit", "-m", "chore: restore baseline (post-demo)", "-q")
        git("push", "origin", "main")

    summary(results)
    print(
        f"\nDashboard:  {C_BOLD}http://localhost:3000{C_RST}  "
        f"(switch to the {C_BOLD}demo{C_RST} tab to see today's predictions)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
