"""Admin operations: model retraining trigger (FR-17)."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.audit import record_audit
from app.core.auth import require_roles
from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.db.models import User, UserRole
from app.db.session import get_db

router = APIRouter(prefix="/admin", tags=["admin"])

_require_admin = require_roles(UserRole.ADMIN)
_log = get_logger("admin.retrain")


def _runs_dir() -> Path:
    base = Path(get_settings().ml_model_dir).resolve().parent.parent / "retrain_runs"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _state_path(run_id: str) -> Path:
    return _runs_dir() / f"{run_id}.json"


def _write_state(run_id: str, **fields) -> None:
    path = _state_path(run_id)
    state = {}
    if path.exists():
        state = json.loads(path.read_text())
    state.update(fields)
    path.write_text(json.dumps(state, default=str))


class RetrainRequest(BaseModel):
    features: str | None = None  # parquet path; defaults to env REAL features
    out_version: str | None = None  # subfolder name under data/artifacts
    tune: bool = False
    n_trials: int = 40


class RetrainStarted(BaseModel):
    run_id: str
    status: str
    started_at: datetime


class RetrainStatus(BaseModel):
    run_id: str
    status: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    return_code: int | None = None
    out_dir: str | None = None
    error: str | None = None
    log_tail: str | None = None


def _run_training(run_id: str, features: str, out_dir: str, tune: bool, n_trials: int) -> None:
    """Spawn the training script as a subprocess and capture stdout/stderr."""
    log_path = _runs_dir() / f"{run_id}.log"
    cmd = [
        sys.executable,
        "-m",
        "training.train",
        "--features",
        features,
        "--out",
        out_dir,
    ]
    if tune:
        cmd.extend(["--tune", "--n-trials", str(n_trials)])
    repo_root = Path(__file__).resolve().parents[3]
    cwd = repo_root / "ml"
    _write_state(run_id, status="running", cmd=" ".join(cmd))
    _log.info("retrain.start", run_id=run_id, cmd=cmd, cwd=str(cwd))
    try:
        with open(log_path, "w") as lf:
            proc = subprocess.run(  # noqa: S603 - args are constructed, not shell-passed
                cmd,
                cwd=str(cwd),
                stdout=lf,
                stderr=subprocess.STDOUT,
                check=False,
            )
        _write_state(
            run_id,
            status="finished" if proc.returncode == 0 else "failed",
            return_code=proc.returncode,
            finished_at=datetime.now(tz=UTC).isoformat(),
            out_dir=out_dir,
        )
        _log.info("retrain.done", run_id=run_id, rc=proc.returncode)
    except Exception as exc:  # noqa: BLE001
        _write_state(
            run_id,
            status="failed",
            error=str(exc),
            finished_at=datetime.now(tz=UTC).isoformat(),
        )
        _log.exception("retrain.crashed", run_id=run_id)


@router.post("/retrain", response_model=RetrainStarted, status_code=202)
def trigger_retrain(
    body: RetrainRequest,
    background: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)],
    actor: Annotated[User, Depends(_require_admin)],
) -> RetrainStarted:
    settings = get_settings()
    artifacts_root = Path(settings.ml_model_dir).resolve().parent
    features = body.features or str(
        artifacts_root.parent / "processed" / "features_real_only.parquet"
    )
    if not Path(features).exists():
        raise HTTPException(status_code=400, detail=f"features file not found: {features}")
    version = body.out_version or f"v_retrain_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}"
    out_dir = str(artifacts_root / version)
    run_id = uuid.uuid4().hex[:12]
    started = datetime.now(tz=UTC)
    _write_state(
        run_id,
        status="queued",
        started_at=started.isoformat(),
        features=features,
        out_dir=out_dir,
        tune=body.tune,
    )
    record_audit(
        db,
        user_id=actor.id,
        action="model.retrain",
        entity_type="model",
        entity_id=run_id,
        payload={"features": features, "out_dir": out_dir, "tune": body.tune},
    )
    db.commit()
    background.add_task(_run_training, run_id, features, out_dir, body.tune, body.n_trials)
    return RetrainStarted(run_id=run_id, status="queued", started_at=started)


@router.get("/retrain/{run_id}", response_model=RetrainStatus)
def get_retrain_status(
    run_id: str,
    _actor: Annotated[User, Depends(_require_admin)],
) -> RetrainStatus:
    path = _state_path(run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="run not found")
    state = json.loads(path.read_text())
    log_path = _runs_dir() / f"{run_id}.log"
    log_tail = None
    if log_path.exists():
        lines = log_path.read_text().splitlines()
        log_tail = "\n".join(lines[-40:])
    return RetrainStatus(
        run_id=run_id,
        status=state.get("status", "unknown"),
        started_at=state.get("started_at"),
        finished_at=state.get("finished_at"),
        return_code=state.get("return_code"),
        out_dir=state.get("out_dir"),
        error=state.get("error"),
        log_tail=log_tail,
    )
