"""Lifecycle endpoints: liveness and readiness."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import Settings
from app.dependencies import get_settings, get_state
from app.schemas import HealthResponse, ReadinessResponse
from app.services.state import AppState

router = APIRouter(tags=["lifecycle"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings: Settings = get_settings(request)
    return HealthResponse(
        status="ok",
        app_env=settings.app_env,
        version="0.6.0",
        time_utc=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/ready", response_model=ReadinessResponse)
def ready(request: Request):
    state: AppState = get_state(request)
    components = state.component_snapshot()
    ready_now = state.is_ready()
    return JSONResponse(
        status_code=200 if ready_now else 503,
        content=ReadinessResponse(
            status="ready" if ready_now else "not_ready",
            model=state.model_info["name"] if state.model_info else "n/a",
            model_version=state.model_info["version"] if state.model_info else None,
            components={k: {"status": v["status"], "details": v["details"]} for k, v in components.items()},
            time_utc=datetime.now(timezone.utc).isoformat(),
        ).model_dump(),
    )