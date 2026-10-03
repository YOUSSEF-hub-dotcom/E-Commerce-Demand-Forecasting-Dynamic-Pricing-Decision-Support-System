"""FastAPI dependency helpers: access application-wide runtime objects."""

from __future__ import annotations

from fastapi import Request

from app.config import Settings
from app.db.database import InferenceStore
from app.services.state import AppState


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_state(request: Request) -> AppState:
    return request.app.state.app_state


def get_store(request: Request) -> InferenceStore:
    return request.app.state.inference_store


def request_id(request: Request) -> str:
    rid = getattr(request.state, "request_id", None)
    if rid is None:
        rid = getattr(request.headers, "x-request-id", "n/a")
    return str(rid)