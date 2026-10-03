"""Central exception handling: structured JSON errors, never stack traces."""

from __future__ import annotations

import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.errors import INTERNAL_ERROR, VALIDATION_ERROR, AppError

logger = logging.getLogger("deploy.api")


def get_request_id(request: Request) -> str | None:
    rid = getattr(request.state, "request_id", None)
    return str(rid) if rid is not None else None


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    logger.warning(
        "app_error",
        extra={
            "code": exc.code,
            "status_code": exc.status_code,
            "request_id": get_request_id(request),
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.message,
            "code": exc.code,
            "request_id": get_request_id(request),
            "extra": exc.extra,
        },
    )


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    field_msgs = []
    for err in exc.errors():
        loc = err.get("loc", ())
        ctx = err.get("ctx")
        piece = " -> ".join(str(p) for p in loc if p not in ("body", "query", "path"))
        msg = str(err.get("msg", "invalid"))
        if ctx is not None and "error" in ctx:
            msg = f"{msg}: {ctx['error']}"
        field_msgs.append({"field": piece, "message": msg})
    logger.info(
        "validation_error",
        extra={
            "request_id": get_request_id(request),
            "errors": field_msgs,
        },
    )
    return JSONResponse(
        status_code=422,
        content={
            "detail": "request failed validation",
            "code": VALIDATION_ERROR,
            "request_id": get_request_id(request),
            "extra": {"fields": field_msgs},
        },
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "unhandled_error",
        exc_info=exc,
        extra={"request_id": get_request_id(request)},
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "internal server error",
            "code": INTERNAL_ERROR,
            "request_id": get_request_id(request),
            "extra": {},
        },
    )