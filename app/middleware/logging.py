"""Structured request logging middleware (single-line JSON-ish log per request)."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.config import Settings

logger = logging.getLogger("deploy.http")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        status_code = None
        error = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            if request.url.path != "/health":
                latency_ms = (time.perf_counter() - start) * 1000
                entry = {
                    "event": "http_request",
                    "method": request.method,
                    "path": request.url.path,
                    "status": status_code,
                    "latency_ms": round(latency_ms, 2),
                    "request_id": getattr(request.state, "request_id", None),
                    "environment": self.settings.app_env,
                    "client_host": request.client.host if request.client else None,
                    "error": error,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                logger.info(json.dumps(entry))