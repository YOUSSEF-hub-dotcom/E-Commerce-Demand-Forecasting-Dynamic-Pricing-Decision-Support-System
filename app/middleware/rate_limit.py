"""Optional per-IP fixed-window rate limiter (default off)."""

from __future__ import annotations

import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import Settings
from app.errors import RATE_LIMITED


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.enabled = settings.rate_limit_enabled
        self.max_requests = settings.rate_limit_max_requests
        self.window = settings.rate_limit_window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if not self.enabled or request.url.path in ("/health", "/ready"):
            return await call_next(request)
        host = request.client.host if request.client else "unknown"
        now = time.monotonic()
        queue = self._hits[host]
        while queue and now - queue[0] > self.window:
            queue.popleft()
        if len(queue) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "rate limit exceeded",
                    "code": RATE_LIMITED,
                    "request_id": getattr(request.state, "request_id", None),
                    "extra": {"limit": self.max_requests, "window_seconds": self.window},
                },
            )
        queue.append(now)
        return await call_next(request)