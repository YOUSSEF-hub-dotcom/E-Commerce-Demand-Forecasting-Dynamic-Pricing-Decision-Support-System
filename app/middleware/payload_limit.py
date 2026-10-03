"""Request body size limit middleware (JSON payloads buffered)."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import Settings
from app.errors import PAYLOAD_TOO_LARGE


class PayloadSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.max_bytes = settings.max_payload_bytes

    async def dispatch(self, request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            try:
                body = await request.body()
            except Exception:
                body = b""
            if len(body) > self.max_bytes:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": f"request body exceeds {self.max_bytes} bytes",
                        "code": PAYLOAD_TOO_LARGE,
                        "request_id": getattr(request.state, "request_id", None),
                        "extra": {},
                    },
                )
        return await call_next(request)