"""Request-ID middleware: propagate/issue a request id and echo it back."""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.config import Settings


class RequestIdMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.header = settings.request_id_header or "X-Request-ID"

    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get(self.header)
        rid = incoming if incoming else uuid.uuid4().hex
        request.state.request_id = rid
        response: Response = await call_next(request)
        response.headers[self.header] = rid
        return response