"""Request ID middleware for tracing."""

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request for tracing."""

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request and add request ID."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.start_time = time.time()

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        # Add processing time
        process_time = time.time() - request.state.start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"

        return response
