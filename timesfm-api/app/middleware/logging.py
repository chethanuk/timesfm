"""Request logging middleware."""

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming requests and responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request and log details."""
        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")

        # Log request
        logger.info(
            f"Request started: {request.method} {request.url.path} "
            f"[{request_id}]"
        )

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"status={response.status_code} time={process_time:.4f}s "
                f"[{request_id}]"
            )

            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"error={str(e)} time={process_time:.4f}s [{request_id}]"
            )
            raise
