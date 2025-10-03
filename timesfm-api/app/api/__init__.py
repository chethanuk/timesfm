"""API routers."""

from .forecast import router as forecast_router
from .health import router as health_router

__all__ = ["forecast_router", "health_router"]
