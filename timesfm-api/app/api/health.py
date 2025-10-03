"""Health check endpoints."""

from datetime import datetime, UTC

from fastapi import APIRouter, Depends, Request

from app.models.response import HealthResponse, LivenessResponse, ReadinessResponse, VersionResponse
from app.services.forecasting import TimesFMService

router = APIRouter(tags=["health"])


def get_model_service(request: Request) -> TimesFMService:
    """Get model service from app state."""
    return request.app.state.model_service


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(UTC).isoformat()
    )


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness(
    service: TimesFMService = Depends(get_model_service)
) -> ReadinessResponse:
    """Readiness check - is the service ready to accept requests?"""
    return ReadinessResponse(
        ready=service.model_loaded,
        model_loaded=service.model_loaded
    )


@router.get("/health/live", response_model=LivenessResponse)
async def liveness() -> LivenessResponse:
    """Liveness check - is the service running?"""
    return LivenessResponse(alive=True)


@router.get("/version", response_model=VersionResponse)
async def version() -> VersionResponse:
    """Get API and model version."""
    return VersionResponse(
        api_version="1.0.0",
        timesfm_version="2.5",
        timesfm_model="200M-pytorch"
    )
