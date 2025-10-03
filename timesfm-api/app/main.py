"""Main FastAPI application for TimesFM forecasting service."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import forecast_router, health_router
from app.config import settings
from app.middleware import RequestIDMiddleware, RequestLoggingMiddleware
from app.services import TimesFMService

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Manages model loading and cleanup.
    """
    logger.info("Starting TimesFM API service...")
    
    # Initialize model service
    model_service = TimesFMService(settings)
    
    try:
        # Load and compile model
        await model_service.initialize()
        
        # Store in app state
        app.state.model_service = model_service
        
        logger.info("TimesFM API service started successfully")
        yield
        
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down TimesFM API service...")
        await model_service.cleanup()
        logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title=settings.API_TITLE,
        version=settings.API_VERSION,
        description=(
            "Production-ready API for time series forecasting using "
            "Google Research's TimesFM 2.5 foundation model. "
            "Supports zero-shot forecasting with quantile predictions."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add request ID middleware (must be before logging)
    app.add_middleware(RequestIDMiddleware)
    
    # Add logging middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    # Include routers
    app.include_router(health_router)
    app.include_router(forecast_router)
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
        reload=False  # Set to True for development
    )
