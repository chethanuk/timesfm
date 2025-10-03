"""Response models for API endpoints."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Timestamp of health check")


class ReadinessResponse(BaseModel):
    """Response model for readiness check."""

    ready: bool = Field(..., description="Whether service is ready")
    model_loaded: bool = Field(..., description="Whether model is loaded")


class LivenessResponse(BaseModel):
    """Response model for liveness check."""

    alive: bool = Field(..., description="Whether service is alive")


class VersionResponse(BaseModel):
    """Response model for version endpoint."""

    api_version: str = Field(..., description="API version")
    timesfm_version: str = Field(..., description="TimesFM model version")
    timesfm_model: str = Field(..., description="TimesFM model variant")


class ModelInfoResponse(BaseModel):
    """Response model for model info endpoint."""

    model_name: str
    model_version: str
    parameters: int
    max_context: int
    max_horizon: int
    patch_size: int
    output_patch_size: int
    quantile_forecast_limit: int
    device: str
    model_loaded: bool


class QuantileForecast(BaseModel):
    """
    Quantile forecasts for uncertainty quantification.
    
    TimesFM produces 10 quantiles:
    - q00: 0% quantile (minimum)
    - q10-q90: 10%-90% quantiles in 10% increments
    - q50 is the median (same as point_forecast)
    """

    q00: List[float] = Field(..., description="0th percentile (minimum) forecast")
    q10: List[float] = Field(..., description="10th percentile forecast")
    q20: List[float] = Field(..., description="20th percentile forecast")
    q30: List[float] = Field(..., description="30th percentile forecast")
    q40: List[float] = Field(..., description="40th percentile forecast")
    q50: List[float] = Field(..., description="50th percentile (median) forecast")
    q60: List[float] = Field(..., description="60th percentile forecast")
    q70: List[float] = Field(..., description="70th percentile forecast")
    q80: List[float] = Field(..., description="80th percentile forecast")
    q90: List[float] = Field(..., description="90th percentile forecast")


class SingleForecast(BaseModel):
    """Forecast for a single time series."""

    point_forecast: List[float] = Field(description="Point forecast (median/q50)")
    quantiles: Optional[QuantileForecast] = Field(
        default=None, description="Quantile forecasts"
    )


class ForecastMetadata(BaseModel):
    """Metadata about the forecast."""

    model_version: str
    inference_time_ms: float
    config_used: Dict[str, bool | int]
    batch_size: int


class ForecastResponse(BaseModel):
    """Response from forecasting endpoint."""

    forecasts: List[SingleForecast]
    metadata: ForecastMetadata


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracing")
