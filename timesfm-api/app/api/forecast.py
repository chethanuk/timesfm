"""Forecasting endpoints."""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from app.models.request import ForecastRequest
from app.models.response import ErrorResponse, ForecastMetadata, ForecastResponse, ModelInfoResponse, QuantileForecast, SingleForecast
from app.services.forecasting import TimesFMService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["forecast"])


def get_model_service(request: Request) -> TimesFMService:
    """Get model service from app state."""
    return request.app.state.model_service


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info(
    service: TimesFMService = Depends(get_model_service)
) -> ModelInfoResponse:
    """Get model information and metadata."""
    info = service.get_model_info()
    return ModelInfoResponse(**info)


@router.post("/forecast", response_model=ForecastResponse)
async def forecast(
    request: ForecastRequest,
    raw_request: Request,
    service: TimesFMService = Depends(get_model_service)
) -> ForecastResponse:
    """
    Generate forecasts for one or more time series.
    
    This endpoint accepts a batch of time series and returns forecasts
    for each series. The forecasts include both point estimates (median)
    and quantile predictions for uncertainty quantification.
    """
    request_id = getattr(raw_request.state, "request_id", "unknown")
    
    try:
        logger.info(
            f"Forecast request received: {len(request.time_series)} series, "
            f"horizon={request.horizon} [{request_id}]"
        )
        
        start_time = time.time()
        
        # Run inference
        point_forecast, quantile_forecast = await service.forecast(
            time_series=request.time_series,
            horizon=request.horizon,
            config_overrides=request.config
        )
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Build response
        forecasts = []
        for i in range(len(request.time_series)):
            # Extract point forecast
            # TimesFM returns:
            # - point_forecast: shape (batch, horizon) - median only (2D)
            # - quantile_forecast: shape (batch, horizon, 10) - all quantiles (3D)
            point = point_forecast[i].tolist()
            
            # Extract quantiles
            # TimesFM quantile indices:
            # 0: 0% (min), 1: 10%, 2: 20%, ..., 5: 50% (median), ..., 9: 90%
            quantiles = QuantileForecast(
                q00=quantile_forecast[i, :, 0].tolist(),  # 0% quantile (minimum)
                q10=quantile_forecast[i, :, 1].tolist(),  # 10% quantile
                q20=quantile_forecast[i, :, 2].tolist(),  # 20% quantile
                q30=quantile_forecast[i, :, 3].tolist(),  # 30% quantile
                q40=quantile_forecast[i, :, 4].tolist(),  # 40% quantile
                q50=quantile_forecast[i, :, 5].tolist(),  # 50% quantile (median)
                q60=quantile_forecast[i, :, 6].tolist(),  # 60% quantile
                q70=quantile_forecast[i, :, 7].tolist(),  # 70% quantile
                q80=quantile_forecast[i, :, 8].tolist(),  # 80% quantile
                q90=quantile_forecast[i, :, 9].tolist(),  # 90% quantile
            )
            
            forecasts.append(
                SingleForecast(
                    point_forecast=point,
                    quantiles=quantiles
                )
            )
        
        metadata = ForecastMetadata(
            model_version="2.5-200m",
            inference_time_ms=inference_time_ms,
            config_used=request.config or {},
            batch_size=len(request.time_series)
        )
        
        logger.info(
            f"Forecast completed: {len(forecasts)} forecasts in "
            f"{inference_time_ms:.2f}ms [{request_id}]"
        )
        
        return ForecastResponse(
            forecasts=forecasts,
            metadata=metadata
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e} [{request_id}]")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {e} [{request_id}]")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e} [{request_id}]")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during forecasting: {str(e)}"
        )
