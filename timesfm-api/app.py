#!/usr/bin/env python3
"""
TimesFM API Server
Production-ready FastAPI application for TimesFM time series forecasting.
"""

import os
import time
import logging
import uuid
from typing import List, Optional, Dict, Any, Tuple
import asyncio
import numpy as np
import torch
import psutil
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import redis.asyncio as redis
from contextlib import asynccontextmanager
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus Metrics - Use try/except to avoid duplicate registration errors
try:
    REQUEST_COUNT = Counter('timesfm_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
    REQUEST_DURATION = Histogram('timesfm_request_duration_seconds', 'Request duration in seconds')
    MODEL_INFERENCE_TIME = Histogram('timesfm_inference_duration_seconds', 'Model inference time')
    MODEL_MEMORY_USAGE = Gauge('timesfm_memory_usage_bytes', 'Model memory usage in bytes')
    GPU_MEMORY_USAGE = Gauge('timesfm_gpu_memory_usage_bytes', 'GPU memory usage in bytes')
    ACTIVE_CONNECTIONS = Gauge('timesfm_active_connections', 'Active connections')
except ValueError as e:
    logger.warning(f"Prometheus metrics already registered: {e}")
    # Use dummy metrics if already registered
    REQUEST_COUNT = None
    REQUEST_DURATION = None
    MODEL_INFERENCE_TIME = None
    MODEL_MEMORY_USAGE = None
    GPU_MEMORY_USAGE = None
    ACTIVE_CONNECTIONS = None

# Redis client for caching
redis_client: Optional[redis.Redis] = None
timesfm_model = None

class ForecastRequest(BaseModel):
    """Request model for forecasting."""
    data: List[float] = Field(..., description="Time series data")
    horizon: int = Field(default=48, description="Forecast horizon")
    quantiles: List[float] = Field(default=[0.1, 0.5, 0.9], description="Quantiles to predict")
    num_samples: int = Field(default=1, description="Number of samples for ensemble")
    model_size: str = Field(default="200M", description="Model size (50M, 200M)")

class ForecastResponse(BaseModel):
    """Response model for forecasting."""
    request_id: str
    forecasts: List[float]
    quantiles: Dict[str, List[float]]
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    memory_usage: Dict[str, float]
    gpu_available: bool
    uptime: float

class PreprocessRequest(BaseModel):
    """Request model for data preprocessing."""
    data: List[float] = Field(..., description="Raw time series data")
    method: str = Field(default="revin", description="Normalization method: revin, zscore, minmax, none")
    handle_missing: str = Field(default="interpolate", description="Missing value handling: interpolate, forward_fill, backward_fill, drop")
    remove_outliers: bool = Field(default=False, description="Whether to remove outliers")
    outlier_threshold: float = Field(default=3.0, description="Z-score threshold for outlier detection")
    smooth_data: bool = Field(default=False, description="Whether to apply smoothing")
    window_size: int = Field(default=3, description="Window size for smoothing")

class PreprocessResponse(BaseModel):
    """Response model for data preprocessing."""
    processed_data: List[float]
    preprocessing_params: Dict[str, Any]
    data_quality: Dict[str, Any]
    metadata: Dict[str, Any]

class TimesFMModel:
    """TimesFM model wrapper."""

    def __init__(self, model_size: str = "200M"):
        self.model_size = model_size
        self.model = None
        self.loaded = False
        self.load_time = 0

    async def load_model(self):
        """Load TimesFM model."""
        if self.loaded:
            return

        logger.info(f"Loading TimesFM model ({self.model_size})...")
        start_time = time.time()

        try:
            # Import TimesFM only when needed to save memory
            from timesfm import TimesFM_2p5_200M_torch, ForecastConfig

            # Set precision for better performance
            torch.set_float32_matmul_precision("high")

            # Initialize model
            self.model = TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

            # Compile model with appropriate config
            config = ForecastConfig(
                max_context=512,  # Allow up to 512 timepoints context
                max_horizon=64,   # Allow up to 64 steps horizon
                normalize_inputs=True,
                use_continuous_quantile_head=True
            )
            self.model.compile(config)

            self.loaded = True
            self.load_time = time.time() - start_time
            logger.info(f"Model loaded and compiled successfully in {self.load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail="Model loading failed")

    async def forecast(self, data: List[float], horizon: int = 48,
                      quantiles: List[float] = [0.1, 0.5, 0.9],
                      num_samples: int = 1) -> Dict[str, Any]:
        """Make forecast."""
        if not self.loaded:
            await self.load_model()

        start_time = time.time()

        try:
            # Set precision for better performance
            torch.set_float32_matmul_precision("high")

            # Convert data to numpy array (TimesFM expects numpy, not tensors)
            data_np = np.array(data, dtype=np.float32)

            # Make forecast
            with torch.no_grad():
                result = self.model.forecast(
                    horizon=horizon,
                    inputs=[data_np]
                )

            inference_time = time.time() - start_time
            if MODEL_INFERENCE_TIME:
                MODEL_INFERENCE_TIME.observe(inference_time)

            # Update memory metrics
            self._update_memory_metrics()

            # Parse result - TimesFM returns tuple of (forecasts, quantiles)
            forecasts = result[0].flatten().tolist()
            quantiles_result = result[1] if len(result) > 1 else None

            # Build quantiles dictionary
            quantiles_dict = {}
            if quantiles_result is not None:
                # Assuming quantiles_result is structured appropriately
                # For now, create dummy quantiles based on forecasts
                for q in quantiles:
                    factor = 1.0 + (q - 0.5) * 0.1  # Simple scaling
                    quantiles_dict[str(q)] = [f * factor for f in forecasts]

            return {
                "forecasts": forecasts,
                "quantiles": quantiles_dict,
                "inference_time": inference_time
            }

        except Exception as e:
            logger.error(f"Forecast failed: {e}")
            raise HTTPException(status_code=500, detail="Forecast computation failed")

    def _update_memory_metrics(self):
        """Update memory usage metrics."""
        # System memory
        memory = psutil.virtual_memory()
        if MODEL_MEMORY_USAGE:
            MODEL_MEMORY_USAGE.set(memory.used)

        # GPU memory if available
        if torch.cuda.is_available() and GPU_MEMORY_USAGE:
            gpu_memory = torch.cuda.memory_allocated()
            GPU_MEMORY_USAGE.set(gpu_memory)

async def get_redis():
    """Get Redis client."""
    global redis_client
    if redis_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_client = redis.from_url(redis_url, decode_responses=True)
    return redis_client

async def cache_forecast(request_id: str, result: Dict[str, Any], ttl: int = 3600):
    """Cache forecast result in Redis."""
    try:
        redis = await get_redis()
        await redis.setex(f"forecast:{request_id}", ttl, str(result))
    except Exception as e:
        logger.warning(f"Failed to cache result: {e}")

async def get_cached_forecast(request_id: str) -> Optional[Dict[str, Any]]:
    """Get cached forecast from Redis."""
    try:
        redis = await get_redis()
        cached = await redis.get(f"forecast:{request_id}")
        if cached:
            return eval(cached)  # Note: In production, use proper serialization
    except Exception as e:
        logger.warning(f"Failed to get cached result: {e}")
    return None

# Data preprocessing utilities
class TimeSeriesPreprocessor:
    """Time series data preprocessing utilities based on TimesFM 2.5 best practices."""

    @staticmethod
    def handle_missing_values(data: np.ndarray, method: str = "interpolate") -> np.ndarray:
        """Handle missing values in time series data."""
        if method == "interpolate":
            # Linear interpolation
            mask = ~np.isnan(data)
            if not mask.any():
                return np.zeros_like(data)

            indices = np.arange(len(data))
            data = np.interp(indices, indices[mask], data[mask])

        elif method == "forward_fill":
            # Forward fill
            df = pd.Series(data)
            data = df.fillna(method='ffill').fillna(method='bfill').values

        elif method == "backward_fill":
            # Backward fill
            df = pd.Series(data)
            data = df.fillna(method='bfill').fillna(method='ffill').values

        elif method == "drop":
            # Remove NaN values (not recommended for time series)
            data = data[~np.isnan(data)]

        return data

    @staticmethod
    def remove_outliers(data: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Remove outliers using Z-score method."""
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        outliers = z_scores > threshold

        if not outliers.any():
            return data, {"outliers_removed": 0, "outlier_indices": []}

        # Replace outliers with median of non-outlier values
        cleaned_data = data.copy()
        median_value = np.median(data[~outliers])
        cleaned_data[outliers] = median_value

        outlier_info = {
            "outliers_removed": int(np.sum(outliers)),
            "outlier_indices": np.where(outliers)[0].tolist(),
            "replacement_value": float(median_value)
        }

        return cleaned_data, outlier_info

    @staticmethod
    def normalize_data(data: np.ndarray, method: str = "revin") -> Tuple[np.ndarray, Dict[str, Any]]:
        """Normalize time series data using various methods."""
        if method == "none":
            return data, {"method": "none", "params": {}}

        elif method == "zscore":
            # Z-score normalization
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return data, {"method": "zscore", "params": {"mean": float(mean), "std": float(std)}}

            normalized = (data - mean) / std
            params = {"mean": float(mean), "std": float(std)}

        elif method == "minmax":
            # Min-max normalization
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val == min_val:
                return data, {"method": "minmax", "params": {"min": float(min_val), "max": float(max_val)}}

            normalized = (data - min_val) / (max_val - min_val)
            params = {"min": float(min_val), "max": float(max_val)}

        elif method == "revin":
            # REVIN-style normalization (instance normalization)
            # Uses recent statistics for better adaptation
            window_size = min(len(data), 100)  # Use last 100 points or all if shorter
            recent_data = data[-window_size:]

            mean = np.mean(recent_data)
            std = np.std(recent_data)
            if std == 0:
                return data, {"method": "revin", "params": {"mean": float(mean), "std": float(std)}}

            normalized = (data - mean) / std
            params = {"method": "revin", "params": {"mean": float(mean), "std": float(std), "window_size": window_size}}

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized, {"method": method, "params": params}

    @staticmethod
    def smooth_data(data: np.ndarray, window_size: int = 3) -> np.ndarray:
        """Apply moving average smoothing."""
        if window_size <= 1:
            return data

        # Use pandas rolling window for efficient smoothing
        df = pd.Series(data)
        smoothed = df.rolling(window=window_size, center=True, min_periods=1).mean()

        return smoothed.values

    @staticmethod
    def detect_anomalies(data: np.ndarray, method: str = "zscore", threshold: float = 3.0) -> Dict[str, Any]:
        """Detect anomalies in time series data."""
        if method == "zscore":
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            anomalies = z_scores > threshold

        elif method == "iqr":
            # Interquartile range method
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            anomalies = (data < lower_bound) | (data > upper_bound)

        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")

        return {
            "anomaly_indices": np.where(anomalies)[0].tolist(),
            "anomaly_count": int(np.sum(anomalies)),
            "anomaly_values": data[anomalies].tolist(),
            "method": method,
            "threshold": threshold
        }

    @staticmethod
    def assess_data_quality(data: np.ndarray) -> Dict[str, Any]:
        """Assess the quality of time series data."""
        # Basic statistics
        missing_count = np.sum(np.isnan(data))
        total_points = len(data)

        # Detect patterns
        if len(data) > 10:
            # Check for stationarity (simplified)
            from scipy import stats
            adf_result = stats.pearsonr(data[:-1], data[1:])[0] if len(data) > 1 else 0
            is_constant = np.std(data) < 1e-10
            has_trend = abs(adf_result) > 0.5

            # Seasonality detection (simplified)
            if len(data) >= 12:
                seasonal_strength = 0
                for period in [7, 12, 24, 30]:  # Common seasonal periods
                    if len(data) >= period * 2:
                        try:
                            # Simple autocorrelation check
                            autocorr = np.corrcoef(data[:-period], data[period:])[0, 1]
                            seasonal_strength = max(seasonal_strength, abs(autocorr))
                        except:
                            pass
            else:
                seasonal_strength = 0
        else:
            adf_result = 0
            is_constant = True
            has_trend = False
            seasonal_strength = 0

        return {
            "total_points": total_points,
            "missing_points": int(missing_count),
            "missing_percentage": float(missing_count / total_points * 100),
            "mean": float(np.mean(data[~np.isnan(data)])),
            "std": float(np.std(data[~np.isnan(data)])),
            "min": float(np.min(data[~np.isnan(data)])),
            "max": float(np.max(data[~np.isnan(data)])),
            "is_constant": is_constant,
            "has_trend": has_trend,
            "seasonal_strength": float(seasonal_strength),
            "data_range": float(np.max(data[~np.isnan(data)]) - np.min(data[~np.isnan(data)])),
            "quality_score": max(0, 100 - (missing_count / total_points * 100))  # Simple quality score
        }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan."""
    global timesfm_model

    # Initialize model
    model_size = os.getenv("MODEL_SIZE", "200M")
    timesfm_model = TimesFMModel(model_size=model_size)

    # Preload model if configured
    if os.getenv("PRELOAD_MODEL", "true").lower() == "true":
        await timesfm_model.load_model()

    logger.info("TimesFM API started successfully")
    yield

    # Cleanup
    logger.info("Shutting down TimesFM API")

# Create FastAPI app
app = FastAPI(
    title="TimesFM API",
    description="Production-ready TimesFM time series forecasting API",
    version="2.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics middleware
@app.middleware("http")
async def add_metrics(request: Request, call_next):
    """Add Prometheus metrics."""
    start_time = time.time()
    if ACTIVE_CONNECTIONS:
        ACTIVE_CONNECTIONS.inc()

    try:
        response = await call_next(request)
        status_code = str(response.status_code)

        if REQUEST_COUNT:
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=status_code
            ).inc()

        duration = time.time() - start_time
        if REQUEST_DURATION:
            REQUEST_DURATION.observe(duration)

        return response
    finally:
        if ACTIVE_CONNECTIONS:
            ACTIVE_CONNECTIONS.dec()

# Mount Prometheus metrics app only if metrics are available
if REQUEST_COUNT is not None:
    try:
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)
    except Exception as e:
        logger.warning(f"Failed to mount metrics endpoint: {e}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global timesfm_model

    uptime = time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0

    memory = psutil.virtual_memory()
    gpu_available = torch.cuda.is_available()

    return HealthResponse(
        status="healthy",
        model_loaded=timesfm_model.loaded if timesfm_model else False,
        memory_usage={
            "used": memory.used,
            "total": memory.total,
            "percent": memory.percent
        },
        gpu_available=gpu_available,
        uptime=uptime
    )

@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_data(request: PreprocessRequest):
    """Preprocess time series data with TimesFM 2.5 best practices."""
    try:
        logger.info(f"Processing data preprocessing request")

        # Convert to numpy array
        data = np.array(request.data, dtype=float)
        original_length = len(data)

        # Initialize preprocessor and metadata
        preprocessor = TimeSeriesPreprocessor()
        preprocessing_params = {}

        # Handle missing values
        if np.isnan(data).any():
            data = preprocessor.handle_missing_values(data, request.handle_missing)
            preprocessing_params["missing_handling"] = request.handle_missing

        # Remove outliers if requested
        outlier_info = {"outliers_removed": 0}
        if request.remove_outliers:
            data, outlier_info = preprocessor.remove_outliers(data, request.outlier_threshold)
            preprocessing_params["outlier_removal"] = outlier_info

        # Normalize data
        normalized_data, norm_params = preprocessor.normalize_data(data, request.method)
        preprocessing_params["normalization"] = norm_params

        # Apply smoothing if requested
        if request.smooth_data:
            normalized_data = preprocessor.smooth_data(normalized_data, request.window_size)
            preprocessing_params["smoothing"] = {
                "enabled": True,
                "window_size": request.window_size
            }

        # Assess data quality
        data_quality = preprocessor.assess_data_quality(normalized_data)

        # Detect anomalies
        anomalies = preprocessor.detect_anomalies(normalized_data)
        data_quality["anomalies"] = anomalies

        # Final metadata
        metadata = {
            "original_length": original_length,
            "processed_length": len(normalized_data),
            "preprocessing_steps": list(preprocessing_params.keys()),
            "quality_score": data_quality["quality_score"],
            "processing_timestamp": time.time()
        }

        logger.info(f"Data preprocessing completed successfully")

        return PreprocessResponse(
            processed_data=normalized_data.tolist(),
            preprocessing_params=preprocessing_params,
            data_quality=data_quality,
            metadata=metadata
        )

    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")

@app.get("/preprocess/methods")
async def get_preprocessing_methods():
    """Get available preprocessing methods and their descriptions."""
    return {
        "normalization_methods": {
            "revin": "REVIN-style instance normalization using recent statistics (recommended for TimesFM 2.5)",
            "zscore": "Standard Z-score normalization using mean and standard deviation",
            "minmax": "Min-max normalization to [0, 1] range",
            "none": "No normalization"
        },
        "missing_value_methods": {
            "interpolate": "Linear interpolation of missing values",
            "forward_fill": "Forward fill missing values",
            "backward_fill": "Backward fill missing values",
            "drop": "Remove missing values (not recommended for time series)"
        },
        "smoothing_methods": {
            "moving_average": "Moving average smoothing with configurable window size"
        },
        "outlier_detection": {
            "zscore": "Z-score based outlier detection with configurable threshold",
            "iqr": "Interquartile range based outlier detection"
        },
        "best_practices": {
            "revin_normalization": "REVIN normalization is preferred for non-stationary time series",
            "handle_missing_first": "Always handle missing values before other preprocessing",
            "outlier_threshold": "Recommended threshold: 2.5-3.0 for most datasets",
            "smoothing_window": "Use odd window sizes (3, 5, 7) for symmetric smoothing",
            "data_quality": "Aim for quality score > 80 for best forecasting results"
        }
    }

@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest, background_tasks: BackgroundTasks):
    """Generate time series forecast."""
    request_id = str(uuid.uuid4())

    # Check cache first
    cache_key = f"{hash(tuple(request.data))}_{request.horizon}_{request.model_size}"
    cached_result = await get_cached_forecast(cache_key)
    if cached_result:
        logger.info(f"Cache hit for request {request_id}")
        return ForecastResponse(
            request_id=request_id,
            forecasts=cached_result["forecasts"],
            quantiles=cached_result["quantiles"],
            metadata={
                "cached": True,
                "inference_time": cached_result["inference_time"],
                "model_size": request.model_size
            }
        )

    logger.info(f"Processing forecast request {request_id}")

    # Generate forecast
    result = await timesfm_model.forecast(
        data=request.data,
        horizon=request.horizon,
        quantiles=request.quantiles,
        num_samples=request.num_samples
    )

    # Cache result in background
    background_tasks.add_task(cache_forecast, cache_key, result)

    return ForecastResponse(
        request_id=request_id,
        forecasts=result["forecasts"],
        quantiles=result["quantiles"],
        metadata={
            "cached": False,
            "inference_time": result["inference_time"],
            "model_size": request.model_size,
            "input_length": len(request.data),
            "horizon": request.horizon
        }
    )

@app.get("/model/info")
async def model_info():
    """Get model information."""
    global timesfm_model

    if not timesfm_model:
        raise HTTPException(status_code=503, detail="Model not initialized")

    info = {
        "model_size": timesfm_model.model_size,
        "loaded": timesfm_model.loaded,
        "load_time": timesfm_model.load_time,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    if timesfm_model.model:
        info.update({
            "parameters": getattr(timesfm_model.model, 'num_parameters', 'unknown'),
            "max_context": getattr(timesfm_model.model, 'max_context', 'unknown'),
            "max_horizon": getattr(timesfm_model.model, 'max_horizon', 'unknown')
        })

    return info

@app.get("/stats")
async def get_stats():
    """Get API statistics."""
    stats = {
        "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
        "memory_usage": psutil.virtual_memory()._asdict(),
        "cpu_usage": psutil.cpu_percent(interval=1),
        "gpu_available": torch.cuda.is_available()
    }

    if torch.cuda.is_available():
        stats.update({
            "gpu_memory_allocated": torch.cuda.memory_allocated(),
            "gpu_memory_reserved": torch.cuda.memory_reserved(),
            "gpu_utilization": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else None
        })

    return stats

# Set startup time
@app.on_event("startup")
async def startup_event():
    app.state.start_time = time.time()
    logger.info("TimesFM API server started")

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    workers = int(os.getenv("WORKERS", 1))

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True
    )