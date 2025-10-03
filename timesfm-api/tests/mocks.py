"""Mock services for testing."""

import asyncio
import numpy as np

from app.config import Settings


class MockTimesFMService:
    """
    Mock TimesFM service for testing without model download.
    
    Simulates the real service behavior but returns synthetic forecasts.
    """

    def __init__(self, config: Settings):
        self.config = config
        self.model = None
        self.forecast_config = None
        self.model_loaded = False
        
    async def initialize(self, timeout: int = 300):
        """Mock initialization - instant, no model download."""
        await asyncio.sleep(0.1)  # Simulate slight delay
        
        # Create mock forecast config
        self.forecast_config = type('ForecastConfig', (), {
            'max_context': self.config.DEFAULT_MAX_CONTEXT,
            'max_horizon': self.config.DEFAULT_MAX_HORIZON,
        })()
        
        self.model_loaded = True

    async def cleanup(self):
        """Mock cleanup."""
        self.model_loaded = False

    async def forecast(self, time_series: list, horizon: int, config_overrides: dict = None):
        """
        Generate mock forecasts.
        
        Returns synthetic forecasts that look realistic:
        - Point forecast: extrapolates the trend
        - Quantile forecast: adds realistic uncertainty bands
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        # Convert to numpy if needed
        series_arrays = [np.array(ts, dtype=np.float32) if not isinstance(ts, np.ndarray) else ts 
                         for ts in time_series]
        
        # Validate inputs (same as real service)
        self._validate_inputs(series_arrays, horizon)
        
        # Generate synthetic forecasts
        batch_size = len(series_arrays)
        point_forecasts = []
        quantile_forecasts = []
        
        for series in series_arrays:
            # Simple forecast: extend the mean + small trend
            mean = np.mean(series[-min(32, len(series)):])
            trend = (series[-1] - series[-min(10, len(series))]) / min(10, len(series))
            
            # TimesFM returns shape (horizon, 10) for each series
            # Where each horizon point has 10 quantile values
            quantiles = np.zeros((horizon, 10))
            std = np.std(series)
            
            for i, q in enumerate([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
                # Base forecast at this quantile
                point = mean + trend * np.arange(1, horizon + 1)
                # Add quantile-specific offset
                offset = (q - 0.5) * std * 2.0
                quantiles[:, i] = point + offset + np.random.randn(horizon) * std * 0.05
            
            # TimesFM returns point forecast as just the median (2D)
            # and full quantiles separately (3D)
            point_forecasts.append(quantiles[:, 5])  # Shape: (horizon,) - median only
            quantile_forecasts.append(quantiles)  # Shape: (horizon, 10) - all quantiles
        
        # Stack into arrays matching real service output
        # point_forecast: shape (batch, horizon) - 2D
        # quantile_forecast: shape (batch, horizon, 10) - 3D
        point_forecast = np.array(point_forecasts, dtype=np.float32)
        quantile_forecast = np.array(quantile_forecasts, dtype=np.float32)
        
        return point_forecast, quantile_forecast

    def _validate_inputs(self, time_series: list, horizon: int):
        """Validate inputs (same as real service)."""
        if not time_series:
            raise ValueError("Empty time series list")
        
        if horizon <= 0:
            raise ValueError(f"Horizon must be positive, got {horizon}")
        
        if horizon > self.forecast_config.max_horizon:
            raise ValueError(
                f"Horizon {horizon} exceeds max_horizon {self.forecast_config.max_horizon}"
            )
        
        for i, series in enumerate(time_series):
            if not isinstance(series, np.ndarray):
                raise ValueError(f"Series {i} is not a numpy array")
            
            if series.ndim != 1:
                raise ValueError(f"Series {i} must be 1-dimensional")
            
            # TimesFM pads shorter series automatically - no minimum required
            
            if len(series) > 16384:
                raise ValueError(
                    f"Time series #{i} is too long (length={len(series)}). "
                    f"Maximum supported: 16,384 time points."
                )
            
            if not np.isfinite(series).all():
                nan_count = np.isnan(series).sum()
                inf_count = np.isinf(series).sum()
                raise ValueError(
                    f"Time series #{i} contains invalid values: "
                    f"{nan_count} NaN values, {inf_count} Inf values."
                )

    def get_model_info(self):
        """Get mock model metadata."""
        return {
            "model_name": "google/timesfm-2.5-200m-pytorch (Mock)",
            "model_version": "2.5-200M-pytorch-mock",
            "parameters": 200_000_000,
            "max_context": self.forecast_config.max_context if self.forecast_config else 0,
            "max_horizon": self.forecast_config.max_horizon if self.forecast_config else 0,
            "patch_size": 32,
            "output_patch_size": 128,
            "quantile_forecast_limit": 1024,
            "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "decode_index": 5,
            "device": "cpu",
            "model_loaded": self.model_loaded,
            "context_limit": 1024,
        }
