"""TimesFM model service for time series forecasting."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from app.config import Settings

logger = logging.getLogger(__name__)


class TimesFMService:
    """
    TimesFM model service following singleton pattern.
    
    Responsibilities:
    - Load model once at startup
    - Manage model compilation
    - Handle inference requests
    - Batch processing
    - Input validation and preprocessing
    """

    def __init__(self, config: Settings):
        self.config = config
        self.model = None
        self.forecast_config = None
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        
        # Log device info
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA not available, using CPU")

    async def initialize(self, timeout: int = 300):
        """
        Load and compile model at startup.
        This is called once during application startup.
        
        Args:
            timeout: Maximum time in seconds to wait for initialization (default: 5 minutes)
            
        Raises:
            asyncio.TimeoutError: If initialization takes longer than timeout
            RuntimeError: If model loading fails
        """
        logger.info(f"Loading TimesFM model: {self.config.MODEL_NAME}")
        
        try:
            async with asyncio.timeout(timeout):
                await self._initialize_model()
        except asyncio.TimeoutError:
            logger.error(f"Model initialization timed out after {timeout} seconds")
            raise RuntimeError(
                f"Failed to initialize TimesFM model within {timeout} seconds. "
                "This could indicate network issues downloading the model or insufficient resources."
            )

    async def _initialize_model(self):
        """Internal method to initialize model components."""
        # Import here to avoid issues if timesfm is not installed yet
        try:
            import timesfm
            from timesfm import ForecastConfig
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import timesfm package: {e}. "
                "Please ensure timesfm is installed: pip install -e ."
            )
        
        # Create cache directory if it doesn't exist
        cache_dir = Path(self.config.MODEL_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model from HuggingFace
        logger.info("Downloading/loading model from HuggingFace...")
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            self.config.MODEL_NAME,
            cache_dir=str(cache_dir),
            torch_compile=self.config.ENABLE_TORCH_COMPILE
        )
        
        # Create default forecast configuration
        self.forecast_config = ForecastConfig(
            max_context=self.config.DEFAULT_MAX_CONTEXT,
            max_horizon=self.config.DEFAULT_MAX_HORIZON,
            normalize_inputs=self.config.DEFAULT_NORMALIZE,
            use_continuous_quantile_head=self.config.DEFAULT_USE_QUANTILE_HEAD,
            force_flip_invariance=self.config.DEFAULT_FORCE_FLIP_INVARIANCE,
            infer_is_positive=self.config.DEFAULT_INFER_IS_POSITIVE,
            fix_quantile_crossing=self.config.DEFAULT_FIX_QUANTILE_CROSSING,
        )
        
        # Compile model with default config
        logger.info("Compiling model...")
        self.model.compile(self.forecast_config)
        
        # Warm up with dummy inference
        logger.info("Warming up model with dummy inference...")
        dummy_series = [np.random.randn(256).astype(np.float32)]
        await self._warm_up(dummy_series)
        
        self.model_loaded = True
        logger.info("Model initialization complete")

    async def _warm_up(self, series: List[np.ndarray]):
        """Warm up model with dummy inference."""
        try:
            _ = await self.forecast(series, horizon=12)
            logger.info("Model warm-up successful")
        except Exception as e:
            logger.warning(f"Warmup inference failed: {e}")

    async def cleanup(self):
        """Cleanup resources."""
        if self.model:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        logger.info("Model service cleaned up")

    async def forecast(
        self,
        time_series: List[List[float]],
        horizon: int,
        config_overrides: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate forecasts for multiple time series.
        
        Args:
            time_series: List of time series (each is a list of floats)
            horizon: Number of time points to forecast
            config_overrides: Optional config overrides (dict of param: value)
            
        Returns:
            Tuple of (point_forecasts, quantile_forecasts)
            - point_forecasts: shape (batch, horizon)
            - quantile_forecasts: shape (batch, horizon, 10) for 10 quantiles
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If inference fails
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        # Convert to numpy arrays
        series_arrays = [np.array(ts, dtype=np.float32) for ts in time_series]
        
        # Validate inputs
        self._validate_inputs(series_arrays, horizon)
        
        # Run inference in executor to not block event loop
        loop = asyncio.get_event_loop()
        point_forecast, quantile_forecast = await loop.run_in_executor(
            None,
            self._run_inference,
            series_arrays,
            horizon
        )
        
        return point_forecast, quantile_forecast

    def _run_inference(
        self,
        time_series: List[np.ndarray],
        horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run model inference (synchronous, called in executor)."""
        start_time = time.time()
        
        try:
            point_forecast, quantile_forecast = self.model.forecast(
                horizon=horizon,
                inputs=time_series
            )
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Model inference failed: {str(e)}")
        
        inference_time = (time.time() - start_time) * 1000  # ms
        logger.info(
            f"Inference completed in {inference_time:.2f}ms "
            f"for {len(time_series)} series"
        )
        
        return point_forecast, quantile_forecast

    def _validate_inputs(self, time_series: List[np.ndarray], horizon: int):
        """Validate input time series and horizon."""
        if not time_series:
            raise ValueError("Empty time series list")
        
        if horizon <= 0:
            raise ValueError(f"Horizon must be positive, got {horizon}")
        
        if horizon > self.forecast_config.max_horizon:
            raise ValueError(
                f"Horizon {horizon} exceeds max_horizon "
                f"{self.forecast_config.max_horizon}"
            )
        
        # Check each series
        for i, series in enumerate(time_series):
            if not isinstance(series, np.ndarray):
                raise ValueError(f"Series {i} is not a numpy array")
            
            if series.ndim != 1:
                raise ValueError(f"Series {i} must be 1-dimensional")
            
            # TimesFM will pad series shorter than max_context with zeros
            # No minimum length requirement - padding is handled automatically
            
            if len(series) > 16384:  # Max context
                raise ValueError(
                    f"Time series #{i} is too long (length={len(series)}). "
                    f"Maximum supported: 16,384 time points. "
                    f"Please truncate your series or use a sliding window approach."
                )
            
            if not np.isfinite(series).all():
                nan_count = np.isnan(series).sum()
                inf_count = np.isinf(series).sum()
                raise ValueError(
                    f"Time series #{i} contains invalid values: "
                    f"{nan_count} NaN values, {inf_count} Inf values. "
                    f"Please clean your data before forecasting."
                )

    def get_model_info(self) -> Dict:
        """Get model metadata from the loaded model."""
        if not self.model_loaded or not self.model:
            return {
                "model_name": self.config.MODEL_NAME,
                "model_loaded": False,
                "error": "Model not loaded yet"
            }
        
        # Extract actual values from the loaded model config
        model_config = self.model.config
        
        return {
            "model_name": self.config.MODEL_NAME,
            "model_version": "2.5-200M-pytorch",
            "parameters": 200_000_000,  # TimesFM 2.5 200M parameter model
            "max_context": self.forecast_config.max_context if self.forecast_config else model_config.context_limit,
            "max_horizon": self.forecast_config.max_horizon if self.forecast_config else 256,
            "patch_size": model_config.input_patch_len,  # From model: 32
            "output_patch_size": model_config.output_patch_len,  # From model: 128
            "quantile_forecast_limit": model_config.output_quantile_len,  # From model: 1024
            "quantiles": model_config.quantiles,  # From model: [0.1, 0.2, ..., 0.9]
            "decode_index": model_config.decode_index,  # From model: 5 (median)
            "device": str(self.device),
            "model_loaded": self.model_loaded,
            "context_limit": model_config.context_limit,  # From model: 16384
        }
