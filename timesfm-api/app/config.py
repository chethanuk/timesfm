"""Configuration management for TimesFM API."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # API Settings
    API_TITLE: str = "TimesFM Forecasting API"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1  # Multiple workers not recommended with GPU

    # Model Settings
    MODEL_NAME: str = "google/timesfm-2.5-200m-pytorch"
    MODEL_CACHE_DIR: str = "./model_cache"
    DEVICE: str = "cuda"  # or "cpu"

    # TimesFM Default Config
    DEFAULT_MAX_CONTEXT: int = 1024
    DEFAULT_MAX_HORIZON: int = 256
    DEFAULT_NORMALIZE: bool = True
    DEFAULT_USE_QUANTILE_HEAD: bool = True
    DEFAULT_FORCE_FLIP_INVARIANCE: bool = True
    DEFAULT_INFER_IS_POSITIVE: bool = True
    DEFAULT_FIX_QUANTILE_CROSSING: bool = True

    # Performance
    ENABLE_TORCH_COMPILE: bool = True

    # Logging
    LOG_LEVEL: str = "INFO"


# Global settings instance
settings = Settings()
