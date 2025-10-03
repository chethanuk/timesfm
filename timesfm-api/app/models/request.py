"""Request models for API endpoints."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ForecastRequest(BaseModel):
    """Request model for forecasting endpoint."""

    time_series: List[List[float]] = Field(
        ...,
        description="List of time series, each as a list of float values",
        min_length=1,
        max_length=100,  # Batch limit
    )

    horizon: int = Field(
        ...,
        description="Number of time points to forecast",
        gt=0,
        le=1024,
    )

    config: Optional[Dict[str, bool | int]] = Field(
        default=None,
        description="Optional forecast configuration overrides",
    )

    @field_validator("time_series")
    @classmethod
    def validate_time_series(cls, v: List[List[float]]) -> List[List[float]]:
        """Validate time series data."""
        for i, ts in enumerate(v):
            # TimesFM automatically pads shorter series - no minimum required
            if len(ts) < 1:
                raise ValueError(
                    f"Time series #{i} is empty. "
                    f"Please provide at least 1 data point."
                )
            if len(ts) > 16384:  # Max context
                raise ValueError(
                    f"Time series #{i} is too long (length={len(ts)}). "
                    f"Maximum supported: 16,384 time points. "
                    f"Please truncate your series or use a sliding window approach."
                )
            if not all(isinstance(x, (int, float)) for x in ts):
                raise ValueError(
                    f"Time series #{i} contains non-numeric values. "
                    f"Please ensure all values are numbers."
                )
        return v

    model_config = {"json_schema_extra": {"example": {
        "time_series": [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 11.0, 12.0, 13.0, 14.0],
        ],
        "horizon": 12,
        "config": {
            "normalize_inputs": True,
            "use_continuous_quantile_head": True,
        },
    }}}
