"""Smoke tests for basic functionality."""

def test_import_app():
    """Test that we can import the app module."""
    from app import main
    assert main is not None


def test_import_models():
    """Test that we can import models."""
    from app.models import ForecastRequest, ForecastResponse
    assert ForecastRequest is not None
    assert ForecastResponse is not None


def test_import_config():
    """Test that we can import config."""
    from app.config import settings
    assert settings is not None
    assert settings.API_TITLE == "TimesFM Forecasting API"


def test_pydantic_request_validation():
    """Test that Pydantic validation works."""
    from app.models import ForecastRequest
    
    # Valid request
    valid_request = ForecastRequest(
        time_series=[[float(i) for i in range(50)]],
        horizon=12
    )
    assert valid_request.horizon == 12
    assert len(valid_request.time_series) == 1
    
    # Test that validation catches short series
    import pytest
    with pytest.raises(ValueError, match="too short"):
        ForecastRequest(
            time_series=[[1.0, 2.0, 3.0]],  # Too short
            horizon=12
        )
