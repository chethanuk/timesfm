"""Integration tests for forecast endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import create_app
from tests.mocks import MockTimesFMService


@pytest.fixture
def client():
    """Create test client with mock service."""
    from contextlib import asynccontextmanager
    from app.main import create_app as _create_app
    
    mock_service = MockTimesFMService(settings)
    
    @asynccontextmanager
    async def mock_lifespan(app):
        await mock_service.initialize()
        app.state.model_service = mock_service
        yield
        await mock_service.cleanup()
    
    app = _create_app()
    app.router.lifespan_context = mock_lifespan
    
    with TestClient(app) as client:
        yield client


def test_model_info_endpoint(client):
    """Test model info endpoint."""
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    
    data = response.json()
    assert "model_name" in data
    assert "model_version" in data
    assert "parameters" in data
    assert data["parameters"] == 200_000_000
    assert "max_context" in data
    assert "max_horizon" in data


def test_forecast_single_series(client):
    """Test forecast with a single time series."""
    # Create a simple time series (50 points)
    time_series = [float(i) for i in range(50)]
    
    request_data = {
        "time_series": [time_series],
        "horizon": 12
    }
    
    response = client.post("/api/v1/forecast", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "forecasts" in data
    assert "metadata" in data
    
    # Check forecasts
    assert len(data["forecasts"]) == 1
    forecast = data["forecasts"][0]
    
    assert "point_forecast" in forecast
    assert len(forecast["point_forecast"]) == 12
    
    assert "quantiles" in forecast
    quantiles = forecast["quantiles"]
    assert "q00" in quantiles  # 0% quantile (minimum)
    assert "q10" in quantiles
    assert "q50" in quantiles  # median
    assert "q90" in quantiles
    assert len(quantiles["q50"]) == 12
    
    # Verify quantile ordering (q00 <= q10 <= ... <= q90)
    for j in range(12):
        assert quantiles["q00"][j] <= quantiles["q10"][j]
        assert quantiles["q10"][j] <= quantiles["q50"][j]
        assert quantiles["q50"][j] <= quantiles["q90"][j]
    
    # Check metadata
    metadata = data["metadata"]
    assert metadata["model_version"] == "2.5-200m"
    assert metadata["batch_size"] == 1
    assert "inference_time_ms" in metadata


def test_forecast_batch(client):
    """Test forecast with multiple time series."""
    # Create 3 time series
    series1 = [float(i) for i in range(50)]
    series2 = [float(i * 2) for i in range(60)]
    series3 = [float(i * 0.5) for i in range(40)]
    
    request_data = {
        "time_series": [series1, series2, series3],
        "horizon": 10
    }
    
    response = client.post("/api/v1/forecast", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert len(data["forecasts"]) == 3
    assert data["metadata"]["batch_size"] == 3
    
    # Check each forecast
    for forecast in data["forecasts"]:
        assert len(forecast["point_forecast"]) == 10
        assert len(forecast["quantiles"]["q50"]) == 10


def test_forecast_validation_short_series(client):
    """Test that short series are rejected."""
    request_data = {
        "time_series": [[1.0, 2.0, 3.0]],  # Too short (< 32)
        "horizon": 12
    }
    
    response = client.post("/api/v1/forecast", json=request_data)
    assert response.status_code == 422  # Validation error


def test_forecast_validation_negative_horizon(client):
    """Test that negative horizon is rejected."""
    request_data = {
        "time_series": [[float(i) for i in range(50)]],
        "horizon": -5
    }
    
    response = client.post("/api/v1/forecast", json=request_data)
    assert response.status_code == 422


def test_forecast_validation_zero_horizon(client):
    """Test that zero horizon is rejected."""
    request_data = {
        "time_series": [[float(i) for i in range(50)]],
        "horizon": 0
    }
    
    response = client.post("/api/v1/forecast", json=request_data)
    assert response.status_code == 422


def test_forecast_with_config_override(client):
    """Test forecast with configuration overrides."""
    request_data = {
        "time_series": [[float(i) for i in range(50)]],
        "horizon": 12,
        "config": {
            "normalize_inputs": True
        }
    }
    
    response = client.post("/api/v1/forecast", json=request_data)
    assert response.status_code == 200


def test_forecast_different_horizons(client):
    """Test forecasting with different horizon lengths."""
    time_series = [float(i) for i in range(100)]
    
    for horizon in [1, 12, 24, 96, 256]:
        request_data = {
            "time_series": [time_series],
            "horizon": horizon
        }
        
        response = client.post("/api/v1/forecast", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["forecasts"][0]["point_forecast"]) == horizon


def test_forecast_request_tracing(client):
    """Test that forecasts include request tracing."""
    request_data = {
        "time_series": [[float(i) for i in range(50)]],
        "horizon": 12
    }
    
    response = client.post("/api/v1/forecast", json=request_data)
    assert response.status_code == 200
    
    # Check headers
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time" in response.headers
