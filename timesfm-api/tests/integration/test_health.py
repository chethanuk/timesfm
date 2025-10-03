"""Integration tests for health endpoints."""

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
    
    # Create mock service
    mock_service = MockTimesFMService(settings)
    
    @asynccontextmanager
    async def mock_lifespan(app):
        await mock_service.initialize()
        app.state.model_service = mock_service
        yield
        await mock_service.cleanup()
    
    # Create app with mock lifespan
    app = _create_app()
    app.router.lifespan_context = mock_lifespan
    
    # Use TestClient with lifespan context
    with TestClient(app) as client:
        yield client


def test_health_endpoint(client):
    """Test basic health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_liveness_endpoint(client):
    """Test liveness endpoint."""
    response = client.get("/health/live")
    assert response.status_code == 200
    
    data = response.json()
    assert data["alive"] is True


def test_readiness_endpoint(client):
    """Test readiness endpoint."""
    response = client.get("/health/ready")
    assert response.status_code == 200
    
    data = response.json()
    assert "ready" in data
    assert "model_loaded" in data


def test_version_endpoint(client):
    """Test version endpoint."""
    response = client.get("/version")
    assert response.status_code == 200
    
    data = response.json()
    assert data["api_version"] == "1.0.0"
    assert data["timesfm_version"] == "2.5"
    assert data["timesfm_model"] == "200M-pytorch"


def test_request_id_header(client):
    """Test that request ID header is added."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0


def test_process_time_header(client):
    """Test that process time header is added."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Process-Time" in response.headers
    
    # Should be a float string
    process_time = float(response.headers["X-Process-Time"])
    assert process_time >= 0
