"""Pytest configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="session")
def app():
    """Create FastAPI app for testing (without model loading)."""
    # Note: For unit/integration tests, we'll mock the model service
    # For E2E tests, we'll need the actual model loaded
    return create_app()


@pytest.fixture(scope="function")
def client(app):
    """Create test client."""
    return TestClient(app)
