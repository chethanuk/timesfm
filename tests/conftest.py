# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytest configuration and shared fixtures for TimesFM tests."""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from pytest import FixtureRequest


# Test configuration constants
TEST_DATA_DIR = Path(__file__).parent / "fixtures"
TEST_OUTPUT_DIR = Path(__file__).parent / "output"
MOCK_MODEL_DIR = TEST_DATA_DIR / "mock_model"

# Ensure test directories exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_OUTPUT_DIR.mkdir(exist_ok=True)
MOCK_MODEL_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Test configuration dictionary."""
    return {
        "model_name": "timesfm-2.5-200m",
        "device": "cpu",
        "batch_size": 32,
        "sequence_length": 512,
        "horizon_length": 96,
        "test_data_points": 1000,
        "random_seed": 42,
    }


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="function")
def mock_logger() -> Mock:
    """Mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture(scope="session")
def numpy_random_seed() -> None:
    """Set numpy random seed for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)


@pytest.fixture(scope="function")
def timing_fixture() -> Generator[Dict[str, float], None, None]:
    """Fixture for timing test execution."""
    timings = {"start": 0.0, "end": 0.0, "duration": 0.0}
    timings["start"] = time.time()
    yield timings
    timings["end"] = time.time()
    timings["duration"] = timings["end"] - timings["start"]


@pytest.fixture(scope="session")
def mock_torch_device() -> str:
    """Mock torch device for testing."""
    return "cpu"


@pytest.fixture(scope="function")
def mock_tensor() -> torch.Tensor:
    """Create a mock tensor for testing."""
    return torch.randn(10, 20)


@pytest.fixture(scope="function")
def mock_model_config() -> Dict[str, Any]:
    """Mock model configuration."""
    return {
        "model_name": "timesfm-2.5-200m",
        "input_dims": 1,
        "output_dims": 1,
        "context_length": 512,
        "horizon_length": 96,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "dropout": 0.1,
        "device": "cpu",
    }


@pytest.fixture(scope="function")
def sample_time_series_data() -> Dict[str, np.ndarray]:
    """Sample time series data for testing."""
    n_points = 1000
    time_steps = np.arange(n_points)

    # Generate synthetic time series with trend, seasonality, and noise
    trend = 0.1 * time_steps
    seasonality = 5 * np.sin(2 * np.pi * time_steps / 24)  # Daily seasonality
    noise = np.random.normal(0, 0.5, n_points)

    values = trend + seasonality + noise

    return {
        "time": time_steps,
        "values": values,
        "trend": trend,
        "seasonality": seasonality,
        "noise": noise,
    }


@pytest.fixture(scope="function")
def batch_time_series_data() -> Dict[str, np.ndarray]:
    """Batch time series data for testing."""
    batch_size = 32
    sequence_length = 512

    # Create batch of time series
    time_data = np.arange(sequence_length)
    values_data = np.random.randn(batch_size, sequence_length, 1)

    return {
        "time": time_data,
        "values": values_data,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
    }


@pytest.fixture(scope="function")
def mock_forecast_config() -> Dict[str, Any]:
    """Mock forecast configuration."""
    return {
        "horizon": 96,
        "context_length": 512,
        "quantiles": [0.1, 0.5, 0.9],
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.9,
        "num_samples": 100,
    }


@pytest.fixture(scope="function")
def mock_api_response() -> Dict[str, Any]:
    """Mock API response structure."""
    return {
        "status": "success",
        "data": {
            "forecasts": np.random.randn(10, 96).tolist(),
            "quantiles": {
                "10": np.random.randn(10, 96).tolist(),
                "50": np.random.randn(10, 96).tolist(),
                "90": np.random.randn(10, 96).tolist(),
            },
            "metadata": {
                "model_version": "2.5.0",
                "timestamp": "2025-01-01T00:00:00Z",
                "processing_time": 0.5,
            },
        },
        "message": "Forecast generated successfully",
    }


@pytest.fixture(scope="function")
def mock_error_response() -> Dict[str, Any]:
    """Mock error response structure."""
    return {
        "status": "error",
        "error": {
            "code": 400,
            "message": "Invalid input data",
            "details": "Input tensor shape mismatch",
        },
        "timestamp": "2025-01-01T00:00:00Z",
    }


@pytest.fixture(scope="function")
def performance_test_config() -> Dict[str, Any]:
    """Configuration for performance tests."""
    return {
        "warmup_iterations": 3,
        "test_iterations": 10,
        "batch_sizes": [1, 8, 16, 32, 64],
        "sequence_lengths": [64, 128, 256, 512],
        "horizon_lengths": [24, 48, 96],
        "max_duration_seconds": 60,
        "memory_threshold_mb": 1024,
    }


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """Test database URL for integration tests."""
    return "sqlite:///:memory:"


@pytest.fixture(scope="session")
def test_redis_url() -> str:
    """Test Redis URL for integration tests."""
    return "redis://localhost:6379/0"


@pytest.fixture(scope="function")
def mock_huggingface_hub() -> Mock:
    """Mock HuggingFace Hub for testing."""
    mock_hub = Mock()
    mock_hub.hf_hub_download = Mock(return_value="mock_model_path")
    mock_hub.snapshot_download = Mock(return_value="mock_snapshot_path")
    return mock_hub


@pytest.fixture(scope="function")
def mock_safetensors() -> Mock:
    """Mock SafeTensors for testing."""
    mock_st = Mock()
    mock_st.load_file = Mock(return_value={"model": torch.randn(100, 100)})
    return mock_st


@pytest.fixture(scope="function")
def environment_variables() -> Generator[Dict[str, str], None, None]:
    """Set up environment variables for testing."""
    original_env = os.environ.copy()

    test_env = {
        "TIMESFM_MODEL_PATH": str(MOCK_MODEL_DIR),
        "TIMESFM_DEVICE": "cpu",
        "TIMESFM_LOG_LEVEL": "DEBUG",
        "TIMESFM_CACHE_DIR": str(TEST_OUTPUT_DIR / "cache"),
        "TIMESFM_MAX_BATCH_SIZE": "32",
    }

    os.environ.update(test_env)
    yield test_env

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(scope="function")
def benchmark_data() -> Dict[str, Any]:
    """Data for benchmarking tests."""
    return {
        "small_batch": {"batch_size": 1, "seq_len": 64, "horizon": 24},
        "medium_batch": {"batch_size": 16, "seq_len": 256, "horizon": 48},
        "large_batch": {"batch_size": 64, "seq_len": 512, "horizon": 96},
    }


# Custom markers
def pytest_configure(config) -> None:
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "async: marks tests as async tests"
    )


# Skip slow tests by default
def pytest_collection_modifyitems(config, items) -> None:
    """Modify test collection to handle markers."""
    if config.option.markexpr != "slow":
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)