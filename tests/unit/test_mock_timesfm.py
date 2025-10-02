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

"""Unit tests for mock TimesFM model."""

import numpy as np
import pytest
import torch

from tests.fixtures.mock_timesfm import (
    MockTimesFM,
    MockTimesFMConfig,
    create_mock_timesfm,
)


class TestMockTimesFMConfig:
    """Test MockTimesFMConfig class."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = MockTimesFMConfig()

        assert config.model_name == "timesfm-2.5-200m"
        assert config.input_dims == 1
        assert config.output_dims == 1
        assert config.hidden_size == 768
        assert config.num_layers == 12
        assert config.num_heads == 12
        assert config.max_seq_len == 2048
        assert config.dropout == 0.1
        assert config.device == "cpu"

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = MockTimesFMConfig(
            model_name="test-model",
            input_dims=3,
            output_dims=2,
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            max_seq_len=1024,
            dropout=0.2,
            device="cuda",
        )

        assert config.model_name == "test-model"
        assert config.input_dims == 3
        assert config.output_dims == 2
        assert config.hidden_size == 512
        assert config.num_layers == 8
        assert config.num_heads == 8
        assert config.max_seq_len == 1024
        assert config.dropout == 0.2
        assert config.device == "cuda"


class TestMockTimesFM:
    """Test MockTimesFM class."""

    def test_initialization(self) -> None:
        """Test model initialization."""
        model = MockTimesFM()

        assert model.config.model_name == "timesfm-2.5-200m"
        assert not model.is_loaded
        assert model.load_time is None
        assert model.model is not None

    def test_initialization_with_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = MockTimesFMConfig(model_name="test-model", hidden_size=512)
        model = MockTimesFM(config=config)

        assert model.config.model_name == "test-model"
        assert model.config.hidden_size == 512

    def test_load_model(self) -> None:
        """Test model loading."""
        model = MockTimesFM()
        assert not model.is_loaded

        model.load()
        assert model.is_loaded
        assert model.load_time is not None

    def test_load_model_force_reload(self) -> None:
        """Test force reload model."""
        model = MockTimesFM()
        model.load()

        first_load_time = model.load_time
        model.load(force_reload=True)

        assert model.is_loaded
        assert model.load_time > first_load_time

    def test_forecast_with_numpy_input(self) -> None:
        """Test forecasting with numpy input."""
        model = MockTimesFM()
        model.load()

        # Create test data
        input_data = np.random.randn(100)

        result = model.forecast(input_data, horizon=24)

        assert "forecasts" in result
        assert "quantiles" in result
        assert "metadata" in result

        assert result["forecasts"].shape == (1, 24, 1)
        assert "0.5" in result["quantiles"]
        assert result["quantiles"]["0.5"].shape == (1, 24, 1)

    def test_forecast_with_tensor_input(self) -> None:
        """Test forecasting with tensor input."""
        model = MockTimesFM()
        model.load()

        # Create test data
        input_data = torch.randn(100)

        result = model.forecast(input_data, horizon=48)

        assert "forecasts" in result
        assert result["forecasts"].shape == (1, 48, 1)

    def test_forecast_with_2d_tensor(self) -> None:
        """Test forecasting with 2D tensor input."""
        model = MockTimesFM()
        model.load()

        # Create test data (seq_len, input_dims)
        input_data = torch.randn(100, 1)

        result = model.forecast(input_data, horizon=96)

        assert result["forecasts"].shape == (1, 96, 1)

    def test_forecast_with_3d_tensor(self) -> None:
        """Test forecasting with 3D tensor input."""
        model = MockTimesFM()
        model.load()

        # Create test data (batch_size, seq_len, input_dims)
        input_data = torch.randn(4, 100, 1)

        result = model.forecast(input_data, horizon=24)

        assert result["forecasts"].shape == (4, 24, 1)

    def test_forecast_multivariate(self) -> None:
        """Test forecasting with multivariate input."""
        config = MockTimesFMConfig(input_dims=3, output_dims=3)
        model = MockTimesFM(config=config)
        model.load()

        # Create test data
        input_data = torch.randn(4, 100, 3)

        result = model.forecast(input_data, horizon=24)

        assert result["forecasts"].shape == (4, 24, 3)

    def test_forecast_quantiles(self) -> None:
        """Test forecasting with specific quantiles."""
        model = MockTimesFM()
        model.load()

        input_data = torch.randn(100)
        quantiles = [0.1, 0.5, 0.9]

        result = model.forecast(input_data, horizon=24, quantiles=quantiles)

        assert "quantiles" in result
        for q in quantiles:
            assert str(q) in result["quantiles"]
            assert result["quantiles"][str(q)].shape == (1, 24, 1)

    def test_forecast_samples(self) -> None:
        """Test forecasting with multiple samples."""
        model = MockTimesFM()
        model.load()

        input_data = torch.randn(100)
        num_samples = 10

        result = model.forecast(input_data, horizon=24, num_samples=num_samples)

        assert result["forecasts"].shape == (1, 10, 24, 1)

    def test_forecast_temperature(self) -> None:
        """Test forecasting with temperature parameter."""
        model = MockTimesFM()
        model.load()

        input_data = torch.randn(100)

        result_low = model.forecast(input_data, horizon=24, temperature=0.1)
        result_high = model.forecast(input_data, horizon=24, temperature=2.0)

        # Both should have valid results but with different characteristics
        assert result_low["forecasts"].shape == result_high["forecasts"].shape

    def test_auto_load_on_forecast(self) -> None:
        """Test automatic model loading on forecast."""
        model = MockTimesFM()
        assert not model.is_loaded

        input_data = torch.randn(100)
        result = model.forecast(input_data, horizon=24)

        assert model.is_loaded
        assert "forecasts" in result

    def test_evaluate_metrics(self) -> None:
        """Test model evaluation with metrics."""
        model = MockTimesFM()
        model.load()

        test_data = torch.randn(10, 100, 1)
        targets = torch.randn(10, 24, 1)

        result = model.evaluate(test_data, targets)

        assert "mae" in result
        assert "mse" in result
        assert "mape" in result
        assert "rmse" in result

        # All metrics should be positive
        for metric_value in result.values():
            assert metric_value >= 0

    def test_evaluate_custom_metrics(self) -> None:
        """Test model evaluation with custom metrics."""
        model = MockTimesFM()
        model.load()

        test_data = torch.randn(10, 100, 1)
        targets = torch.randn(10, 24, 1)
        metrics = ["mae", "mse"]

        result = model.evaluate(test_data, targets, metrics=metrics)

        assert len(result) == 2
        assert "mae" in result
        assert "mse" in result
        assert "mape" not in result

    def test_get_model_info(self) -> None:
        """Test getting model information."""
        model = MockTimesFM()
        model.load()

        info = model.get_model_info()

        assert "model_name" in info
        assert "input_dims" in info
        assert "output_dims" in info
        assert "hidden_size" in info
        assert "num_layers" in info
        assert "num_heads" in info
        assert "max_seq_len" in info
        assert "device" in info
        assert "is_loaded" in info
        assert "load_time" in info
        assert "parameters" in info

        assert info["model_name"] == "timesfm-2.5-200m"
        assert info["is_loaded"] is True
        assert info["parameters"] > 0


class TestCreateMockTimesFM:
    """Test factory function for creating mock models."""

    def test_create_default_model(self) -> None:
        """Test creating default mock model."""
        model = create_mock_timesfm()

        assert isinstance(model, MockTimesFM)
        assert model.config.model_name == "timesfm-2.5-200m"
        assert model.config.hidden_size == 768

    def test_create_50m_model(self) -> None:
        """Test creating 50M parameter model."""
        model = create_mock_timesfm(model_size="50m")

        assert model.config.model_name == "timesfm-2.5-50m"
        assert model.config.hidden_size == 384
        assert model.config.num_layers == 6
        assert model.config.num_heads == 6

    def test_create_200m_model(self) -> None:
        """Test creating 200M parameter model."""
        model = create_mock_timesfm(model_size="200m")

        assert model.config.model_name == "timesfm-2.5-200m"
        assert model.config.hidden_size == 768
        assert model.config.num_layers == 12
        assert model.config.num_heads == 12

    def test_create_1b_model(self) -> None:
        """Test creating 1B parameter model."""
        model = create_mock_timesfm(model_size="1b")

        assert model.config.model_name == "timesfm-2.5-1b"
        assert model.config.hidden_size == 2048
        assert model.config.num_layers == 24
        assert model.config.num_heads == 32

    def test_create_model_with_device(self) -> None:
        """Test creating model with specific device."""
        model = create_mock_timesfm(device="cuda")

        assert model.config.device == "cuda"

    def test_create_model_with_custom_config(self) -> None:
        """Test creating model with custom configuration."""
        model = create_mock_timesfm(
            model_size="200m",
            device="cpu",
            dropout=0.2,
            max_seq_len=1024
        )

        assert model.config.device == "cpu"
        assert model.config.dropout == 0.2
        assert model.config.max_seq_len == 1024

    def test_invalid_model_size(self) -> None:
        """Test creating model with invalid size."""
        with pytest.raises(ValueError, match="Unknown model size"):
            create_mock_timesfm(model_size="invalid")


class TestMockTimesFMIntegration:
    """Integration tests for mock TimesFM."""

    def test_end_to_end_forecast_workflow(self) -> None:
        """Test complete forecast workflow."""
        # Create and load model
        model = create_mock_timesfm(model_size="200m", device="cpu")
        model.load()

        # Generate test data
        batch_size = 4
        seq_len = 512
        input_data = torch.randn(batch_size, seq_len, 1)

        # Make forecast
        result = model.forecast(
            input_data,
            horizon=96,
            quantiles=[0.1, 0.5, 0.9],
            num_samples=1,
            temperature=1.0
        )

        # Validate results
        assert result["forecasts"].shape == (batch_size, 96, 1)
        assert len(result["quantiles"]) == 3
        for q in ["0.1", "0.5", "0.9"]:
            assert result["quantiles"][q].shape == (batch_size, 96, 1)

        # Check metadata
        metadata = result["metadata"]
        assert metadata["model_name"] == "timesfm-2.5-200m"
        assert metadata["sequence_length"] == seq_len
        assert metadata["horizon"] == 96
        assert metadata["batch_size"] == batch_size

    def test_performance_evaluation(self) -> None:
        """Test model performance evaluation."""
        model = create_mock_timesfm()
        model.load()

        # Generate test dataset
        test_data = torch.randn(20, 256, 1)
        targets = torch.randn(20, 48, 1)

        # Evaluate performance
        metrics = model.evaluate(test_data, targets, metrics=["mae", "mse", "rmse"])

        # Validate metrics
        for metric_name, metric_value in metrics.items():
            assert isinstance(metric_value, (int, float))
            assert metric_value >= 0
            assert not np.isnan(metric_value)
            assert not np.isinf(metric_value)

    def test_model_info_completeness(self) -> None:
        """Test model information completeness."""
        model = create_mock_timesfm(model_size="1b")
        model.load()

        info = model.get_model_info()

        # Check all required fields are present
        required_fields = [
            "model_name", "input_dims", "output_dims", "hidden_size",
            "num_layers", "num_heads", "max_seq_len", "device",
            "is_loaded", "load_time", "parameters"
        ]

        for field in required_fields:
            assert field in info, f"Missing field: {field}"

        # Check field types and values
        assert isinstance(info["parameters"], int)
        assert info["parameters"] > 0
        assert info["is_loaded"] is True
        assert isinstance(info["load_time"], float)
        assert info["load_time"] > 0