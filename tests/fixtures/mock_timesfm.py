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

"""Mock TimesFM model for testing."""

import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from tests.utils.test_helpers import TestDataGenerator


class MockTimesFMConfig:
    """Mock configuration for TimesFM model."""

    def __init__(
        self,
        model_name: str = "timesfm-2.5-200m",
        input_dims: int = 1,
        output_dims: int = 1,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        """Initialize mock configuration.

        Args:
            model_name: Name of the model.
            input_dims: Input dimension.
            output_dims: Output dimension.
            hidden_size: Hidden layer size.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            max_seq_len: Maximum sequence length.
            dropout: Dropout rate.
            device: Device to run on.
        """
        self.model_name = model_name
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.device = device


class MockTransformerLayer(nn.Module):
    """Mock transformer layer for TimesFM."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        """Initialize mock transformer layer.

        Args:
            hidden_size: Hidden layer size.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Mock attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Mock feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).
            mask: Optional attention mask.

        Returns:
            Output tensor of same shape as input.
        """
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)

        # Feed-forward network with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class MockTimesFMModel(nn.Module):
    """Mock TimesFM model for testing."""

    def __init__(self, config: MockTimesFMConfig):
        """Initialize mock model.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config

        # Input embedding layer
        self.input_embedding = nn.Linear(config.input_dims, config.hidden_size)

        # Position encoding (simplified)
        self.pos_encoding = self._create_position_encoding(
            config.max_seq_len, config.hidden_size
        )

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            MockTransformerLayer(config.hidden_size, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(config.hidden_size, config.output_dims)

        # Mock data generator for realistic outputs
        self.data_gen = TestDataGenerator(random_seed=42)

        # Move to device
        self.to(config.device)

    def _create_position_encoding(self, max_len: int, hidden_size: int) -> torch.Tensor:
        """Create positional encoding.

        Args:
            max_len: Maximum sequence length.
            hidden_size: Hidden dimension size.

        Returns:
            Positional encoding tensor.
        """
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() *
                           (-np.log(10000.0) / hidden_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.to(self.config.device)

    def forward(
        self,
        x: torch.Tensor,
        horizon: Optional[int] = None,
        num_samples: int = 1,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through mock model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dims).
            horizon: Forecast horizon length.
            num_samples: Number of samples to generate.
            temperature: Sampling temperature.

        Returns:
            Dictionary containing predictions and metadata.
        """
        batch_size, seq_len, input_dims = x.shape

        # Set default horizon if not provided
        if horizon is None:
            horizon = min(96, seq_len)

        # Input embedding
        x = self.input_embedding(x)

        # Add position encoding
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0)
        x = x + pos_enc

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Generate mock forecasts based on input patterns
        if num_samples == 1:
            # Deterministic forecast
            forecasts = self._generate_deterministic_forecast(x, horizon)
        else:
            # Sample multiple forecasts
            forecasts = self._generate_sampled_forecasts(x, horizon, num_samples, temperature)

        # Generate quantiles
        quantiles = self._generate_quantiles(forecasts)

        return {
            "forecasts": forecasts,
            "quantiles": quantiles,
            "hidden_states": x,
            "metadata": {
                "model_name": self.config.model_name,
                "sequence_length": seq_len,
                "horizon": horizon,
                "batch_size": batch_size,
                "num_samples": num_samples,
                "device": str(self.config.device),
            }
        }

    def _generate_deterministic_forecast(
        self,
        hidden_states: torch.Tensor,
        horizon: int
    ) -> torch.Tensor:
        """Generate deterministic forecast.

        Args:
            hidden_states: Hidden states from transformer.
            horizon: Forecast horizon.

        Returns:
            Forecast tensor.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Simple pattern: continue the last observed pattern with some modification
        last_hidden = hidden_states[:, -1:, :]  # (batch_size, 1, hidden_size)

        # Create forecast by applying simple transformations
        forecasts = []
        current_hidden = last_hidden

        for i in range(horizon):
            # Apply some non-linear transformation
            step_output = self.output_projection(current_hidden)
            forecasts.append(step_output)

            # Update hidden state for next step (simplified)
            current_hidden = current_hidden + 0.1 * torch.randn_like(current_hidden) / self.config.hidden_size

        forecasts = torch.cat(forecasts, dim=1)  # (batch_size, horizon, output_dims)

        # Add some realistic time series patterns
        forecasts = self._add_time_series_patterns(forecasts, seq_len)

        return forecasts

    def _generate_sampled_forecast(
        self,
        hidden_states: torch.Tensor,
        horizon: int,
        num_samples: int,
        temperature: float
    ) -> torch.Tensor:
        """Generate sampled forecasts.

        Args:
            hidden_states: Hidden states from transformer.
            horizon: Forecast horizon.
            num_samples: Number of samples.
            temperature: Sampling temperature.

        Returns:
            Sampled forecasts tensor.
        """
        # Generate base forecast
        base_forecast = self._generate_deterministic_forecast(hidden_states, horizon)

        # Add noise scaled by temperature
        noise = torch.randn_like(base_forecast) * temperature * 0.1

        # Create multiple samples
        samples = []
        for _ in range(num_samples):
            sample_noise = torch.randn_like(base_forecast) * temperature * 0.1
            sample = base_forecast + sample_noise
            samples.append(sample)

        return torch.stack(samples, dim=1)  # (batch_size, num_samples, horizon, output_dims)

    def _add_time_series_patterns(
        self,
        forecasts: torch.Tensor,
        input_length: int
    ) -> torch.Tensor:
        """Add realistic time series patterns to forecasts.

        Args:
            forecasts: Base forecast tensor.
            input_length: Length of input sequence.

        Returns:
            Forecasts with added patterns.
        """
        batch_size, horizon, output_dims = forecasts.shape
        device = forecasts.device

        # Create time indices for forecast
        time_indices = torch.arange(input_length, input_length + horizon, device=device)

        # Add trend component
        trend = 0.05 * time_indices.float().unsqueeze(0).unsqueeze(-1) / horizon

        # Add seasonal component (daily pattern)
        daily_pattern = 0.5 * torch.sin(2 * np.pi * time_indices.float() / 24).unsqueeze(0).unsqueeze(-1)

        # Combine patterns
        patterns = trend + daily_pattern

        return forecasts + patterns

    def _generate_quantiles(self, forecasts: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate quantile forecasts.

        Args:
            forecasts: Point forecasts.

        Returns:
            Dictionary of quantile forecasts.
        """
        # Generate quantiles by adding and subtracting scaled noise
        quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
        quantiles = {}

        # Base point forecast (median)
        median_forecast = forecasts
        quantiles["0.5"] = median_forecast

        # Generate other quantiles
        for q in quantile_levels:
            if q == 0.5:
                continue

            # Scale factor based on distance from median
            scale = abs(q - 0.5) * 2

            # Add scaled noise
            noise = torch.randn_like(median_forecast) * 0.2 * scale

            if q < 0.5:
                quantile_forecast = median_forecast - noise.abs()
            else:
                quantile_forecast = median_forecast + noise.abs()

            quantiles[str(q)] = quantile_forecast

        return quantiles


class MockTimesFM:
    """Mock TimesFM API that mimics the real TimesFM interface."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[MockTimesFMConfig] = None,
        **kwargs
    ):
        """Initialize mock TimesFM.

        Args:
            model_path: Path to model (ignored in mock).
            config: Model configuration.
            **kwargs: Additional arguments.
        """
        if config is None:
            config = MockTimesFMConfig(**kwargs)

        self.config = config
        self.model = MockTimesFMModel(config)
        self.is_loaded = False
        self.load_time = None

    def load(self, force_reload: bool = False) -> None:
        """Mock loading the model.

        Args:
            force_reload: Whether to force reload.
        """
        if not self.is_loaded or force_reload:
            # Simulate loading time
            time.sleep(0.1)
            self.is_loaded = True
            self.load_time = time.time()

    def forecast(
        self,
        time_series: Union[np.ndarray, torch.Tensor],
        horizon: int = 96,
        quantiles: Optional[List[float]] = None,
        num_samples: int = 1,
        temperature: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate forecasts using mock model.

        Args:
            time_series: Input time series data.
            horizon: Forecast horizon.
            quantiles: Quantile levels to compute.
            num_samples: Number of samples to generate.
            temperature: Sampling temperature.
            **kwargs: Additional arguments.

        Returns:
            Dictionary containing forecasts and metadata.
        """
        if not self.is_loaded:
            self.load()

        # Convert input to tensor if needed
        if isinstance(time_series, np.ndarray):
            time_series = torch.from_numpy(time_series).float()

        # Ensure 3D shape (batch_size, seq_len, input_dims)
        if time_series.dim() == 1:
            time_series = time_series.unsqueeze(0).unsqueeze(-1)
        elif time_series.dim() == 2:
            time_series = time_series.unsqueeze(-1)

        # Move to device
        time_series = time_series.to(self.config.device)

        # Generate forecasts
        with torch.no_grad():
            results = self.model(
                time_series,
                horizon=horizon,
                num_samples=num_samples,
                temperature=temperature
            )

        # Convert results to numpy for consistency
        output = {
            "forecasts": results["forecasts"].cpu().numpy(),
            "quantiles": {
                q: tensor.cpu().numpy()
                for q, tensor in results["quantiles"].items()
            },
            "metadata": results["metadata"],
        }

        # Filter quantiles if requested
        if quantiles is not None:
            output["quantiles"] = {
                str(q): output["quantiles"][str(q)]
                for q in quantiles
                if str(q) in output["quantiles"]
            }

        return output

    def evaluate(
        self,
        test_data: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        metrics: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Evaluate model performance on test data.

        Args:
            test_data: Test input data.
            targets: Target values.
            metrics: Metrics to compute.
            **kwargs: Additional arguments.

        Returns:
            Dictionary of evaluation metrics.
        """
        if metrics is None:
            metrics = ["mae", "mse", "mape"]

        # Generate forecasts
        forecasts = self.forecast(test_data, **kwargs)
        forecast_values = forecasts["forecasts"]

        # Convert to numpy if needed
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        # Compute metrics
        results = {}
        for metric in metrics:
            if metric == "mae":
                results[metric] = np.mean(np.abs(forecast_values - targets))
            elif metric == "mse":
                results[metric] = np.mean((forecast_values - targets) ** 2)
            elif metric == "mape":
                # Avoid division by zero
                mask = np.abs(targets) > 1e-8
                results[metric] = np.mean(np.abs((forecast_values[mask] - targets[mask]) / targets[mask])) * 100
            elif metric == "rmse":
                results[metric] = np.sqrt(np.mean((forecast_values - targets) ** 2))

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary with model metadata.
        """
        return {
            "model_name": self.config.model_name,
            "input_dims": self.config.input_dims,
            "output_dims": self.config.output_dims,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_layers,
            "num_heads": self.config.num_heads,
            "max_seq_len": self.config.max_seq_len,
            "device": self.config.device,
            "is_loaded": self.is_loaded,
            "load_time": self.load_time,
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }


# Factory function for creating mock models
def create_mock_timesfm(
    model_size: str = "200m",
    device: str = "cpu",
    **kwargs
) -> MockTimesFM:
    """Create a mock TimesFM model.

    Args:
        model_size: Size of model ("50m", "200m", "1b").
        device: Device to run on.
        **kwargs: Additional configuration.

    Returns:
        Mock TimesFM instance.
    """
    # Default configurations for different model sizes
    size_configs = {
        "50m": {"hidden_size": 384, "num_layers": 6, "num_heads": 6},
        "200m": {"hidden_size": 768, "num_layers": 12, "num_heads": 12},
        "1b": {"hidden_size": 2048, "num_layers": 24, "num_heads": 32},
    }

    if model_size not in size_configs:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(size_configs.keys())}")

    config = MockTimesFMConfig(
        model_name=f"timesfm-2.5-{model_size}",
        device=device,
        **size_configs[model_size],
        **kwargs
    )

    return MockTimesFM(config=config)