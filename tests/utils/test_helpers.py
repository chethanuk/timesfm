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

"""Test helper utilities for TimesFM tests."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray


class TestDataGenerator:
    """Generate realistic test data for time series forecasting."""

    def __init__(self, random_seed: int = 42):
        """Initialize the data generator.

        Args:
            random_seed: Seed for random number generation.
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def generate_trend(
        self,
        n_points: int,
        trend_type: str = "linear",
        slope: float = 0.1
    ) -> NDArray[np.float64]:
        """Generate trend component.

        Args:
            n_points: Number of time points.
            trend_type: Type of trend ('linear', 'exponential', 'logarithmic').
            slope: Slope parameter for trend.

        Returns:
            Array of trend values.
        """
        time_steps = np.arange(n_points, dtype=np.float64)

        if trend_type == "linear":
            return slope * time_steps
        elif trend_type == "exponential":
            return np.exp(slope * time_steps / n_points)
        elif trend_type == "logarithmic":
            return slope * np.log(time_steps + 1)
        else:
            raise ValueError(f"Unknown trend type: {trend_type}")

    def generate_seasonality(
        self,
        n_points: int,
        period: int = 24,
        amplitude: float = 5.0,
        phase: float = 0.0
    ) -> NDArray[np.float64]:
        """Generate seasonal component.

        Args:
            n_points: Number of time points.
            period: Period of seasonality.
            amplitude: Amplitude of seasonal variation.
            phase: Phase offset.

        Returns:
            Array of seasonal values.
        """
        time_steps = np.arange(n_points, dtype=np.float64)
        return amplitude * np.sin(2 * np.pi * time_steps / period + phase)

    def generate_noise(
        self,
        n_points: int,
        noise_type: str = "gaussian",
        noise_level: float = 0.5,
        outlier_prob: float = 0.01,
        outlier_magnitude: float = 10.0
    ) -> NDArray[np.float64]:
        """Generate noise component.

        Args:
            n_points: Number of time points.
            noise_type: Type of noise ('gaussian', 'uniform', 'student_t').
            noise_level: Standard deviation of noise.
            outlier_prob: Probability of outliers.
            outlier_magnitude: Magnitude of outliers.

        Returns:
            Array of noise values.
        """
        if noise_type == "gaussian":
            noise = np.random.normal(0, noise_level, n_points)
        elif noise_type == "uniform":
            noise = np.random.uniform(-noise_level, noise_level, n_points)
        elif noise_type == "student_t":
            # Student's t-distribution with 3 degrees of freedom
            noise = np.random.standard_t(3, n_points) * noise_level
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        # Add outliers
        outlier_mask = np.random.random(n_points) < outlier_prob
        outlier_values = np.random.choice([-1, 1], n_points) * outlier_magnitude
        noise[outlier_mask] = outlier_values[outlier_mask]

        return noise

    def generate_time_series(
        self,
        n_points: int = 1000,
        trend_type: str = "linear",
        trend_slope: float = 0.1,
        seasonal_periods: List[int] = [24, 168],  # Daily and weekly
        seasonal_amplitudes: List[float] = [5.0, 3.0],
        noise_level: float = 0.5,
        noise_type: str = "gaussian",
        missing_data_prob: float = 0.01,
        value_range: Optional[Tuple[float, float]] = None
    ) -> Dict[str, NDArray[np.float64]]:
        """Generate a complete time series with multiple components.

        Args:
            n_points: Number of time points.
            trend_type: Type of trend component.
            trend_slope: Slope of trend component.
            seasonal_periods: List of seasonal periods.
            seasonal_amplitudes: List of seasonal amplitudes.
            noise_level: Level of noise.
            noise_type: Type of noise.
            missing_data_prob: Probability of missing data points.
            value_range: Optional range to clip values to.

        Returns:
            Dictionary containing time series components.
        """
        time_steps = np.arange(n_points, dtype=np.float64)

        # Generate components
        trend = self.generate_trend(n_points, trend_type, trend_slope)

        seasonality = np.zeros(n_points)
        for period, amplitude in zip(seasonal_periods, seasonal_amplitudes):
            seasonality += self.generate_seasonality(n_points, period, amplitude)

        noise = self.generate_noise(n_points, noise_type, noise_level)

        # Combine components
        values = trend + seasonality + noise

        # Add missing data
        if missing_data_prob > 0:
            missing_mask = np.random.random(n_points) < missing_data_prob
            values[missing_mask] = np.nan

        # Clip to value range if specified
        if value_range:
            values = np.clip(values, value_range[0], value_range[1])

        return {
            "time": time_steps,
            "values": values,
            "trend": trend,
            "seasonality": seasonality,
            "noise": noise,
            "has_missing": missing_data_prob > 0,
        }

    def generate_batch_data(
        self,
        batch_size: int = 32,
        sequence_length: int = 512,
        horizon_length: int = 96,
        num_features: int = 1,
        pattern: str = "mixed"
    ) -> Dict[str, torch.Tensor]:
        """Generate batch of time series data.

        Args:
            batch_size: Number of time series in batch.
            sequence_length: Length of input sequence.
            horizon_length: Length of forecast horizon.
            num_features: Number of features per time step.
            pattern: Pattern type ('trend', 'seasonal', 'mixed', 'random').

        Returns:
            Dictionary containing batch tensors.
        """
        total_length = sequence_length + horizon_length
        batch_data = []

        for _ in range(batch_size):
            if pattern == "trend":
                ts = self.generate_time_series(
                    total_length,
                    seasonal_periods=[],
                    noise_level=0.1
                )
            elif pattern == "seasonal":
                ts = self.generate_time_series(
                    total_length,
                    trend_slope=0.01,
                    noise_level=0.1
                )
            elif pattern == "mixed":
                ts = self.generate_time_series(total_length)
            elif pattern == "random":
                ts = self.generate_time_series(
                    total_length,
                    trend_slope=0.0,
                    seasonal_periods=[],
                    noise_level=1.0
                )
            else:
                raise ValueError(f"Unknown pattern: {pattern}")

            batch_data.append(ts["values"])

        batch_array = np.array(batch_data)

        # Split into input and target
        input_data = batch_array[:, :sequence_length]
        target_data = batch_array[:, sequence_length:]

        # Convert to tensors
        if num_features == 1:
            input_tensor = torch.from_numpy(input_data).float().unsqueeze(-1)
            target_tensor = torch.from_numpy(target_data).float().unsqueeze(-1)
        else:
            # For multiple features, replicate the signal
            input_tensor = torch.from_numpy(input_data).float().unsqueeze(-1).repeat(1, 1, num_features)
            target_tensor = torch.from_numpy(target_data).float().unsqueeze(-1).repeat(1, 1, num_features)

        return {
            "input": input_tensor,
            "target": target_tensor,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "horizon_length": horizon_length,
            "num_features": num_features,
        }


class ModelTestUtils:
    """Utilities for testing model components."""

    @staticmethod
    def compare_tensors(
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
        tolerance: float = 1e-6
    ) -> bool:
        """Compare two tensors within tolerance.

        Args:
            tensor1: First tensor.
            tensor2: Second tensor.
            tolerance: Tolerance for comparison.

        Returns:
            True if tensors are close within tolerance.
        """
        if tensor1.shape != tensor2.shape:
            return False
        return torch.allclose(tensor1, tensor2, atol=tolerance)

    @staticmethod
    def create_mock_model(
        input_dims: int = 1,
        output_dims: int = 1,
        hidden_size: int = 768,
        num_layers: int = 12,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Create mock model for testing.

        Args:
            input_dims: Input dimension.
            output_dims: Output dimension.
            hidden_size: Hidden layer size.
            num_layers: Number of layers.
            device: Device to place tensors on.

        Returns:
            Mock model dictionary.
        """
        return {
            "parameters": {
                "input_dims": input_dims,
                "output_dims": output_dims,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "device": device,
            },
            "weights": {
                "embedding": torch.randn(input_dims, hidden_size),
                "transformer_weights": [
                    {
                        "attention": torch.randn(hidden_size, hidden_size),
                        "ffn": torch.randn(hidden_size, hidden_size * 4),
                    }
                    for _ in range(num_layers)
                ],
                "output": torch.randn(hidden_size, output_dims),
            },
            "metadata": {
                "model_version": "test-model-v1.0",
                "training_data": "synthetic",
                "created_at": "2025-01-01T00:00:00Z",
            },
        }

    @staticmethod
    def forward_pass_time(
        model: Any,
        input_tensor: torch.Tensor,
        num_runs: int = 100
    ) -> Tuple[float, torch.Tensor]:
        """Measure forward pass time.

        Args:
            model: Model to test.
            input_tensor: Input tensor.
            num_runs: Number of runs for averaging.

        Returns:
            Tuple of (average_time, output_tensor).
        """
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_tensor)

        # Timing
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                output = model(input_tensor)
                end_time = time.time()
                times.append(end_time - start_time)

        avg_time = np.mean(times)
        return avg_time, output


class AsyncTestUtils:
    """Utilities for async testing."""

    @staticmethod
    async def run_with_timeout(
        coro,
        timeout: float = 5.0
    ) -> Any:
        """Run coroutine with timeout.

        Args:
            coro: Coroutine to run.
            timeout: Timeout in seconds.

        Returns:
            Result of coroutine.

        Raises:
            asyncio.TimeoutError: If coroutine doesn't complete in time.
        """
        return await asyncio.wait_for(coro, timeout=timeout)

    @staticmethod
    async def gather_with_concurrency(
        coros: List,
        max_concurrency: int = 10
    ) -> List[Any]:
        """Run coroutines with limited concurrency.

        Args:
            coros: List of coroutines to run.
            max_concurrency: Maximum concurrent coroutines.

        Returns:
            List of results.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def run_with_semaphore(coro):
            async with semaphore:
                return await coro

        return await asyncio.gather(
            *[run_with_semaphore(coro) for coro in coros]
        )


class PerformanceTestUtils:
    """Utilities for performance testing."""

    @staticmethod
    def measure_memory_usage() -> Dict[str, float]:
        """Measure current memory usage.

        Returns:
            Dictionary with memory statistics.
        """
        import psutil
        process = psutil.Process()

        return {
            "rss_mb": process.memory_info().rss / 1024 / 1024,
            "vms_mb": process.memory_info().vms / 1024 / 1024,
            "percent": process.memory_percent(),
        }

    @staticmethod
    def benchmark_function(
        func,
        *args,
        num_runs: int = 10,
        warmup_runs: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """Benchmark function performance.

        Args:
            func: Function to benchmark.
            *args: Function arguments.
            num_runs: Number of benchmark runs.
            warmup_runs: Number of warmup runs.
            **kwargs: Function keyword arguments.

        Returns:
            Dictionary with benchmark results.
        """
        # Warmup
        for _ in range(warmup_runs):
            func(*args, **kwargs)

        # Benchmark
        times = []
        memory_before = PerformanceTestUtils.measure_memory_usage()

        for _ in range(num_runs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            times.append(end_time - start_time)

        memory_after = PerformanceTestUtils.measure_memory_usage()

        return {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "total_time": np.sum(times),
            "memory_before": memory_before,
            "memory_after": memory_after,
            "memory_delta": memory_after["rss_mb"] - memory_before["rss_mb"],
            "result": result,
        }


class FileTestUtils:
    """Utilities for file-based testing."""

    @staticmethod
    def create_temp_model_file(
        model_dir: Path,
        model_data: Dict[str, Any]
    ) -> Path:
        """Create temporary model file.

        Args:
            model_dir: Directory to create model in.
            model_data: Model data to save.

        Returns:
            Path to created model file.
        """
        model_file = model_dir / "test_model.safetensors"

        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {}
        for key, value in model_data.items():
            if isinstance(value, torch.Tensor):
                serializable_data[key] = value.numpy().tolist()
            elif isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            else:
                serializable_data[key] = value

        with open(model_file, "w") as f:
            json.dump(serializable_data, f, indent=2)

        return model_file

    @staticmethod
    def cleanup_temp_files(temp_dir: Path) -> None:
        """Clean up temporary files.

        Args:
            temp_dir: Directory to clean up.
        """
        if temp_dir.exists():
            for file in temp_dir.iterdir():
                if file.is_file():
                    file.unlink()
            temp_dir.rmdir()


# Global instances for convenience
data_generator = TestDataGenerator()
model_utils = ModelTestUtils()
async_utils = AsyncTestUtils()
perf_utils = PerformanceTestUtils()
file_utils = FileTestUtils()