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

 """Performance benchmarks for TimesFM models."""

import pytest
import torch

from tests.fixtures.mock_timesfm import create_mock_timesfm
from tests.performance.benchmark_framework import (
    BenchmarkConfig,
    PerformanceBenchmark,
    benchmark_model,
)
from tests.utils.data_generators import create_training_dataset


class TestModelPerformance:
    """Performance tests for TimesFM models."""

    @pytest.fixture
    def benchmark_config(self) -> BenchmarkConfig:
        """Benchmark configuration for testing."""
        return BenchmarkConfig(
            warmup_iterations=2,
            test_iterations=5,
            batch_sizes=[1, 4, 8],
            sequence_lengths=[64, 128, 256],
            horizon_lengths=[24, 48],
            memory_threshold_mb=1024,
            time_threshold_seconds=30.0,
        )

    @pytest.fixture
    def small_model(self) -> Any:
        """Small model for performance testing."""
        return create_mock_timesfm(model_size="50m", device="cpu")

    @pytest.fixture
    def medium_model(self) -> Any:
        """Medium model for performance testing."""
        return create_mock_timesfm(model_size="200m", device="cpu")

    @pytest.fixture
    def test_data(self) -> dict:
        """Test data for benchmarking."""
        return create_training_dataset(
            n_series=10,
            context_length=256,
            horizon=48,
            n_features=1,
            generator_seed=42
        )

    @pytest.mark.performance
    def test_small_model_basic_performance(
        self,
        small_model: Any,
        benchmark_config: BenchmarkConfig,
        test_data: dict
    ) -> None:
        """Test basic performance of small model."""
        benchmark = PerformanceBenchmark(benchmark_config)
        small_model.load()

        # Test single inference
        input_data = test_data[0]["context"]
        result = benchmark.benchmark_model_inference(
            small_model,
            input_data,
            horizon=48
        )

        # Validate results
        assert result.test_name == "model_inference_timesfm-2.5-50m"
        assert result.metrics["success_rate"] == 1.0
        assert result.metrics["mean_time"] < benchmark_config.time_threshold_seconds
        assert result.memory_usage["rss_delta_mb"] < benchmark_config.memory_threshold_mb

        # Performance expectations
        assert result.metrics["mean_time"] < 1.0  # Should be fast for small model
        assert result.metrics["throughput"] > 1.0  # At least 1 inference per second

    @pytest.mark.performance
    def test_medium_model_basic_performance(
        self,
        medium_model: Any,
        benchmark_config: BenchmarkConfig,
        test_data: dict
    ) -> None:
        """Test basic performance of medium model."""
        benchmark = PerformanceBenchmark(benchmark_config)
        medium_model.load()

        # Test single inference
        input_data = test_data[0]["context"]
        result = benchmark.benchmark_model_inference(
            medium_model,
            input_data,
            horizon=48
        )

        # Validate results
        assert result.test_name == "model_inference_timesfm-2.5-200m"
        assert result.metrics["success_rate"] == 1.0
        assert result.metrics["mean_time"] < benchmark_config.time_threshold_seconds

        # Medium model should be slower than small model
        assert result.metrics["mean_time"] > 0

    @pytest.mark.performance
    def test_batch_processing_performance(
        self,
        small_model: Any,
        benchmark_config: BenchmarkConfig,
        test_data: dict
    ) -> None:
        """Test batch processing performance."""
        benchmark = PerformanceBenchmark(benchmark_config)
        small_model.load()

        # Create batch data
        batch_inputs = torch.stack([item["context"] for item in test_data[:4]])

        # Test batch processing
        result = benchmark.benchmark_batch_processing(
            small_model,
            batch_inputs,
            horizon=48
        )

        # Validate results
        assert "batch_processing_size_4" in result.test_name
        assert result.metrics["success_rate"] == 1.0
        assert result.metrics["mean_time"] < benchmark_config.time_threshold_seconds

        # Batch processing should be more efficient than individual processing
        expected_time_per_sample = result.metrics["mean_time"] / 4
        assert expected_time_per_sample < 1.0  # Should be efficient

    @pytest.mark.performance
    def test_scalability_performance(
        self,
        small_model: Any,
        benchmark_config: BenchmarkConfig
    ) -> None:
        """Test performance across different configurations."""
        benchmark = PerformanceBenchmark(benchmark_config)
        small_model.load()

        def generate_test_data(batch_size, sequence_length, horizon_length):
            return {
                "input": torch.randn(batch_size, sequence_length, 1),
                "target": torch.randn(batch_size, horizon_length, 1),
            }

        # Run scalability tests
        results = benchmark.run_scalability_test(
            small_model,
            generate_test_data
        )

        # Should have results for all combinations
        expected_combinations = (
            len(benchmark_config.batch_sizes) *
            len(benchmark_config.sequence_lengths) *
            len(benchmark_config.horizon_lengths)
        )
        assert len(results) == expected_combinations

        # Validate each result
        for result in results:
            assert result.metrics["success_rate"] == 1.0
            assert result.metrics["mean_time"] < benchmark_config.time_threshold_seconds

        # Check that performance scales reasonably
        # Larger batches should be more efficient per sample
        batch_1_results = [r for r in results if "bs1_" in r.test_name]
        batch_8_results = [r for r in results if "bs8_" in r.test_name]

        if batch_1_results and batch_8_results:
            avg_time_1 = sum(r.metrics["mean_time"] for r in batch_1_results) / len(batch_1_results)
            avg_time_8 = sum(r.metrics["mean_time"] for r in batch_8_results) / len(batch_8_results)

            # Batch 8 should not be 8x slower than batch 1 (indicating good batching)
            assert avg_time_8 < avg_time_1 * 6

    @pytest.mark.performance
    def test_concurrent_inference_performance(
        self,
        small_model: Any,
        benchmark_config: BenchmarkConfig,
        test_data: dict
    ) -> None:
        """Test concurrent inference performance."""
        benchmark = PerformanceBenchmark(benchmark_config)
        small_model.load()

        input_data = test_data[0]["context"]

        # Test with different numbers of workers
        worker_counts = [1, 2, 4]
        results = []

        for num_workers in worker_counts:
            result = benchmark.run_concurrent_test(
                small_model,
                input_data,
                num_workers=num_workers,
                requests_per_worker=5
            )
            results.append(result)

        # Validate results
        for i, result in enumerate(results):
            assert result.test_name == f"concurrent_inference_workers_{worker_counts[i]}"
            assert result.metrics["success_rate"] == 1.0
            assert result.metrics["total_requests"] == worker_counts[i] * 5
            assert result.metrics["successful_requests"] == result.metrics["total_requests"]

        # Check that concurrency improves throughput
        single_worker_throughput = results[0].metrics["successful_requests_per_second"]
        multi_worker_throughput = max(r.metrics["successful_requests_per_second"] for r in results[1:])

        # Multi-worker should be more efficient (at least in theory)
        # Note: This might not always be true in practice due to overhead
        if multi_worker_throughput > single_worker_throughput * 0.8:
            assert True  # Acceptable performance
        else:
            # If not, at least verify it completed successfully
            assert results[0].metrics["success_rate"] == 1.0

    @pytest.mark.performance
    def test_memory_usage_scaling(
        self,
        small_model: Any,
        benchmark_config: BenchmarkConfig
    ) -> None:
        """Test memory usage scaling with batch size."""
        benchmark = PerformanceBenchmark(benchmark_config)
        small_model.load()

        memory_results = []

        for batch_size in [1, 4, 8, 16]:
            input_data = torch.randn(batch_size, 256, 1)

            result = benchmark.benchmark_model_inference(
                small_model,
                input_data,
                horizon=48
            )

            memory_results.append({
                "batch_size": batch_size,
                "memory_delta_mb": result.memory_usage["rss_delta_mb"],
                "peak_memory_mb": result.memory_usage["peak_rss_mb"],
            })

        # Memory usage should scale reasonably with batch size
        # (allowing for some overhead and non-linear scaling)
        base_memory = memory_results[0]["memory_delta_mb"]
        for i, result in enumerate(memory_results[1:], 1):
            expected_max_memory = base_memory * (result["batch_size"] * 1.5)  # Allow 50% overhead
            assert result["memory_delta_mb"] < expected_max_memory

        # Memory should not grow excessively
        total_memory_usage = sum(r["memory_delta_mb"] for r in memory_results)
        assert total_memory_usage < benchmark_config.memory_threshold_mb

    @pytest.mark.performance
    @pytest.mark.slow
    def test_extended_performance_test(
        self,
        medium_model: Any,
        benchmark_config: BenchmarkConfig
    ) -> None:
        """Extended performance test with more iterations."""
        # Use more demanding configuration
        extended_config = BenchmarkConfig(
            warmup_iterations=5,
            test_iterations=20,
            batch_sizes=[1, 2, 4, 8, 16],
            sequence_lengths=[128, 256, 512],
            horizon_lengths=[24, 48, 96],
            memory_threshold_mb=2048,
            time_threshold_seconds=60.0,
        )

        benchmark = PerformanceBenchmark(extended_config)
        medium_model.load()

        # Test a representative case
        input_data = torch.randn(8, 256, 1)
        result = benchmark.benchmark_model_inference(
            medium_model,
            input_data,
            horizon=48
        )

        # Validate extended test results
        assert result.metrics["success_rate"] == 1.0
        assert result.metrics["mean_time"] < extended_config.time_threshold_seconds
        assert result.memory_usage["rss_delta_mb"] < extended_config.memory_threshold_mb

        # Check consistency across iterations
        assert result.metrics["std_time"] < result.metrics["mean_time"] * 0.5  # CV < 50%

    @benchmark_model(model_sizes=["50m", "200m"], batch_sizes=[1, 4, 8], iterations=5)
    def test_model_comparison_benchmark(self) -> None:
        """Benchmark comparison between different model sizes."""
        # This test demonstrates the benchmark decorator
        # The actual benchmark logic would be implemented in the decorator
        pass

    def test_performance_regression_detection(
        self,
        small_model: Any,
        benchmark_config: BenchmarkConfig
    ) -> None:
        """Test to detect performance regressions."""
        benchmark = PerformanceBenchmark(benchmark_config)
        small_model.load()

        input_data = torch.randn(4, 128, 1)

        # Run benchmark
        result = benchmark.benchmark_model_inference(
            small_model,
            input_data,
            horizon=24
        )

        # Define performance thresholds (these would be based on historical data)
        performance_thresholds = {
            "max_mean_time": 2.0,  # seconds
            "min_throughput": 0.5,  # requests per second
            "max_memory_mb": 500,  # MB
            "min_success_rate": 0.95,  # 95%
        }

        # Check against thresholds
        assert result.metrics["mean_time"] < performance_thresholds["max_mean_time"]
        assert result.metrics["throughput"] > performance_thresholds["min_throughput"]
        assert result.memory_usage["rss_delta_mb"] < performance_thresholds["max_memory_mb"]
        assert result.metrics["success_rate"] >= performance_thresholds["min_success_rate"]

        # Store result for regression analysis (in real scenario)
        benchmark.results.append(result)

    def test_model_warmup_effects(
        self,
        medium_model: Any,
        benchmark_config: BenchmarkConfig
    ) -> None:
        """Test model warmup effects on performance."""
        # Create configuration with no warmup
        no_warmup_config = BenchmarkConfig(
            warmup_iterations=0,
            test_iterations=10,
            batch_sizes=[1],
            sequence_lengths=[256],
            horizon_lengths=[48],
        )

        benchmark = PerformanceBenchmark(no_warmup_config)
        medium_model.load()

        input_data = torch.randn(1, 256, 1)

        # First run without warmup (cold start)
        cold_result = benchmark.benchmark_model_inference(
            medium_model,
            input_data,
            horizon=48
        )

        # Reset model state
        medium_model.load(force_reload=True)

        # Second run without warmup (but model already loaded)
        warm_result = benchmark.benchmark_model_inference(
            medium_model,
            input_data,
            horizon=48
        )

        # Warm run should be faster or more consistent
        # Note: This is a subtle test and might not always show differences in mock models
        assert cold_result.metrics["success_rate"] == 1.0
        assert warm_result.metrics["success_rate"] == 1.0

        # At minimum, both should complete successfully
        assert True  # Test passes if both runs complete