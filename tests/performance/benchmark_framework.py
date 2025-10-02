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

"""Performance testing and benchmarking framework for TimesFM."""

import asyncio
import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil
import pytest
import torch
from memory_profiler import profile
from torch.profiler import profile as torch_profile, record_function

from tests.fixtures.mock_timesfm import create_mock_timesfm
from tests.utils.data_generators import create_training_dataset


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests."""
    warmup_iterations: int = 3
    test_iterations: int = 10
    batch_sizes: List[int] = None
    sequence_lengths: List[int] = None
    horizon_lengths: List[int] = None
    device: str = "cpu"
    num_threads: int = 4
    memory_threshold_mb: float = 2048
    time_threshold_seconds: float = 60.0
    collect_gpu_metrics: bool = False
    collect_detailed_profiling: bool = False

    def __post_init__(self):
        """Set default values."""
        if self.batch_sizes is None:
            self.batch_sizes = [1, 8, 16, 32, 64]
        if self.sequence_lengths is None:
            self.sequence_lengths = [64, 128, 256, 512]
        if self.horizon_lengths is None:
            self.horizon_lengths = [24, 48, 96]


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    test_name: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    timestamps: Dict[str, float]
    memory_usage: Dict[str, float]
    gpu_usage: Optional[Dict[str, float]] = None
    profiling_data: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None


class MemoryProfiler:
    """Memory usage profiler."""

    def __init__(self):
        """Initialize memory profiler."""
        self.process = psutil.Process()

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage.

        Returns:
            Dictionary with memory statistics.
        """
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": memory_percent,
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
        }

    def start_profiling(self) -> Dict[str, float]:
        """Start memory profiling.

        Returns:
            Initial memory state.
        """
        gc.collect()  # Force garbage collection
        return self.get_memory_info()

    def end_profiling(self, start_state: Dict[str, float]) -> Dict[str, float]:
        """End memory profiling.

        Args:
            start_state: Initial memory state.

        Returns:
            Memory usage delta.
        """
        gc.collect()
        end_state = self.get_memory_info()

        return {
            "rss_delta_mb": end_state["rss_mb"] - start_state["rss_mb"],
            "vms_delta_mb": end_state["vms_delta_mb"] - start_state["vms_mb"],
            "peak_rss_mb": end_state["rss_mb"],
            "end_rss_mb": end_state["rss_mb"],
        }


class GPUProfiler:
    """GPU usage profiler for CUDA devices."""

    def __init__(self):
        """Initialize GPU profiler."""
        self.device_count = torch.cuda.device_count()
        self.has_cuda = self.device_count > 0

    def get_gpu_info(self) -> Optional[Dict[str, float]]:
        """Get GPU usage information.

        Returns:
            GPU statistics or None if CUDA unavailable.
        """
        if not self.has_cuda:
            return None

        try:
            device = torch.cuda.current_device()
            memory_allocated = torch.cuda.memory_allocated(device)
            memory_reserved = torch.cuda.memory_reserved(device)

            return {
                "memory_allocated_mb": memory_allocated / 1024 / 1024,
                "memory_reserved_mb": memory_reserved / 1024 / 1024,
                "device": device,
            }
        except Exception:
            return None

    def reset_peak_memory(self) -> None:
        """Reset peak memory counter."""
        if self.has_cuda:
            torch.cuda.reset_peak_memory_stats()

    def get_peak_memory(self) -> Optional[float]:
        """Get peak GPU memory usage.

        Returns:
            Peak memory in MB or None if CUDA unavailable.
        """
        if not self.has_cuda:
            return None

        try:
            device = torch.cuda.current_device()
            peak_memory = torch.cuda.max_memory_allocated(device)
            return peak_memory / 1024 / 1024
        except Exception:
            return None


class PerformanceBenchmark:
    """Main performance benchmark framework."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark framework.

        Args:
            config: Benchmark configuration.
        """
        self.config = config
        self.memory_profiler = MemoryProfiler()
        self.gpu_profiler = GPUProfiler() if config.collect_gpu_metrics else None
        self.results: List[BenchmarkResult] = []

    def benchmark_function(
        self,
        func: Callable,
        *args,
        test_name: str = "unknown",
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark a function execution.

        Args:
            func: Function to benchmark.
            *args: Function arguments.
            test_name: Name of the test.
            **kwargs: Function keyword arguments.

        Returns:
            Benchmark result.
        """
        # Start memory profiling
        memory_start = self.memory_profiler.start_profiling()
        gpu_start = self.gpu_profiler.get_gpu_info() if self.gpu_profiler else None

        timestamps = {"start": time.time()}

        # Warmup iterations
        for _ in range(self.config.warmup_iterations):
            try:
                func(*args, **kwargs)
            except Exception as e:
                timestamps["warmup_error"] = time.time()
                return BenchmarkResult(
                    test_name=test_name,
                    config=asdict(self.config),
                    metrics={"error": 1},
                    timestamps=timestamps,
                    memory_usage=memory_start,
                    errors=[f"Warmup failed: {str(e)}"]
                )

        timestamps["warmup_end"] = time.time()

        # Benchmark iterations
        execution_times = []
        memory_peaks = []

        for i in range(self.config.test_iterations):
            iter_start = time.time()

            # Profile memory for this iteration
            iter_memory_start = self.memory_profiler.get_memory_info()

            try:
                result = func(*args, **kwargs)
                execution_times.append(time.time() - iter_start)

                # Check memory usage after iteration
                iter_memory_end = self.memory_profiler.get_memory_info()
                memory_peaks.append(iter_memory_end["rss_mb"])

            except Exception as e:
                execution_times.append(float('inf'))
                if i == 0:  # Stop early if first iteration fails
                    timestamps["benchmark_error"] = time.time()
                    memory_end = self.memory_profiler.end_profiling(memory_start)
                    return BenchmarkResult(
                        test_name=test_name,
                        config=asdict(self.config),
                        metrics={"error": 1},
                        timestamps=timestamps,
                        memory_usage=memory_end,
                        errors=[f"Benchmark failed: {str(e)}"]
                    )

        timestamps["benchmark_end"] = time.time()

        # End memory profiling
        memory_end = self.memory_profiler.end_profiling(memory_start)
        gpu_end = self.gpu_profiler.get_gpu_info() if self.gpu_profiler else None

        # Calculate metrics
        valid_times = [t for t in execution_times if t != float('inf')]
        metrics = {
            "mean_time": np.mean(valid_times) if valid_times else float('inf'),
            "std_time": np.std(valid_times) if valid_times else 0,
            "min_time": np.min(valid_times) if valid_times else float('inf'),
            "max_time": np.max(valid_times) if valid_times else float('inf'),
            "median_time": np.median(valid_times) if valid_times else float('inf'),
            "total_time": np.sum(execution_times),
            "throughput": len(valid_times) / np.sum(valid_times) if valid_times else 0,
            "success_rate": len(valid_times) / len(execution_times),
        }

        return BenchmarkResult(
            test_name=test_name,
            config=asdict(self.config),
            metrics=metrics,
            timestamps=timestamps,
            memory_usage=memory_end,
            gpu_usage=gpu_end,
        )

    def benchmark_model_inference(
        self,
        model: Any,
        input_data: torch.Tensor,
        horizon: int = 96,
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark model inference.

        Args:
            model: Model to benchmark.
            input_data: Input tensor.
            horizon: Forecast horizon.
            **kwargs: Additional inference arguments.

        Returns:
            Benchmark result.
        """
        def inference():
            return model.forecast(input_data, horizon=horizon, **kwargs)

        return self.benchmark_function(
            inference,
            test_name=f"model_inference_{model.config.model_name}",
        )

    def benchmark_batch_processing(
        self,
        model: Any,
        batch_data: torch.Tensor,
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark batch processing.

        Args:
            model: Model to benchmark.
            batch_data: Batch input data.
            **kwargs: Additional inference arguments.

        Returns:
            Benchmark result.
        """
        batch_size = batch_data.shape[0]

        def batch_inference():
            return model.forecast(batch_data, **kwargs)

        return self.benchmark_function(
            batch_inference,
            test_name=f"batch_processing_size_{batch_size}",
        )

    def run_scalability_test(
        self,
        model: Any,
        data_generator: Callable,
        **kwargs
    ) -> List[BenchmarkResult]:
        """Run scalability tests across different configurations.

        Args:
            model: Model to test.
            data_generator: Function to generate test data.
            **kwargs: Additional arguments.

        Returns:
            List of benchmark results.
        """
        results = []

        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.sequence_lengths:
                for horizon in self.config.horizon_lengths:
                    # Generate test data
                    test_data = data_generator(
                        batch_size=batch_size,
                        sequence_length=seq_len,
                        horizon_length=horizon
                    )

                    result = self.benchmark_model_inference(
                        model,
                        test_data["input"],
                        horizon=horizon,
                        **kwargs
                    )
                    result.test_name = f"scalability_bs{batch_size}_sl{seq_len}_h{horizon}"
                    results.append(result)

        return results

    def run_concurrent_test(
        self,
        model: Any,
        input_data: torch.Tensor,
        num_workers: int = 4,
        requests_per_worker: int = 10,
        **kwargs
    ) -> BenchmarkResult:
        """Run concurrent inference test.

        Args:
            model: Model to test.
            input_data: Input tensor.
            num_workers: Number of concurrent workers.
            requests_per_worker: Requests per worker.
            **kwargs: Additional inference arguments.

        Returns:
            Benchmark result.
        """
        def worker_inference():
            return model.forecast(input_data, **kwargs)

        timestamps = {"start": time.time()}
        memory_start = self.memory_profiler.start_profiling()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for _ in range(requests_per_worker * num_workers):
                future = executor.submit(worker_inference)
                futures.append(future)

            results = []
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})

        timestamps["end"] = time.time()
        memory_end = self.memory_profiler.end_profiling(memory_start)

        success_count = sum(1 for r in results if "error" not in r)
        total_time = timestamps["end"] - timestamps["start"]

        metrics = {
            "total_requests": len(results),
            "successful_requests": success_count,
            "failed_requests": len(results) - success_count,
            "success_rate": success_count / len(results),
            "total_time": total_time,
            "requests_per_second": len(results) / total_time,
            "successful_requests_per_second": success_count / total_time,
            "concurrent_workers": num_workers,
        }

        return BenchmarkResult(
            test_name=f"concurrent_inference_workers_{num_workers}",
            config=asdict(self.config),
            metrics=metrics,
            timestamps=timestamps,
            memory_usage=memory_end,
        )

    def save_results(self, filepath: Union[str, Path]) -> None:
        """Save benchmark results to file.

        Args:
            filepath: Path to save results.
        """
        filepath = Path(filepath)
        results_data = []

        for result in self.results:
            result_dict = asdict(result)
            # Convert numpy types to Python native types
            result_dict["metrics"] = {
                k: float(v) if isinstance(v, np.number) else v
                for k, v in result_dict["metrics"].items()
            }
            result_dict["memory_usage"] = {
                k: float(v) if isinstance(v, np.number) else v
                for k, v in result_dict["memory_usage"].items()
            }
            results_data.append(result_dict)

        with open(filepath, "w") as f:
            json.dump(results_data, f, indent=2)

    def generate_report(self, output_dir: Union[str, Path]) -> pd.DataFrame:
        """Generate performance report.

        Args:
            output_dir: Directory to save report.

        Returns:
            DataFrame with performance summary.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Create summary DataFrame
        data = []
        for result in self.results:
            row = {
                "test_name": result.test_name,
                "mean_time": result.metrics.get("mean_time"),
                "std_time": result.metrics.get("std_time"),
                "min_time": result.metrics.get("min_time"),
                "max_time": result.metrics.get("max_time"),
                "success_rate": result.metrics.get("success_rate"),
                "throughput": result.metrics.get("throughput"),
                "memory_peak_mb": result.memory_usage.get("peak_rss_mb"),
                "memory_delta_mb": result.memory_usage.get("rss_delta_mb"),
            }
            data.append(row)

        df = pd.DataFrame(data)

        # Save CSV report
        csv_path = output_dir / "performance_report.csv"
        df.to_csv(csv_path, index=False)

        # Save detailed JSON results
        json_path = output_dir / "detailed_results.json"
        self.save_results(json_path)

        return df


class LoadTester:
    """Load testing framework for API endpoints."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize load tester.

        Args:
            base_url: Base URL for API testing.
        """
        self.base_url = base_url
        self.results = []

    async def run_load_test(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        concurrent_users: int = 10,
        requests_per_user: int = 100,
        test_duration: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run load test on API endpoint.

        Args:
            endpoint: API endpoint to test.
            payload: Request payload.
            concurrent_users: Number of concurrent users.
            requests_per_user: Requests per user.
            test_duration: Maximum test duration in seconds.
            **kwargs: Additional request arguments.

        Returns:
            Load test results.
        """
        import aiohttp

        url = f"{self.base_url}{endpoint}"
        timestamps = {"start": time.time()}

        async def user_session(session, user_id):
            """Simulate user session."""
            user_results = []
            user_start = time.time()

            for request_id in range(requests_per_user):
                request_start = time.time()
                try:
                    async with session.post(url, json=payload, **kwargs) as response:
                        if response.status == 200:
                            result = await response.json()
                            user_results.append({
                                "user_id": user_id,
                                "request_id": request_id,
                                "status": "success",
                                "response_time": time.time() - request_start,
                                "status_code": response.status,
                            })
                        else:
                            user_results.append({
                                "user_id": user_id,
                                "request_id": request_id,
                                "status": "error",
                                "response_time": time.time() - request_start,
                                "status_code": response.status,
                            })
                except Exception as e:
                    user_results.append({
                        "user_id": user_id,
                        "request_id": request_id,
                        "status": "exception",
                        "response_time": time.time() - request_start,
                        "error": str(e),
                    })

                # Check if test duration exceeded
                if test_duration and time.time() - timestamps["start"] > test_duration:
                    break

            return user_results

        # Run concurrent users
        connector = aiohttp.TCPConnector(limit=concurrent_users * 2)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                user_session(session, user_id)
                for user_id in range(concurrent_users)
            ]
            user_results = await asyncio.gather(*tasks, return_exceptions=True)

        timestamps["end"] = time.time()

        # Aggregate results
        all_requests = []
        for user_result in user_results:
            if isinstance(user_result, list):
                all_requests.extend(user_result)

        # Calculate metrics
        successful_requests = [r for r in all_requests if r["status"] == "success"]
        failed_requests = [r for r in all_requests if r["status"] != "success"]

        if successful_requests:
            response_times = [r["response_time"] for r in successful_requests]
        else:
            response_times = []

        metrics = {
            "total_requests": len(all_requests),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": len(successful_requests) / len(all_requests) if all_requests else 0,
            "total_time": timestamps["end"] - timestamps["start"],
            "requests_per_second": len(all_requests) / (timestamps["end"] - timestamps["start"]),
            "avg_response_time": np.mean(response_times) if response_times else 0,
            "median_response_time": np.median(response_times) if response_times else 0,
            "p95_response_time": np.percentile(response_times, 95) if response_times else 0,
            "p99_response_time": np.percentile(response_times, 99) if response_times else 0,
            "concurrent_users": concurrent_users,
        }

        return {
            "metrics": metrics,
            "timestamps": timestamps,
            "raw_results": all_requests,
        }


# Benchmark decorators for pytest
def benchmark_model(
    model_sizes: List[str] = None,
    batch_sizes: List[int] = None,
    iterations: int = 10
):
    """Decorator for benchmarking models.

    Args:
        model_sizes: Model sizes to test.
        batch_sizes: Batch sizes to test.
        iterations: Number of iterations.
    """
    def decorator(func):
        @pytest.mark.performance
        def wrapper(*args, **kwargs):
            # Implementation for model benchmarking
            return func(*args, **kwargs)
        return wrapper
    return decorator


def benchmark_api(
    endpoints: List[str] = None,
    concurrent_users: int = 10,
    requests_per_user: int = 100
):
    """Decorator for benchmarking API endpoints.

    Args:
        endpoints: API endpoints to test.
        concurrent_users: Number of concurrent users.
        requests_per_user: Requests per user.
    """
    def decorator(func):
        @pytest.mark.performance
        @pytest.mark.slow
        async def async_wrapper(*args, **kwargs):
            # Implementation for API benchmarking
            return await func(*args, **kwargs)

        @pytest.mark.performance
        def sync_wrapper(*args, **kwargs):
            # Implementation for API benchmarking
            return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator