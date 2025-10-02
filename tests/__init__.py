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

"""TimesFM test framework package.

This package provides a comprehensive testing framework for the TimesFM API
including unit tests, integration tests, and performance benchmarks.

The framework includes:
- Mock TimesFM models for testing without model loading
- Realistic time series data generators
- In-memory database and storage for integration tests
- Performance benchmarking and load testing utilities
- Async testing capabilities
- Coverage reporting and test discovery

Example usage:
    # Create a mock model for testing
    from tests.fixtures.mock_timesfm import create_mock_timesfm
    model = create_mock_timesfm(model_size="200m", device="cpu")

    # Generate test data
    from tests.utils.data_generators import create_training_dataset
    dataset = create_training_dataset(n_series=100, context_length=512, horizon=96)

    # Run performance benchmarks
    from tests.performance.benchmark_framework import PerformanceBenchmark, BenchmarkConfig
    config = BenchmarkConfig(test_iterations=10, batch_sizes=[1, 4, 8])
    benchmark = PerformanceBenchmark(config)
    result = benchmark.benchmark_model_inference(model, test_data)

For more detailed examples, see the test files in the respective directories.
"""

__version__ = "1.0.0"

from .fixtures import create_mock_timesfm
from .utils.data_generators import create_training_dataset, create_test_dataset

__all__ = [
    "create_mock_timesfm",
    "create_training_dataset",
    "create_test_dataset",
]