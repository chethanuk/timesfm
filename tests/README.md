# TimesFM Test Framework

A comprehensive testing framework for the TimesFM API that supports unit tests, integration tests, and performance benchmarks with realistic test data and comprehensive coverage capabilities.

## Overview

The TimesFM test framework provides:

- **Mock TimesFM Models**: Realistic mock models for testing without requiring actual model loading
- **Data Generators**: Advanced utilities for generating realistic time series test data
- **Integration Testing**: In-memory database and storage for integration tests
- **Performance Benchmarking**: Comprehensive performance testing and load testing utilities
- **Async Testing**: Full support for asynchronous testing capabilities
- **Coverage Reporting**: Integrated coverage reporting with configurable thresholds

## Structure

```
tests/
├── __init__.py                 # Package initialization
├── conftest.py                 # Pytest configuration and shared fixtures
├── README.md                   # This file
├── fixtures/                   # Test fixtures and mock objects
│   ├── __init__.py
│   └── mock_timesfm.py         # Mock TimesFM model implementation
├── unit/                       # Unit tests
│   ├── __init__.py
│   └── test_mock_timesfm.py    # Unit tests for mock model
├── integration/                # Integration tests
│   ├── __init__.py
│   └── test_data_pipeline.py   # Integration tests for data pipeline
├── performance/                # Performance tests and benchmarks
│   ├── __init__.py
│   ├── benchmark_framework.py  # Performance testing framework
│   └── test_model_benchmarks.py # Model performance benchmarks
└── utils/                      # Test utilities and helpers
    ├── __init__.py
    ├── test_helpers.py         # General test utilities
    ├── data_generators.py      # Advanced data generation
    └── integration_helpers.py  # Integration test helpers
```

## Installation

Install the test dependencies:

```bash
# Install development dependencies
poetry install --with dev

# Or install specific test dependencies
pip install pytest pytest-asyncio pytest-cov pytest-benchmark pytest-mock
pip install aioresponses aiosqlite psutil memory-profiler redis fastapi httpx
```

## Quick Start

### 1. Using Mock Models

```python
from tests.fixtures.mock_timesfm import create_mock_timesfm
import torch

# Create a mock model
model = create_mock_timesfm(model_size="200m", device="cpu")
model.load()

# Generate forecasts
input_data = torch.randn(100, 1)
result = model.forecast(input_data, horizon=48, quantiles=[0.1, 0.5, 0.9])

print(f"Forecast shape: {result['forecasts'].shape}")
print(f"Quantiles: {list(result['quantiles'].keys())}")
```

### 2. Generating Test Data

```python
from tests.utils.data_generators import create_training_dataset

# Create training dataset
dataset = create_training_dataset(
    n_series=100,
    context_length=512,
    horizon=96,
    n_features=1
)

print(f"Dataset size: {len(dataset)}")
print(f"Sample shape: {dataset[0]['context'].shape}")
```

### 3. Running Performance Benchmarks

```python
from tests.performance.benchmark_framework import PerformanceBenchmark, BenchmarkConfig

# Configure benchmark
config = BenchmarkConfig(
    test_iterations=10,
    batch_sizes=[1, 4, 8],
    sequence_lengths=[256, 512],
    horizon_lengths=[48, 96]
)

# Run benchmarks
benchmark = PerformanceBenchmark(config)
result = benchmark.benchmark_model_inference(model, test_data)

print(f"Mean inference time: {result.metrics['mean_time']:.3f}s")
print(f"Throughput: {result.metrics['throughput']:.1f} req/s")
```

## Running Tests

### Basic Test Commands

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/timesfm --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m performance   # Performance tests only
pytest -m "not slow"    # Skip slow tests

# Run with parallel execution
pytest -n auto

# Run specific test file
pytest tests/unit/test_mock_timesfm.py
```

### Performance Testing

```bash
# Run performance benchmarks
pytest -m performance

# Run extended performance tests
pytest -m performance -v --benchmark-only

# Generate performance report
pytest tests/performance/ --benchmark-json=benchmark_results.json
```

### Integration Testing

```bash
# Run integration tests
pytest -m integration

# Run async tests
pytest -m async

# Run API tests
pytest -m api
```

## Test Categories

### Unit Tests (`-m unit`)

Test individual components in isolation:

- Mock model functionality
- Data generation utilities
- Helper functions
- Configuration validation

### Integration Tests (`-m integration`)

Test component interactions:

- End-to-end data pipelines
- Database operations
- API endpoint integration
- Async workflows

### Performance Tests (`-m performance`)

Benchmark model performance:

- Inference latency
- Memory usage
- Throughput
- Scalability
- Concurrent processing

### Async Tests (`-m async`)

Test asynchronous functionality:

- Async database operations
- Concurrent model inference
- Async API calls

## Configuration

### Pytest Configuration

The framework uses comprehensive pytest configuration in `pyproject.toml`:

- Test discovery settings
- Coverage reporting
- Async testing support
- Custom markers
- Warning filters

### Benchmark Configuration

Configure performance tests using `BenchmarkConfig`:

```python
config = BenchmarkConfig(
    warmup_iterations=3,        # Warmup runs before benchmarking
    test_iterations=10,         # Number of benchmark runs
    batch_sizes=[1, 4, 8],      # Batch sizes to test
    sequence_lengths=[256, 512], # Input sequence lengths
    horizon_lengths=[48, 96],   # Forecast horizons
    memory_threshold_mb=2048,   # Memory usage threshold
    time_threshold_seconds=60,  # Time threshold per test
)
```

## Mock TimesFM Models

The framework includes realistic mock TimesFM models that:

- Mimic the real API interface
- Generate plausible forecasting results
- Support multiple model sizes (50M, 200M, 1B parameters)
- Include realistic time series patterns
- Support quantile forecasts and sampling

### Available Model Sizes

```python
# 50M parameter model
model_50m = create_mock_timesfm(model_size="50m")

# 200M parameter model
model_200m = create_mock_timesfm(model_size="200m")

# 1B parameter model
model_1b = create_mock_timesfm(model_size="1b")
```

## Data Generation

### Advanced Time Series Generation

The framework provides sophisticated data generators:

```python
from tests.utils.data_generators import AdvancedTimeSeriesGenerator

generator = AdvancedTimeSeriesGenerator(random_seed=42)

# Multivariate time series with patterns
df = generator.generate_multivariate_time_series(
    n_samples=1000,
    n_features=3,
    patterns=["trend_seasonal", "cyclical", "random_walk"],
    missing_data_rate=0.01,
    outlier_rate=0.005
)

# Forecasting dataset
dataset = generator.generate_forecasting_dataset(
    n_series=100,
    context_length=512,
    horizon=96,
    n_features=1,
    dataset_type="various"
)

# Anomaly data
anomaly_data, labels, metadata = generator.generate_anomaly_data(
    n_samples=1000,
    anomaly_types=["spike", "dip", "drift", "level_shift"],
    anomaly_rate=0.02
)
```

### Pre-built Datasets

```python
# Training dataset with diverse patterns
training_data = create_training_dataset(
    n_series=1000,
    context_length=512,
    horizon=96
)

# Test dataset organized by pattern
test_data = create_test_dataset(
    patterns=["trend_seasonal", "cyclical", "random_walk"],
    n_series_per_pattern=20
)
```

## Performance Benchmarking

### Benchmark Types

1. **Inference Benchmarks**: Measure model inference performance
2. **Scalability Tests**: Test performance across different configurations
3. **Memory Profiling**: Monitor memory usage and leaks
4. **Concurrent Tests**: Test performance under load
5. **Regression Tests**: Detect performance regressions

### Example Benchmark Usage

```python
from tests.performance.benchmark_framework import PerformanceBenchmark, BenchmarkConfig

# Configure comprehensive benchmark
config = BenchmarkConfig(
    warmup_iterations=5,
    test_iterations=20,
    batch_sizes=[1, 2, 4, 8, 16],
    sequence_lengths=[128, 256, 512],
    horizon_lengths=[24, 48, 96],
)

# Run benchmark suite
benchmark = PerformanceBenchmark(config)

# Single model inference benchmark
result = benchmark.benchmark_model_inference(model, test_data, horizon=48)

# Scalability test
results = benchmark.run_scalability_test(model, data_generator)

# Concurrent inference test
concurrent_result = benchmark.run_concurrent_test(
    model, input_data, num_workers=4, requests_per_worker=100
)

# Save results
benchmark.save_results("benchmark_results.json")
df = benchmark.generate_report("benchmark_output")
```

## Integration Testing

### Database Testing

```python
from tests.utils.integration_helpers import create_test_database

# Create in-memory database
db = create_test_database()

# Insert test data
user_id = db.insert_test_user()
request_id = db.insert_forecast_request(
    user_id=user_id,
    model_name="timesfm-2.5-200m",
    input_shape="(512, 1)",
    horizon=96
)

# Test operations
request = db.get_forecast_request(request_id)
db.update_forecast_request(request_id, status="completed")
```

### Async Testing

```python
import pytest
from tests.utils.integration_helpers import AsyncInMemoryDatabase

@pytest.mark.asyncio
async def test_async_operations():
    async_db = AsyncInMemoryDatabase()
    await async_db.connect()

    async with async_db.get_connection() as conn:
        result = await conn.execute("SELECT 1")
        assert (await result.fetchone())[0] == 1

    await async_db.close()
```

## Coverage

The framework includes comprehensive coverage reporting:

```bash
# Generate coverage report
pytest --cov=src/timesfm --cov-report=html

# Coverage with minimum threshold
pytest --cov=src/timesfm --cov-fail-under=80

# Coverage for specific modules
pytest --cov=src/timesfm.models --cov=src/timesfm.api
```

Coverage settings:
- Target: 80% minimum coverage
- Excludes: test files, migrations, scripts
- Reports: Terminal, HTML, XML
- Exclude lines: pragma no cover, abstract methods, etc.

## Best Practices

### 1. Test Organization

- Use descriptive test names that explain what is being tested
- Group related tests in classes
- Use fixtures for common setup/teardown
- Mark tests with appropriate markers

### 2. Mock Usage

- Use mock models for unit tests to avoid loading real models
- Mock external dependencies (databases, APIs)
- Use realistic test data that matches production characteristics

### 3. Performance Testing

- Always include warmup iterations for stable measurements
- Test across different input sizes and configurations
- Monitor both time and memory metrics
- Use performance thresholds to catch regressions

### 4. Data Generation

- Use consistent random seeds for reproducible tests
- Include edge cases and realistic scenarios
- Test with missing data, outliers, and anomalies
- Validate data shapes and properties

### 5. Async Testing

- Use proper async test decorators
- Test both success and failure scenarios
- Use timeouts to prevent hanging tests
- Test concurrent operations

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Use CPU device for tests or reduce batch sizes
2. **Slow Tests**: Skip slow tests with `-m "not slow"`
3. **Coverage Issues**: Check exclude patterns and import paths
4. **Async Test Failures**: Ensure proper async/await usage and fixtures

### Debug Mode

Run tests with verbose output and debugging:

```bash
pytest -v -s --tb=long
pytest --pdb  # Drop into debugger on failure
```

## Contributing

When adding new tests:

1. Follow the existing directory structure
2. Use appropriate markers for test categorization
3. Include comprehensive test data
4. Add performance benchmarks for new features
5. Update documentation as needed

## License

This test framework is part of the TimesFM project and is licensed under the Apache License 2.0.