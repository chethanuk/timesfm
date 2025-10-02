#!/usr/bin/env python3
"""
Example usage of the TimesFM test framework.

This script demonstrates how to use various components of the test framework
including mock models, data generation, and performance benchmarking.
"""

import time
from pathlib import Path

import torch

# Import test framework components
from tests.fixtures.mock_timesfm import create_mock_timesfm
from tests.performance.benchmark_framework import BenchmarkConfig, PerformanceBenchmark
from tests.utils.data_generators import create_training_dataset, AdvancedTimeSeriesGenerator


def main():
    """Demonstrate test framework usage."""
    print("🚀 TimesFM Test Framework Demo")
    print("=" * 50)

    # 1. Create and use mock models
    print("\n1. Mock Model Demo")
    print("-" * 20)

    # Create different model sizes
    models = {
        "50M": create_mock_timesfm(model_size="50m", device="cpu"),
        "200M": create_mock_timesfm(model_size="200m", device="cpu"),
    }

    for name, model in models.items():
        print(f"Loading {name} model...")
        model.load()

        # Generate sample data
        input_data = torch.randn(100, 1)
        start_time = time.time()

        # Make forecast
        result = model.forecast(
            input_data,
            horizon=48,
            quantiles=[0.1, 0.5, 0.9],
            num_samples=1
        )

        inference_time = time.time() - start_time

        print(f"  ✅ {name}: {result['forecasts'].shape} "
              f"in {inference_time:.3f}s")
        print(f"  📊 Quantiles: {list(result['quantiles'].keys())}")

        # Get model info
        info = model.get_model_info()
        print(f"  ℹ️  Parameters: {info['parameters']:,}")

    # 2. Data generation demo
    print("\n2. Data Generation Demo")
    print("-" * 25)

    # Create advanced data generator
    generator = AdvancedTimeSeriesGenerator(random_seed=42)

    # Generate multivariate time series
    print("Generating multivariate time series...")
    df = generator.generate_multivariate_time_series(
        n_samples=500,
        n_features=3,
        freq="H",
        patterns=["trend_seasonal", "cyclical", "random_walk"],
        missing_data_rate=0.02,
        outlier_rate=0.01
    )

    print(f"  ✅ Generated: {df.shape}")
    print(f"  📈 Features: {list(df.columns)}")
    print(f"  ⏰ Time range: {df.index[0]} to {df.index[-1]}")
    print(f"  🔍 Missing data: {df.isnull().sum().sum()} values")

    # Generate forecasting dataset
    print("\nGenerating forecasting dataset...")
    dataset = create_training_dataset(
        n_series=20,
        context_length=256,
        horizon=64,
        n_features=1,
        generator_seed=42
    )

    print(f"  ✅ Dataset size: {len(dataset)} series")
    print(f"  📊 Context shape: {dataset[0]['context'].shape}")
    print(f"  🎯 Target shape: {dataset[0]['target'].shape}")

    # 3. Performance benchmarking demo
    print("\n3. Performance Benchmarking Demo")
    print("-" * 35)

    # Configure benchmark
    config = BenchmarkConfig(
        warmup_iterations=2,
        test_iterations=5,
        batch_sizes=[1, 4],
        sequence_lengths=[128, 256],
        horizon_lengths=[24, 48],
        memory_threshold_mb=1024,
        time_threshold_seconds=30.0,
    )

    # Create benchmark instance
    benchmark = PerformanceBenchmark(config)

    # Use medium model for benchmarking
    model = models["200M"]

    # Single inference benchmark
    print("Running single inference benchmark...")
    input_data = torch.randn(1, 256, 1)
    result = benchmark.benchmark_model_inference(
        model,
        input_data,
        horizon=48
    )

    print(f"  ⏱️  Mean time: {result.metrics['mean_time']:.3f}s ± {result.metrics['std_time']:.3f}s")
    print(f"  🚀 Throughput: {result.metrics['throughput']:.1f} req/s")
    print(f"  💾 Memory delta: {result.memory_usage['rss_delta_mb']:.1f} MB")

    # Batch processing benchmark
    print("\nRunning batch processing benchmark...")
    batch_input = torch.randn(4, 256, 1)
    batch_result = benchmark.benchmark_batch_processing(
        model,
        batch_input,
        horizon=48
    )

    print(f"  📦 Batch size: 4")
    print(f"  ⏱️  Mean time: {batch_result.metrics['mean_time']:.3f}s")
    print(f"  🚀 Throughput: {batch_result.metrics['throughput']:.1f} req/s")
    print(f"  💾 Memory delta: {batch_result.memory_usage['rss_delta_mb']:.1f} MB")

    # Scalability test
    print("\nRunning scalability test...")
    def generate_test_data(batch_size, sequence_length, horizon_length):
        return {
            "input": torch.randn(batch_size, sequence_length, 1),
            "target": torch.randn(batch_size, horizon_length, 1),
        }

    scalability_results = benchmark.run_scalability_test(model, generate_test_data)
    print(f"  📊 Tested {len(scalability_results)} configurations")

    # Show some scalability results
    for i, result in enumerate(scalability_results[:3]):  # Show first 3
        print(f"    {i+1}. {result.test_name}: {result.metrics['mean_time']:.3f}s")

    # 4. Save benchmark results
    print("\n4. Saving Results")
    print("-" * 15)

    # Create output directory
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)

    # Save benchmark results
    benchmark.results.extend([result, batch_result] + scalability_results)
    benchmark.save_results(output_dir / "benchmark_results.json")

    # Generate performance report
    report_df = benchmark.generate_report(output_dir)
    print(f"  💾 Results saved to: {output_dir}")
    print(f"  📊 Performance report: {len(report_df)} test results")
    print(f"  📈 Average inference time: {report_df['mean_time'].mean():.3f}s")

    # 5. Model evaluation demo
    print("\n5. Model Evaluation Demo")
    print("-" * 22)

    # Generate test data for evaluation
    test_data = torch.randn(10, 256, 1)
    targets = torch.randn(10, 48, 1)

    # Evaluate model performance
    metrics = model.evaluate(test_data, targets, metrics=["mae", "mse", "mape", "rmse"])

    print("  📊 Model Performance Metrics:")
    for metric, value in metrics.items():
        print(f"    {metric.upper()}: {value:.4f}")

    # Get detailed model information
    info = model.get_model_info()
    print(f"\n  🔧 Model Configuration:")
    print(f"    Name: {info['model_name']}")
    print(f"    Hidden size: {info['hidden_size']}")
    print(f"    Layers: {info['num_layers']}")
    print(f"    Parameters: {info['parameters']:,}")
    print(f"    Device: {info['device']}")

    print("\n✅ Demo completed successfully!")
    print(f"📁 Check {output_dir} for detailed results.")

if __name__ == "__main__":
    main()