#!/usr/bin/env python3
"""
Simple demo of the TimesFM test framework focusing on core functionality.
"""

import numpy as np
import torch
import time

# Import core components
from tests.fixtures.mock_timesfm import create_mock_timesfm


def main():
    """Simple demonstration of the test framework."""
    print("🚀 TimesFM Test Framework - Simple Demo")
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

    # 2. Model evaluation demo
    print("\n2. Model Evaluation Demo")
    print("-" * 22)

    # Use medium model for evaluation
    model = models["200M"]

    # Generate test data for evaluation
    test_data = torch.randn(10, 256, 1)
    targets = torch.randn(10, 96, 1)  # Match the default horizon

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

    # 3. Basic performance testing
    print("\n3. Basic Performance Test")
    print("-" * 25)

    # Simple timing test
    input_data = torch.randn(1, 256, 1)
    n_iterations = 10

    times = []
    for i in range(n_iterations):
        start_time = time.time()
        result = model.forecast(input_data, horizon=48)
        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = 1.0 / avg_time

    print(f"  ⏱️  Average inference time: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"  🚀 Throughput: {throughput:.1f} req/s")
    print(f"  📈 Min time: {min(times):.3f}s, Max time: {max(times):.3f}s")

    # 4. Test with different input sizes
    print("\n4. Scalability Test")
    print("-" * 17)

    batch_sizes = [1, 4, 8]
    sequence_lengths = [64, 128, 256]

    for batch_size in batch_sizes:
        for seq_len in sequence_lengths:
            input_data = torch.randn(batch_size, seq_len, 1)
            start_time = time.time()
            result = model.forecast(input_data, horizon=48)
            end_time = time.time()

            time_per_sample = (end_time - start_time) / batch_size
            print(f"  📊 Batch {batch_size}, Seq {seq_len}: "
                  f"{time_per_sample:.3f}s per sample")

    print("\n✅ Demo completed successfully!")
    print("\n📝 To run the full test framework:")
    print("   pytest tests/unit/                    # Unit tests")
    print("   pytest tests/integration/             # Integration tests")
    print("   pytest tests/performance/             # Performance tests")
    print("   pytest --help                         # More options")


if __name__ == "__main__":
    main()