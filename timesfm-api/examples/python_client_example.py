#!/usr/bin/env python3
"""
TimesFM API Python Client Example
Demonstrates how to use the TimesFM API from Python applications.
"""

import json
import time
from typing import List, Dict, Any
import requests
import numpy as np
import pandas as pd


class TimesFMClient:
    """A simple Python client for the TimesFM API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the TimesFM client.

        Args:
            base_url: Base URL of the TimesFM API
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def health_check(self) -> Dict[str, Any]:
        """Check API health and status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        response = self.session.get(f"{self.base_url}/model/info")
        response.raise_for_status()
        return response.json()

    def get_stats(self) -> Dict[str, Any]:
        """Get API statistics."""
        response = self.session.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()

    def forecast(
        self,
        data: List[float],
        horizon: int = 48,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        model_size: str = "200M"
    ) -> Dict[str, Any]:
        """
        Make a forecast request.

        Args:
            data: Time series data as a list of floats
            horizon: Number of steps to forecast
            quantiles: List of quantiles to predict
            model_size: Model size to use

        Returns:
            Forecast response containing predictions and metadata
        """
        payload = {
            "data": data,
            "horizon": horizon,
            "quantiles": quantiles,
            "model_size": model_size
        }

        response = self.session.post(
            f"{self.base_url}/forecast",
            json=payload
        )
        response.raise_for_status()
        return response.json()


def generate_synthetic_data(
    pattern: str = "trend_seasonal",
    n_points: int = 100,
    noise_level: float = 0.1
) -> List[float]:
    """
    Generate synthetic time series data for testing.

    Args:
        pattern: Type of pattern to generate
        n_points: Number of data points to generate
        noise_level: Amount of random noise to add

    Returns:
        List of synthetic time series values
    """
    np.random.seed(42)  # For reproducible results

    if pattern == "trend_seasonal":
        # Trend + seasonal pattern
        trend = np.linspace(100, 200, n_points)
        seasonal = 10 * np.sin(np.linspace(0, 4 * np.pi, n_points))
        noise = np.random.normal(0, noise_level * 10, n_points)
        data = trend + seasonal + noise

    elif pattern == "stock_like":
        # Simulated stock price movement
        data = [100.0]
        for i in range(1, n_points):
            change = np.random.normal(0.001, 0.02)  # Daily return
            data.append(max(data[-1] * (1 + change), 1.0))  # Prevent negative prices

    elif pattern == "seasonal_only":
        # Pure seasonal pattern
        data = 100 + 20 * np.sin(np.linspace(0, 8 * np.pi, n_points))
        data += np.random.normal(0, noise_level * 5, n_points)

    else:  # random_walk
        # Random walk
        data = [100.0]
        for i in range(1, n_points):
            step = np.random.normal(0, 1)
            data.append(max(data[-1] + step, 1.0))

    return [float(x) for x in data]


def demonstrate_basic_usage(client: TimesFMClient):
    """Demonstrate basic usage of the TimesFM client."""
    print("🎯 Basic Usage Examples")
    print("=" * 50)

    # Check API health
    print("1. Checking API health...")
    health = client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Model loaded: {health['model_loaded']}")
    print(f"   GPU available: {health['gpu_available']}")
    print()

    # Get model info
    print("2. Model information...")
    model_info = client.get_model_info()
    print(f"   Model size: {model_info['model_size']}")
    print(f"   Device: {model_info['device']}")
    print(f"   Load time: {model_info['load_time']:.2f}s")
    print()


def demonstrate_forecasting(client: TimesFMClient):
    """Demonstrate different forecasting scenarios."""
    print("📊 Forecasting Examples")
    print("=" * 50)

    scenarios = [
        {
            "name": "Simple trend data",
            "data": [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110],
            "horizon": 6,
            "quantiles": [0.1, 0.5, 0.9]
        },
        {
            "name": "Stock price simulation",
            "data": generate_synthetic_data("stock_like", 50),
            "horizon": 10,
            "quantiles": [0.05, 0.5, 0.95]
        },
        {
            "name": "Seasonal pattern",
            "data": generate_synthetic_data("seasonal_only", 60),
            "horizon": 20,
            "quantiles": [0.25, 0.5, 0.75]
        },
        {
            "name": "Long-term trend with seasonality",
            "data": generate_synthetic_data("trend_seasonal", 150),
            "horizon": 30,
            "quantiles": [0.1, 0.5, 0.9]
        }
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['name']}")
        print(f"   Data points: {len(scenario['data'])}")
        print(f"   Horizon: {scenario['horizon']}")
        print(f"   Quantiles: {scenario['quantiles']}")

        start_time = time.time()
        try:
            result = client.forecast(
                data=scenario['data'],
                horizon=scenario['horizon'],
                quantiles=scenario['quantiles']
            )
            request_time = time.time() - start_time

            # Display results
            forecasts = result['forecasts']
            metadata = result['metadata']

            print(f"   ✅ Forecast successful!")
            print(f"   📈 Forecast (first 5 values): {forecasts[:5]}")
            print(f"   ⏱️  Request time: {request_time:.3f}s")
            print(f"   🧠 Inference time: {metadata['inference_time']:.3f}s")
            print(f"   💾 Cached: {metadata['cached']}")

            # Show quantile ranges for first prediction
            if '0.1' in result['quantiles'] and '0.9' in result['quantiles']:
                q10 = result['quantiles']['0.1'][0]
                q50 = result['quantiles']['0.5'][0]
                q90 = result['quantiles']['0.9'][0]
                print(f"   📊 90% confidence interval for step 1: [{q10:.2f}, {q90:.2f}] (median: {q50:.2f})")

        except Exception as e:
            print(f"   ❌ Forecast failed: {e}")

        print()


def demonstrate_caching_behavior(client: TimesFMClient):
    """Demonstrate caching behavior of the API."""
    print("🚀 Caching Behavior Demo")
    print("=" * 50)

    # Use the same data for multiple requests
    test_data = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]

    print("Making first request (uncached)...")
    start_time = time.time()
    result1 = client.forecast(data=test_data, horizon=5, quantiles=[0.5])
    first_time = time.time() - start_time

    print(f"   Request time: {first_time:.3f}s")
    print(f"   Cached: {result1['metadata']['cached']}")
    print()

    print("Making second request (same data, should be cached)...")
    start_time = time.time()
    result2 = client.forecast(data=test_data, horizon=5, quantiles=[0.5])
    second_time = time.time() - start_time

    print(f"   Request time: {second_time:.3f}s")
    print(f"   Cached: {result2['metadata']['cached']}")
    print()

    # Verify results are identical
    forecasts_match = result1['forecasts'] == result2['forecasts']
    print(f"   Results identical: {forecasts_match}")
    print(f"   Speedup: {first_time/second_time:.1f}x faster")
    print()


def demonstrate_performance_testing(client: TimesFMClient):
    """Demonstrate performance testing with different data sizes."""
    print("⚡ Performance Testing")
    print("=" * 50)

    test_sizes = [50, 100, 200, 500]
    horizon = 20

    print(f"Testing performance with horizon={horizon}...")
    print()

    for size in test_sizes:
        print(f"Data size: {size} points")

        # Generate test data
        test_data = generate_synthetic_data("trend_seasonal", size)

        # Make request and measure time
        start_time = time.time()
        try:
            result = client.forecast(
                data=test_data,
                horizon=horizon,
                quantiles=[0.1, 0.5, 0.9]
            )
            total_time = time.time() - start_time

            # Calculate performance metrics
            inference_time = result['metadata']['inference_time']
            throughput = len(test_data) / total_time if total_time > 0 else 0

            print(f"   ✅ Total time: {total_time:.3f}s")
            print(f"   🧠 Inference time: {inference_time:.3f}s")
            print(f"   📊 Throughput: {throughput:.1f} points/sec")
            print(f"   💾 Cached: {result['metadata']['cached']}")

        except Exception as e:
            print(f"   ❌ Request failed: {e}")

        print()


def demonstrate_error_handling(client: TimesFMClient):
    """Demonstrate error handling."""
    print("⚠️  Error Handling Examples")
    print("=" * 50)

    error_cases = [
        {
            "name": "Empty data",
            "data": [],
            "horizon": 5
        },
        {
            "name": "Negative horizon",
            "data": [1, 2, 3],
            "horizon": -1
        },
        {
            "name": "Zero horizon",
            "data": [1, 2, 3],
            "horizon": 0
        }
    ]

    for case in error_cases:
        print(f"Testing: {case['name']}")
        try:
            result = client.forecast(
                data=case['data'],
                horizon=case['horizon']
            )
            print(f"   ⚠️  Unexpected success: {result.get('detail', 'No error')}")
        except Exception as e:
            print(f"   ✅ Correctly handled error: {e}")
        print()


def create_forecast_report(client: TimesFMClient, data: List[float], title: str = "Time Series Forecast"):
    """
    Create a detailed forecast report.

    Args:
        client: TimesFM client instance
        data: Time series data to forecast
        title: Title for the report
    """
    print(f"📋 {title}")
    print("=" * len(title))
    print()

    # Basic statistics
    print("📊 Input Data Statistics:")
    print(f"   Length: {len(data)} points")
    print(f"   Min: {min(data):.2f}")
    print(f"   Max: {max(data):.2f}")
    print(f"   Mean: {np.mean(data):.2f}")
    print(f"   Std: {np.std(data):.2f}")
    print()

    # Make forecast
    print("🔮 Generating forecast...")
    start_time = time.time()
    result = client.forecast(
        data=data,
        horizon=20,
        quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
    request_time = time.time() - start_time

    # Display forecast results
    forecasts = result['forecasts']
    quantiles = result['quantiles']
    metadata = result['metadata']

    print(f"   ✅ Forecast completed in {request_time:.3f}s")
    print(f"   🧠 Inference time: {metadata['inference_time']:.3f}s")
    print()

    # Forecast summary
    print("📈 Forecast Summary (next 5 steps):")
    print("   Step | 5%ile | 25%ile | Median | 75%ile | 95%ile")
    print("   -----|--------|--------|--------|--------|--------")

    for i in range(min(5, len(forecasts))):
        q5 = quantiles['0.05'][i]
        q25 = quantiles['0.25'][i]
        q50 = quantiles['0.5'][i]
        q75 = quantiles['0.75'][i]
        q95 = quantiles['0.95'][i]
        print(f"   {i+1:4d} | {q5:6.2f} | {q25:6.2f} | {q50:6.2f} | {q75:6.2f} | {q95:6.2f}")

    print()

    # Last value vs first forecast
    last_actual = data[-1]
    first_forecast = forecasts[0]
    change = first_forecast - last_actual
    change_pct = (change / last_actual) * 100

    print("🔄 Trend Analysis:")
    print(f"   Last actual value: {last_actual:.2f}")
    print(f"   First forecast: {first_forecast:.2f}")
    print(f"   Expected change: {change:+.2f} ({change_pct:+.1f}%)")
    print()


def main():
    """Main function to run all examples."""
    print("🚀 TimesFM API Python Client Examples")
    print("=" * 50)
    print()

    # Initialize client
    client = TimesFMClient()

    # Check if API is available
    try:
        health = client.health_check()
        if health['status'] != 'healthy':
            print("❌ API is not healthy. Please check the API server.")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to TimesFM API at http://localhost:8000")
        print("Please make sure the API is running:")
        print("  cd /workspace/timesfm/timesfm-api")
        print("  python3 app.py")
        return
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        return

    print("✅ Successfully connected to TimesFM API!")
    print()

    # Run all demonstrations
    demonstrate_basic_usage(client)
    demonstrate_forecasting(client)
    demonstrate_caching_behavior(client)
    demonstrate_performance_testing(client)
    demonstrate_error_handling(client)

    # Create a detailed forecast report
    sample_data = generate_synthetic_data("trend_seasonal", 100)
    create_forecast_report(client, sample_data, "Sample Forecast Report")

    print("🎉 All examples completed successfully!")
    print()
    print("Next steps:")
    print("1. Replace sample data with your own time series")
    print("2. Adjust horizon and quantiles based on your needs")
    print("3. Integrate the client into your applications")
    print("4. Monitor API performance and adjust accordingly")


if __name__ == "__main__":
    main()