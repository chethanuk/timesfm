"""
Table-driven tests with real public datasets for TimesFM API
"""
import pytest
import requests
import json
import time
from pathlib import Path

# Load real datasets
DATASETS_PATH = Path(__file__).parent.parent.parent / "datasets" / "test_datasets.json"
with open(DATASETS_PATH) as f:
    TEST_DATASETS = json.load(f)

BASE_URL = "http://localhost:8000"

class TestTimesFMRealDatasets:
    """Test TimesFM API with real public datasets"""

    @pytest.fixture
    def api_client(self):
        """Create API client for testing"""
        return requests.Session()

    @pytest.mark.parametrize("dataset_name,expected_behavior", [
        ("sunspots_full", "long_sequence_forecasting"),
        ("sunspots_recent", "medium_sequence_forecasting"),
        ("sunspots_short", "short_sequence_forecasting"),
        ("linear_trend", "trend_forecasting"),
        ("seasonal", "seasonal_pattern_forecasting"),
        ("random_walk", "stochastic_forecasting"),
        ("constant", "stable_forecasting"),
        ("exponential_growth", "growth_pattern_forecasting")
    ])
    def test_dataset_forecasting_capabilities(self, api_client, dataset_name, expected_behavior):
        """Test forecasting capabilities across different dataset types"""
        dataset = TEST_DATASETS[dataset_name]

        # Test different horizon lengths
        for horizon in [1, 5, 10, 20]:
            with api_client.post(f"{BASE_URL}/forecast", json={
                "data": dataset,
                "horizon": horizon,
                "quantiles": [0.1, 0.5, 0.9]
            }) as response:

                assert response.status_code == 200, f"Failed for {dataset_name} with horizon {horizon}"

                data = response.json()
                assert "forecast" in data
                assert "quantiles" in data
                assert len(data["forecast"]) == horizon

                # Validate quantile structure
                if "quantiles" in data and data["quantiles"]:
                    for quantile, values in data["quantiles"].items():
                        assert len(values) == horizon
                        assert all(isinstance(v, (int, float)) for v in values)

    @pytest.mark.parametrize("dataset_name", ["sunspots_recent", "seasonal", "random_walk"])
    def test_dataset_caching_behavior(self, api_client, dataset_name):
        """Test caching behavior with real datasets"""
        dataset = TEST_DATASETS[dataset_name]

        # First request
        start_time = time.time()
        response1 = api_client.post(f"{BASE_URL}/forecast", json={
            "data": dataset,
            "horizon": 5
        })
        first_duration = time.time() - start_time

        # Second request (should be cached)
        start_time = time.time()
        response2 = api_client.post(f"{BASE_URL}/forecast", json={
            "data": dataset,
            "horizon": 5
        })
        second_duration = time.time() - start_time

        assert response1.status_code == 200
        assert response2.status_code == 200

        # Cached response should be faster
        assert second_duration < first_duration, "Caching should improve performance"

    @pytest.mark.parametrize("dataset_name,horizon,expected_performance", [
        ("sunspots_tiny", 1, "< 0.5s"),
        ("sunspots_short", 5, "< 1.0s"),
        ("sunspots_recent", 10, "< 2.0s"),
        ("sunspots_full", 20, "< 5.0s")
    ])
    def test_dataset_performance_requirements(self, api_client, dataset_name, horizon, expected_performance):
        """Test performance requirements for different dataset sizes"""
        dataset = TEST_DATASETS[dataset_name]
        max_time = float(expected_performance.replace("< ", "").replace("s", ""))

        start_time = time.time()
        response = api_client.post(f"{BASE_URL}/forecast", json={
            "data": dataset,
            "horizon": horizon,
            "quantiles": [0.5]
        })
        duration = time.time() - start_time

        assert response.status_code == 200
        assert duration < max_time, f"Performance requirement failed: {duration:.3f}s > {max_time}s"

    @pytest.mark.parametrize("dataset_name,data_characteristics", [
        ("sunspots_full", {"length": 2820, "has_missing": False, "pattern": "seasonal"}),
        ("linear_trend", {"length": 100, "has_missing": False, "pattern": "trend"}),
        ("seasonal", {"length": 100, "has_missing": False, "pattern": "seasonal"}),
        ("random_walk", {"length": 100, "has_missing": False, "pattern": "stochastic"})
    ])
    def test_dataset_characteristics_handling(self, api_client, dataset_name, data_characteristics):
        """Test handling of different data characteristics"""
        dataset = TEST_DATASETS[dataset_name]

        response = api_client.post(f"{BASE_URL}/forecast", json={
            "data": dataset,
            "horizon": 5,
            "quantiles": [0.25, 0.5, 0.75]
        })

        assert response.status_code == 200
        data = response.json()

        # Validate forecast characteristics based on data pattern
        forecast = data["forecast"]
        assert len(forecast) == 5
        assert all(isinstance(v, (int, float)) for v in forecast)

        # For seasonal data, check that forecast shows reasonable variation
        if data_characteristics["pattern"] == "seasonal":
            assert len(set(forecast)) > 1, "Seasonal data should produce varying forecasts"

    def test_dataset_error_handling(self, api_client):
        """Test error handling with edge cases from real datasets"""
        # Test with empty dataset
        response = api_client.post(f"{BASE_URL}/forecast", json={
            "data": [],
            "horizon": 5
        })
        assert response.status_code == 422

        # Test with insufficient data
        response = api_client.post(f"{BASE_URL}/forecast", json={
            "data": [1.0],
            "horizon": 10
        })
        assert response.status_code == 422

        # Test with invalid data types
        response = api_client.post(f"{BASE_URL}/forecast", json={
            "data": ["not_a_number"],
            "horizon": 5
        })
        assert response.status_code == 422

    @pytest.mark.parametrize("dataset_name", ["sunspots_recent", "seasonal"])
    def test_dataset_quantile_consistency(self, api_client, dataset_name):
        """Test quantile forecast consistency"""
        dataset = TEST_DATASETS[dataset_name]

        response = api_client.post(f"{BASE_URL}/forecast", json={
            "data": dataset,
            "horizon": 10,
            "quantiles": [0.1, 0.5, 0.9]
        })

        assert response.status_code == 200
        data = response.json()

        quantiles = data["quantiles"]
        q10 = quantiles["0.1"]
        q50 = quantiles["0.5"]
        q90 = quantiles["0.9"]

        # Check quantile ordering
        for i in range(len(q10)):
            assert q10[i] <= q50[i] <= q90[i], f"Quantile ordering violated at index {i}"

    def test_dataset_memory_efficiency(self, api_client):
        """Test memory efficiency with large datasets"""
        # Use the largest dataset
        large_dataset = TEST_DATASETS["sunspots_full"]

        # Make multiple requests to test memory management
        for i in range(5):
            response = api_client.post(f"{BASE_URL}/forecast", json={
                "data": large_dataset,
                "horizon": 10
            })
            assert response.status_code == 200

            # Add small delay to allow memory cleanup
            time.sleep(0.1)

    @pytest.mark.parametrize("dataset_name", ["linear_trend", "constant", "exponential_growth"])
    def test_dataset_pattern_recognition(self, api_client, dataset_name):
        """Test pattern recognition for different dataset types"""
        dataset = TEST_DATASETS[dataset_name]

        response = api_client.post(f"{BASE_URL}/forecast", json={
            "data": dataset,
            "horizon": 10,
            "quantiles": [0.5]
        })

        assert response.status_code == 200
        data = response.json()
        forecast = data["forecast"]

        # Validate forecast based on expected pattern
        if dataset_name == "linear_trend":
            # Should continue the trend
            assert forecast[-1] > forecast[0], "Linear trend should continue upward"
        elif dataset_name == "constant":
            # Should remain relatively stable
            variance = max(forecast) - min(forecast)
            assert variance < 10, "Constant data should produce stable forecasts"