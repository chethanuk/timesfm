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

"""
End-to-end integration tests for TimesFM API.
Tests real API functionality with actual data and no mocking.
"""

import json
import time
from typing import Dict, List, Any
import pytest
import requests
import numpy as np


class TestTimesFMEndpoint:
    """Test TimesFM API endpoints with real data."""

    BASE_URL = "http://localhost:8000"

    @classmethod
    def setup_class(cls):
        """Setup test class - verify API is running."""
        try:
            response = requests.get(f"{cls.BASE_URL}/health", timeout=10)
            assert response.status_code == 200
            cls.api_available = True
        except requests.exceptions.RequestException:
            cls.api_available = False
            pytest.skip("TimesFM API not available at http://localhost:8000")

    @pytest.mark.parametrize("test_case", [
        {
            "name": "minimal_data",
            "data": [100.0, 102.0, 99.0, 103.0, 105.0],
            "horizon": 3,
            "quantiles": [0.5],
            "expected_forecast_length": 3,
            "expected_quantiles": ["0.5"]
        },
        {
            "name": "synthetic_trend_data",
            "data": [100 + i * 0.5 + np.sin(i * 0.1) * 2 for i in range(50)],
            "horizon": 10,
            "quantiles": [0.1, 0.5, 0.9],
            "expected_forecast_length": 10,
            "expected_quantiles": ["0.1", "0.5", "0.9"]
        },
        {
            "name": "seasonal_pattern",
            "data": [100 + 10 * np.sin(i * 0.2) + i * 0.1 for i in range(100)],
            "horizon": 20,
            "quantiles": [0.25, 0.5, 0.75],
            "expected_forecast_length": 20,
            "expected_quantiles": ["0.25", "0.5", "0.75"]
        },
        {
            "name": "real_world_stock_like",
            "data": [150, 152, 148, 155, 153, 158, 156, 160, 162, 159, 163, 165, 161, 167, 169, 166, 170, 172, 168, 174],
            "horizon": 8,
            "quantiles": [0.1, 0.5, 0.9],
            "expected_forecast_length": 8,
            "expected_quantiles": ["0.1", "0.5", "0.9"]
        },
        {
            "name": "long_series_data",
            "data": [100 + np.random.normal(0, 5) for _ in range(200)],
            "horizon": 32,
            "quantiles": [0.05, 0.5, 0.95],
            "expected_forecast_length": 32,
            "expected_quantiles": ["0.05", "0.5", "0.95"]
        }
    ])
    def test_forecast_endpoint_varied_data(self, test_case):
        """Test forecast endpoint with different data patterns and configurations."""

        # Prepare request
        payload = {
            "data": test_case["data"],
            "horizon": test_case["horizon"],
            "quantiles": test_case["quantiles"]
        }

        # Make request
        response = requests.post(
            f"{self.BASE_URL}/forecast",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        # Verify response
        assert response.status_code == 200, f"API request failed: {response.text}"

        result = response.json()

        # Verify response structure
        assert "request_id" in result
        assert "forecasts" in result
        assert "quantiles" in result
        assert "metadata" in result

        # Verify forecast length
        forecasts = result["forecasts"]
        assert len(forecasts) == test_case["expected_forecast_length"], \
            f"Expected {test_case['expected_forecast_length']} forecasts, got {len(forecasts)}"

        # Verify all forecasts are numbers
        for i, forecast in enumerate(forecasts):
            assert isinstance(forecast, (int, float)), f"Forecast {i} is not a number: {forecast}"
            assert not np.isnan(forecast), f"Forecast {i} is NaN"
            assert not np.isinf(forecast), f"Forecast {i} is infinite"

        # Verify quantiles structure
        quantiles_result = result["quantiles"]
        for quantile_key in test_case["expected_quantiles"]:
            assert quantile_key in quantiles_result, f"Missing quantile {quantile_key}"
            quantile_values = quantiles_result[quantile_key]
            assert len(quantile_values) == test_case["expected_forecast_length"], \
                f"Quantile {quantile_key} has wrong length: {len(quantile_values)}"

            # Verify quantile values are numbers
            for i, q_value in enumerate(quantile_values):
                assert isinstance(q_value, (int, float)), f"Quantile {quantile_key}[{i}] is not a number"
                assert not np.isnan(q_value), f"Quantile {quantile_key}[{i}] is NaN"
                assert not np.isinf(q_value), f"Quantile {quantile_key}[{i}] is infinite"

        # Verify metadata
        metadata = result["metadata"]
        assert metadata["cached"] is False  # First request should not be cached
        assert metadata["model_size"] == "200M"
        assert metadata["input_length"] == len(test_case["data"])
        assert metadata["horizon"] == test_case["horizon"]
        assert "inference_time" in metadata
        assert metadata["inference_time"] > 0

        # Print test info for debugging
        print(f"✓ Test case '{test_case['name']}' passed")
        print(f"  - Input length: {len(test_case['data'])}")
        print(f"  - Horizon: {test_case['horizon']}")
        print(f"  - Inference time: {metadata['inference_time']:.3f}s")

    def test_health_endpoint_detailed(self):
        """Test health endpoint with detailed verification."""

        response = requests.get(f"{self.BASE_URL}/health", timeout=10)
        assert response.status_code == 200

        health = response.json()

        # Verify health response structure
        required_fields = ["status", "model_loaded", "memory_usage", "gpu_available", "uptime"]
        for field in required_fields:
            assert field in health, f"Missing field in health response: {field}"

        # Verify health values
        assert health["status"] == "healthy"
        assert health["model_loaded"] is True
        assert health["gpu_available"] is True
        assert health["uptime"] >= 0

        # Verify memory usage
        memory = health["memory_usage"]
        assert "used" in memory
        assert "total" in memory
        assert "percent" in memory
        assert memory["used"] > 0
        assert memory["total"] > 0
        assert 0 <= memory["percent"] <= 100

    def test_model_info_endpoint(self):
        """Test model info endpoint."""

        response = requests.get(f"{self.BASE_URL}/model/info", timeout=10)
        assert response.status_code == 200

        info = response.json()

        # Verify model info structure
        required_fields = ["model_size", "loaded", "load_time", "device"]
        for field in required_fields:
            assert field in info, f"Missing field in model info: {field}"

        # Verify model info values
        assert info["model_size"] == "200M"
        assert info["loaded"] is True
        assert info["device"] in ["cuda", "cpu"]
        assert info["load_time"] > 0

    def test_forecast_caching_behavior(self):
        """Test that repeated requests with same data are cached."""

        # Prepare request data
        test_data = [100, 102, 99, 103, 105, 108, 106, 109, 111, 108]
        payload = {
            "data": test_data,
            "horizon": 5,
            "quantiles": [0.5]
        }

        # Make first request
        start_time = time.time()
        response1 = requests.post(
            f"{self.BASE_URL}/forecast",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        first_request_time = time.time() - start_time

        assert response1.status_code == 200
        result1 = response1.json()

        # Make second request with same data
        start_time = time.time()
        response2 = requests.post(
            f"{self.BASE_URL}/forecast",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        second_request_time = time.time() - start_time

        assert response2.status_code == 200
        result2 = response2.json()

        # Verify results are identical
        assert result1["forecasts"] == result2["forecasts"]
        assert result1["quantiles"] == result2["quantiles"]

        # Verify caching behavior (second request should be cached)
        assert result1["metadata"]["cached"] is False
        assert result2["metadata"]["cached"] is True

        # Verify second request is faster (due to caching)
        assert second_request_time < first_request_time, \
            f"Cached request ({second_request_time:.3f}s) should be faster than first ({first_request_time:.3f}s)"

        print(f"✓ Caching test passed")
        print(f"  - First request: {first_request_time:.3f}s (uncached)")
        print(f"  - Second request: {second_request_time:.3f}s (cached)")

    @pytest.mark.parametrize("invalid_request", [
        {
            "name": "empty_data",
            "payload": {"data": [], "horizon": 5, "quantiles": [0.5]},
            "expected_error": "Data validation failed"
        },
        {
            "name": "negative_horizon",
            "payload": {"data": [1, 2, 3], "horizon": -1, "quantiles": [0.5]},
            "expected_error": "Invalid horizon"
        },
        {
            "name": "invalid_quantiles",
            "payload": {"data": [1, 2, 3], "horizon": 5, "quantiles": [1.5]},
            "expected_error": "Invalid quantile"
        },
        {
            "name": "zero_horizon",
            "payload": {"data": [1, 2, 3], "horizon": 0, "quantiles": [0.5]},
            "expected_error": "Invalid horizon"
        }
    ])
    def test_forecast_error_handling(self, invalid_request):
        """Test forecast endpoint error handling with invalid requests."""

        response = requests.post(
            f"{self.BASE_URL}/forecast",
            json=invalid_request["payload"],
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        # Should return error status
        assert response.status_code in [400, 422, 500], \
            f"Expected error status for {invalid_request['name']}, got {response.status_code}"

        result = response.json()
        assert "detail" in result, f"Error response should have 'detail' field for {invalid_request['name']}"

        print(f"✓ Error handling test passed for '{invalid_request['name']}'")

    def test_concurrent_requests(self):
        """Test API handling of concurrent requests."""
        import threading
        import queue

        # Prepare test data
        test_data = [100 + i for i in range(20)]
        payload = {
            "data": test_data,
            "horizon": 5,
            "quantiles": [0.5]
        }

        results_queue = queue.Queue()
        errors = []

        def make_request(request_id):
            try:
                response = requests.post(
                    f"{self.BASE_URL}/forecast",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                if response.status_code == 200:
                    results_queue.put((request_id, response.json()))
                else:
                    errors.append(f"Request {request_id} failed with status {response.status_code}")
            except Exception as e:
                errors.append(f"Request {request_id} exception: {str(e)}")

        # Launch concurrent requests
        num_requests = 5
        threads = []
        for i in range(num_requests):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all requests to complete
        for thread in threads:
            thread.join(timeout=60)

        # Verify results
        assert not errors, f"Concurrent requests had errors: {errors}"
        assert results_queue.qsize() == num_requests, \
            f"Expected {num_requests} results, got {results_queue.qsize()}"

        # Collect and verify all results
        all_results = []
        while not results_queue.empty():
            all_results.append(results_queue.get()[1])

        # All results should have valid forecasts
        for i, result in enumerate(all_results):
            assert "forecasts" in result, f"Result {i} missing forecasts"
            assert len(result["forecasts"]) == 5, f"Result {i} has wrong forecast length"

        print(f"✓ Concurrent requests test passed - {num_requests} requests handled successfully")

    def test_api_performance_requirements(self):
        """Test that API meets reasonable performance requirements."""

        # Test with moderate data size
        test_data = [100 + np.random.normal(0, 5) for _ in range(100)]
        payload = {
            "data": test_data,
            "horizon": 20,
            "quantiles": [0.1, 0.5, 0.9]
        }

        # Make request and measure time
        start_time = time.time()
        response = requests.post(
            f"{self.BASE_URL}/forecast",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        request_time = time.time() - start_time

        assert response.status_code == 200
        result = response.json()

        # Performance requirements
        assert request_time < 10.0, f"Request took too long: {request_time:.3f}s"
        assert result["metadata"]["inference_time"] < 5.0, \
            f"Inference took too long: {result['metadata']['inference_time']:.3f}s"

        print(f"✓ Performance test passed")
        print(f"  - Total request time: {request_time:.3f}s")
        print(f"  - Inference time: {result['metadata']['inference_time']:.3f}s")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])