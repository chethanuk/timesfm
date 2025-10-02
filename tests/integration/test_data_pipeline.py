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

"""Integration tests for data pipeline components."""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
import torch

from tests.utils.data_generators import (
    AdvancedTimeSeriesGenerator,
    create_training_dataset,
    create_test_dataset,
)
from tests.utils.integration_helpers import (
    AsyncInMemoryDatabase,
    InMemoryDatabase,
    TestDataManager,
    create_test_database,
    create_test_redis,
)


class TestAdvancedTimeSeriesGenerator:
    """Test advanced time series data generation."""

    def test_generate_multivariate_time_series(self) -> None:
        """Test multivariate time series generation."""
        generator = AdvancedTimeSeriesGenerator(random_seed=42)

        df = generator.generate_multivariate_time_series(
            n_samples=100,
            n_features=3,
            freq="H",
            start_date="2024-01-01",
            patterns=["trend_seasonal", "cyclical", "random_walk"],
            missing_data_rate=0.05,
            outlier_rate=0.02
        )

        # Validate structure
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 100
        assert df.shape[1] == 3
        assert list(df.columns) == ["feature_0", "feature_1", "feature_2"]

        # Validate datetime index
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.freq_str == "H"

        # Validate missing data
        missing_count = df.isnull().sum().sum()
        total_values = df.shape[0] * df.shape[1]
        missing_rate = missing_count / total_values
        assert 0.03 <= missing_rate <= 0.07  # Allow some variance

        # Validate outliers (values far from normal range)
        outlier_count = ((np.abs(df) > 10).sum().sum())
        assert outlier_count > 0  # Should have some outliers

    def test_generate_forecasting_dataset(self) -> None:
        """Test forecasting dataset generation."""
        generator = AdvancedTimeSeriesGenerator(random_seed=42)

        dataset = generator.generate_forecasting_dataset(
            n_series=10,
            min_length=200,
            max_length=800,
            context_length=128,
            horizon=32,
            n_features=1,
            dataset_type="various"
        )

        # Validate dataset structure
        assert len(dataset) == 10

        for item in dataset:
            assert "context" in item
            assert "target" in item
            assert "metadata" in item

            # Validate tensor shapes
            assert item["context"].shape[1] == 128  # context_length
            assert item["target"].shape[1] == 32    # horizon
            assert item["context"].shape[2] == 1    # n_features
            assert item["target"].shape[2] == 1

            # Validate metadata
            metadata = item["metadata"]
            assert "series_id" in metadata
            assert "length" in metadata
            assert "pattern" in metadata
            assert 200 <= metadata["length"] <= 800

    def test_generate_anomaly_data(self) -> None:
        """Test anomaly data generation."""
        generator = AdvancedTimeSeriesGenerator(random_seed=42)

        data, labels, metadata = generator.generate_anomaly_data(
            n_samples=1000,
            anomaly_types=["spike", "dip", "drift", "level_shift"],
            anomaly_rate=0.02,
            anomaly_magnitude=3.0
        )

        # Validate data shapes
        assert data.shape == (1000,)
        assert labels.shape == (1000,)

        # Validate anomaly rate
        actual_anomaly_rate = np.sum(labels) / len(labels)
        assert 0.015 <= actual_anomaly_rate <= 0.025  # Allow some variance

        # Validate metadata
        assert "n_anomalies" in metadata
        assert "anomaly_rate" in metadata
        assert "anomaly_types" in metadata
        assert "anomaly_metadata" in metadata

        # Validate anomaly metadata
        anomaly_info = metadata["anomaly_metadata"]
        assert len(anomaly_info) == metadata["n_anomalies"]
        for anomaly in anomaly_info:
            assert "position" in anomaly
            assert "type" in anomaly
            assert "magnitude" in anomaly

    def test_create_forecasting_tensor(self) -> None:
        """Test forecasting tensor creation."""
        generator = AdvancedTimeSeriesGenerator(random_seed=42)

        # Generate sample data
        data = np.random.randn(1000, 2)  # 1000 time points, 2 features

        result = generator.create_forecasting_tensor(
            data=data,
            context_length=256,
            horizon=64,
            stride=128,
            n_features=2
        )

        # Validate tensor shapes
        assert "input" in result
        assert "target" in result
        assert result["input"].shape[1] == 256  # context_length
        assert result["target"].shape[1] == 64   # horizon
        assert result["input"].shape[2] == 2    # n_features
        assert result["target"].shape[2] == 2

        # Validate number of windows
        expected_windows = (1000 - 256 - 64) // 128 + 1
        assert result["n_windows"] == expected_windows
        assert result["input"].shape[0] == expected_windows
        assert result["target"].shape[0] == expected_windows

    def test_scale_data(self) -> None:
        """Test data scaling functionality."""
        generator = AdvancedTimeSeriesGenerator(random_seed=42)

        # Generate test data
        data = np.random.randn(100, 3) * 10 + 5  # Non-standard distribution

        # Test different scaling methods
        methods = ["standard", "minmax", "robust"]
        for method in methods:
            scaled_data = generator.scale_data(data, method=method)

            # Validate output shape
            assert scaled_data.shape == data.shape

            # Validate scaling effect
            if method == "standard":
                # Standard scaling should have mean ~0 and std ~1
                assert np.abs(np.mean(scaled_data, axis=0)) < 0.1
                assert np.all(np.abs(np.std(scaled_data, axis=0) - 1) < 0.1)
            elif method == "minmax":
                # Min-max scaling should be in [0, 1]
                assert np.all(scaled_data >= -0.01)  # Small tolerance
                assert np.all(scaled_data <= 1.01)

    def test_save_and_load_dataset(self) -> None:
        """Test dataset saving functionality."""
        generator = AdvancedTimeSeriesGenerator(random_seed=42)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Generate test dataset
            dataset = generator.generate_forecasting_dataset(
                n_series=5,
                context_length=64,
                horizon=16
            )

            # Test JSON format
            json_path = temp_path / "test_dataset.json"
            generator.save_dataset(dataset, json_path, format="json")
            assert json_path.exists()

            # Load and verify
            with open(json_path, "r") as f:
                loaded_data = json.load(f)
            assert len(loaded_data) == 5

            # Test CSV format (convert to DataFrame first)
            csv_data = []
            for item in dataset:
                context_flat = item["context"].numpy().flatten()
                target_flat = item["target"].numpy().flatten()
                csv_data.append({
                    "context": context_flat.tolist(),
                    "target": target_flat.tolist(),
                })

            df = pd.DataFrame(csv_data)
            csv_path = temp_path / "test_dataset.csv"
            df.to_csv(csv_path, index=False)
            assert csv_path.exists()


class TestIntegrationHelpers:
    """Test integration helper utilities."""

    def test_in_memory_database(self) -> None:
        """Test in-memory database functionality."""
        db = create_test_database()

        # Test table creation
        assert db.connection is not None

        # Test user insertion
        user_id = db.insert_test_user(
            username="testuser",
            email="test@example.com",
            api_key="test_key_123"
        )
        assert user_id > 0

        # Test forecast request insertion
        request_id = db.insert_forecast_request(
            user_id=user_id,
            model_name="timesfm-2.5-200m",
            input_shape="(512, 1)",
            horizon=96,
            status="pending"
        )
        assert request_id > 0

        # Test request retrieval
        request = db.get_forecast_request(request_id)
        assert request is not None
        assert request["model_name"] == "timesfm-2.5-200m"
        assert request["status"] == "pending"

        # Test request update
        db.update_forecast_request(
            request_id=request_id,
            status="completed",
            result='{"forecast": [1, 2, 3]}',
            processing_time=0.5
        )

        updated_request = db.get_forecast_request(request_id)
        assert updated_request["status"] == "completed"
        assert updated_request["processing_time"] == 0.5

        db.close()

    @pytest.mark.asyncio
    async def test_async_in_memory_database(self) -> None:
        """Test async in-memory database functionality."""
        async_db = AsyncInMemoryDatabase(":memory:")
        await async_db.connect()
        await async_db.setup_tables()

        # Test connection
        assert async_db.connection is not None

        # Test connection context manager
        async with async_db.get_connection() as conn:
            assert conn is not None
            # Execute a simple query
            cursor = await conn.execute("SELECT 1")
            result = await cursor.fetchone()
            assert result[0] == 1

        await async_db.close()

    def test_mock_redis(self) -> None:
        """Test mock Redis functionality."""
        redis_client = create_test_redis()

        # Test basic operations
        assert not redis_client.exists("test_key")

        # Test set and get
        assert redis_client.set("test_key", "test_value")
        assert redis_client.get("test_key") == "test_value"
        assert redis_client.exists("test_key")

        # Test expiration
        assert redis_client.set("expire_key", "expire_value", ex=1)
        assert redis_client.get("expire_key") == "expire_value"
        # Note: In real scenario, we'd wait for expiration

        # Test delete
        assert redis_client.delete("test_key") == 1
        assert not redis_client.exists("test_key")

        # Test keys with pattern
        redis_client.set("user:1", "value1")
        redis_client.set("user:2", "value2")
        redis_client.set("product:1", "value3")

        user_keys = redis_client.keys("user:*")
        assert len(user_keys) == 2
        assert "user:1" in user_keys
        assert "user:2" in user_keys

        # Test flush all
        assert redis_client.flushall()
        assert not redis_client.exists("user:1")
        assert not redis_client.exists("product:1")

    def test_test_data_manager(self) -> None:
        """Test test data manager functionality."""
        manager = TestDataManager()

        # Test model file creation
        model_config = {
            "model_name": "test-model",
            "hidden_size": 512,
            "num_layers": 8,
        }
        model_weights = {
            "layer1.weight": torch.randn(512, 256),
            "layer2.weight": torch.randn(256, 128),
        }

        model_dir = manager.create_test_model_file(
            model_name="test_model",
            model_config=model_config,
            model_weights=model_weights
        )

        assert model_dir.exists()
        assert (model_dir / "config.json").exists()
        assert (model_dir / "model.safetensors").exists()

        # Test dataset file creation
        dataset_data = {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "target": [1.1, 2.1, 3.1, 4.1, 5.1],
        }

        csv_path = manager.create_test_dataset_file(
            dataset_name="test_data",
            data=dataset_data,
            format="csv"
        )
        assert csv_path.exists()

        json_path = manager.create_test_dataset_file(
            dataset_name="test_data",
            data=dataset_data,
            format="json"
        )
        assert json_path.exists()

        # Test caching
        test_result = {"forecast": [1, 2, 3], "metadata": {"model": "test"}}
        manager.cache_forecast_result("test_cache_key", test_result, ttl=60)

        cached_result = manager.get_cached_result("test_cache_key")
        assert cached_result == test_result

        # Test cleanup
        manager.cleanup()
        assert not model_dir.exists()


class TestDataPipelineIntegration:
    """Integration tests for complete data pipeline."""

    def test_end_to_end_data_pipeline(self) -> None:
        """Test complete data pipeline from generation to storage."""
        # Initialize components
        generator = AdvancedTimeSeriesGenerator(random_seed=42)
        db = create_test_database()
        redis_client = create_test_redis()
        manager = TestDataManager()

        try:
            # 1. Generate training dataset
            dataset = generator.generate_forecasting_dataset(
                n_series=20,
                context_length=256,
                horizon=64,
                n_features=2,
                dataset_type="various"
            )
            assert len(dataset) == 20

            # 2. Save dataset to file
            dataset_file = manager.create_test_dataset_file(
                dataset_name="integration_test",
                data={
                    "context": [item["context"].numpy().tolist() for item in dataset],
                    "target": [item["target"].numpy().tolist() for item in dataset],
                    "metadata": [item["metadata"] for item in dataset],
                },
                format="json"
            )
            assert dataset_file.exists()

            # 3. Store metadata in database
            user_id = db.insert_test_user()
            request_id = db.insert_forecast_request(
                user_id=user_id,
                model_name="timesfm-2.5-200m",
                input_shape=f"({len(dataset)}, 256, 2)",
                horizon=64,
                status="data_ready"
            )
            assert request_id > 0

            # 4. Cache dataset info in Redis
            cache_key = f"dataset_{request_id}"
            redis_client.set(
                cache_key,
                json.dumps({
                    "file_path": str(dataset_file),
                    "n_series": len(dataset),
                    "context_length": 256,
                    "horizon": 64,
                    "n_features": 2,
                }),
                ex=3600  # 1 hour
            )
            assert redis_client.exists(cache_key)

            # 5. Retrieve and validate cached data
            cached_info = json.loads(redis_client.get(cache_key))
            assert cached_info["n_series"] == 20
            assert cached_info["context_length"] == 256

            # 6. Update request status
            db.update_forecast_request(
                request_id=request_id,
                status="processed",
                result='{"status": "success"}',
                processing_time=1.5
            )

            # 7. Verify final state
            final_request = db.get_forecast_request(request_id)
            assert final_request["status"] == "processed"
            assert final_request["processing_time"] == 1.5

        finally:
            # Cleanup
            db.close()
            manager.cleanup()

    def test_pipeline_with_different_data_types(self) -> None:
        """Test pipeline with different data types and patterns."""
        generator = AdvancedTimeSeriesGenerator(random_seed=42)

        # Test multivariate data
        multivariate_df = generator.generate_multivariate_time_series(
            n_samples=500,
            n_features=4,
            patterns=["trend_seasonal", "cyclical", "random_walk", "intermittent"],
            missing_data_rate=0.02,
            outlier_rate=0.01
        )

        # Convert to tensors
        data = multivariate_df.values
        tensor_result = generator.create_forecasting_tensor(
            data=data,
            context_length=128,
            horizon=32,
            stride=64,
            n_features=4
        )

        assert tensor_result["input"].shape[2] == 4
        assert tensor_result["target"].shape[2] == 4

        # Test anomaly data
        anomaly_data, anomaly_labels, anomaly_metadata = generator.generate_anomaly_data(
            n_samples=1000,
            anomaly_types=["spike", "dip", "drift", "level_shift", "outlier"],
            anomaly_rate=0.03
        )

        assert np.sum(anomaly_labels) > 0  # Should have anomalies
        assert len(anomaly_metadata["anomaly_metadata"]) > 0

        # Test scaling on anomaly data
        scaled_anomaly_data = generator.scale_data(anomaly_data, method="robust")
        assert scaled_anomaly_data.shape == anomaly_data.shape

    @pytest.mark.asyncio
    async def test_async_pipeline_operations(self) -> None:
        """Test asynchronous pipeline operations."""
        async_db = AsyncInMemoryDatabase(":memory:")
        await async_db.connect()
        await async_db.setup_tables()

        try:
            # Simulate async data processing
            async def process_data_batch(batch_id: int):
                """Simulate async data processing."""
                await asyncio.sleep(0.01)  # Simulate processing time
                return {"batch_id": batch_id, "status": "processed"}

            # Process multiple batches concurrently
            tasks = [
                process_data_batch(i)
                for i in range(10)
            ]
            results = await asyncio.gather(*tasks)

            # Validate results
            assert len(results) == 10
            for result in results:
                assert result["status"] == "processed"

        finally:
            await async_db.close()