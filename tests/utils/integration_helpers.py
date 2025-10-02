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

"""Integration test helpers for TimesFM API testing."""

import asyncio
import json
import sqlite3
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import aiosqlite
import pandas as pd
import redis
from aioresponses import aioresponses
from fastapi.testclient import TestClient

# Mock database and Redis for testing


class InMemoryDatabase:
    """In-memory SQLite database for testing."""

    def __init__(self, database_url: str = ":memory:"):
        """Initialize in-memory database.

        Args:
            database_url: Database URL (defaults to in-memory).
        """
        self.database_url = database_url
        self.connection = None

    def connect(self) -> sqlite3.Connection:
        """Connect to database."""
        self.connection = sqlite3.connect(
            self.database_url,
            check_same_thread=False,
            timeout=30
        )
        self.connection.row_factory = sqlite3.Row
        return self.connection

    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()

    def setup_tables(self) -> None:
        """Setup test database tables."""
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                api_key TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Forecast requests table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS forecast_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                input_shape TEXT NOT NULL,
                horizon INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                result TEXT,
                error_message TEXT,
                processing_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        # Model metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                test_data_shape TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Usage statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                request_count INTEGER DEFAULT 1,
                total_processing_time REAL DEFAULT 0,
                last_request TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                UNIQUE(user_id, model_name)
            )
        """)

        # Cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS forecast_cache (
                cache_key TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                result TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        """)

        self.connection.commit()

    def insert_test_user(
        self,
        username: str = "testuser",
        email: str = "test@example.com",
        api_key: str = "test_api_key_12345"
    ) -> int:
        """Insert a test user.

        Args:
            username: Username.
            email: Email address.
            api_key: API key.

        Returns:
            User ID.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO users (username, email, api_key) VALUES (?, ?, ?)",
            (username, email, api_key)
        )
        self.connection.commit()
        return cursor.lastrowid

    def insert_forecast_request(
        self,
        user_id: int,
        model_name: str,
        input_shape: str,
        horizon: int,
        status: str = "pending",
        result: Optional[str] = None,
        processing_time: Optional[float] = None
    ) -> int:
        """Insert a forecast request.

        Args:
            user_id: User ID.
            model_name: Model name.
            input_shape: Input shape as string.
            horizon: Forecast horizon.
            status: Request status.
            result: Result data.
            processing_time: Processing time in seconds.

        Returns:
            Request ID.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO forecast_requests
            (user_id, model_name, input_shape, horizon, status, result, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, model_name, input_shape, horizon, status, result, processing_time)
        )
        self.connection.commit()
        return cursor.lastrowid

    def get_forecast_request(self, request_id: int) -> Optional[Dict[str, Any]]:
        """Get forecast request by ID.

        Args:
            request_id: Request ID.

        Returns:
            Request data or None if not found.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT * FROM forecast_requests WHERE id = ?",
            (request_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def update_forecast_request(
        self,
        request_id: int,
        status: str,
        result: Optional[str] = None,
        error_message: Optional[str] = None,
        processing_time: Optional[float] = None
    ) -> None:
        """Update forecast request.

        Args:
            request_id: Request ID.
            status: New status.
            result: Result data.
            error_message: Error message.
            processing_time: Processing time.
        """
        cursor = self.connection.cursor()
        update_fields = ["status = ?", "completed_at = CURRENT_TIMESTAMP"]
        update_values = [status]

        if result is not None:
            update_fields.append("result = ?")
            update_values.append(result)

        if error_message is not None:
            update_fields.append("error_message = ?")
            update_values.append(error_message)

        if processing_time is not None:
            update_fields.append("processing_time = ?")
            update_values.append(processing_time)

        update_values.append(request_id)

        cursor.execute(
            f"UPDATE forecast_requests SET {', '.join(update_fields)} WHERE id = ?",
            update_values
        )
        self.connection.commit()


class AsyncInMemoryDatabase:
    """Async in-memory database using aiosqlite."""

    def __init__(self, database_url: str = ":memory:"):
        """Initialize async database.

        Args:
            database_url: Database URL.
        """
        self.database_url = database_url
        self.connection = None

    async def connect(self) -> aiosqlite.Connection:
        """Connect to async database."""
        self.connection = await aiosqlite.connect(self.database_url)
        await self.connection.execute("PRAGMA foreign_keys = ON")
        return self.connection

    async def close(self) -> None:
        """Close database connection."""
        if self.connection:
            await self.connection.close()

    async def setup_tables(self) -> None:
        """Setup async database tables."""
        if not self.connection:
            await self.connect()

        # Use same table definitions as synchronous version
        db = InMemoryDatabase()
        db.connect()
        db.setup_tables()
        db.close()

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Get database connection context manager."""
        if not self.connection:
            await self.connect()
        yield self.connection


class MockRedis:
    """Mock Redis client for testing."""

    def __init__(self):
        """Initialize mock Redis."""
        self._data = {}
        self._expires = {}

    def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        if key in self._expires and time.time() > self._expires[key]:
            self.delete(key)
            return None
        return self._data.get(key)

    def set(
        self,
        key: str,
        value: str,
        ex: Optional[int] = None,
        px: Optional[int] = None
    ) -> bool:
        """Set key-value pair."""
        self._data[key] = value
        if ex:
            self._expires[key] = time.time() + ex
        elif px:
            self._expires[key] = time.time() + px / 1000
        return True

    def delete(self, key: str) -> int:
        """Delete key."""
        deleted = 0
        if key in self._data:
            del self._data[key]
            deleted += 1
        if key in self._expires:
            del self._expires[key]
        return deleted

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        if key in self._expires and time.time() > self._expires[key]:
            self.delete(key)
            return False
        return key in self._data

    def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        # Simple pattern matching (only * supported)
        if pattern == "*":
            return list(self._data.keys())
        else:
            prefix = pattern.replace("*", "")
            return [k for k in self._data.keys() if k.startswith(prefix)]

    def flushall(self) -> bool:
        """Clear all data."""
        self._data.clear()
        self._expires.clear()
        return True


class APITestClient:
    """Test client for TimesFM API."""

    def __init__(
        self,
        app: Any,
        db: InMemoryDatabase,
        redis_client: MockRedis,
        base_url: str = "http://testserver"
    ):
        """Initialize API test client.

        Args:
            app: FastAPI application.
            db: Database instance.
            redis_client: Redis client.
            base_url: Base URL for API.
        """
        self.app = app
        self.db = db
        self.redis_client = redis_client
        self.base_url = base_url
        self.client = None

    def setup_client(self) -> TestClient:
        """Setup FastAPI test client."""
        # Override dependencies
        from fastapi import FastAPI
        if hasattr(self.app, 'dependency_overrides'):
            self.app.dependency_overrides.clear()

        # Mock database dependency
        def mock_get_db():
            try:
                yield self.db.connection
            finally:
                pass

        # Mock Redis dependency
        def mock_get_redis():
            return self.redis_client

        # Apply overrides (assuming these are the dependency names)
        self.app.dependency_overrides["get_db"] = mock_get_db
        self.app.dependency_overrides["get_redis"] = mock_get_redis

        self.client = TestClient(self.app)
        return self.client

    def create_test_user_session(
        self,
        username: str = "testuser",
        api_key: str = "test_api_key_12345"
    ) -> Dict[str, str]:
        """Create test user session headers.

        Args:
            username: Username.
            api_key: API key.

        Returns:
            Headers dictionary.
        """
        return {
            "X-API-Key": api_key,
            "Content-Type": "application/json",
        }

    def post_forecast(
        self,
        data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ) -> Any:
        """Make POST request to forecast endpoint.

        Args:
            data: Request data.
            headers: Request headers.

        Returns:
            Response object.
        """
        if not self.client:
            self.setup_client()

        url = f"{self.base_url}/api/v1/forecast"
        return self.client.post(url, json=data, headers=headers)

    def get_forecast(
        self,
        request_id: str,
        headers: Optional[Dict[str, str]] = None
    ) -> Any:
        """Make GET request to forecast endpoint.

        Args:
            request_id: Forecast request ID.
            headers: Request headers.

        Returns:
            Response object.
        """
        if not self.client:
            self.setup_client()

        url = f"{self.base_url}/api/v1/forecast/{request_id}"
        return self.client.get(url, headers=headers)

    def get_model_info(
        self,
        model_name: str,
        headers: Optional[Dict[str, str]] = None
    ) -> Any:
        """Get model information.

        Args:
            model_name: Model name.
            headers: Request headers.

        Returns:
            Response object.
        """
        if not self.client:
            self.setup_client()

        url = f"{self.base_url}/api/v1/models/{model_name}"
        return self.client.get(url, headers=headers)


class AsyncAPITestClient:
    """Async API test client with mocked HTTP requests."""

    def __init__(self, base_url: str = "http://testserver"):
        """Initialize async API test client.

        Args:
            base_url: Base URL for API.
        """
        self.base_url = base_url
        self.mock_session = None

    @asynccontextmanager
    async def get_mock_session(self) -> AsyncGenerator[aioresponses, None]:
        """Get mocked HTTP session.

        Yields:
            aioresponses instance.
        """
        with aioresponses() as mocked:
            yield mocked

    async def mock_forecast_response(
        self,
        mock_session: aioresponses,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any],
        status_code: int = 200
    ) -> None:
        """Mock forecast API response.

        Args:
            mock_session: aioresponses instance.
            request_data: Expected request data.
            response_data: Response data to return.
            status_code: HTTP status code.
        """
        mock_session.post(
            f"{self.base_url}/api/v1/forecast",
            payload=response_data,
            status=status_code
        )

    async def mock_model_info_response(
        self,
        mock_session: aioresponses,
        model_name: str,
        response_data: Dict[str, Any],
        status_code: int = 200
    ) -> None:
        """Mock model info API response.

        Args:
            mock_session: aioresponses instance.
            model_name: Model name.
            response_data: Response data to return.
            status_code: HTTP status code.
        """
        mock_session.get(
            f"{self.base_url}/api/v1/models/{model_name}",
            payload=response_data,
            status=status_code
        )


class TestDataManager:
    """Manager for test data during integration tests."""

    def __init__(self, temp_dir: Optional[Path] = None):
        """Initialize test data manager.

        Args:
            temp_dir: Temporary directory for test data.
        """
        if temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp())
        else:
            self.temp_dir = temp_dir

        self.data_files = {}
        self.cached_results = {}

    def create_test_model_file(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        model_weights: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Create test model file.

        Args:
            model_name: Name of model.
            model_config: Model configuration.
            model_weights: Model weights.

        Returns:
            Path to model file.
        """
        model_dir = self.temp_dir / model_name
        model_dir.mkdir(exist_ok=True)

        # Save config
        config_file = model_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(model_config, f, indent=2)

        # Save weights if provided
        if model_weights:
            weights_file = model_dir / "model.safetensors"
            # Convert to serializable format
            serializable_weights = {}
            for key, value in model_weights.items():
                if hasattr(value, 'numpy'):
                    serializable_weights[key] = value.numpy().tolist()
                elif hasattr(value, 'tolist'):
                    serializable_weights[key] = value.tolist()
                else:
                    serializable_weights[key] = value

            with open(weights_file, "w") as f:
                json.dump(serializable_weights, f, indent=2)

        self.data_files[model_name] = model_dir
        return model_dir

    def create_test_dataset_file(
        self,
        dataset_name: str,
        data: Union[pd.DataFrame, Dict[str, Any]],
        format: str = "csv"
    ) -> Path:
        """Create test dataset file.

        Args:
            dataset_name: Name of dataset.
            data: Dataset data.
            format: File format ('csv', 'json', 'parquet').

        Returns:
            Path to dataset file.
        """
        file_path = self.temp_dir / f"{dataset_name}.{format}"

        if format == "csv":
            if isinstance(data, dict):
                data = pd.DataFrame(data)
            data.to_csv(file_path, index=False)
        elif format == "json":
            if isinstance(data, pd.DataFrame):
                data = data.to_dict()
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
        elif format == "parquet":
            if isinstance(data, dict):
                data = pd.DataFrame(data)
            data.to_parquet(file_path, index=False)

        self.data_files[dataset_name] = file_path
        return file_path

    def cache_forecast_result(
        self,
        cache_key: str,
        result: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> None:
        """Cache forecast result.

        Args:
            cache_key: Cache key.
            result: Result to cache.
            ttl: Time to live in seconds.
        """
        self.cached_results[cache_key] = {
            "result": result,
            "cached_at": time.time(),
            "ttl": ttl,
        }

    def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result.

        Args:
            cache_key: Cache key.

        Returns:
            Cached result or None if expired/not found.
        """
        if cache_key not in self.cached_results:
            return None

        cached_item = self.cached_results[cache_key]
        if cached_item["ttl"] and time.time() - cached_item["cached_at"] > cached_item["ttl"]:
            del self.cached_results[cache_key]
            return None

        return cached_item["result"]

    def cleanup(self) -> None:
        """Clean up test data."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.data_files.clear()
        self.cached_results.clear()


# Factory functions for easy setup
def create_test_database() -> InMemoryDatabase:
    """Create and setup test database.

    Returns:
        Configured test database.
    """
    db = InMemoryDatabase()
    db.connect()
    db.setup_tables()
    return db


def create_test_redis() -> MockRedis:
    """Create test Redis client.

    Returns:
        Mock Redis client.
    """
    return MockRedis()


def create_api_test_client(app: Any) -> APITestClient:
    """Create API test client.

    Args:
        app: FastAPI application.

    Returns:
        Configured test client.
    """
    db = create_test_database()
    redis_client = create_test_redis()
    return APITestClient(app, db, redis_client)


def create_test_data_manager() -> TestDataManager:
    """Create test data manager.

    Returns:
        Test data manager instance.
    """
    return TestDataManager()