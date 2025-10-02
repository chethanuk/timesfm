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

"""Advanced test data generators for realistic time series data."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class AdvancedTimeSeriesGenerator:
    """Advanced time series data generator with realistic patterns."""

    def __init__(self, random_seed: int = 42):
        """Initialize generator with random seed."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.scalers = {}

    def generate_multivariate_time_series(
        self,
        n_samples: int = 1000,
        n_features: int = 3,
        freq: str = "H",
        start_date: str = "2024-01-01",
        patterns: Optional[List[str]] = None,
        correlation_matrix: Optional[NDArray] = None,
        noise_levels: Optional[List[float]] = None,
        missing_data_rate: float = 0.01,
        outlier_rate: float = 0.005,
        value_ranges: Optional[List[Tuple[float, float]]] = None
    ) -> pd.DataFrame:
        """Generate multivariate time series with realistic patterns.

        Args:
            n_samples: Number of time points.
            n_features: Number of features/variables.
            freq: Time frequency (e.g., 'H' for hourly, 'D' for daily).
            start_date: Start date for the time series.
            patterns: List of pattern types for each feature.
            correlation_matrix: Correlation matrix between features.
            noise_levels: Noise levels for each feature.
            missing_data_rate: Rate of missing data.
            outlier_rate: Rate of outliers.
            value_ranges: Min/max value ranges for each feature.

        Returns:
            DataFrame with multivariate time series data.
        """
        # Set default values
        if patterns is None:
            patterns = ["trend_seasonal", "cyclical", "random_walk"][:n_features]

        if correlation_matrix is None:
            correlation_matrix = np.eye(n_features)
            # Add some correlation between features
            if n_features > 1:
                correlation_matrix[0, 1] = 0.3
                correlation_matrix[1, 0] = 0.3

        if noise_levels is None:
            noise_levels = [0.1, 0.2, 0.05][:n_features]

        if value_ranges is None:
            value_ranges = [(-10, 10)] * n_features

        # Generate time index
        time_index = pd.date_range(start=start_date, periods=n_samples, freq=freq)

        # Generate base patterns
        data = np.zeros((n_samples, n_features))

        for i in range(n_features):
            pattern = patterns[i % len(patterns)]
            data[:, i] = self._generate_pattern(
                n_samples, pattern, noise_levels[i], value_ranges[i]
            )

        # Apply correlation
        data = data @ correlation_matrix

        # Add missing data
        missing_mask = np.random.random((n_samples, n_features)) < missing_data_rate
        data[missing_mask] = np.nan

        # Add outliers
        outlier_mask = np.random.random((n_samples, n_features)) < outlier_rate
        outlier_magnitudes = np.random.choice([-1, 1], size=(n_samples, n_features)) * 10
        data[outlier_mask] += outlier_magnitudes[outlier_mask]

        # Create DataFrame
        columns = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(data, index=time_index, columns=columns)

        return df

    def _generate_pattern(
        self,
        n_samples: int,
        pattern_type: str,
        noise_level: float,
        value_range: Tuple[float, float]
    ) -> NDArray[np.float64]:
        """Generate a specific time series pattern.

        Args:
            n_samples: Number of samples.
            pattern_type: Type of pattern.
            noise_level: Noise level.
            value_range: Value range.

        Returns:
            Generated pattern.
        """
        t = np.arange(n_samples, dtype=float)

        if pattern_type == "trend_seasonal":
            # Linear trend + seasonal pattern
            trend = 0.01 * t
            seasonal = 5 * np.sin(2 * np.pi * t / 24) + 3 * np.sin(2 * np.pi * t / (24 * 7))
            signal = trend + seasonal

        elif pattern_type == "cyclical":
            # Cyclical pattern with varying amplitude
            amplitude = 5 + 2 * np.sin(2 * np.pi * t / (24 * 30))
            signal = amplitude * np.sin(2 * np.pi * t / 24)

        elif pattern_type == "random_walk":
            # Random walk with drift
            noise = np.random.normal(0, noise_level, n_samples)
            drift = 0.001 * t
            signal = np.cumsum(noise) + drift

        elif pattern_type == "regime_switching":
            # Regime switching behavior
            regime1 = 5 * np.sin(2 * np.pi * t / 24)
            regime2 = -3 * np.cos(2 * np.pi * t / 12)
            switch_prob = 0.01
            regime = np.zeros(n_samples)
            for i in range(1, n_samples):
                if np.random.random() < switch_prob:
                    regime[i] = 1 - regime[i-1]
                else:
                    regime[i] = regime[i-1]
            signal = regime1 * (1 - regime) + regime2 * regime

        elif pattern_type == "intermittent":
            # Intermittent time series (common in demand forecasting)
            base_signal = 10 + 2 * np.sin(2 * np.pi * t / 24)
            spikes = np.random.poisson(0.1, n_samples) * 5
            signal = base_signal + spikes
            # Add zeros randomly
            zero_mask = np.random.random(n_samples) < 0.7
            signal[zero_mask] = 0

        else:
            # Default to simple sine wave
            signal = 5 * np.sin(2 * np.pi * t / 24)

        # Add noise
        noise = np.random.normal(0, noise_level, n_samples)
        signal = signal + noise

        # Clip to value range
        signal = np.clip(signal, value_range[0], value_range[1])

        return signal

    def generate_forecasting_dataset(
        self,
        n_series: int = 100,
        min_length: int = 100,
        max_length: int = 1000,
        context_length: int = 512,
        horizon: int = 96,
        n_features: int = 1,
        dataset_type: str = "various"
    ) -> List[Dict[str, Union[torch.Tensor, Dict[str, Any]]]]:
        """Generate a dataset for forecasting tasks.

        Args:
            n_series: Number of time series to generate.
            min_length: Minimum length of each series.
            max_length: Maximum length of each series.
            context_length: Context length for forecasting.
            horizon: Forecast horizon.
            n_features: Number of features.
            dataset_type: Type of dataset ('various', 'similar', 'synthetic').

        Returns:
            List of time series with context/target splits.
        """
        dataset = []

        for i in range(n_series):
            # Generate time series
            length = np.random.randint(min_length, max_length + 1)

            if dataset_type == "various":
                pattern_types = ["trend_seasonal", "cyclical", "random_walk", "regime_switching"]
                pattern = np.random.choice(pattern_types)
            elif dataset_type == "similar":
                pattern = "trend_seasonal"
            else:
                pattern = "synthetic"

            data = self._generate_pattern(
                length, pattern, noise_level=0.1, value_range=(-10, 10)
            )

            # Convert to multivariate if needed
            if n_features > 1:
                data = np.tile(data.reshape(-1, 1), (1, n_features))
                # Add small variations for each feature
                for j in range(1, n_features):
                    data[:, j] += np.random.normal(0, 0.1, length)

            # Ensure we have enough data for context + horizon
            if length < context_length + horizon:
                # Pad with zeros if needed
                padding = context_length + horizon - length
                if n_features == 1:
                    data = np.pad(data, (0, padding), mode='constant')
                else:
                    data = np.pad(data, ((0, padding), (0, 0)), mode='constant')

            # Split into context and target
            context = data[-context_length-horizon:-horizon]
            target = data[-horizon:]

            # Convert to tensors
            context_tensor = torch.from_numpy(context).float()
            if n_features == 1:
                context_tensor = context_tensor.unsqueeze(-1)

            target_tensor = torch.from_numpy(target).float()
            if n_features == 1:
                target_tensor = target_tensor.unsqueeze(-1)

            dataset.append({
                "context": context_tensor,
                "target": target_tensor,
                "metadata": {
                    "series_id": i,
                    "length": length,
                    "pattern": pattern,
                    "n_features": n_features,
                    "context_length": context_length,
                    "horizon": horizon,
                }
            })

        return dataset

    def generate_anomaly_data(
        self,
        n_samples: int = 1000,
        anomaly_types: Optional[List[str]] = None,
        anomaly_rate: float = 0.01,
        anomaly_magnitude: float = 5.0
    ) -> Tuple[NDArray, NDArray, Dict[str, Any]]:
        """Generate time series with anomalies.

        Args:
            n_samples: Number of samples.
            anomaly_types: Types of anomalies to include.
            anomaly_rate: Rate of anomalies.
            anomaly_magnitude: Magnitude of anomalies.

        Returns:
            Tuple of (data, labels, metadata).
        """
        if anomaly_types is None:
            anomaly_types = ["spike", "dip", "drift", "level_shift", "outlier"]

        # Generate normal time series
        normal_data = self._generate_pattern(n_samples, "trend_seasonal", 0.1, (-10, 10))

        # Initialize labels (0 = normal, 1 = anomaly)
        labels = np.zeros(n_samples, dtype=int)

        # Add anomalies
        n_anomalies = int(n_samples * anomaly_rate)
        anomaly_positions = np.random.choice(n_samples, n_anomalies, replace=False)

        anomaly_metadata = []

        for pos in anomaly_positions:
            anomaly_type = np.random.choice(anomaly_types)

            if anomaly_type == "spike":
                normal_data[pos] += anomaly_magnitude
            elif anomaly_type == "dip":
                normal_data[pos] -= anomaly_magnitude
            elif anomaly_type == "drift":
                # Gradual change over several points
                drift_length = min(10, n_samples - pos)
                for i in range(drift_length):
                    normal_data[pos + i] += anomaly_magnitude * (i / drift_length)
            elif anomaly_type == "level_shift":
                # Sudden level change
                shift_start = pos
                shift_end = min(pos + 50, n_samples)
                normal_data[shift_start:shift_end] += anomaly_magnitude
                labels[shift_start:shift_end] = 1
            elif anomaly_type == "outlier":
                # Extreme value
                normal_data[pos] += anomaly_magnitude * 5

            labels[pos] = 1
            anomaly_metadata.append({
                "position": pos,
                "type": anomaly_type,
                "magnitude": anomaly_magnitude,
                "original_value": normal_data[pos]
            })

        metadata = {
            "n_anomalies": n_anomalies,
            "anomaly_rate": anomaly_rate,
            "anomaly_types": anomaly_types,
            "anomaly_metadata": anomaly_metadata,
        }

        return normal_data, labels, metadata

    def create_forecasting_tensor(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        context_length: int = 512,
        horizon: int = 96,
        stride: int = 1,
        n_features: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Create tensors suitable for model training/inference.

        Args:
            data: Input time series data.
            context_length: Length of context window.
            horizon: Length of forecast horizon.
            stride: Stride between windows.
            n_features: Number of features.

        Returns:
            Dictionary with input and target tensors.
        """
        if isinstance(data, pd.DataFrame):
            values = data.values
        else:
            values = data

        # Ensure values is 2D
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        n_samples, n_features_actual = values.shape
        if n_features is None:
            n_features = n_features_actual

        # Calculate number of windows
        n_windows = (n_samples - context_length - horizon) // stride + 1

        # Initialize tensors
        input_windows = np.zeros((n_windows, context_length, n_features))
        target_windows = np.zeros((n_windows, horizon, n_features))

        # Create sliding windows
        for i in range(n_windows):
            start_idx = i * stride
            end_idx = start_idx + context_length
            target_end_idx = end_idx + horizon

            input_window = values[start_idx:end_idx]
            target_window = values[end_idx:target_end_idx]

            # Handle feature mismatch
            if n_features_actual != n_features:
                # Tile or truncate features
                if n_features_actual < n_features:
                    # Tile existing features
                    input_window = np.tile(input_window, (1, n_features // n_features_actual))
                    target_window = np.tile(target_window, (1, n_features // n_features_actual))
                    # Pad remaining features with zeros
                    if n_features % n_features_actual != 0:
                        padding = n_features - input_window.shape[1]
                        input_window = np.pad(input_window, ((0, 0), (0, padding)), mode='constant')
                        target_window = np.pad(target_window, ((0, 0), (0, padding)), mode='constant')
                else:
                    # Truncate features
                    input_window = input_window[:, :n_features]
                    target_window = target_window[:, :n_features]

            input_windows[i] = input_window
            target_windows[i] = target_window

        return {
            "input": torch.from_numpy(input_windows).float(),
            "target": torch.from_numpy(target_windows).float(),
            "n_windows": n_windows,
            "context_length": context_length,
            "horizon": horizon,
            "stride": stride,
        }

    def scale_data(
        self,
        data: Union[np.ndarray, torch.Tensor],
        method: str = "standard",
        fit_on: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """Scale data using specified method.

        Args:
            data: Data to scale.
            method: Scaling method ('standard', 'minmax', 'robust').
            fit_on: Data to fit scaler on (defaults to data).

        Returns:
            Scaled data.
        """
        if fit_on is None:
            fit_on = data

        # Convert to numpy if tensor
        if isinstance(fit_on, torch.Tensor):
            fit_data = fit_on.numpy()
        else:
            fit_data = fit_on

        if isinstance(data, torch.Tensor):
            input_data = data.numpy()
            return_tensor = True
        else:
            input_data = data
            return_tensor = False

        # Reshape for scaling (flatten if needed)
        original_shape = input_data.shape
        if len(original_shape) > 2:
            fit_data = fit_data.reshape(-1, fit_data.shape[-1])
            input_data = input_data.reshape(-1, input_data.shape[-1])

        # Create and fit scaler
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        scaler.fit(fit_data)
        scaled_data = scaler.transform(input_data)

        # Store scaler for later use
        self.scalers[method] = scaler

        # Reshape back to original shape
        scaled_data = scaled_data.reshape(original_shape)

        if return_tensor:
            return torch.from_numpy(scaled_data).float()
        else:
            return scaled_data

    def save_dataset(
        self,
        dataset: Union[List[Dict], pd.DataFrame],
        filepath: Union[str, Path],
        format: str = "json"
    ) -> None:
        """Save generated dataset to file.

        Args:
            dataset: Dataset to save.
            filepath: Path to save file.
            format: File format ('json', 'csv', 'pt').
        """
        filepath = Path(filepath)

        if format == "json":
            if isinstance(dataset, pd.DataFrame):
                dataset = dataset.to_dict()
            with open(filepath, "w") as f:
                json.dump(dataset, f, indent=2, default=str)

        elif format == "csv":
            if isinstance(dataset, list):
                # Convert to DataFrame
                df_data = {}
                for i, item in enumerate(dataset):
                    if "context" in item:
                        context = item["context"].numpy() if isinstance(item["context"], torch.Tensor) else item["context"]
                        df_data[f"series_{i}_context"] = context.flatten()
                    if "target" in item:
                        target = item["target"].numpy() if isinstance(item["target"], torch.Tensor) else item["target"]
                        df_data[f"series_{i}_target"] = target.flatten()
                dataset = pd.DataFrame(df_data)

            dataset.to_csv(filepath, index=False)

        elif format == "pt":
            if isinstance(dataset, list):
                # Save as torch tensors
                tensors = {}
                for i, item in enumerate(dataset):
                    if "context" in item:
                        tensors[f"context_{i}"] = item["context"]
                    if "target" in item:
                        tensors[f"target_{i}"] = item["target"]
                torch.save(tensors, filepath)

        else:
            raise ValueError(f"Unknown format: {format}")


# Factory functions for common datasets
def create_training_dataset(
    n_series: int = 1000,
    context_length: int = 512,
    horizon: int = 96,
    n_features: int = 1,
    generator_seed: int = 42
) -> List[Dict[str, torch.Tensor]]:
    """Create a training dataset with diverse patterns.

    Args:
        n_series: Number of series.
        context_length: Context window size.
        horizon: Forecast horizon.
        n_features: Number of features.
        generator_seed: Random seed.

    Returns:
        List of training examples.
    """
    generator = AdvancedTimeSeriesGenerator(generator_seed)
    return generator.generate_forecasting_dataset(
        n_series=n_series,
        context_length=context_length,
        horizon=horizon,
        n_features=n_features,
        dataset_type="various"
    )


def create_test_dataset(
    patterns: List[str] = None,
    n_series_per_pattern: int = 20,
    context_length: int = 512,
    horizon: int = 96,
    n_features: int = 1,
    generator_seed: int = 42
) -> Dict[str, List[Dict[str, torch.Tensor]]]:
    """Create test dataset organized by pattern.

    Args:
        patterns: List of patterns to generate.
        n_series_per_pattern: Number of series per pattern.
        context_length: Context window size.
        horizon: Forecast horizon.
        n_features: Number of features.
        generator_seed: Random seed.

    Returns:
        Dictionary mapping pattern names to datasets.
    """
    if patterns is None:
        patterns = ["trend_seasonal", "cyclical", "random_walk", "regime_switching"]

    generator = AdvancedTimeSeriesGenerator(generator_seed)
    dataset_by_pattern = {}

    for pattern in patterns:
        dataset = generator.generate_forecasting_dataset(
            n_series=n_series_per_pattern,
            context_length=context_length,
            horizon=horizon,
            n_features=n_features,
            dataset_type="similar"
        )
        dataset_by_pattern[pattern] = dataset

    return dataset_by_pattern


# Global instance
advanced_generator = AdvancedTimeSeriesGenerator()