"""Feature engineering for EV charging demand forecasting.

This module provides the FeatureEngineer class for generating temporal and
lagged features from raw time-series data.
"""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Feature engineering component for QRC-EV.

    Generates temporal (sin/cos) and lagged features from time-series data
    to provide informative input vectors for the quantum reservoir.

    Attributes:
        lag_steps: List of integer lag steps to generate (e.g., [1, 2, 24]).
    """

    def __init__(self, lag_steps: list[int] | None = None):
        """Initialize FeatureEngineer with configurable lag steps.

        Args:
            lag_steps: List of lag steps. Defaults to [1, 2, 4, 12, 24].
        """
        self.lag_steps = lag_steps or [1, 2, 4, 12, 24]

    def add_temporal_features(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Generate sin/cos encodings for hour-of-day and day-of-week.

        Args:
            timestamps: Index of pandas DatetimeIndex.

        Returns:
            NumPy array of shape (T, 4) containing:
            [sin_hour, cos_hour, sin_day, cos_day]
        """
        hours = timestamps.hour + timestamps.minute / 60.0
        days = timestamps.dayofweek

        sin_hour = np.sin(2 * np.pi * hours / 24.0)
        cos_hour = np.cos(2 * np.pi * hours / 24.0)
        sin_day = np.sin(2 * np.pi * days / 7.0)
        cos_day = np.cos(2 * np.pi * days / 7.0)

        return np.stack([sin_hour, cos_hour, sin_day, cos_day], axis=1)

    def add_lag_features(self, series: np.ndarray) -> np.ndarray:
        """Create lagged copies of the target variable at configurable steps.

        Args:
            series: 1D NumPy array of time-series values of shape (T,).

        Returns:
            NumPy array of shape (T, len(lag_steps)). Missing values at the
            beginning of the series are filled with 0.0.
        """
        lags = []
        for k in self.lag_steps:
            # Shift the series by k steps
            lagged = np.roll(series, k)
            # Nullify the first k elements (since shift is circular in np.roll)
            lagged[:k] = 0.0
            lags.append(lagged)

        return np.stack(lags, axis=1)

    def engineer(
        self, series: np.ndarray, timestamps: pd.DatetimeIndex
    ) -> np.ndarray:
        """Full feature engineering pipeline.

        Combines temporal features and lagged features into a single array.

        Args:
            series: 1D NumPy array of target values.
            timestamps: Corresponding DatetimeIndex for the series.

        Returns:
            2D NumPy array of shape (T, feature_dim).
        """
        temporal = self.add_temporal_features(timestamps)
        lags = self.add_lag_features(series)
        return np.concatenate([temporal, lags], axis=1)

    @property
    def feature_dim(self) -> int:
        """Total dimension of the engineered feature vector."""
        return 4 + len(self.lag_steps)
