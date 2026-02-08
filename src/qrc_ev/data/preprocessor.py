"""Data preprocessing pipeline for QRC-EV.

This module provides the Preprocessor class for transforming raw datasets
into normalized, windowed feature arrays ready for quantum encoding.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from qrc_ev.utils.config import DataConfig

logger = logging.getLogger(__name__)


class Preprocessor:
    """Preprocessor for EV charging data.

    Transforms raw session-level data or time-series into normalized,
    windowed samples for the quantum reservoir.

    Attributes:
        config: Data configuration object.
    """

    def __init__(self, config: DataConfig):
        """Initialize the preprocessor with configuration.

        Args:
            config: DataConfig object containing pipeline parameters.
        """
        self.config = config
        self._train_min: np.ndarray | None = None
        self._train_max: np.ndarray | None = None

    def aggregate_sessions(
        self, sessions: pd.DataFrame, resolution: str | None = None
    ) -> pd.Series:
        """Aggregate session-level data to fixed time bins.

        Args:
            sessions: DataFrame with session records. Must contain
                'connectionTime' and 'energyDone' columns.
            resolution: Time bin resolution (e.g., '15min', '1h').
                If None, uses self.config.resolution.

        Returns:
            pd.Series: Aggregated energy demand time-series.
        """
        res = resolution or self.config.resolution
        
        # Ensure connectionTime is datetime
        sessions = sessions.copy()
        sessions["connectionTime"] = pd.to_datetime(sessions["connectionTime"])
        
        # Set connectionTime as index for resampling
        sessions = sessions.set_index("connectionTime")
        
        # Resample and sum energy done
        # Using 'sum' because we want total demand in the bin
        ts = sessions["energyDone"].resample(res).sum()
        
        return ts

    def handle_missing(self, series: pd.Series, max_gap: int = 4) -> pd.Series:
        """Forward-fill missing values and detect large gaps.

        Args:
            series: Time-series with potential missing values.
            max_gap: Maximum number of consecutive missing values to interpolate.
                Gaps exceeding this will be logged as warnings.

        Returns:
            pd.Series: Interpolated time-series.
        """
        # Detect gaps before filling
        is_missing = series.isna()
        if is_missing.any():
            # Check for large gaps
            # Calculate gap lengths
            gap_mask = is_missing.astype(int)
            gap_groups = (gap_mask != gap_mask.shift()).cumsum()
            gap_lengths = gap_mask.groupby(gap_groups).transform("sum")
            
            large_gaps = (gap_lengths > max_gap) & is_missing
            if large_gaps.any():
                gap_starts = series.index[large_gaps][0] # Just report the first one for brevity
                logger.warning(
                    f"Gap exceeding threshold {max_gap} detected starting at {gap_starts}"
                )
        
        # Forward fill
        return series.ffill()

    def clip_outliers(self, series: pd.Series, n_sigma: float = 3.0) -> pd.Series:
        """Clip values beyond n_sigma standard deviations.

        Args:
            series: Input time-series.
            n_sigma: Number of standard deviations for clipping boundary.

        Returns:
            pd.Series: Clipped time-series.
        """
        mean = series.mean()
        std = series.std()
        
        lower_bound = mean - n_sigma * std
        upper_bound = mean + n_sigma * std
        
        return series.clip(lower=lower_bound, upper=upper_bound)

    def split_chronological(
        self, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train, validation, and test sets chronologically.

        Args:
            data: Input array of shape (T, d) or (T,).

        Returns:
            tuple: (train, val, test) arrays.
        """
        T = len(data)
        train_end = int(T * self.config.train_ratio)
        val_end = int(T * (self.config.train_ratio + self.config.val_ratio))
        
        train = data[:train_end]
        val = data[train_end:val_end]
        test = data[val_end:]
        
        return train, val, test

    def fit_normalize(self, train_data: np.ndarray) -> None:
        """Compute normalization statistics from training data.

        Args:
            train_data: Training set features of shape (T_train, d).
        """
        if train_data.ndim == 1:
            train_data = train_data.reshape(-1, 1)
            
        self._train_min = np.nanmin(train_data, axis=0)
        self._train_max = np.nanmax(train_data, axis=0)

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply min-max normalization to [0, 1] with clipping.

        Args:
            data: Input features to normalize.

        Returns:
            np.ndarray: Normalized features in range [0, 1].

        Raises:
            RuntimeError: If fit_normalize has not been called.
        """
        if self._train_min is None or self._train_max is None:
            raise RuntimeError("Preprocessor.fit_normalize must be called before normalize")
            
        is_1d = data.ndim == 1
        if is_1d:
            data = data.reshape(-1, 1)
            
        # Avoid division by zero
        range_val = self._train_max - self._train_min
        range_val[range_val == 0] = 1.0
        
        normalized = (data - self._train_min) / range_val
        
        # Clip to [0, 1] as per Requirement 12.8
        normalized_clipped: np.ndarray = np.clip(normalized, 0.0, 1.0)
        
        if is_1d:
            result: np.ndarray = normalized_clipped.flatten()
            return result
            
        return normalized_clipped

    def create_windows(
        self, features: np.ndarray, targets: np.ndarray, window_size: int, horizon: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate sliding window samples for supervised learning.

        Args:
            features: Feature array of shape (T, d).
            targets: Target array of shape (T,) or (T, h_raw).
            window_size: Length of the lookback window.
            horizon: Forecast horizon (number of steps ahead).

        Returns:
            tuple: (X, y) where X is (N, window_size, d) and y is (N, horizon).
        """
        # Ensure features is 2D
        if features.ndim == 1:
            features = features.reshape(-1, 1)
            
        T = len(features)
        # Number of segments: T - window_size - horizon + 1
        num_samples = T - window_size - horizon + 1
        
        if num_samples <= 0:
            return np.array([]), np.array([])
            
        d = features.shape[1]
        X = np.zeros((num_samples, window_size, d))
        
        # Handle multi-horizon targets if targets is already 2D
        if targets.ndim == 1:
            y = np.zeros((num_samples, horizon))
            for i in range(num_samples):
                X[i] = features[i : i + window_size]
                y[i] = targets[i + window_size : i + window_size + horizon]
        else:
            # If targets are already windowed or formatted differently, 
            # this might need adjustment, but according to spec:
            # y has shape (h,) per sample.
            y = np.zeros((num_samples, horizon))
            for i in range(num_samples):
                X[i] = features[i : i + window_size]
                # Assuming targets[t] is the value to forecast at time t
                # and we want steps [t+W, ..., t+W+h-1]
                y[i] = targets[i + window_size : i + window_size + horizon, 0] # Take first col if 2D
                
        return X, y
