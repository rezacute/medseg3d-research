"""Configuration system for QRC-EV.

This module provides structured configuration objects using dataclasses
for experiment parameters, model hyperparameters, and data pipeline settings.
"""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Configuration for data preprocessing and loading.
    
    Attributes:
        dataset: Name of the dataset to use.
        resolution: Time resolution (e.g., '15min', '1h').
        window_size: Number of past timesteps in each input window.
        forecast_horizon: Number of future timesteps to forecast.
        train_ratio: Ratio of data used for training.
        val_ratio: Ratio of data used for validation.
        test_ratio: Ratio of data used for testing.
    """
    dataset: str = "synthetic"
    resolution: str = "1h"
    window_size: int = 24
    forecast_horizon: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
