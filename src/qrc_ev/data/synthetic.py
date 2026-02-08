"""Synthetic data generation for testing QRC-EV pipeline."""

import numpy as np


class SyntheticGenerator:
    """Generate synthetic time-series data for testing.
    
    Provides sinusoidal patterns and EV charging patterns with configurable
    parameters and reproducible seeds.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize generator with a random seed.
        
        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
    
    def sinusoidal(
        self,
        length: int = 500,
        n_features: int = 4,
        amplitude: float = 1.0,
        frequency: float = 0.1,
        noise_std: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate sinusoidal time-series data.
        
        Creates synthetic data with sinusoidal patterns and Gaussian noise.
        Each feature has a different phase offset.
        
        Args:
            length: Number of timesteps (T).
            n_features: Number of features (d).
            amplitude: Amplitude of sinusoidal signal.
            frequency: Frequency of sinusoidal signal.
            noise_std: Standard deviation of Gaussian noise.
        
        Returns:
            Tuple of (features, targets):
                - features: Array of shape (T, d) with sinusoidal patterns
                - targets: Array of shape (T,) with target values
        """
        t = np.arange(length)
        
        # Generate features with different phase offsets
        features = np.zeros((length, n_features))
        for i in range(n_features):
            phase = 2 * np.pi * i / n_features
            signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
            noise = self.rng.normal(0, noise_std, length)
            features[:, i] = signal + noise
        
        # Target is the mean of all features plus some noise
        targets = np.mean(features, axis=1) + self.rng.normal(0, noise_std, length)
        
        return features, targets
    
    def ev_charging_pattern(
        self,
        length: int = 720,
        n_features: int = 4,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic EV charging pattern with realistic characteristics.
        
        Creates time-series exhibiting:
        - Daily periodicity (morning peak ~8am, evening peak ~6pm)
        - Weekly periodicity (weekday vs weekend variation)
        - Gaussian noise overlay
        
        Args:
            length: Number of timesteps (T). Default 720 = 30 days at hourly resolution.
            n_features: Number of features (d).
        
        Returns:
            Tuple of (features, targets):
                - features: Array of shape (T, d) with EV charging patterns
                - targets: Array of shape (T,) with demand values
        """
        # Assume hourly resolution
        hours = np.arange(length)
        
        # Daily pattern: morning peak at hour 8, evening peak at hour 18
        hour_of_day = hours % 24
        daily_pattern = (
            0.3 * np.exp(-((hour_of_day - 8) ** 2) / (2 * 2**2))  # Morning peak
            + 0.7 * np.exp(-((hour_of_day - 18) ** 2) / (2 * 2**2))  # Evening peak
            + 0.1  # Baseline
        )
        
        # Weekly pattern: weekday vs weekend
        day_of_week = (hours // 24) % 7
        is_weekday = day_of_week < 5
        weekly_multiplier = np.where(is_weekday, 1.2, 0.8)
        
        # Combine patterns
        base_demand = daily_pattern * weekly_multiplier
        
        # Generate features with variations
        features = np.zeros((length, n_features))
        for i in range(n_features):
            # Each feature is a variation of the base pattern
            phase_shift = self.rng.uniform(-2, 2)
            amplitude_scale = self.rng.uniform(0.8, 1.2)
            noise = self.rng.normal(0, 0.05, length)
            
            # Shift the pattern slightly for each feature
            shifted_hour = (hour_of_day + phase_shift) % 24
            daily_var = (
                0.3 * np.exp(-((shifted_hour - 8) ** 2) / (2 * 2**2))
                + 0.7 * np.exp(-((shifted_hour - 18) ** 2) / (2 * 2**2))
                + 0.1
            )
            
            features[:, i] = amplitude_scale * daily_var * weekly_multiplier + noise
        
        # Target is the base demand pattern with noise
        targets = base_demand + self.rng.normal(0, 0.05, length)
        
        # Clip to ensure non-negative (EV charging demand cannot be negative)
        targets = np.maximum(targets, 0.0)
        
        return features, targets
