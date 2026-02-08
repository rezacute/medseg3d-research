import numpy as np


class SyntheticGenerator:
    """Utility for producing synthetic time-series data for testing.

    Requirement 14.
    """

    def __init__(self, seed: int = 42):
        """Initialize with a seed for reproducibility.

        Requirement 14.3.
        """
        self.rng = np.random.default_rng(seed)

    def sinusoidal(
        self,
        length: int = 500,
        n_features: int = 4,
        amplitude: float = 1.0,
        frequency: float = 1.0,
        noise_std: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a sinusoidal dataset.

        Requirement 14.1, 14.4.
        """
        t = np.linspace(0, 2 * np.pi * frequency, length)
        # Target is a simple sine wave
        targets = amplitude * np.sin(t) + self.rng.normal(0, noise_std, length)

        # Features are multiple sine waves with different phases/frequencies
        features = np.zeros((length, n_features))
        for i in range(n_features):
            phase = i * (np.pi / n_features)
            freq_offset = 1.0 + (i * 0.1)
            features[:, i] = amplitude * np.sin(t * freq_offset + phase) + self.rng.normal(
                0, noise_std, length
            )

        return features, targets

    def ev_charging_pattern(
        self, length: int = 720, n_features: int = 4, noise_std: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic EV charging patterns with daily and weekly periodicity.

        Requirement 14.2, 14.4.
        """
        # Assume hourly data (720 hours = 30 days)
        t = np.arange(length)

        # Daily periodicity: peaks at ~8am (hour 8) and ~6pm (hour 18)
        # Using a mixture of Gaussians for daily profile
        def daily_profile(hour):
            # Morning peak
            m_peak = np.exp(-0.5 * ((hour - 8) / 2) ** 2)
            # Evening peak
            e_peak = np.exp(-0.5 * ((hour - 18) / 3) ** 2)
            return 0.4 * m_peak + 0.6 * e_peak

        day_of_week = (t // 24) % 7
        hour_of_day = t % 24

        # Weekly periodicity: Weekdays (0-4) higher than Weekends (5-6)
        weekly_factor = np.where(day_of_week < 5, 1.0, 0.4)

        targets = np.array(
            [daily_profile(h) * w for h, w in zip(hour_of_day, weekly_factor)]
        )
        targets += self.rng.normal(0, noise_std, length)
        # Ensure non-negative
        targets = np.maximum(0, targets)

        # Features could be temporal indicators or related patterns
        features = np.zeros((length, n_features))
        for i in range(n_features):
            # Shifted or scaled versions of the pattern
            shift = i * 2
            feat_hour = (hour_of_day + shift) % 24
            features[:, i] = np.array(
                [daily_profile(h) * w for h, w in zip(feat_hour, weekly_factor)]
            )
            features[:, i] += self.rng.normal(0, noise_std, length)

        return features, targets
