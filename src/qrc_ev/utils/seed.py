"""Seed management for reproducible experiments."""

import hashlib
import logging
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SeedManager:
    """Manages random seeds for reproducible experiments.
    
    The SeedManager provides deterministic seed management across Python's
    random module, NumPy, and component-specific random states. It supports
    automatic seed generation with logging for reproducibility.
    
    Attributes:
        global_seed: The master seed used for all random number generation.
    
    Example:
        >>> sm = SeedManager(42)
        >>> sm.seed_all()
        >>> reservoir_seed = sm.derive_seed("reservoir")
        >>> data_seed = sm.derive_seed("data_split")
    """
    
    def __init__(self, global_seed: Optional[int] = None):
        """Initialize the seed manager.
        
        Args:
            global_seed: Master seed for all random number generation.
                If None, generates a random seed and logs it for reproducibility.
        """
        if global_seed is None:
            global_seed = int(np.random.default_rng().integers(0, 2**31))
            logger.info(f"Generated random seed: {global_seed}")
        self.global_seed = global_seed
    
    def seed_all(self) -> None:
        """Seed Python random and NumPy with the global seed.
        
        This method should be called at the start of an experiment to ensure
        reproducibility across all random number generation.
        """
        random.seed(self.global_seed)
        np.random.seed(self.global_seed)
    
    def derive_seed(self, component: str) -> int:
        """Derive a deterministic child seed for a named component.
        
        Uses SHA-256 hashing to generate component-specific seeds from the
        global seed. This avoids seed correlation between different components
        while maintaining reproducibility.
        
        Args:
            component: Name of the component (e.g., "reservoir", "data_split").
        
        Returns:
            A deterministic integer seed in the range [0, 2^31).
        
        Example:
            >>> sm = SeedManager(42)
            >>> seed1 = sm.derive_seed("reservoir")
            >>> seed2 = sm.derive_seed("data_split")
            >>> seed1 != seed2  # Different components get different seeds
            True
        """
        h = hashlib.sha256(f"{self.global_seed}:{component}".encode())
        return int.from_bytes(h.digest()[:4], "big") % (2**31)
