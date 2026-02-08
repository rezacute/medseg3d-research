"""Shared pytest fixtures for QRC-EV test suite.

This module provides common fixtures for backend instances, sample configurations,
and seed values used across multiple test modules.
"""

import numpy as np
import pytest
from dataclasses import dataclass
from typing import Any


@pytest.fixture
def test_seed() -> int:
    """Provide a fixed seed for reproducible tests."""
    return 42


@pytest.fixture
def alternative_seed() -> int:
    """Provide an alternative seed for testing seed-dependent behavior."""
    return 123


@pytest.fixture
def sample_seeds() -> list[int]:
    """Provide a list of seeds for multi-seed tests."""
    return [42, 123, 456]


@pytest.fixture
def small_qubit_count() -> int:
    """Provide a small qubit count for fast tests."""
    return 4


@pytest.fixture
def medium_qubit_count() -> int:
    """Provide a medium qubit count for standard tests."""
    return 8


@pytest.fixture
def sample_time_series(small_qubit_count: int) -> np.ndarray:
    """Generate a small sample time-series for testing.
    
    Returns:
        Array of shape (10, 4) with values in [0, 1].
    """
    rng = np.random.default_rng(42)
    return rng.uniform(0, 1, size=(10, small_qubit_count))


@pytest.fixture
def sample_targets() -> np.ndarray:
    """Generate sample target values for testing.
    
    Returns:
        Array of shape (10,) with values in [0, 1].
    """
    rng = np.random.default_rng(42)
    return rng.uniform(0, 1, size=10)


@pytest.fixture
def sample_backend_config() -> dict[str, Any]:
    """Provide a sample backend configuration."""
    return {
        "name": "pennylane",
        "device": "default.qubit",
        "shots": 0,
    }


@pytest.fixture
def sample_quantum_model_config(small_qubit_count: int) -> dict[str, Any]:
    """Provide a sample quantum model configuration."""
    return {
        "arch": "standard",
        "n_qubits": small_qubit_count,
        "n_layers": 2,
        "evolution_steps": 1,
        "encoding": "angle",
        "observables": "pauli_z",
    }


@pytest.fixture
def sample_data_config() -> dict[str, Any]:
    """Provide a sample data configuration."""
    return {
        "dataset": "synthetic",
        "resolution": "1h",
        "window_size": 24,
        "forecast_horizon": 1,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
    }


@pytest.fixture
def sample_experiment_config() -> dict[str, Any]:
    """Provide a sample experiment configuration."""
    return {
        "name": "test_experiment",
        "seeds": [42, 123],
        "metrics": ["rmse", "r2"],
    }


@pytest.fixture
def sample_full_config(
    sample_experiment_config: dict[str, Any],
    sample_quantum_model_config: dict[str, Any],
    sample_backend_config: dict[str, Any],
    sample_data_config: dict[str, Any],
) -> dict[str, Any]:
    """Provide a complete sample configuration."""
    return {
        "experiment": sample_experiment_config,
        "quantum_model": sample_quantum_model_config,
        "backend": sample_backend_config,
        "data": sample_data_config,
    }


@pytest.fixture
def temp_config_file(tmp_path: Any, sample_full_config: dict[str, Any]) -> str:
    """Create a temporary YAML config file for testing.
    
    Args:
        tmp_path: pytest's temporary directory fixture.
        sample_full_config: Sample configuration dictionary.
        
    Returns:
        Path to the temporary config file.
    """
    import yaml
    
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_full_config, f)
    
    return str(config_path)
