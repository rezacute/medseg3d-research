"""End-to-end training pipeline for QRC-EV.

This module orchestrates the complete workflow from configuration loading
through data preprocessing, quantum reservoir processing, and prediction.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from qrc_ev.backends.pennylane_backend import PennyLaneBackend
from qrc_ev.backends import CUDAQ_AVAILABLE
from qrc_ev.data.preprocessor import Preprocessor
from qrc_ev.data.synthetic import SyntheticGenerator
from qrc_ev.readout.ridge import RidgeReadout
from qrc_ev.reservoirs.factory import create_reservoir
from qrc_ev.utils.config import load_config
from qrc_ev.utils.seed import SeedManager

logger = logging.getLogger(__name__)


def run_pipeline(config_path: str) -> dict[str, Any]:
    """Execute the end-to-end QRC-EV pipeline.

    Orchestrates the complete workflow:
    1. Load configuration from YAML
    2. Initialize seed manager and seed all RNGs
    3. Generate or load data
    4. Preprocess data (normalize, window)
    5. Create quantum backend and reservoir
    6. Process features through reservoir
    7. Fit ridge regression readout
    8. Generate predictions
    9. Compute evaluation metrics

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary containing:
            - predictions: NumPy array of predictions on test set
            - metrics: Dictionary of evaluation metrics (RMSE, R²)
            - config: The loaded configuration object

    Example:
        >>> results = run_pipeline("configs/demo.yaml")
        >>> print(f"Test R²: {results['metrics']['test_r2']:.4f}")
    """
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    # Use the first seed from the config
    seed = config.experiment.seeds[0]
    logger.info(f"Initializing seed manager with seed {seed}")
    seed_manager = SeedManager(seed)
    seed_manager.seed_all()

    # Generate synthetic data
    logger.info("Generating synthetic data")
    data_seed = seed_manager.derive_seed("synthetic_data")
    generator = SyntheticGenerator(seed=data_seed)

    if config.data.dataset == "synthetic":
        # Use sinusoidal pattern for testing
        features, targets = generator.sinusoidal(
            length=500,
            n_features=config.quantum_model.n_qubits,  # Match qubit count
            noise_std=0.1,
        )
    elif config.data.dataset == "ev_charging":
        features, targets = generator.ev_charging_pattern(
            length=720,
            n_features=config.quantum_model.n_qubits,
        )
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}")

    # Preprocess data
    logger.info("Preprocessing data")
    preprocessor = Preprocessor(config.data)

    # Split chronologically
    train_features, val_features, test_features = preprocessor.split_chronological(
        features
    )
    train_targets, val_targets, test_targets = preprocessor.split_chronological(
        targets
    )

    # Fit normalization on training data only
    preprocessor.fit_normalize(train_features)

    # Normalize all splits
    train_features_norm = preprocessor.normalize(train_features)
    val_features_norm = preprocessor.normalize(val_features)
    test_features_norm = preprocessor.normalize(test_features)

    # Create windowed samples
    # For simplicity, we'll process each timestep independently (window_size=1)
    # This matches the reservoir's process() method which takes (T, d) input
    # In a full implementation, windowing would be more sophisticated

    # Since the reservoir processes (T, d) directly, we don't need windowing here
    # We'll use the normalized features directly
    X_train = train_features_norm
    y_train = train_targets
    X_test = test_features_norm
    y_test = test_targets

    # Create quantum backend
    logger.info(
        f"Creating {config.backend.name} backend with device {config.backend.device}"
    )
    if config.backend.name == "pennylane":
        backend = PennyLaneBackend(
            device_name=config.backend.device, shots=config.backend.shots
        )
    elif config.backend.name == "cudaq":
        if not CUDAQ_AVAILABLE:
            raise RuntimeError(
                "CUDA-Quantum backend requested but not available. "
                "Install with: pip install cuda-quantum (requires CUDA toolkit)"
            )
        from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
        backend = CUDAQuantumBackend(
            target=config.backend.device, shots=config.backend.shots
        )
    else:
        raise ValueError(f"Unsupported backend: {config.backend.name}")

    # Create reservoir via factory
    logger.info(f"Creating {config.quantum_model.arch} reservoir")
    reservoir_seed = seed_manager.derive_seed("reservoir")
    reservoir = create_reservoir(
        arch=config.quantum_model.arch,
        backend=backend,
        n_qubits=config.quantum_model.n_qubits,
        n_layers=config.quantum_model.n_layers,
        evolution_steps=config.quantum_model.evolution_steps,
        seed=reservoir_seed,
    )

    # Process features through reservoir
    logger.info("Processing training features through reservoir")
    train_reservoir_features = reservoir.process(X_train)

    logger.info("Processing test features through reservoir")
    test_reservoir_features = reservoir.process(X_test)

    # Fit ridge readout
    logger.info("Fitting ridge regression readout")
    readout = RidgeReadout(alpha=1e-4)
    readout.fit(train_reservoir_features, y_train)

    # Generate predictions
    logger.info("Generating predictions")
    train_predictions = readout.predict(train_reservoir_features)
    test_predictions = readout.predict(test_reservoir_features)

    # Compute metrics
    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_predictions)))
    train_r2 = float(r2_score(y_train, train_predictions))

    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_predictions)))
    test_r2 = float(r2_score(y_test, test_predictions))

    metrics = {
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
    }

    logger.info(f"Training RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
    logger.info(f"Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

    return {
        "predictions": test_predictions,
        "metrics": metrics,
        "config": config,
    }
