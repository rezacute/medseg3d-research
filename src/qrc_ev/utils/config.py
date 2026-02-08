"""Configuration system for QRC-EV experiments.

Provides YAML-based configuration with inheritance support, validation,
and serialization for reproducible experiments.
"""

from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any
import yaml  # type: ignore


class ConfigError(ValueError):
    """Custom exception for configuration errors."""
    pass


@dataclass
class ExperimentConfig:
    """Experiment metadata and settings.

    Attributes:
        name: Experiment name identifier
        seeds: List of random seeds for reproducibility
        metrics: List of evaluation metrics to compute
    """
    name: str
    seeds: list[int]
    metrics: list[str] = field(default_factory=lambda: ["rmse", "r2"])


@dataclass
class QuantumModelConfig:
    """Quantum model architecture configuration.

    Attributes:
        arch: Architecture name (e.g., "standard")
        n_qubits: Number of qubits in the reservoir
        n_layers: Number of reservoir layers
        evolution_steps: Number of evolution steps per timestep
        encoding: Encoding strategy (e.g., "angle")
        observables: Observable measurement strategy (e.g., "pauli_z")
    """
    arch: str
    n_qubits: int
    n_layers: int = 4
    evolution_steps: int = 1
    encoding: str = "angle"
    observables: str = "pauli_z"


@dataclass
class BackendConfig:
    """Quantum backend configuration.

    Attributes:
        name: Backend name (e.g., "pennylane", "qiskit")
        device: Device name (e.g., "default.qubit", "lightning.qubit")
        shots: Number of shots for sampling (0 for exact statevector)
    """
    name: str = "pennylane"
    device: str = "default.qubit"
    shots: int = 0


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


@dataclass
class QRCConfig:
    """Top-level QRC-EV configuration.

    Attributes:
        experiment: Experiment configuration
        quantum_model: Quantum model configuration
        backend: Backend configuration
        data: Data pipeline configuration
    """
    experiment: ExperimentConfig
    quantum_model: QuantumModelConfig
    backend: BackendConfig
    data: DataConfig


def _validate_config_dict(config_dict: dict[str, Any], dataclass_type: type) -> None:
    """Validate that config dict has required fields and no unknown fields.

    Args:
        config_dict: Dictionary to validate
        dataclass_type: Dataclass type to validate against

    Raises:
        ConfigError: If required fields are missing or unknown fields present
    """
    if config_dict is None:
        raise ConfigError(f"Missing required section for {dataclass_type.__name__}")

    # Get field names from dataclass
    valid_fields = {f.name for f in fields(dataclass_type)}
    provided_fields = set(config_dict.keys())

    # Check for unknown fields
    unknown_fields = provided_fields - valid_fields
    if unknown_fields:
        raise ConfigError(
            f"Unknown configuration field(s) in {dataclass_type.__name__}: "
            f"{', '.join(sorted(unknown_fields))}"
        )

    # Check for missing required fields (fields without defaults)
    # A field is required if it has no default value and no default_factory
    from dataclasses import MISSING
    required_fields = {
        f.name for f in fields(dataclass_type)
        if f.default is MISSING and f.default_factory is MISSING
    }
    missing_fields = required_fields - provided_fields
    if missing_fields:
        raise ConfigError(
            f"Missing required field(s) in {dataclass_type.__name__}: "
            f"{', '.join(sorted(missing_fields))}"
        )


def _merge_configs(base: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Merge base and child configurations, with child taking precedence.

    Args:
        base: Base configuration dictionary
        child: Child configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    merged = base.copy()

    for key, value in child.items():
        if key == "extends":
            # Don't propagate extends field
            continue
        elif isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            # Recursively merge nested dicts
            merged[key] = _merge_configs(merged[key], value)
        else:
            # Child value overrides base
            merged[key] = value

    return merged


def load_config(path: str) -> QRCConfig:
    """Load configuration from YAML file with inheritance support.

    Supports configuration inheritance via the 'extends' field. When a config
    specifies 'extends: base.yaml', the base config is loaded first and merged
    with the child config, with child values taking precedence.

    Args:
        path: Path to YAML configuration file

    Returns:
        Parsed and validated QRCConfig object

    Raises:
        ConfigError: If file not found, YAML parsing fails, validation fails,
                    or base config not found
    """
    config_path = Path(path)

    # Check if file exists
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {path}")

    # Load YAML
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML: {e}")
    except Exception as e:
        raise ConfigError(f"Failed to load configuration: {e}")

    if config_dict is None:
        raise ConfigError(f"Empty configuration file: {path}")

    # Handle inheritance
    if "extends" in config_dict:
        base_path = config_path.parent / config_dict["extends"]
        if not base_path.exists():
            raise ConfigError(f"Base config not found: {base_path}")

        # Load base config recursively
        base_config_dict = load_config(str(base_path))
        base_dict = asdict(base_config_dict)

        # Merge base and child
        config_dict = _merge_configs(base_dict, config_dict)

    # Validate top-level structure
    top_level_fields = {"experiment", "quantum_model", "backend", "data"}
    provided_top_level = set(config_dict.keys()) - {"extends"}

    unknown_top_level = provided_top_level - top_level_fields
    if unknown_top_level:
        raise ConfigError(
            f"Unknown top-level configuration field(s): "
            f"{', '.join(sorted(unknown_top_level))}"
        )

    # Validate each section
    _validate_config_dict(config_dict.get("experiment"), ExperimentConfig)
    _validate_config_dict(config_dict.get("quantum_model"), QuantumModelConfig)
    _validate_config_dict(config_dict.get("backend"), BackendConfig)
    _validate_config_dict(config_dict.get("data"), DataConfig)

    # Construct dataclass instances
    try:
        experiment = ExperimentConfig(**config_dict["experiment"])
        quantum_model = QuantumModelConfig(**config_dict["quantum_model"])
        backend = BackendConfig(**config_dict["backend"])
        data = DataConfig(**config_dict["data"])

        return QRCConfig(
            experiment=experiment,
            quantum_model=quantum_model,
            backend=backend,
            data=data,
        )
    except TypeError as e:
        raise ConfigError(f"Configuration validation failed: {e}")


def dump_config(config: QRCConfig) -> str:
    """Serialize configuration to YAML string.

    Args:
        config: QRCConfig object to serialize

    Returns:
        YAML string representation of the configuration
    """
    config_dict = asdict(config)
    result = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
    return str(result)
