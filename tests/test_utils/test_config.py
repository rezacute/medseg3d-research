"""Tests for configuration system."""

import pytest
import tempfile
from pathlib import Path
from dataclasses import asdict
import yaml
from hypothesis import given, settings, strategies as st

from qrc_ev.utils.config import (
    ConfigError,
    ExperimentConfig,
    QuantumModelConfig,
    BackendConfig,
    DataConfig,
    QRCConfig,
    load_config,
    dump_config,
)


@pytest.fixture
def valid_config_dict():
    """Valid configuration dictionary."""
    return {
        "experiment": {
            "name": "test_experiment",
            "seeds": [42, 43, 44],
            "metrics": ["rmse", "r2", "mae"],
        },
        "quantum_model": {
            "arch": "standard",
            "n_qubits": 8,
            "n_layers": 4,
            "evolution_steps": 1,
            "encoding": "angle",
            "observables": "pauli_z",
        },
        "backend": {
            "name": "pennylane",
            "device": "default.qubit",
            "shots": 0,
        },
        "data": {
            "dataset": "synthetic",
            "resolution": "1h",
            "window_size": 24,
            "forecast_horizon": 1,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        },
    }


@pytest.fixture
def minimal_config_dict():
    """Minimal valid configuration with only required fields."""
    return {
        "experiment": {
            "name": "minimal_test",
            "seeds": [42],
        },
        "quantum_model": {
            "arch": "standard",
            "n_qubits": 4,
        },
        "backend": {},
        "data": {},
    }


def test_experiment_config_creation():
    """Test ExperimentConfig dataclass creation."""
    config = ExperimentConfig(
        name="test",
        seeds=[42, 43],
        metrics=["rmse", "r2"],
    )
    assert config.name == "test"
    assert config.seeds == [42, 43]
    assert config.metrics == ["rmse", "r2"]


def test_experiment_config_defaults():
    """Test ExperimentConfig default values."""
    config = ExperimentConfig(name="test", seeds=[42])
    assert config.metrics == ["rmse", "r2"]


def test_quantum_model_config_creation():
    """Test QuantumModelConfig dataclass creation."""
    config = QuantumModelConfig(
        arch="standard",
        n_qubits=8,
        n_layers=4,
        evolution_steps=1,
        encoding="angle",
        observables="pauli_z",
    )
    assert config.arch == "standard"
    assert config.n_qubits == 8
    assert config.n_layers == 4


def test_quantum_model_config_defaults():
    """Test QuantumModelConfig default values."""
    config = QuantumModelConfig(arch="standard", n_qubits=8)
    assert config.n_layers == 4
    assert config.evolution_steps == 1
    assert config.encoding == "angle"
    assert config.observables == "pauli_z"


def test_backend_config_defaults():
    """Test BackendConfig default values."""
    config = BackendConfig()
    assert config.name == "pennylane"
    assert config.device == "default.qubit"
    assert config.shots == 0


def test_data_config_defaults():
    """Test DataConfig default values."""
    config = DataConfig()
    assert config.dataset == "synthetic"
    assert config.resolution == "1h"
    assert config.window_size == 24
    assert config.forecast_horizon == 1
    assert config.train_ratio == 0.7
    assert config.val_ratio == 0.15
    assert config.test_ratio == 0.15


def test_load_config_valid(valid_config_dict):
    """Test loading a valid configuration."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(valid_config_dict, f)
        config_path = f.name
    
    try:
        config = load_config(config_path)
        
        assert isinstance(config, QRCConfig)
        assert config.experiment.name == "test_experiment"
        assert config.experiment.seeds == [42, 43, 44]
        assert config.quantum_model.arch == "standard"
        assert config.quantum_model.n_qubits == 8
        assert config.backend.name == "pennylane"
        assert config.data.dataset == "synthetic"
    finally:
        Path(config_path).unlink()


def test_load_config_minimal(minimal_config_dict):
    """Test loading a minimal configuration with defaults."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(minimal_config_dict, f)
        config_path = f.name
    
    try:
        config = load_config(config_path)
        
        assert config.experiment.name == "minimal_test"
        assert config.experiment.metrics == ["rmse", "r2"]  # default
        assert config.quantum_model.n_layers == 4  # default
        assert config.backend.device == "default.qubit"  # default
    finally:
        Path(config_path).unlink()


def test_load_config_file_not_found():
    """Test loading non-existent configuration file."""
    with pytest.raises(ConfigError, match="Configuration file not found"):
        load_config("nonexistent.yaml")


def test_load_config_invalid_yaml():
    """Test loading invalid YAML."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content: [")
        config_path = f.name
    
    try:
        with pytest.raises(ConfigError, match="Failed to parse YAML"):
            load_config(config_path)
    finally:
        Path(config_path).unlink()


def test_load_config_empty_file():
    """Test loading empty configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_path = f.name
    
    try:
        with pytest.raises(ConfigError, match="Empty configuration file"):
            load_config(config_path)
    finally:
        Path(config_path).unlink()


def test_load_config_missing_required_field():
    """Test loading config with missing required field."""
    config_dict = {
        "experiment": {
            "name": "test",
            # Missing required 'seeds' field
        },
        "quantum_model": {
            "arch": "standard",
            "n_qubits": 4,
        },
        "backend": {},
        "data": {},
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        config_path = f.name
    
    try:
        with pytest.raises(ConfigError, match="Missing required field.*seeds"):
            load_config(config_path)
    finally:
        Path(config_path).unlink()


def test_load_config_unknown_field():
    """Test loading config with unknown field."""
    config_dict = {
        "experiment": {
            "name": "test",
            "seeds": [42],
            "unknown_field": "value",
        },
        "quantum_model": {
            "arch": "standard",
            "n_qubits": 4,
        },
        "backend": {},
        "data": {},
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        config_path = f.name
    
    try:
        with pytest.raises(ConfigError, match="Unknown configuration field.*unknown_field"):
            load_config(config_path)
    finally:
        Path(config_path).unlink()


def test_load_config_unknown_top_level_field():
    """Test loading config with unknown top-level field."""
    config_dict = {
        "experiment": {
            "name": "test",
            "seeds": [42],
        },
        "quantum_model": {
            "arch": "standard",
            "n_qubits": 4,
        },
        "backend": {},
        "data": {},
        "unknown_section": {},
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        config_path = f.name
    
    try:
        with pytest.raises(ConfigError, match="Unknown top-level configuration field.*unknown_section"):
            load_config(config_path)
    finally:
        Path(config_path).unlink()


def test_load_config_missing_section():
    """Test loading config with missing required section."""
    config_dict = {
        "experiment": {
            "name": "test",
            "seeds": [42],
        },
        # Missing quantum_model section
        "backend": {},
        "data": {},
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        config_path = f.name
    
    try:
        with pytest.raises(ConfigError, match="Missing required section"):
            load_config(config_path)
    finally:
        Path(config_path).unlink()


def test_config_inheritance(valid_config_dict):
    """Test configuration inheritance via extends field."""
    # Create base config
    base_config = {
        "experiment": {
            "name": "base_experiment",
            "seeds": [1, 2, 3],
            "metrics": ["rmse"],
        },
        "quantum_model": {
            "arch": "standard",
            "n_qubits": 4,
            "n_layers": 2,
        },
        "backend": {
            "name": "pennylane",
            "device": "default.qubit",
        },
        "data": {
            "dataset": "synthetic",
            "window_size": 12,
        },
    }
    
    # Create child config that extends base
    child_config = {
        "extends": "base.yaml",
        "experiment": {
            "name": "child_experiment",  # Override
            "seeds": [42],  # Override
        },
        "quantum_model": {
            "n_qubits": 8,  # Override
            # n_layers should be inherited from base
        },
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "base.yaml"
        child_path = Path(tmpdir) / "child.yaml"
        
        with open(base_path, 'w') as f:
            yaml.dump(base_config, f)
        
        with open(child_path, 'w') as f:
            yaml.dump(child_config, f)
        
        config = load_config(str(child_path))
        
        # Check overridden values
        assert config.experiment.name == "child_experiment"
        assert config.experiment.seeds == [42]
        assert config.quantum_model.n_qubits == 8
        
        # Check inherited values
        assert config.experiment.metrics == ["rmse"]
        assert config.quantum_model.n_layers == 2
        assert config.quantum_model.arch == "standard"
        assert config.backend.name == "pennylane"
        assert config.data.window_size == 12


def test_config_inheritance_base_not_found():
    """Test error when base config not found."""
    child_config = {
        "extends": "nonexistent_base.yaml",
        "experiment": {
            "name": "test",
            "seeds": [42],
        },
        "quantum_model": {
            "arch": "standard",
            "n_qubits": 4,
        },
        "backend": {},
        "data": {},
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(child_config, f)
        config_path = f.name
    
    try:
        with pytest.raises(ConfigError, match="Base config not found"):
            load_config(config_path)
    finally:
        Path(config_path).unlink()


def test_dump_config(valid_config_dict):
    """Test configuration serialization to YAML."""
    # Load config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(valid_config_dict, f)
        config_path = f.name
    
    try:
        config = load_config(config_path)
        
        # Dump to YAML string
        yaml_str = dump_config(config)
        
        # Parse the dumped YAML
        dumped_dict = yaml.safe_load(yaml_str)
        
        # Verify structure
        assert "experiment" in dumped_dict
        assert "quantum_model" in dumped_dict
        assert "backend" in dumped_dict
        assert "data" in dumped_dict
        
        # Verify values
        assert dumped_dict["experiment"]["name"] == "test_experiment"
        assert dumped_dict["quantum_model"]["n_qubits"] == 8
        assert dumped_dict["backend"]["device"] == "default.qubit"
    finally:
        Path(config_path).unlink()


def test_config_round_trip(valid_config_dict):
    """Test configuration round-trip: load -> dump -> load."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(valid_config_dict, f)
        config_path = f.name
    
    try:
        # Load original
        config1 = load_config(config_path)
        
        # Dump to YAML
        yaml_str = dump_config(config1)
        
        # Write to new file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f2:
            f2.write(yaml_str)
            config_path2 = f2.name
        
        try:
            # Load again
            config2 = load_config(config_path2)
            
            # Compare
            assert config1.experiment.name == config2.experiment.name
            assert config1.experiment.seeds == config2.experiment.seeds
            assert config1.experiment.metrics == config2.experiment.metrics
            assert config1.quantum_model.arch == config2.quantum_model.arch
            assert config1.quantum_model.n_qubits == config2.quantum_model.n_qubits
            assert config1.backend.name == config2.backend.name
            assert config1.data.dataset == config2.data.dataset
        finally:
            Path(config_path2).unlink()
    finally:
        Path(config_path).unlink()


# ============================================================================
# Property-Based Tests
# ============================================================================

# Hypothesis strategies for generating valid configurations
@st.composite
def valid_experiment_configs(draw):
    """Generate valid ExperimentConfig instances."""
    name = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_-'
    )))
    seeds = draw(st.lists(st.integers(min_value=0, max_value=2**31-1), min_size=1, max_size=10))
    metrics = draw(st.lists(
        st.sampled_from(["rmse", "r2", "mae", "mape", "mse"]),
        min_size=1,
        max_size=5,
        unique=True
    ))
    return ExperimentConfig(name=name, seeds=seeds, metrics=metrics)


@st.composite
def valid_quantum_model_configs(draw):
    """Generate valid QuantumModelConfig instances."""
    arch = draw(st.sampled_from(["standard", "recurrence_free", "multi_timescale"]))
    n_qubits = draw(st.integers(min_value=2, max_value=20))
    n_layers = draw(st.integers(min_value=1, max_value=10))
    evolution_steps = draw(st.integers(min_value=1, max_value=10))
    encoding = draw(st.sampled_from(["angle", "amplitude", "iqp"]))
    observables = draw(st.sampled_from(["pauli_z", "pauli_x", "pauli_y"]))
    return QuantumModelConfig(
        arch=arch,
        n_qubits=n_qubits,
        n_layers=n_layers,
        evolution_steps=evolution_steps,
        encoding=encoding,
        observables=observables,
    )


@st.composite
def valid_backend_configs(draw):
    """Generate valid BackendConfig instances."""
    name = draw(st.sampled_from(["pennylane", "qiskit", "cudaq"]))
    device = draw(st.sampled_from(["default.qubit", "lightning.qubit", "aer_simulator"]))
    shots = draw(st.integers(min_value=0, max_value=10000))
    return BackendConfig(name=name, device=device, shots=shots)


@st.composite
def valid_data_configs(draw):
    """Generate valid DataConfig instances."""
    dataset = draw(st.sampled_from(["synthetic", "acn_data", "palo_alto"]))
    resolution = draw(st.sampled_from(["15min", "30min", "1h"]))
    window_size = draw(st.integers(min_value=1, max_value=168))
    forecast_horizon = draw(st.integers(min_value=1, max_value=24))
    
    # Generate ratios that sum to 1.0 with better numerical stability
    train_ratio = draw(st.floats(min_value=0.5, max_value=0.8))
    remaining = 1.0 - train_ratio
    # Ensure val_ratio has valid bounds
    if remaining <= 0.2:
        val_ratio = remaining / 2.0
    else:
        val_ratio = draw(st.floats(min_value=0.1, max_value=remaining - 0.1))
    test_ratio = 1.0 - train_ratio - val_ratio
    
    return DataConfig(
        dataset=dataset,
        resolution=resolution,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )


@st.composite
def valid_qrc_configs(draw):
    """Generate valid QRCConfig instances."""
    experiment = draw(valid_experiment_configs())
    quantum_model = draw(valid_quantum_model_configs())
    backend = draw(valid_backend_configs())
    data = draw(valid_data_configs())
    return QRCConfig(
        experiment=experiment,
        quantum_model=quantum_model,
        backend=backend,
        data=data,
    )


# Feature: phase1-foundation-setup, Property 9: Configuration round-trip
@given(config=valid_qrc_configs())
@settings(max_examples=100, deadline=None)
def test_property_config_round_trip(config):
    """Property 9: Configuration round-trip.
    
    **Validates: Requirements 8.7**
    
    For any valid QRCConfig object, serializing it to YAML via dump_config()
    and then parsing the result via load_config() should produce an equivalent
    configuration object.
    """
    # Dump config to YAML string
    yaml_str = dump_config(config)
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_str)
        config_path = f.name
    
    try:
        # Load config back
        loaded_config = load_config(config_path)
        
        # Verify all fields match
        # Experiment config
        assert loaded_config.experiment.name == config.experiment.name
        assert loaded_config.experiment.seeds == config.experiment.seeds
        assert loaded_config.experiment.metrics == config.experiment.metrics
        
        # Quantum model config
        assert loaded_config.quantum_model.arch == config.quantum_model.arch
        assert loaded_config.quantum_model.n_qubits == config.quantum_model.n_qubits
        assert loaded_config.quantum_model.n_layers == config.quantum_model.n_layers
        assert loaded_config.quantum_model.evolution_steps == config.quantum_model.evolution_steps
        assert loaded_config.quantum_model.encoding == config.quantum_model.encoding
        assert loaded_config.quantum_model.observables == config.quantum_model.observables
        
        # Backend config
        assert loaded_config.backend.name == config.backend.name
        assert loaded_config.backend.device == config.backend.device
        assert loaded_config.backend.shots == config.backend.shots
        
        # Data config
        assert loaded_config.data.dataset == config.data.dataset
        assert loaded_config.data.resolution == config.data.resolution
        assert loaded_config.data.window_size == config.data.window_size
        assert loaded_config.data.forecast_horizon == config.data.forecast_horizon
        assert abs(loaded_config.data.train_ratio - config.data.train_ratio) < 1e-10
        assert abs(loaded_config.data.val_ratio - config.data.val_ratio) < 1e-10
        assert abs(loaded_config.data.test_ratio - config.data.test_ratio) < 1e-10
    finally:
        Path(config_path).unlink()


# Feature: phase1-foundation-setup, Property 10: Configuration inheritance merge
@given(
    base_config=valid_qrc_configs(),
    child_overrides=st.fixed_dictionaries(
        {},
        optional={
            "experiment": st.fixed_dictionaries(
                {},
                optional={
                    "name": st.text(min_size=1, max_size=30),
                    "seeds": st.lists(st.integers(0, 1000), min_size=1, max_size=5),
                }
            ),
            "quantum_model": st.fixed_dictionaries(
                {},
                optional={
                    "n_qubits": st.integers(2, 16),
                    "n_layers": st.integers(1, 8),
                }
            ),
            "backend": st.fixed_dictionaries(
                {},
                optional={
                    "device": st.sampled_from(["default.qubit", "lightning.qubit"]),
                }
            ),
            "data": st.fixed_dictionaries(
                {},
                optional={
                    "window_size": st.integers(1, 48),
                }
            ),
        }
    )
)
@settings(max_examples=100, deadline=None)
def test_property_config_inheritance_merge(base_config, child_overrides):
    """Property 10: Configuration inheritance merge.
    
    **Validates: Requirements 8.2**
    
    For any base config and child config where the child specifies an 'extends'
    field, the merged result should contain all base values not overridden by
    the child, and all child values should take precedence over base values.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "base.yaml"
        child_path = Path(tmpdir) / "child.yaml"
        
        # Write base config
        base_dict = asdict(base_config)
        with open(base_path, 'w') as f:
            yaml.dump(base_dict, f)
        
        # Build child config with extends and overrides
        child_dict = {"extends": "base.yaml"}
        
        # Apply overrides, removing empty dicts
        for section, overrides in child_overrides.items():
            if overrides:
                child_dict[section] = overrides
        
        # Write child config
        with open(child_path, 'w') as f:
            yaml.dump(child_dict, f)
        
        # Load merged config
        merged_config = load_config(str(child_path))
        
        # Verify inheritance: check that base values are present unless overridden
        
        # Experiment section
        if "experiment" in child_dict and "name" in child_dict["experiment"]:
            assert merged_config.experiment.name == child_dict["experiment"]["name"]
        else:
            assert merged_config.experiment.name == base_config.experiment.name
        
        if "experiment" in child_dict and "seeds" in child_dict["experiment"]:
            assert merged_config.experiment.seeds == child_dict["experiment"]["seeds"]
        else:
            assert merged_config.experiment.seeds == base_config.experiment.seeds
        
        # metrics should always be inherited from base (not in child_overrides)
        assert merged_config.experiment.metrics == base_config.experiment.metrics
        
        # Quantum model section
        if "quantum_model" in child_dict and "n_qubits" in child_dict["quantum_model"]:
            assert merged_config.quantum_model.n_qubits == child_dict["quantum_model"]["n_qubits"]
        else:
            assert merged_config.quantum_model.n_qubits == base_config.quantum_model.n_qubits
        
        if "quantum_model" in child_dict and "n_layers" in child_dict["quantum_model"]:
            assert merged_config.quantum_model.n_layers == child_dict["quantum_model"]["n_layers"]
        else:
            assert merged_config.quantum_model.n_layers == base_config.quantum_model.n_layers
        
        # arch should always be inherited from base (not in child_overrides)
        assert merged_config.quantum_model.arch == base_config.quantum_model.arch
        assert merged_config.quantum_model.evolution_steps == base_config.quantum_model.evolution_steps
        assert merged_config.quantum_model.encoding == base_config.quantum_model.encoding
        assert merged_config.quantum_model.observables == base_config.quantum_model.observables
        
        # Backend section
        if "backend" in child_dict and "device" in child_dict["backend"]:
            assert merged_config.backend.device == child_dict["backend"]["device"]
        else:
            assert merged_config.backend.device == base_config.backend.device
        
        # name and shots should always be inherited from base (not in child_overrides)
        assert merged_config.backend.name == base_config.backend.name
        assert merged_config.backend.shots == base_config.backend.shots
        
        # Data section
        if "data" in child_dict and "window_size" in child_dict["data"]:
            assert merged_config.data.window_size == child_dict["data"]["window_size"]
        else:
            assert merged_config.data.window_size == base_config.data.window_size
        
        # Other data fields should always be inherited from base (not in child_overrides)
        assert merged_config.data.dataset == base_config.data.dataset
        assert merged_config.data.resolution == base_config.data.resolution
        assert merged_config.data.forecast_horizon == base_config.data.forecast_horizon
        assert abs(merged_config.data.train_ratio - base_config.data.train_ratio) < 1e-10
        assert abs(merged_config.data.val_ratio - base_config.data.val_ratio) < 1e-10
        assert abs(merged_config.data.test_ratio - base_config.data.test_ratio) < 1e-10


# Feature: phase1-foundation-setup, Property 11: Invalid configuration raises ConfigError
@given(
    section=st.sampled_from(["experiment", "quantum_model", "backend", "data"]),
    error_type=st.sampled_from(["missing_required", "unknown_field"]),
    base_config=valid_qrc_configs(),
)
@settings(max_examples=100, deadline=None)
def test_property_invalid_config_raises_error(section, error_type, base_config):
    """Property 11: Invalid configuration raises ConfigError.
    
    **Validates: Requirements 8.3, 8.4**
    
    For any YAML config that is missing a required field or contains an unknown
    field, the Config_System should raise a ConfigError with a descriptive message.
    """
    # Start with a valid config dict
    config_dict = asdict(base_config)
    
    if error_type == "missing_required":
        # Remove a required field from the selected section
        if section == "experiment":
            # 'name' and 'seeds' are required fields
            del config_dict["experiment"]["name"]
            expected_pattern = "Missing required field.*name"
        elif section == "quantum_model":
            # 'arch' and 'n_qubits' are required fields
            del config_dict["quantum_model"]["arch"]
            expected_pattern = "Missing required field.*arch"
        elif section == "backend":
            # All fields have defaults, so we need to test with None
            # Instead, we'll test missing section entirely
            del config_dict["backend"]
            expected_pattern = "Missing required section"
        elif section == "data":
            # All fields have defaults, so we need to test with None
            # Instead, we'll test missing section entirely
            del config_dict["data"]
            expected_pattern = "Missing required section"
    
    elif error_type == "unknown_field":
        # Add an unknown field to the selected section
        unknown_field_name = "unknown_field_xyz_123"
        config_dict[section][unknown_field_name] = "invalid_value"
        expected_pattern = f"Unknown configuration field.*{unknown_field_name}"
    
    # Write invalid config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        config_path = f.name
    
    try:
        # Attempt to load config - should raise ConfigError
        with pytest.raises(ConfigError, match=expected_pattern):
            load_config(config_path)
    finally:
        Path(config_path).unlink()


@given(
    unknown_top_level=st.text(
        min_size=1,
        max_size=30,
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_')
    ).filter(lambda x: x not in {"experiment", "quantum_model", "backend", "data", "extends"}),
    base_config=valid_qrc_configs(),
)
@settings(max_examples=100, deadline=None)
def test_property_unknown_top_level_field_raises_error(unknown_top_level, base_config):
    """Property 11 (variant): Unknown top-level field raises ConfigError.
    
    **Validates: Requirements 8.3, 8.4**
    
    For any YAML config that contains an unknown top-level field, the Config_System
    should raise a ConfigError with a descriptive message.
    """
    # Start with a valid config dict
    config_dict = asdict(base_config)
    
    # Add unknown top-level field
    config_dict[unknown_top_level] = {"some": "value"}
    
    # Write invalid config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        config_path = f.name
    
    try:
        # Attempt to load config - should raise ConfigError
        with pytest.raises(ConfigError, match=f"Unknown top-level configuration field.*{unknown_top_level}"):
            load_config(config_path)
    finally:
        Path(config_path).unlink()
