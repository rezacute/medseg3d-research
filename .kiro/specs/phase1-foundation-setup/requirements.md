# Requirements Document

## Introduction

Phase 1 Foundation Setup establishes the core infrastructure for the QRC-EV framework. This phase delivers a minimal end-to-end pipeline: YAML configuration loading → data preprocessing → data encoding → quantum reservoir processing → observable extraction → ridge regression prediction. It includes a data preprocessing pipeline with feature engineering and synthetic data generation for testing. PennyLane serves as the primary backend, with the abstraction layer designed for future Qiskit and CUDA Quantum integration. The naive quantum implementation uses A1 Standard Gate-Based QRC with angle encoding and Pauli-Z observables.

## Glossary

- **QRC_Framework**: The top-level Python package (`qrc_ev`) providing quantum reservoir computing for EV charging demand forecasting
- **Backend_Abstraction_Layer**: The set of abstract base classes (`QuantumBackend`, `QuantumReservoir`) that decouple quantum circuit logic from specific simulator libraries
- **PennyLane_Backend**: The concrete implementation of `QuantumBackend` using PennyLane's `default.qubit` and `lightning.qubit` devices
- **A1_Reservoir**: The Standard Gate-Based QRC architecture using a fixed random Ising unitary for reservoir evolution
- **Angle_Encoder**: The encoding strategy that maps each input feature to a single-qubit Ry rotation scaled to [0, π]
- **Pauli_Z_Observables**: The measurement strategy extracting single-qubit ⟨Zᵢ⟩ expectation values from the reservoir state
- **Ridge_Readout**: The classical readout layer using Tikhonov-regularized linear regression with closed-form solution
- **Config_System**: The YAML-based configuration loader that drives all experiment parameters
- **Seed_Manager**: The reproducibility framework that controls random number generation across NumPy, Python, and quantum backends
- **Architecture_Factory**: The factory function that instantiates reservoir objects from string identifiers and configuration dictionaries
- **Preprocessor**: The data pipeline component that transforms raw EV charging datasets into normalized, windowed feature arrays ready for quantum encoding
- **Feature_Engineer**: The component that generates temporal encodings (sin/cos) and lagged features from raw time-series data
- **Synthetic_Generator**: The utility that produces synthetic time-series data mimicking EV charging patterns for testing purposes

## Requirements

### Requirement 1: Project Scaffolding

**User Story:** As a developer, I want a well-structured Python package with proper dependency management, so that I can develop and test the QRC-EV framework reliably.

#### Acceptance Criteria

1. THE QRC_Framework SHALL provide a `pyproject.toml` with package metadata, Python ≥3.10 requirement, and dependency specifications for PennyLane, NumPy, SciPy, scikit-learn, PyYAML, and pytest
2. THE QRC_Framework SHALL organize source code under `src/qrc_ev/` with subpackages for `backends`, `encoding`, `reservoirs`, `readout`, `baselines`, `training`, and `utils`
3. THE QRC_Framework SHALL be installable via `pip install -e .` in development mode without errors
4. WHEN the package is imported, THE QRC_Framework SHALL expose a `__version__` attribute containing the current version string

### Requirement 2: Backend Abstraction Layer

**User Story:** As a developer, I want a backend-agnostic interface for quantum circuit operations, so that I can swap between PennyLane, Qiskit, and CUDA Quantum without changing reservoir logic.

#### Acceptance Criteria

1. THE Backend_Abstraction_Layer SHALL define a `QuantumBackend` abstract base class with abstract methods: `create_circuit(n_qubits)`, `apply_encoding(circuit, data, strategy)`, `apply_reservoir(circuit, params)`, `measure_observables(circuit, obs_set)`, and `execute(circuit, shots)`
2. THE Backend_Abstraction_Layer SHALL define a `QuantumReservoir` abstract base class with abstract methods: `encode(x)`, `evolve(steps)`, `measure()`, `process(time_series)`, and `reset()`
3. WHEN a method on `QuantumBackend` or `QuantumReservoir` is called without a concrete implementation, THE Backend_Abstraction_Layer SHALL raise a `NotImplementedError`
4. THE Backend_Abstraction_Layer SHALL define a `ReservoirParams` dataclass containing fields for `n_qubits`, `n_layers`, `coupling_strengths`, `rotation_angles`, and `seed`

### Requirement 3: PennyLane Backend Implementation

**User Story:** As a researcher, I want a working PennyLane backend, so that I can run quantum reservoir simulations using PennyLane's `default.qubit` and `lightning.qubit` devices.

#### Acceptance Criteria

1. THE PennyLane_Backend SHALL implement all abstract methods of `QuantumBackend`
2. WHEN `create_circuit(n_qubits)` is called, THE PennyLane_Backend SHALL initialize a PennyLane device with the configured device name and qubit count
3. WHEN `apply_encoding(circuit, data, "angle")` is called, THE PennyLane_Backend SHALL apply Ry(π × xᵢ) rotations to each qubit where xᵢ is the corresponding element of the input data array
4. WHEN `apply_reservoir(circuit, params)` is called, THE PennyLane_Backend SHALL apply a fixed random Ising-type unitary using the coupling strengths and rotation angles from `params`
5. WHEN `measure_observables(circuit, "pauli_z")` is called, THE PennyLane_Backend SHALL return a NumPy array of ⟨Zᵢ⟩ expectation values with length equal to the number of qubits
6. WHEN `execute(circuit, shots=0)` is called, THE PennyLane_Backend SHALL perform exact statevector simulation
7. WHEN `execute(circuit, shots=N)` is called with N > 0, THE PennyLane_Backend SHALL perform shot-based sampling with N shots
8. THE PennyLane_Backend SHALL support both `default.qubit` and `lightning.qubit` device names via configuration

### Requirement 4: Angle Encoding

**User Story:** As a researcher, I want angle encoding that maps input features to qubit rotations, so that I can prepare quantum states from classical time-series data.

#### Acceptance Criteria

1. WHEN an input vector x of dimension d is provided, THE Angle_Encoder SHALL apply Ry(π × xᵢ) to qubit i for each i in [0, d)
2. WHEN the input dimension d exceeds the number of available qubits n, THE Angle_Encoder SHALL raise a `ValueError` with a descriptive message
3. WHEN the input dimension d is less than the number of qubits n, THE Angle_Encoder SHALL encode d features on the first d qubits and leave the remaining qubits in the |0⟩ state
4. THE Angle_Encoder SHALL accept input values in the range [0, 1] and scale them to [0, π] for the Ry rotation angle

### Requirement 5: A1 Standard Gate-Based QRC

**User Story:** As a researcher, I want the baseline quantum reservoir implementation, so that I can process time-series data through a fixed random quantum circuit and extract features.

#### Acceptance Criteria

1. THE A1_Reservoir SHALL implement all abstract methods of `QuantumReservoir`
2. WHEN initialized with a seed, THE A1_Reservoir SHALL generate fixed random Ising coupling strengths Jᵢⱼ and single-qubit rotation angles θᵢ that remain constant across all subsequent calls
3. WHEN `encode(x)` is called, THE A1_Reservoir SHALL delegate to the Angle_Encoder to prepare the input quantum state
4. WHEN `evolve(steps)` is called, THE A1_Reservoir SHALL apply the fixed reservoir unitary `steps` times in sequence
5. WHEN `measure()` is called, THE A1_Reservoir SHALL delegate to the Pauli_Z_Observables extractor and return a NumPy array of expectation values
6. WHEN `process(time_series)` is called with a 2D array of shape (T, d), THE A1_Reservoir SHALL process each timestep sequentially through encode → evolve → measure and return a 2D feature array of shape (T, n_qubits)
7. WHEN `reset()` is called, THE A1_Reservoir SHALL restore the reservoir to its initial |0⟩⊗ⁿ state without changing the fixed random parameters

### Requirement 6: Pauli-Z Observable Extraction

**User Story:** As a researcher, I want to extract single-qubit Pauli-Z expectation values from the reservoir state, so that I can build a classical feature vector for the readout layer.

#### Acceptance Criteria

1. WHEN called with an n-qubit reservoir state, THE Pauli_Z_Observables extractor SHALL return a NumPy array of n expectation values ⟨Zᵢ⟩
2. THE Pauli_Z_Observables extractor SHALL return values in the range [-1, 1] for each observable
3. WHEN the reservoir is in the |0⟩⊗ⁿ state, THE Pauli_Z_Observables extractor SHALL return values equal to 1.0 for all qubits

### Requirement 7: Ridge Regression Readout

**User Story:** As a researcher, I want a ridge regression readout layer, so that I can map quantum reservoir features to demand predictions using a globally optimal closed-form solution.

#### Acceptance Criteria

1. WHEN `fit(features, targets)` is called, THE Ridge_Readout SHALL compute weights using the closed-form solution W = (XᵀX + αI)⁻¹Xᵀy where α is the regularization parameter
2. WHEN `predict(features)` is called, THE Ridge_Readout SHALL return predictions as features @ W
3. THE Ridge_Readout SHALL accept a configurable regularization parameter α with a default value
4. WHEN `fit` is called with mismatched feature and target dimensions, THE Ridge_Readout SHALL raise a `ValueError`
5. WHEN `predict` is called before `fit`, THE Ridge_Readout SHALL raise a `RuntimeError`

### Requirement 8: YAML Configuration System

**User Story:** As a researcher, I want a YAML-based configuration system, so that I can define experiment parameters declaratively and reproduce experiments exactly.

#### Acceptance Criteria

1. WHEN a YAML configuration file path is provided, THE Config_System SHALL parse the file and return a structured configuration object
2. WHEN a configuration file references a base configuration via an `extends` field, THE Config_System SHALL merge the base configuration with the child configuration, with child values taking precedence
3. WHEN a required configuration field is missing, THE Config_System SHALL raise a `ConfigError` with a message identifying the missing field
4. WHEN an unknown configuration field is present, THE Config_System SHALL raise a `ConfigError` with a message identifying the unrecognized field
5. THE Config_System SHALL support nested configuration sections for `experiment`, `quantum_model`, and `backend`
6. THE Config_System SHALL serialize a configuration object back to a valid YAML string
7. FOR ALL valid configuration objects, serializing to YAML and then parsing the resulting YAML SHALL produce an equivalent configuration object (round-trip property)

### Requirement 9: Seed Management and Reproducibility

**User Story:** As a researcher, I want deterministic seed management, so that I can reproduce identical results given the same seed, backend, and device.

#### Acceptance Criteria

1. WHEN a global seed is set, THE Seed_Manager SHALL seed Python's `random` module, NumPy's random generator, and any backend-specific random state
2. WHEN the same seed is used with the same backend and device, THE A1_Reservoir SHALL produce identical output feature arrays for identical input data
3. THE Seed_Manager SHALL derive child seeds from the global seed for different components (reservoir parameters, data splitting) to avoid seed correlation
4. WHEN no seed is provided, THE Seed_Manager SHALL generate a random seed and log the generated value for later reproduction

### Requirement 10: Architecture Factory

**User Story:** As a developer, I want a factory function that creates reservoir instances from configuration, so that I can instantiate any architecture by name without importing specific classes.

#### Acceptance Criteria

1. WHEN called with `arch="standard"` and a valid configuration, THE Architecture_Factory SHALL return an instance of A1_Reservoir
2. WHEN called with an unrecognized architecture name, THE Architecture_Factory SHALL raise a `ValueError` listing the available architecture names
3. THE Architecture_Factory SHALL pass all configuration parameters from the config object to the reservoir constructor
4. THE Architecture_Factory SHALL accept a `backend` parameter and instantiate the reservoir with the specified backend implementation

### Requirement 11: Test Infrastructure and Backend Verification

**User Story:** As a developer, I want a pytest-based test suite with backend verification, so that I can validate that the framework components work correctly and the PennyLane backend is functional.

#### Acceptance Criteria

1. THE QRC_Framework SHALL include a pytest configuration in `pyproject.toml` with test discovery under `tests/`
2. THE QRC_Framework SHALL include a backend verification script that confirms PennyLane is installed and can execute a simple circuit
3. WHEN all unit tests are executed, THE QRC_Framework SHALL report test results with pass/fail status for each component: backend, encoding, reservoir, readout, config, and seed management
4. THE QRC_Framework SHALL include type annotations on all public functions and classes compatible with mypy strict mode

### Requirement 12: Data Preprocessing Pipeline

**User Story:** As a researcher, I want a preprocessing pipeline that transforms raw EV charging datasets into quantum-ready feature vectors, so that I can feed real-world data into the reservoir.

#### Acceptance Criteria

1. WHEN raw session-level data (ACN-Data or Palo Alto format) is provided, THE Preprocessor SHALL aggregate sessions into fixed-interval time bins (configurable: 15-min or hourly)
2. WHEN time-series data contains missing values, THE Preprocessor SHALL apply forward-fill interpolation and detect gaps exceeding a configurable threshold
3. WHEN time-series data contains outliers beyond ±3 standard deviations, THE Preprocessor SHALL clip the values to the ±3σ boundary
4. WHEN a dataset is provided, THE Preprocessor SHALL split it chronologically into train, validation, and test sets using a configurable ratio (default 70/15/15) without shuffling
5. WHEN normalizing features, THE Preprocessor SHALL fit normalization statistics on the training set only and apply the same transformation to validation and test sets
6. WHEN angle encoding normalization is selected, THE Preprocessor SHALL scale features to the [0, 1] range using min-max normalization fitted on training data
7. WHEN a window size W and forecast horizon h are specified, THE Preprocessor SHALL generate (X, y) sample pairs where X has shape (W, d) and y has shape (h,)
8. IF normalization statistics from the training set are applied to validation or test data and a value falls outside the training range, THEN THE Preprocessor SHALL clip the value to the [0, 1] range

### Requirement 13: Feature Engineering

**User Story:** As a researcher, I want temporal and lagged features engineered from the raw time-series, so that the reservoir receives informative input vectors.

#### Acceptance Criteria

1. WHEN temporal feature engineering is enabled, THE Feature_Engineer SHALL generate sin/cos encodings for hour-of-day and day-of-week
2. WHEN lagged feature generation is enabled, THE Feature_Engineer SHALL create lagged copies of the target variable at configurable lag steps (default: t-1, t-2, t-4, t-12, t-24)
3. WHEN feature engineering is applied, THE Feature_Engineer SHALL return a 2D NumPy array where each row is a timestep and each column is a feature
4. THE Feature_Engineer SHALL report the total feature dimension d so that qubit requirements can be determined before reservoir initialization

### Requirement 14: Synthetic Data Generation

**User Story:** As a developer, I want synthetic data generators that produce realistic EV charging patterns, so that I can test the full pipeline without requiring real datasets.

#### Acceptance Criteria

1. WHEN a synthetic sinusoidal dataset is requested, THE Synthetic_Generator SHALL produce a time-series with configurable amplitude, frequency, noise level, and length
2. WHEN a synthetic EV charging pattern is requested, THE Synthetic_Generator SHALL produce a time-series exhibiting daily periodicity (morning and evening peaks) and weekly periodicity (weekday vs weekend variation)
3. WHEN a seed is provided, THE Synthetic_Generator SHALL produce identical synthetic data across repeated calls
4. THE Synthetic_Generator SHALL return data in the same format as the Preprocessor output: a 2D NumPy array of shape (T, d) for features and a 1D array of shape (T,) for targets

### Requirement 15: End-to-End Pipeline Integration

**User Story:** As a researcher, I want a minimal end-to-end pipeline, so that I can verify the complete flow from configuration loading through data preprocessing and quantum reservoir processing to prediction output.

#### Acceptance Criteria

1. WHEN a valid YAML configuration specifying A1 reservoir, angle encoding, PennyLane backend, and ridge readout is provided, THE QRC_Framework SHALL execute the full pipeline: load config → preprocess data → initialize backend → create reservoir → process input data → fit readout → produce predictions
2. WHEN the end-to-end pipeline is run twice with the same seed and configuration, THE QRC_Framework SHALL produce identical prediction arrays
3. WHEN the end-to-end pipeline is given synthetic EV charging data, THE Ridge_Readout SHALL achieve a non-trivial fit (R² > 0.0) on the training portion
