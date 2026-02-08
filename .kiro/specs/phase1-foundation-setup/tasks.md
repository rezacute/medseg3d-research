# Implementation Plan: Phase 1 Foundation Setup

## Overview

Incremental implementation of the QRC-EV foundation: project scaffolding → backend abstraction → PennyLane backend → encoding → reservoir → observables → readout → config → seed → data pipeline → factory → integration. Each task builds on the previous, with tests wired in alongside implementation.

## Tasks

- [x] 1. Project scaffolding and package structure
  - [x] 1.1 Create `pyproject.toml` with package metadata, Python ≥3.10, dependencies (pennylane, numpy, scipy, scikit-learn, pyyaml, hypothesis), dev dependencies (pytest, pytest-cov, mypy, black, isort), and pytest configuration
    - _Requirements: 1.1, 11.1_
  - [x] 1.2 Create `src/qrc_ev/__init__.py` with `__version__` attribute and subpackage `__init__.py` files for `backends`, `encoding`, `reservoirs`, `readout`, `baselines`, `training`, `utils`, and `data`
    - _Requirements: 1.2, 1.4_
  - [x] 1.3 Create `tests/conftest.py` with shared fixtures for backend instances, sample configs, and seed values
    - _Requirements: 11.1_

- [x] 2. Backend abstraction layer
  - [x] 2.1 Create `src/qrc_ev/backends/base.py` with `ReservoirParams` dataclass, `QuantumBackend` ABC (abstract methods: `create_circuit`, `apply_encoding`, `apply_reservoir`, `measure_observables`, `execute`), and `QuantumReservoir` ABC (abstract methods: `encode`, `evolve`, `measure`, `process`, `reset`)
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  - [x] 2.2 Write unit tests in `tests/test_backends/test_base.py` verifying ABC contract: direct instantiation raises `TypeError`, and calling abstract methods on a minimal subclass raises `NotImplementedError`
    - _Requirements: 2.3_

- [x] 3. PennyLane backend implementation
  - [x] 3.1 Create `src/qrc_ev/backends/pennylane_backend.py` implementing `PennyLaneBackend(QuantumBackend)` with `create_circuit` (device init), `apply_encoding` (delegates to angle encoder), `apply_reservoir` (Ising unitary with Rz + CNOT+Rz couplings), `measure_observables` (Pauli-Z expectations), and `execute` (statevector or shot-based)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_
  - [ ]* 3.2 Write property test for observable output dimension
    - **Property 3: Observable output dimension matches qubit count**
    - **Validates: Requirements 3.5, 6.1**
  - [ ]* 3.3 Write property test for Pauli-Z value bounds
    - **Property 4: Pauli-Z values bounded in [-1, 1]**
    - **Validates: Requirements 6.2**
  - [x] 3.4 Create `src/qrc_ev/utils/check_backends.py` backend verification script that confirms PennyLane is installed and can execute a simple circuit
    - _Requirements: 11.2_

- [ ] 4. Angle encoding
  - [ ] 4.1 Create `src/qrc_ev/encoding/angle.py` with `angle_encode(data, n_qubits)` function that applies `Ry(π × xᵢ)` to qubit i, raises `ValueError` when `d > n_qubits`, and leaves unused qubits in |0⟩
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  - [ ]* 4.2 Write property test for angle encoding correctness
    - **Property 1: Angle encoding produces correct quantum state**
    - **Validates: Requirements 4.1, 4.3, 4.4**
  - [ ]* 4.3 Write property test for angle encoding rejection of oversized input
    - **Property 2: Angle encoding rejects oversized input**
    - **Validates: Requirements 4.2**

- [ ] 5. Pauli-Z observables and ridge readout
  - [ ] 5.1 Create `src/qrc_ev/readout/observables.py` with `pauli_z_observables(n_qubits)` returning PennyLane observable list for single-qubit Z expectations
    - _Requirements: 6.1, 6.2, 6.3_
  - [ ] 5.2 Create `src/qrc_ev/readout/ridge.py` with `RidgeReadout` class: `__init__(alpha)`, `fit(features, targets)` using closed-form `(XᵀX + αI)⁻¹Xᵀy`, `predict(features)` returning `X @ W`, with `ValueError` on dimension mismatch and `RuntimeError` on predict-before-fit
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  - [ ]* 5.3 Write property test for ridge regression correctness
    - **Property 7: Ridge regression solves regularized least squares**
    - **Validates: Requirements 7.1, 7.2**
  - [ ]* 5.4 Write property test for ridge dimension mismatch rejection
    - **Property 8: Ridge regression rejects mismatched dimensions**
    - **Validates: Requirements 7.4**

- [ ] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. A1 Standard reservoir
  - [ ] 7.1 Create `src/qrc_ev/reservoirs/standard.py` with `StandardReservoir(QuantumReservoir)`: `__init__` generates fixed random Ising params from seed, `encode` delegates to angle encoder, `evolve` applies reservoir unitary N steps, `measure` extracts Pauli-Z observables, `process(time_series)` loops encode→evolve→measure per timestep returning (T, n_qubits), `reset` restores |0⟩⊗ⁿ
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_
  - [ ]* 7.2 Write property test for reservoir output shape
    - **Property 5: Reservoir process output shape invariant**
    - **Validates: Requirements 5.6**
  - [ ]* 7.3 Write property test for reservoir reset
    - **Property 6: Reservoir reset restores initial state**
    - **Validates: Requirements 5.7**

- [ ] 8. Configuration system
  - [ ] 8.1 Create `src/qrc_ev/utils/config.py` with dataclasses (`ExperimentConfig`, `QuantumModelConfig`, `BackendConfig`, `DataConfig`, `QRCConfig`), custom `ConfigError` exception, `load_config(path)` with YAML parsing and `extends` inheritance support, `dump_config(config)` for serialization, and validation for required/unknown fields
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_
  - [ ]* 8.2 Write property test for configuration round-trip
    - **Property 9: Configuration round-trip**
    - **Validates: Requirements 8.7**
  - [ ]* 8.3 Write property test for configuration inheritance
    - **Property 10: Configuration inheritance merge**
    - **Validates: Requirements 8.2**
  - [ ]* 8.4 Write property test for invalid configuration rejection
    - **Property 11: Invalid configuration raises ConfigError**
    - **Validates: Requirements 8.3, 8.4**

- [ ] 9. Seed management
  - [ ] 9.1 Create `src/qrc_ev/utils/seed.py` with `SeedManager`: `__init__(global_seed)` auto-generates if None and logs, `seed_all()` seeds Python random + NumPy, `derive_seed(component)` uses SHA-256 hash for deterministic child seeds
    - _Requirements: 9.1, 9.2, 9.3, 9.4_
  - [ ]* 9.2 Write property test for seed reproducibility
    - **Property 12: Seed reproducibility — reservoir outputs**
    - **Validates: Requirements 9.2, 15.2**
  - [ ]* 9.3 Write property test for seed derivation distinctness
    - **Property 13: Seed derivation produces distinct child seeds**
    - **Validates: Requirements 9.3**

- [ ] 10. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. Data preprocessing pipeline
  - [ ] 11.1 Create `src/qrc_ev/data/preprocessor.py` with `Preprocessor` class: `aggregate_sessions` (session→time bins), `handle_missing` (forward-fill + gap detection), `clip_outliers` (±3σ clipping), `split_chronological` (train/val/test by ratio), `fit_normalize` (min-max on train only), `normalize` (apply + clip to [0,1]), `create_windows` (sliding window X,y pairs)
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8_
  - [ ]* 11.2 Write property test for outlier clipping invariant
    - **Property 14: Outlier clipping invariant**
    - **Validates: Requirements 12.3**
  - [ ]* 11.3 Write property test for chronological split
    - **Property 15: Chronological split preserves order and ratios**
    - **Validates: Requirements 12.4**
  - [ ]* 11.4 Write property test for normalization output range
    - **Property 16: Normalization output range invariant**
    - **Validates: Requirements 12.5, 12.6, 12.8**
  - [ ]* 11.5 Write property test for windowed sample shapes
    - **Property 17: Windowed sample shapes**
    - **Validates: Requirements 12.7**

- [x] 12. Feature engineering
  - [x] 12.1 Create `src/qrc_ev/data/feature_engineer.py` with `FeatureEngineer` class: `add_temporal_features` (sin/cos hour + day-of-week), `add_lag_features` (configurable lag steps), `engineer` (full pipeline returning (T, d)), `feature_dim` property
    - _Requirements: 13.1, 13.2, 13.3, 13.4_
  - [x]* 12.2 Write property test for temporal feature bounds
    - **Property 18: Temporal features bounded in [-1, 1]**
    - **Validates: Requirements 13.1**
  - [x]* 12.3 Write property test for lag feature correctness
    - **Property 19: Lag feature correctness**
    - **Validates: Requirements 13.2**
  - [x]* 12.4 Write property test for feature dimension consistency
    - **Property 20: Feature dimension consistency**
    - **Validates: Requirements 13.3, 13.4**

- [ ] 13. Synthetic data generation
  - [ ] 13.1 Create `src/qrc_ev/data/synthetic.py` with `SyntheticGenerator` class: `sinusoidal` (configurable amplitude/frequency/noise/length), `ev_charging_pattern` (daily morning+evening peaks, weekday/weekend variation), both returning (features (T,d), targets (T,))
    - _Requirements: 14.1, 14.2, 14.3, 14.4_
  - [ ]* 13.2 Write property test for synthetic data shape
    - **Property 21: Synthetic data shape and format**
    - **Validates: Requirements 14.1, 14.4**
  - [ ]* 13.3 Write property test for synthetic data reproducibility
    - **Property 22: Synthetic data reproducibility**
    - **Validates: Requirements 14.3**

- [ ] 14. Architecture factory
  - [ ] 14.1 Create `src/qrc_ev/reservoirs/factory.py` with `_REGISTRY` dict mapping `"standard"` → `StandardReservoir`, and `create_reservoir(arch, backend, **kwargs)` that looks up the registry, raises `ValueError` for unknown names, and passes config params to constructor
    - _Requirements: 10.1, 10.2, 10.3, 10.4_
  - [ ]* 14.2 Write property test for factory rejection of unknown architectures
    - **Property 23: Factory rejects unknown architectures**
    - **Validates: Requirements 10.2**

- [ ] 15. End-to-end pipeline integration
  - [ ] 15.1 Create `src/qrc_ev/training/trainer.py` with a `run_pipeline(config_path)` function that orchestrates: load config → seed all → create synthetic data → preprocess → create backend → create reservoir via factory → process features → fit ridge readout → predict → return predictions and metrics
    - _Requirements: 15.1, 15.2, 15.3_
  - [ ]* 15.2 Write integration test in `tests/test_integration/test_pipeline.py` verifying end-to-end pipeline runs with a sample YAML config, produces predictions, achieves R² > 0.0 on synthetic data, and is reproducible across two runs with the same seed
    - _Requirements: 15.1, 15.2, 15.3_

- [ ] 16. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests use Hypothesis with `max_examples=100`
- PennyLane is the only backend implemented in this phase; the abstraction layer is ready for Qiskit and CUDA-Q in later phases
