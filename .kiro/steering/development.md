---
inclusion: auto
---

# Development Workflow

## Current Implementation Status

### Phase 1: Foundation Setup (In Progress)

**Completed:**
- ✅ Task 1: Project scaffolding and package structure
  - Package configuration (pyproject.toml)
  - Source package structure (src/qrc_ev/)
  - Test infrastructure (tests/conftest.py)
  - CI/CD pipeline (.github/workflows/ci.yml)

**Next Tasks:**
- Task 2: Backend abstraction layer
- Task 3: PennyLane backend implementation
- Task 4: Angle encoding
- Task 5: Pauli-Z observables and ridge readout
- Task 6-16: See `.kiro/specs/phase1-foundation-setup/tasks.md`

## Development Guidelines

### Code Quality Standards

1. **Type Annotations**: All public functions and methods must have type hints
2. **Docstrings**: Use Google-style docstrings for all public APIs
3. **Testing**: Write tests alongside implementation (not after)
4. **Property-Based Testing**: Use Hypothesis for universal properties (100 examples minimum)
5. **Code Formatting**: Run `black` and `isort` before committing

### Testing Strategy

#### Unit Tests
- Test specific examples and edge cases
- Co-locate with source files when possible
- Use descriptive test names: `test_<component>_<behavior>_<condition>`

#### Property-Based Tests
- Test universal invariants across all valid inputs
- Annotate with requirement links: `**Validates: Requirements X.Y**`
- Use smart generators that constrain to valid input space
- Document failing counterexamples for debugging

#### Integration Tests
- Test end-to-end workflows
- Verify component interactions
- Use synthetic data for reproducibility

### Git Workflow

1. **Branch Naming**: `<type>/<short-description>`
   - `feat/backend-abstraction`
   - `fix/angle-encoding-bounds`
   - `test/reservoir-properties`

2. **Commit Messages**: Follow conventional commits
   ```
   <type>: <short summary>
   
   <detailed description>
   
   Validates Requirements X.Y, X.Z
   ```

3. **Types**: `feat`, `fix`, `test`, `docs`, `refactor`, `perf`, `ci`

### Implementation Order

Follow the task list in `.kiro/specs/phase1-foundation-setup/tasks.md`:

1. Implement subtasks before parent tasks
2. Write code before running tests
3. Update task status using taskStatus tool
4. Commit after completing each major task
5. Run diagnostics to verify no compilation errors

### Error Handling Patterns

```python
# Custom exceptions for domain-specific errors
class ConfigError(ValueError):
    """Raised when configuration is invalid."""
    pass

# Descriptive error messages with context
if d > n_qubits:
    raise ValueError(
        f"Input dimension {d} exceeds qubit count {n_qubits}"
    )

# Validate inputs early
def fit(self, features: np.ndarray, targets: np.ndarray) -> "RidgeReadout":
    if features.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Feature and target sample counts must match: "
            f"{features.shape[0]} != {targets.shape[0]}"
        )
```

### Documentation Standards

#### Module Docstrings
```python
"""Module description.

This module provides [brief description of purpose].
"""
```

#### Function Docstrings
```python
def angle_encode(data: np.ndarray, n_qubits: int) -> None:
    """Apply Ry(pi * x_i) to qubit i for each feature.

    Args:
        data: Input vector of shape (d,) with values in [0, 1].
        n_qubits: Total number of qubits available.

    Raises:
        ValueError: If d > n_qubits.
        
    Example:
        >>> data = np.array([0.5, 0.3, 0.8])
        >>> angle_encode(data, n_qubits=4)
    """
```

#### Class Docstrings
```python
class RidgeReadout:
    """Ridge regression readout layer.
    
    Implements Tikhonov-regularized linear regression with closed-form
    solution: W = (X^T X + αI)^{-1} X^T y
    
    Attributes:
        alpha: Regularization parameter (default: 1e-4).
        
    Example:
        >>> readout = RidgeReadout(alpha=0.01)
        >>> readout.fit(features, targets)
        >>> predictions = readout.predict(test_features)
    """
```

## Continuous Integration

The CI pipeline runs on every push and PR:

1. **Compile Check**: Verify all Python files compile
2. **Import Check**: Verify package imports correctly
3. **Type Check**: Run mypy on source code
4. **Test Suite**: Run pytest with coverage reporting

### Local CI Verification

Before pushing, run locally:

```bash
# Compile check
python -m compileall src/ tests/ -q

# Import check
python -c "from qrc_ev import __version__; print(__version__)"

# Type check
mypy src/qrc_ev/ --ignore-missing-imports

# Test suite
pytest tests/ -v --cov=src/qrc_ev
```

## Debugging Tips

### PennyLane Circuits
```python
# Print circuit structure
print(qml.draw(circuit)())

# Check device capabilities
print(dev.capabilities())
```

### Hypothesis Tests
```python
# Reproduce failing example
@given(st.integers(min_value=1, max_value=10))
@seed(12345)  # Use seed from failure
def test_property(value):
    ...
```

### NumPy Debugging
```python
# Check for NaN/Inf
assert not np.isnan(result).any()
assert not np.isinf(result).any()

# Verify shapes
print(f"Expected: {expected_shape}, Got: {result.shape}")
```

## Performance Considerations

### Quantum Simulation
- Use `lightning.qubit` for faster CPU simulation
- Limit qubit count to ≤12 for reasonable simulation time
- Use statevector mode (shots=0) for exact results during development

### Data Processing
- Vectorize operations with NumPy instead of loops
- Use pandas for time-series aggregation
- Cache preprocessed data to avoid recomputation

### Testing
- Mark slow tests with `@pytest.mark.slow`
- Use small datasets in unit tests
- Run property tests with `max_examples=100` (not 1000)

## Common Issues and Solutions

### Import Errors
```bash
# Ensure package is installed in editable mode
pip install -e .

# Verify PYTHONPATH includes src/
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Type Checking Failures
```python
# Use TYPE_CHECKING for circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qrc_ev.backends.base import QuantumBackend
```

### Test Fixtures Not Found
```python
# Ensure conftest.py is in tests/ directory
# Use fixture names exactly as defined
def test_example(test_seed, sample_time_series):
    ...
```

## Next Steps

After completing Phase 1 Foundation Setup:

1. **Phase 2**: Advanced architectures (A2-A6)
2. **Phase 3**: Classical baselines (B1-B3)
3. **Phase 4**: Data loaders and real datasets
4. **Phase 5**: Experiment orchestration and benchmarking
5. **Phase 6**: Statistical analysis and visualization
6. **Phase 7**: Hardware validation on IBM Quantum

See `docs/ROADMAP.md` for detailed timeline.
