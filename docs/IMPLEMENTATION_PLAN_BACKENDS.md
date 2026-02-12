# Implementation Plan: Qiskit and CUDA-Quantum Backends

**Branch:** `feature/qiskit-cudaq-backends`  
**Author:** Qubit  
**Date:** 2026-02-08  
**Estimated Effort:** 3-4 days

---

## Overview

Implement two additional quantum backends to complement the existing PennyLane backend:

1. **QiskitBackend** — Qiskit Aer (local simulation) + IBM Runtime (hardware)
2. **CUDAQuantumBackend** — NVIDIA CUDA-Quantum for GPU-accelerated simulation

Both must implement the `QuantumBackend` abstract interface defined in `src/qrc_ev/backends/base.py`.

---

## Interface Contract

Each backend must implement these 5 methods:

```python
class QuantumBackend(ABC):
    def create_circuit(self, n_qubits: int) -> Any
    def apply_encoding(self, circuit: Any, data: np.ndarray, strategy: str = "angle") -> Any
    def apply_reservoir(self, circuit: Any, params: ReservoirParams) -> Any
    def measure_observables(self, circuit: Any, obs_set: str = "pauli_z") -> Any
    def execute(self, circuit: Any, shots: int = 0) -> Any
```

---

## Part 1: QiskitBackend

### 1.1 File Structure

```
src/qrc_ev/backends/
├── qiskit_backend.py      # NEW
└── ...

tests/test_backends/
├── test_qiskit.py         # NEW
└── ...
```

### 1.2 Dependencies

Add to `pyproject.toml`:
```toml
"qiskit>=1.0",
"qiskit-aer>=0.14",
"qiskit-ibm-runtime>=0.20",  # Optional, for hardware
```

### 1.3 Class Design

```python
class QiskitBackend(QuantumBackend):
    """Qiskit implementation supporting Aer simulators and IBM hardware."""
    
    def __init__(
        self,
        device_name: str = "aer_simulator",  # or "ibm_torino", etc.
        shots: int | None = None,            # None = statevector
        optimization_level: int = 1,         # Transpilation level
        ibm_channel: str | None = None,      # "ibm_quantum" or "ibm_cloud"
    ):
        ...
```

### 1.4 Implementation Details

#### `create_circuit(n_qubits)`
- Return a `QuantumCircuit(n_qubits, n_qubits)` (include classical registers for measurement)
- Store `n_qubits` internally

#### `apply_encoding(circuit, data, strategy="angle")`
- **angle:** `circuit.ry(np.pi * x[i], i)` for each feature
- **amplitude:** Use `initialize()` with L2-normalized amplitudes (Phase 2)
- **iqp:** Hadamard + diagonal Rzz encoding (Phase 2)
- Return the modified circuit

#### `apply_reservoir(circuit, params)`
- For each layer in `params.n_layers`:
  1. Single-qubit Rz: `circuit.rz(params.rotation_angles[l, q], q)`
  2. Two-qubit coupling: `circuit.cx(i, j)` + `circuit.rz(params.coupling_strengths[l, i, j], j)`
- Skip couplings where strength ≈ 0

#### `measure_observables(circuit, obs_set="pauli_z")`
- **Statevector mode (shots=None):**
  - Use `Statevector.from_instruction(circuit)`
  - Compute `⟨Z_i⟩ = statevector.expectation_value(Pauli("Z"), [i])`
- **Shot-based mode:**
  - Add measurement gates: `circuit.measure_all()`
  - Execute with Aer, parse counts → estimate `⟨Z_i⟩ = (n_0 - n_1) / total`

#### `execute(circuit, shots)`
- **Aer simulator:**
  ```python
  backend = AerSimulator(method="statevector" if shots is None else "automatic")
  job = backend.run(transpile(circuit, backend), shots=shots or 1)
  return job.result()
  ```
- **IBM Runtime (hardware):**
  ```python
  service = QiskitRuntimeService(channel=self.ibm_channel)
  backend = service.backend(self.device_name)
  sampler = Sampler(backend)
  job = sampler.run([circuit], shots=shots)
  return job.result()
  ```

### 1.5 Test Cases (`test_qiskit.py`)

Mirror `test_pennylane.py` structure:

| Test | Description |
|------|-------------|
| `test_initialization_default` | Default params (aer_simulator, shots=None) |
| `test_initialization_custom` | Custom device, shots, optimization_level |
| `test_create_circuit` | Verify QuantumCircuit created with correct qubits |
| `test_apply_encoding_angle` | Verify Ry gates applied, check Z expectations |
| `test_apply_encoding_unsupported` | ValueError for unknown strategy |
| `test_apply_encoding_oversized` | ValueError when len(data) > n_qubits |
| `test_apply_reservoir_single_layer` | Verify Rz + CNOT-Rz gates |
| `test_apply_reservoir_multiple_layers` | Multiple layers produce valid outputs |
| `test_measure_observables_statevector` | Exact ⟨Z⟩ in statevector mode |
| `test_measure_observables_shots` | Shot-based ⟨Z⟩ within tolerance |
| `test_full_circuit_workflow` | encode → reservoir → measure end-to-end |
| `test_reproducibility` | Same params → identical results |
| `test_cross_backend_consistency` | Compare Qiskit vs PennyLane on same circuit |

### 1.6 Edge Cases

- **Zero coupling strengths:** Skip CNOT-Rz to reduce gate count
- **Empty data array:** No encoding gates, just reservoir + measure
- **IBM hardware errors:** Graceful fallback or clear error messages
- **Transpilation:** Use `optimization_level=3` for hardware, 1 for simulation

---

## Part 2: CUDAQuantumBackend

### 2.1 File Structure

```
src/qrc_ev/backends/
├── cudaq_backend.py       # NEW
└── ...

tests/test_backends/
├── test_cudaq.py          # NEW
└── ...
```

### 2.2 Dependencies

CUDA-Quantum requires separate installation (not pip-installable in standard way):
```bash
# Conda or system install
pip install cuda-quantum  # Requires CUDA 12.x toolkit
```

Add optional dependency group:
```toml
[project.optional-dependencies]
cudaq = ["cuda-quantum>=0.9"]
```

### 2.3 Class Design

```python
class CUDAQuantumBackend(QuantumBackend):
    """CUDA-Quantum implementation for GPU-accelerated simulation."""
    
    def __init__(
        self,
        target: str = "nvidia",           # "nvidia", "nvidia-mgpu", "qpp-cpu"
        shots: int | None = None,         # None = statevector
    ):
        ...
```

### 2.4 Implementation Details

CUDA-Quantum uses a different paradigm — kernel functions rather than circuit objects.

#### `create_circuit(n_qubits)`
- Store n_qubits
- Set target: `cudaq.set_target(self.target)`
- Return a placeholder (CUDA-Q builds circuits inside kernels)

#### `apply_encoding(circuit, data, strategy="angle")`
- Store data for use in kernel
- Return data reference (actual gates applied in kernel)

#### `apply_reservoir(circuit, params)`
- Store params for use in kernel
- Return params reference

#### `measure_observables(circuit, obs_set="pauli_z")` + `execute()`
- Build and execute kernel:

```python
@cudaq.kernel
def reservoir_kernel(data: list[float], rotation_angles: list[float], 
                     coupling_strengths: list[float], n_qubits: int, n_layers: int):
    qubits = cudaq.qvector(n_qubits)
    
    # Angle encoding
    for i, x in enumerate(data):
        ry(np.pi * x, qubits[i])
    
    # Reservoir layers
    idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            rz(rotation_angles[layer * n_qubits + q], qubits[q])
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                strength = coupling_strengths[idx]
                if abs(strength) > 1e-10:
                    cx(qubits[i], qubits[j])
                    rz(strength, qubits[j])
                idx += 1

# Execute
if shots is None:
    # Statevector mode
    state = cudaq.get_state(reservoir_kernel, *args)
    expectations = [cudaq.spin.z(i).expectation(state) for i in range(n_qubits)]
else:
    # Shot-based
    results = cudaq.sample(reservoir_kernel, *args, shots_count=shots)
    expectations = compute_expectations_from_counts(results, n_qubits)
```

### 2.5 Test Cases (`test_cudaq.py`)

| Test | Description |
|------|-------------|
| `test_initialization` | Verify target set correctly |
| `test_create_circuit` | Placeholder returned, target configured |
| `test_apply_encoding_angle` | Data stored correctly |
| `test_apply_reservoir` | Params stored correctly |
| `test_measure_statevector` | GPU statevector expectations |
| `test_measure_shots` | Shot-based sampling |
| `test_full_workflow` | End-to-end on GPU |
| `test_cross_backend_consistency` | Compare vs PennyLane |
| `test_multi_gpu` | Test `nvidia-mgpu` target (if available) |

### 2.6 GPU Considerations

- **Fallback:** If CUDA not available, raise clear ImportError with instructions
- **Memory:** Large qubit counts may OOM — add warning for n_qubits > 28
- **Target detection:** Auto-detect available targets at init

```python
def _detect_target(self) -> str:
    """Auto-detect best available CUDA-Q target."""
    try:
        cudaq.set_target("nvidia")
        return "nvidia"
    except:
        try:
            cudaq.set_target("qpp-cpu")
            return "qpp-cpu"
        except:
            raise RuntimeError("No CUDA-Quantum targets available")
```

---

## Part 3: Integration

### 3.1 Update `__init__.py`

```python
from qrc_ev.backends.base import QuantumBackend, QuantumReservoir, ReservoirParams
from qrc_ev.backends.pennylane_backend import PennyLaneBackend
from qrc_ev.backends.qiskit_backend import QiskitBackend

# Conditional import for CUDA-Q (requires GPU)
try:
    from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
    _CUDAQ_AVAILABLE = True
except ImportError:
    _CUDAQ_AVAILABLE = False

__all__ = [
    "QuantumBackend",
    "QuantumReservoir", 
    "ReservoirParams",
    "PennyLaneBackend",
    "QiskitBackend",
]

if _CUDAQ_AVAILABLE:
    __all__.append("CUDAQuantumBackend")
```

### 3.2 Update Backend Factory in `trainer.py`

```python
def _create_backend(config) -> QuantumBackend:
    """Create quantum backend from configuration."""
    if config.backend.name == "pennylane":
        return PennyLaneBackend(
            device_name=config.backend.device,
            shots=config.backend.shots,
        )
    elif config.backend.name == "qiskit":
        return QiskitBackend(
            device_name=config.backend.device,
            shots=config.backend.shots,
            optimization_level=config.backend.get("optimization_level", 1),
        )
    elif config.backend.name == "cudaq":
        return CUDAQuantumBackend(
            target=config.backend.device,
            shots=config.backend.shots,
        )
    else:
        raise ValueError(f"Unknown backend: {config.backend.name}")
```

### 3.3 Update `check_backends.py`

```python
def check_all_backends() -> dict[str, bool]:
    """Check availability of all quantum backends."""
    results = {}
    
    # PennyLane
    try:
        import pennylane as qml
        dev = qml.device("default.qubit", wires=2)
        results["pennylane"] = True
        results["pennylane_lightning"] = "lightning.qubit" in qml.devices
    except:
        results["pennylane"] = False
    
    # Qiskit
    try:
        from qiskit_aer import AerSimulator
        sim = AerSimulator()
        results["qiskit_aer"] = True
    except:
        results["qiskit_aer"] = False
    
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        results["qiskit_ibm"] = True
    except:
        results["qiskit_ibm"] = False
    
    # CUDA-Quantum
    try:
        import cudaq
        cudaq.set_target("nvidia")
        results["cudaq_nvidia"] = True
    except:
        results["cudaq_nvidia"] = False
    
    try:
        import cudaq
        cudaq.set_target("qpp-cpu")
        results["cudaq_cpu"] = True
    except:
        results["cudaq_cpu"] = False
    
    return results
```

### 3.4 Cross-Backend Consistency Test

Add to `tests/test_integration/test_cross_backend.py`:

```python
"""Cross-backend consistency tests.

Verify that all backends produce identical (or statistically equivalent) 
results for the same circuit configuration.
"""

@pytest.mark.integration
def test_pennylane_vs_qiskit_statevector():
    """Test that PennyLane and Qiskit produce identical statevector results."""
    # Same params, same data
    # Assert np.allclose(pl_result, qiskit_result, atol=1e-6)

@pytest.mark.integration  
@pytest.mark.skipif(not _CUDAQ_AVAILABLE, reason="CUDA-Q not installed")
def test_pennylane_vs_cudaq_statevector():
    """Test that PennyLane and CUDA-Q produce identical statevector results."""
    # Same params, same data
    # Assert np.allclose(pl_result, cudaq_result, atol=1e-6)

@pytest.mark.integration
def test_all_backends_shot_based():
    """Test that shot-based results are statistically consistent across backends."""
    # Use enough shots (10000) that results should be close
    # Assert within 3-sigma of each other
```

---

## Part 4: Task Breakdown

### Phase A: Qiskit Backend (2 days)

| Task | Est. | Priority |
|------|------|----------|
| A1. Create `qiskit_backend.py` with class skeleton | 1h | P0 |
| A2. Implement `create_circuit()` | 30m | P0 |
| A3. Implement `apply_encoding()` (angle only) | 1h | P0 |
| A4. Implement `apply_reservoir()` | 1.5h | P0 |
| A5. Implement `measure_observables()` statevector | 1.5h | P0 |
| A6. Implement `measure_observables()` shot-based | 1h | P0 |
| A7. Implement `execute()` for Aer | 1h | P0 |
| A8. Write unit tests (mirror PennyLane tests) | 2h | P0 |
| A9. Add IBM Runtime support (optional) | 2h | P1 |
| A10. Integration test vs PennyLane | 1h | P0 |

### Phase B: CUDA-Quantum Backend (1.5 days)

| Task | Est. | Priority |
|------|------|----------|
| B1. Create `cudaq_backend.py` with class skeleton | 1h | P0 |
| B2. Implement kernel-based circuit execution | 2h | P0 |
| B3. Implement statevector expectations | 1.5h | P0 |
| B4. Implement shot-based sampling | 1h | P0 |
| B5. Add target auto-detection | 30m | P1 |
| B6. Write unit tests | 2h | P0 |
| B7. Integration test vs PennyLane | 1h | P0 |
| B8. Multi-GPU test (if hardware available) | 1h | P2 |

### Phase C: Integration (0.5 days)

| Task | Est. | Priority |
|------|------|----------|
| C1. Update `__init__.py` exports | 15m | P0 |
| C2. Update trainer.py backend factory | 30m | P0 |
| C3. Update `check_backends.py` | 30m | P0 |
| C4. Add cross-backend consistency tests | 1h | P0 |
| C5. Update README with backend instructions | 30m | P1 |
| C6. Update pyproject.toml dependencies | 15m | P0 |

---

## Acceptance Criteria

### Must Have (P0)
- [ ] `QiskitBackend` passes all unit tests
- [ ] `CUDAQuantumBackend` passes all unit tests (on GPU machine)
- [ ] Cross-backend statevector results match within `atol=1e-6`
- [ ] `run_pipeline()` works with `backend.name = "qiskit"`
- [ ] `python -m qrc_ev.utils.check_backends` reports all available backends

### Should Have (P1)
- [ ] IBM Runtime integration tested (requires account)
- [ ] CUDA-Q target auto-detection works
- [ ] Documentation updated

### Nice to Have (P2)
- [ ] Multi-GPU support tested
- [ ] Benchmark: PennyLane vs Qiskit vs CUDA-Q execution times
- [ ] Amplitude and IQP encoding support

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| CUDA-Q installation issues | High | Document conda/system install; add CPU fallback |
| Qiskit API changes (v1.0 migration) | Medium | Pin to `qiskit>=1.0,<2.0` |
| Cross-backend numerical differences | Medium | Use `atol=1e-6` tolerance; document expected precision |
| IBM Runtime quota limits | Low | Use Aer for tests; hardware is Phase 5 |
| GPU not available in CI | Medium | Mark GPU tests with `@pytest.mark.skipif`; use CPU target |

---

## Next Steps

1. **Start with Qiskit** — more familiar API, no GPU dependency
2. **Test on synthetic data** — verify `run_pipeline()` works
3. **Add CUDA-Q** — requires GPU access for full testing
4. **Cross-validate** — ensure all backends produce consistent results

Ready to implement?
