"""Tests for CUDA-Quantum backend implementation.

These tests verify the CUDAQuantumBackend class implements the QuantumBackend
interface correctly. Tests are skipped if CUDA-Quantum is not available.
"""

import numpy as np
import pytest

from qrc_ev.backends.base import ReservoirParams
from qrc_ev.backends.cudaq_backend import (
    CUDAQuantumBackend,
    get_available_targets,
    is_cudaq_available,
)


# Skip all tests if CUDA-Quantum is not available
pytestmark = pytest.mark.skipif(
    not is_cudaq_available(),
    reason="CUDA-Quantum is not installed"
)


@pytest.fixture
def cudaq_backend():
    """Create a CUDA-Quantum backend with available target."""
    targets = get_available_targets()
    if not targets:
        pytest.skip("No CUDA-Quantum targets available")
    # Prefer nvidia, fall back to qpp-cpu
    target = "nvidia" if "nvidia" in targets else targets[0]
    return CUDAQuantumBackend(target=target, shots=None)


@pytest.fixture
def cudaq_backend_shots():
    """Create a shot-based CUDA-Quantum backend."""
    targets = get_available_targets()
    if not targets:
        pytest.skip("No CUDA-Quantum targets available")
    target = "nvidia" if "nvidia" in targets else targets[0]
    return CUDAQuantumBackend(target=target, shots=1000)


@pytest.fixture
def simple_reservoir_params():
    """Create simple reservoir parameters for testing."""
    n_qubits = 3
    n_layers = 2
    rng = np.random.default_rng(42)
    return ReservoirParams(
        n_qubits=n_qubits,
        n_layers=n_layers,
        coupling_strengths=rng.uniform(-np.pi, np.pi, (n_layers, n_qubits, n_qubits)),
        rotation_angles=rng.uniform(-np.pi, np.pi, (n_layers, n_qubits)),
        seed=42,
    )


class TestCUDAQuantumAvailability:
    """Tests for CUDA-Quantum availability checking."""
    
    def test_is_cudaq_available_returns_bool(self):
        """Test that is_cudaq_available returns a boolean."""
        result = is_cudaq_available()
        assert isinstance(result, bool)
    
    def test_get_available_targets_returns_list(self):
        """Test that get_available_targets returns a list."""
        result = get_available_targets()
        assert isinstance(result, list)
    
    @pytest.mark.skipif(not is_cudaq_available(), reason="CUDA-Q not installed")
    def test_get_available_targets_not_empty(self):
        """Test that at least one target is available when cudaq is installed."""
        targets = get_available_targets()
        assert len(targets) > 0, "Expected at least one CUDA-Q target"


class TestCUDAQuantumBackend:
    """Test suite for CUDAQuantumBackend."""
    
    def test_initialization(self, cudaq_backend):
        """Test backend initialization."""
        assert cudaq_backend.shots is None
        assert cudaq_backend._n_qubits == 0
    
    def test_initialization_with_shots(self, cudaq_backend_shots):
        """Test backend initialization with custom shots."""
        assert cudaq_backend_shots.shots == 1000
    
    def test_invalid_target_raises_error(self):
        """Test that invalid target raises RuntimeError."""
        with pytest.raises(RuntimeError, match="not available"):
            CUDAQuantumBackend(target="invalid_target_xyz")
    
    def test_create_circuit(self, cudaq_backend):
        """Test circuit creation."""
        result = cudaq_backend.create_circuit(n_qubits=4)
        
        assert result is not None
        assert result["n_qubits"] == 4
        assert result["backend"] == "cudaq"
        assert cudaq_backend._n_qubits == 4
    
    def test_apply_encoding_angle(self, cudaq_backend):
        """Test angle encoding storage."""
        cudaq_backend.create_circuit(n_qubits=4)
        
        data = np.array([0.1, 0.5, 0.9])
        result = cudaq_backend.apply_encoding(None, data, strategy="angle")
        
        assert result is None  # Returns the circuit (None in this case)
        assert cudaq_backend._encoded_data is not None
        assert len(cudaq_backend._encoded_data) == 3
    
    def test_apply_encoding_unsupported_strategy(self, cudaq_backend):
        """Test that unsupported encoding strategy raises ValueError."""
        cudaq_backend.create_circuit(n_qubits=4)
        
        with pytest.raises(ValueError, match="Unsupported encoding strategy"):
            cudaq_backend.apply_encoding(None, np.array([0.5]), strategy="amplitude")
    
    def test_apply_encoding_oversized_input(self, cudaq_backend):
        """Test that oversized input raises ValueError."""
        cudaq_backend.create_circuit(n_qubits=2)
        
        with pytest.raises(ValueError, match="exceeds qubit count"):
            cudaq_backend.apply_encoding(None, np.array([0.1, 0.2, 0.3]), strategy="angle")
    
    def test_apply_reservoir(self, cudaq_backend, simple_reservoir_params):
        """Test reservoir parameter storage."""
        cudaq_backend.create_circuit(n_qubits=3)
        
        result = cudaq_backend.apply_reservoir(None, simple_reservoir_params)
        
        assert cudaq_backend._reservoir_params is not None
        assert cudaq_backend._reservoir_params.n_qubits == 3
    
    def test_measure_observables_without_params_raises(self, cudaq_backend):
        """Test that measure without reservoir params raises RuntimeError."""
        cudaq_backend.create_circuit(n_qubits=3)
        cudaq_backend.apply_encoding(None, np.array([0.5, 0.5, 0.5]), "angle")
        
        with pytest.raises(RuntimeError, match="Reservoir parameters not set"):
            cudaq_backend.measure_observables(None, obs_set="pauli_z")
    
    def test_measure_observables_unsupported_obs_set(self, cudaq_backend, simple_reservoir_params):
        """Test that unsupported observable set raises ValueError."""
        cudaq_backend.create_circuit(n_qubits=3)
        cudaq_backend.apply_encoding(None, np.array([0.5, 0.5, 0.5]), "angle")
        cudaq_backend.apply_reservoir(None, simple_reservoir_params)
        
        with pytest.raises(ValueError, match="Unsupported observable set"):
            cudaq_backend.measure_observables(None, obs_set="pauli_x")
    
    def test_measure_statevector_mode(self, cudaq_backend, simple_reservoir_params):
        """Test measurement in statevector mode."""
        cudaq_backend.create_circuit(n_qubits=3)
        cudaq_backend.apply_encoding(None, np.array([0.0, 0.5, 1.0]), "angle")
        cudaq_backend.apply_reservoir(None, simple_reservoir_params)
        
        result = cudaq_backend.measure_observables(None, obs_set="pauli_z")
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert all(-1.0 <= val <= 1.0 for val in result)
    
    def test_measure_shot_based_mode(self, cudaq_backend_shots, simple_reservoir_params):
        """Test measurement in shot-based mode."""
        cudaq_backend_shots.create_circuit(n_qubits=3)
        cudaq_backend_shots.apply_encoding(None, np.array([0.0, 0.5, 1.0]), "angle")
        cudaq_backend_shots.apply_reservoir(None, simple_reservoir_params)
        
        result = cudaq_backend_shots.measure_observables(None, obs_set="pauli_z")
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert all(-1.0 <= val <= 1.0 for val in result)
    
    def test_execute_updates_shots(self, cudaq_backend):
        """Test that execute updates shots setting."""
        cudaq_backend.create_circuit(n_qubits=2)
        
        result = cudaq_backend.execute(None, shots=500)
        
        assert cudaq_backend.shots == 500
        assert result["shots"] == 500
    
    def test_reset_clears_state(self, cudaq_backend, simple_reservoir_params):
        """Test that reset clears stored state."""
        cudaq_backend.create_circuit(n_qubits=3)
        cudaq_backend.apply_encoding(None, np.array([0.5, 0.5, 0.5]), "angle")
        cudaq_backend.apply_reservoir(None, simple_reservoir_params)
        
        cudaq_backend.reset()
        
        assert cudaq_backend._encoded_data is None
        assert cudaq_backend._reservoir_params is None
    
    def test_full_workflow(self, cudaq_backend, simple_reservoir_params):
        """Test complete workflow: create → encode → reservoir → measure."""
        # Create circuit
        cudaq_backend.create_circuit(n_qubits=3)
        
        # Encode data
        data = np.array([0.2, 0.5, 0.8])
        cudaq_backend.apply_encoding(None, data, strategy="angle")
        
        # Apply reservoir
        cudaq_backend.apply_reservoir(None, simple_reservoir_params)
        
        # Measure
        result = cudaq_backend.measure_observables(None, obs_set="pauli_z")
        
        # Verify
        assert len(result) == 3
        assert all(-1.0 <= val <= 1.0 for val in result)
    
    def test_reproducibility_statevector(self, simple_reservoir_params):
        """Test that statevector results are reproducible."""
        targets = get_available_targets()
        target = "nvidia" if "nvidia" in targets else targets[0]
        
        # Run twice with same params
        results = []
        for _ in range(2):
            backend = CUDAQuantumBackend(target=target, shots=None)
            backend.create_circuit(n_qubits=3)
            backend.apply_encoding(None, np.array([0.3, 0.6, 0.9]), "angle")
            backend.apply_reservoir(None, simple_reservoir_params)
            results.append(backend.measure_observables(None, obs_set="pauli_z"))
        
        np.testing.assert_array_almost_equal(
            results[0], results[1], decimal=10,
            err_msg="Statevector results should be identical across runs"
        )
    
    def test_ground_state_expectations(self, cudaq_backend):
        """Test that ground state gives expected Z values."""
        # Create simple params with no rotations (identity reservoir)
        params = ReservoirParams(
            n_qubits=3,
            n_layers=1,
            coupling_strengths=np.zeros((1, 3, 3)),  # No couplings
            rotation_angles=np.zeros((1, 3)),  # No rotations
            seed=0,
        )
        
        cudaq_backend.create_circuit(n_qubits=3)
        # No encoding → all qubits in |0⟩
        cudaq_backend._encoded_data = np.array([])
        cudaq_backend.apply_reservoir(None, params)
        
        result = cudaq_backend.measure_observables(None, obs_set="pauli_z")
        
        # All qubits in |0⟩ should give ⟨Z⟩ = 1.0
        np.testing.assert_array_almost_equal(
            result, [1.0, 1.0, 1.0], decimal=6,
            err_msg="Ground state should have ⟨Z⟩ = 1 for all qubits"
        )


class TestCUDAQuantumBackendEdgeCases:
    """Test edge cases for CUDAQuantumBackend."""
    
    def test_single_qubit(self, cudaq_backend):
        """Test with single qubit."""
        params = ReservoirParams(
            n_qubits=1,
            n_layers=1,
            coupling_strengths=np.zeros((1, 1, 1)),
            rotation_angles=np.array([[0.5]]),
            seed=0,
        )
        
        cudaq_backend.create_circuit(n_qubits=1)
        cudaq_backend.apply_encoding(None, np.array([0.5]), "angle")
        cudaq_backend.apply_reservoir(None, params)
        
        result = cudaq_backend.measure_observables(None, obs_set="pauli_z")
        
        assert len(result) == 1
        assert -1.0 <= result[0] <= 1.0
    
    def test_empty_encoding(self, cudaq_backend, simple_reservoir_params):
        """Test with empty encoding data."""
        cudaq_backend.create_circuit(n_qubits=3)
        cudaq_backend.apply_encoding(None, np.array([]), "angle")
        cudaq_backend.apply_reservoir(None, simple_reservoir_params)
        
        result = cudaq_backend.measure_observables(None, obs_set="pauli_z")
        
        assert len(result) == 3
    
    def test_sparse_couplings(self, cudaq_backend):
        """Test with mostly zero couplings."""
        params = ReservoirParams(
            n_qubits=4,
            n_layers=2,
            coupling_strengths=np.zeros((2, 4, 4)),  # All zero
            rotation_angles=np.random.uniform(-np.pi, np.pi, (2, 4)),
            seed=0,
        )
        # Add just one non-zero coupling
        params.coupling_strengths[0, 0, 1] = 0.5
        
        cudaq_backend.create_circuit(n_qubits=4)
        cudaq_backend.apply_encoding(None, np.array([0.5, 0.5, 0.5, 0.5]), "angle")
        cudaq_backend.apply_reservoir(None, params)
        
        result = cudaq_backend.measure_observables(None, obs_set="pauli_z")
        
        assert len(result) == 4
        assert all(-1.0 <= val <= 1.0 for val in result)


# Tests that don't require CUDA-Q
class TestCUDAQuantumWithoutCUDAQ:
    """Tests that run regardless of CUDA-Q availability."""
    
    def test_is_cudaq_available_no_crash(self):
        """Test that availability check never crashes."""
        # This should never raise
        result = is_cudaq_available()
        assert isinstance(result, bool)
    
    def test_get_available_targets_no_crash(self):
        """Test that target listing never crashes."""
        # This should never raise
        result = get_available_targets()
        assert isinstance(result, list)
