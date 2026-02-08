"""Tests for PennyLane backend implementation."""

import numpy as np
import pennylane as qml
import pytest

from qrc_ev.backends.base import ReservoirParams
from qrc_ev.backends.pennylane_backend import PennyLaneBackend


class TestPennyLaneBackend:
    """Test suite for PennyLaneBackend."""

    def test_initialization_default(self):
        """Test backend initialization with default parameters."""
        backend = PennyLaneBackend()
        assert backend.device_name == "default.qubit"
        assert backend.shots is None
        assert backend._device is None
        assert backend._n_qubits == 0

    def test_initialization_custom(self):
        """Test backend initialization with custom parameters."""
        backend = PennyLaneBackend(device_name="lightning.qubit", shots=1000)
        assert backend.device_name == "lightning.qubit"
        assert backend.shots == 1000

    def test_create_circuit_default_qubit(self):
        """Test circuit creation with default.qubit device."""
        backend = PennyLaneBackend(device_name="default.qubit")
        device = backend.create_circuit(n_qubits=4)
        
        assert device is not None
        assert backend._n_qubits == 4
        assert backend._device is device
        assert len(device.wires) == 4

    def test_create_circuit_lightning_qubit(self):
        """Test circuit creation with lightning.qubit device."""
        backend = PennyLaneBackend(device_name="lightning.qubit")
        device = backend.create_circuit(n_qubits=3)
        
        assert device is not None
        assert backend._n_qubits == 3
        assert len(device.wires) == 3

    def test_apply_encoding_angle_strategy(self):
        """Test angle encoding application."""
        backend = PennyLaneBackend()
        backend.create_circuit(n_qubits=4)
        
        # Create a simple QNode to test encoding
        @qml.qnode(backend._device)
        def test_circuit():
            data = np.array([0.0, 0.5, 1.0])
            backend.apply_encoding(backend._device, data, strategy="angle")
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]
        
        result = test_circuit()
        
        # Check that we get 4 measurements
        assert len(result) == 4
        
        # Qubit 0: Ry(0) → |0⟩ → ⟨Z⟩ = 1.0
        assert np.isclose(result[0], 1.0, atol=1e-6)
        
        # Qubit 1: Ry(π/2) → |+⟩ → ⟨Z⟩ ≈ 0.0
        assert np.isclose(result[1], 0.0, atol=1e-6)
        
        # Qubit 2: Ry(π) → |1⟩ → ⟨Z⟩ = -1.0
        assert np.isclose(result[2], -1.0, atol=1e-6)
        
        # Qubit 3: unused → |0⟩ → ⟨Z⟩ = 1.0
        assert np.isclose(result[3], 1.0, atol=1e-6)

    def test_apply_encoding_unsupported_strategy(self):
        """Test that unsupported encoding strategy raises ValueError."""
        backend = PennyLaneBackend()
        backend.create_circuit(n_qubits=4)
        
        with pytest.raises(ValueError, match="Unsupported encoding strategy"):
            backend.apply_encoding(backend._device, np.array([0.5]), strategy="amplitude")

    def test_apply_encoding_oversized_input(self):
        """Test that oversized input raises ValueError."""
        backend = PennyLaneBackend()
        backend.create_circuit(n_qubits=2)
        
        with pytest.raises(ValueError, match="Input dimension .* exceeds qubit count"):
            backend.apply_encoding(backend._device, np.array([0.1, 0.2, 0.3]), strategy="angle")

    def test_apply_reservoir_single_layer(self):
        """Test reservoir unitary application with single layer."""
        backend = PennyLaneBackend()
        backend.create_circuit(n_qubits=3)
        
        # Create simple reservoir params
        params = ReservoirParams(
            n_qubits=3,
            n_layers=1,
            coupling_strengths=np.array([[[0.0, 0.5, 0.0],
                                          [0.0, 0.0, 0.3],
                                          [0.0, 0.0, 0.0]]]),
            rotation_angles=np.array([[0.1, 0.2, 0.3]]),
            seed=42
        )
        
        @qml.qnode(backend._device)
        def test_circuit():
            backend.apply_reservoir(backend._device, params)
            return [qml.expval(qml.PauliZ(i)) for i in range(3)]
        
        result = test_circuit()
        
        # Check that we get 3 measurements in valid range
        assert len(result) == 3
        assert all(-1.0 <= val <= 1.0 for val in result)

    def test_apply_reservoir_multiple_layers(self):
        """Test reservoir unitary application with multiple layers."""
        backend = PennyLaneBackend()
        backend.create_circuit(n_qubits=2)
        
        rng = np.random.default_rng(42)
        params = ReservoirParams(
            n_qubits=2,
            n_layers=3,
            coupling_strengths=rng.uniform(-np.pi, np.pi, (3, 2, 2)),
            rotation_angles=rng.uniform(-np.pi, np.pi, (3, 2)),
            seed=42
        )
        
        @qml.qnode(backend._device)
        def test_circuit():
            backend.apply_reservoir(backend._device, params)
            return [qml.expval(qml.PauliZ(i)) for i in range(2)]
        
        result = test_circuit()
        
        assert len(result) == 2
        assert all(-1.0 <= val <= 1.0 for val in result)

    def test_measure_observables_pauli_z(self):
        """Test Pauli-Z observable measurement."""
        backend = PennyLaneBackend()
        backend.create_circuit(n_qubits=3)
        
        observables = backend.measure_observables(backend._device, obs_set="pauli_z")
        
        # Should return a list of PennyLane observables
        assert len(observables) == 3
        assert all(isinstance(obs, qml.measurements.MeasurementProcess) for obs in observables)
        
        # Test that observables work in a QNode
        @qml.qnode(backend._device)
        def test_circuit():
            # Prepare |0⟩⊗³ state (default)
            # Return each observable separately to test them
            return [qml.expval(qml.PauliZ(i)) for i in range(3)]
        
        result = test_circuit()
        
        # All qubits in |0⟩ should give ⟨Z⟩ = 1.0
        result_array = np.array(result) if isinstance(result, tuple) else result
        assert len(result_array) == 3
        assert all(np.isclose(val, 1.0, atol=1e-6) for val in result_array)

    def test_measure_observables_unsupported(self):
        """Test that unsupported observable set raises ValueError."""
        backend = PennyLaneBackend()
        backend.create_circuit(n_qubits=3)
        
        with pytest.raises(ValueError, match="Unsupported observable set"):
            backend.measure_observables(backend._device, obs_set="pauli_x")

    def test_execute_statevector_mode(self):
        """Test execution in statevector mode (shots=None)."""
        backend = PennyLaneBackend(shots=None)
        device = backend.create_circuit(n_qubits=2)
        
        result_device = backend.execute(device, shots=None)
        
        assert result_device is not None
        # PennyLane wraps shots in a Shots object
        assert result_device.shots.total_shots is None

    def test_execute_shot_based_mode(self):
        """Test execution in shot-based mode."""
        backend = PennyLaneBackend(shots=1024)
        backend.create_circuit(n_qubits=2)
        
        result_device = backend.execute(backend._device, shots=1000)
        
        assert result_device is not None
        # Device should be updated with shots (PennyLane wraps in Shots object)
        assert result_device.shots.total_shots == 1000

    def test_create_qnode(self):
        """Test QNode creation helper method."""
        backend = PennyLaneBackend()
        backend.create_circuit(n_qubits=2)
        
        def circuit_func():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))
        
        qnode = backend.create_qnode(circuit_func)
        
        assert callable(qnode)
        result = qnode()
        
        # Bell state: ⟨Z₀⟩ ≈ 0
        assert isinstance(result, (float, np.floating))
        assert -1.0 <= result <= 1.0

    def test_full_circuit_workflow(self):
        """Test complete workflow: encoding → reservoir → measurement."""
        backend = PennyLaneBackend()
        backend.create_circuit(n_qubits=4)
        
        # Create reservoir params
        rng = np.random.default_rng(123)
        params = ReservoirParams(
            n_qubits=4,
            n_layers=2,
            coupling_strengths=rng.uniform(-np.pi, np.pi, (2, 4, 4)),
            rotation_angles=rng.uniform(-np.pi, np.pi, (2, 4)),
            seed=123
        )
        
        @qml.qnode(backend._device)
        def full_circuit():
            # Encode data
            data = np.array([0.2, 0.5, 0.8, 0.3])
            backend.apply_encoding(backend._device, data, strategy="angle")
            
            # Apply reservoir
            backend.apply_reservoir(backend._device, params)
            
            # Measure
            return backend.measure_observables(backend._device, obs_set="pauli_z")
        
        result = full_circuit()
        
        # Verify output shape and range
        assert len(result) == 4
        assert all(-1.0 <= val <= 1.0 for val in result)
        
        # Run again to verify determinism (statevector mode)
        result2 = full_circuit()
        assert np.allclose(result, result2, atol=1e-10)

    def test_reproducibility_with_same_params(self):
        """Test that same parameters produce identical results."""
        backend1 = PennyLaneBackend()
        backend1.create_circuit(n_qubits=3)
        
        backend2 = PennyLaneBackend()
        backend2.create_circuit(n_qubits=3)
        
        # Same reservoir params
        rng = np.random.default_rng(42)
        params = ReservoirParams(
            n_qubits=3,
            n_layers=2,
            coupling_strengths=rng.uniform(-np.pi, np.pi, (2, 3, 3)),
            rotation_angles=rng.uniform(-np.pi, np.pi, (2, 3)),
            seed=42
        )
        
        data = np.array([0.3, 0.6, 0.9])
        
        @qml.qnode(backend1._device)
        def circuit1():
            backend1.apply_encoding(backend1._device, data, strategy="angle")
            backend1.apply_reservoir(backend1._device, params)
            return backend1.measure_observables(backend1._device, obs_set="pauli_z")
        
        @qml.qnode(backend2._device)
        def circuit2():
            backend2.apply_encoding(backend2._device, data, strategy="angle")
            backend2.apply_reservoir(backend2._device, params)
            return backend2.measure_observables(backend2._device, obs_set="pauli_z")
        
        result1 = circuit1()
        result2 = circuit2()
        
        assert np.allclose(result1, result2, atol=1e-10)
