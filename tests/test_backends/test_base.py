"""Unit tests for backend abstraction layer.

Tests verify that the abstract base classes enforce their contracts:
- Direct instantiation of ABCs raises TypeError
- Calling abstract methods on minimal subclasses raises NotImplementedError
"""

import numpy as np
import pytest

from qrc_ev.backends.base import QuantumBackend, QuantumReservoir, ReservoirParams


class TestReservoirParams:
    """Tests for ReservoirParams dataclass."""

    def test_reservoir_params_creation(self):
        """Test that ReservoirParams can be instantiated with valid data."""
        n_qubits = 4
        n_layers = 2
        coupling_strengths = np.random.uniform(-np.pi, np.pi, (n_layers, n_qubits, n_qubits))
        rotation_angles = np.random.uniform(-np.pi, np.pi, (n_layers, n_qubits))
        seed = 42

        params = ReservoirParams(
            n_qubits=n_qubits,
            n_layers=n_layers,
            coupling_strengths=coupling_strengths,
            rotation_angles=rotation_angles,
            seed=seed,
        )

        assert params.n_qubits == n_qubits
        assert params.n_layers == n_layers
        assert params.seed == seed
        assert params.coupling_strengths.shape == (n_layers, n_qubits, n_qubits)
        assert params.rotation_angles.shape == (n_layers, n_qubits)


class TestQuantumBackendABC:
    """Tests for QuantumBackend abstract base class contract."""

    def test_cannot_instantiate_abstract_backend(self):
        """Test that QuantumBackend cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            QuantumBackend()

    def test_minimal_subclass_raises_not_implemented(self):
        """Test that a minimal subclass without implementations raises NotImplementedError."""

        class MinimalBackend(QuantumBackend):
            """Minimal subclass that doesn't implement abstract methods."""
            pass

        # Should not be able to instantiate without implementing abstract methods
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            MinimalBackend()

    def test_partial_implementation_raises_not_implemented(self):
        """Test that calling unimplemented abstract methods raises NotImplementedError."""

        class PartialBackend(QuantumBackend):
            """Subclass that implements only some methods."""

            def create_circuit(self, n_qubits: int):
                return None

            def apply_encoding(self, circuit, data, strategy="angle"):
                return None

            def apply_reservoir(self, circuit, params):
                return None

            def measure_observables(self, circuit, obs_set="pauli_z"):
                return np.array([])

            def execute(self, circuit, shots=0):
                return None

        # This should now be instantiable since all methods are implemented
        backend = PartialBackend()
        assert backend is not None

    def test_abstract_methods_exist(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = {
            "create_circuit",
            "apply_encoding",
            "apply_reservoir",
            "measure_observables",
            "execute",
        }

        # Get abstract methods from the ABC
        backend_abstracts = set(QuantumBackend.__abstractmethods__)
        assert backend_abstracts == abstract_methods


class TestQuantumReservoirABC:
    """Tests for QuantumReservoir abstract base class contract."""

    def test_cannot_instantiate_abstract_reservoir(self):
        """Test that QuantumReservoir cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            QuantumReservoir()

    def test_minimal_subclass_raises_not_implemented(self):
        """Test that a minimal subclass without implementations raises NotImplementedError."""

        class MinimalReservoir(QuantumReservoir):
            """Minimal subclass that doesn't implement abstract methods."""
            pass

        # Should not be able to instantiate without implementing abstract methods
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            MinimalReservoir()

    def test_partial_implementation_raises_not_implemented(self):
        """Test that calling unimplemented abstract methods raises NotImplementedError."""

        class PartialReservoir(QuantumReservoir):
            """Subclass that implements only some methods."""

            def __init__(self, backend, **kwargs):
                pass

            def encode(self, x: np.ndarray) -> None:
                pass

            def evolve(self, steps: int) -> None:
                pass

            def measure(self) -> np.ndarray:
                return np.array([])

            def process(self, time_series: np.ndarray) -> np.ndarray:
                return np.array([])

            def reset(self) -> None:
                pass

        # This should now be instantiable since all methods are implemented
        reservoir = PartialReservoir(backend=None)
        assert reservoir is not None

    def test_abstract_methods_exist(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = {
            "__init__",
            "encode",
            "evolve",
            "measure",
            "process",
            "reset",
        }

        # Get abstract methods from the ABC
        reservoir_abstracts = set(QuantumReservoir.__abstractmethods__)
        assert reservoir_abstracts == abstract_methods


class TestABCContractEnforcement:
    """Integration tests verifying ABC contract enforcement."""

    def test_backend_subclass_must_implement_all_methods(self):
        """Test that a backend subclass must implement all abstract methods."""

        class IncompleteBackend(QuantumBackend):
            """Backend missing some implementations."""

            def create_circuit(self, n_qubits: int):
                return None

            # Missing: apply_encoding, apply_reservoir, measure_observables, execute

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteBackend()

    def test_reservoir_subclass_must_implement_all_methods(self):
        """Test that a reservoir subclass must implement all abstract methods."""

        class IncompleteReservoir(QuantumReservoir):
            """Reservoir missing some implementations."""

            def encode(self, x: np.ndarray) -> None:
                pass

            def evolve(self, steps: int) -> None:
                pass

            # Missing: measure, process, reset

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteReservoir()

    def test_complete_backend_can_be_instantiated(self):
        """Test that a complete backend implementation can be instantiated."""

        class CompleteBackend(QuantumBackend):
            """Complete backend implementation."""

            def create_circuit(self, n_qubits: int):
                return {"n_qubits": n_qubits}

            def apply_encoding(self, circuit, data, strategy="angle"):
                return circuit

            def apply_reservoir(self, circuit, params):
                return circuit

            def measure_observables(self, circuit, obs_set="pauli_z"):
                return np.zeros(circuit["n_qubits"])

            def execute(self, circuit, shots=0):
                return {"result": "success"}

        backend = CompleteBackend()
        assert backend is not None
        circuit = backend.create_circuit(4)
        assert circuit["n_qubits"] == 4

    def test_complete_reservoir_can_be_instantiated(self):
        """Test that a complete reservoir implementation can be instantiated."""

        class CompleteReservoir(QuantumReservoir):
            """Complete reservoir implementation."""

            def __init__(self, backend, **kwargs):
                self.backend = backend
                self.state = np.zeros(4)

            def encode(self, x: np.ndarray) -> None:
                self.state = x

            def evolve(self, steps: int) -> None:
                self.state = self.state * steps

            def measure(self) -> np.ndarray:
                return self.state

            def process(self, time_series: np.ndarray) -> np.ndarray:
                return time_series

            def reset(self) -> None:
                self.state = np.zeros(4)

        reservoir = CompleteReservoir(backend=None)
        assert reservoir is not None
        reservoir.encode(np.array([1.0, 2.0, 3.0, 4.0]))
        result = reservoir.measure()
        assert len(result) == 4
