"""A1 Standard Gate-Based Quantum Reservoir Computing.

This module implements the standard QRC architecture using a fixed random
Ising-type unitary for reservoir evolution, angle encoding for input, and
Pauli-Z observables for feature extraction.
"""

from typing import TYPE_CHECKING

import numpy as np
import pennylane as qml

from qrc_ev.backends.base import QuantumBackend, QuantumReservoir, ReservoirParams
from qrc_ev.encoding.angle import angle_encode
from qrc_ev.readout.observables import pauli_z_observables

if TYPE_CHECKING:
    from qrc_ev.backends.pennylane_backend import PennyLaneBackend


class StandardReservoir(QuantumReservoir):
    """A1 Standard Gate-Based QRC with fixed random Ising unitary.

    Processes time-series data by encoding each timestep into a quantum state
    via angle encoding, evolving through a fixed random Ising unitary, and
    extracting Pauli-Z expectation values as classical features.

    Attributes:
        backend: Quantum backend for circuit execution.
        n_qubits: Number of qubits in the reservoir.
        n_layers: Number of layers in the reservoir unitary.
        evolution_steps: Number of times the reservoir unitary is applied.
        params: Fixed random Ising parameters generated from seed.
    """

    def __init__(
        self,
        backend: "PennyLaneBackend",
        n_qubits: int,
        n_layers: int = 4,
        evolution_steps: int = 1,
        seed: int = 42,
    ):
        """Initialize the standard reservoir with fixed random parameters.

        Args:
            backend: PennyLane backend implementation.
            n_qubits: Number of qubits in the reservoir.
            n_layers: Number of layers in the Ising unitary. Default: 4.
            evolution_steps: Number of unitary applications per timestep. Default: 1.
            seed: Random seed for generating fixed Ising parameters. Default: 42.
        """
        self.backend = backend
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.evolution_steps = evolution_steps
        self.params = self._generate_fixed_params(seed)
        self._current_input: np.ndarray | None = None
        self._evolution_steps_override = evolution_steps

        # Initialize the backend device
        self.backend.create_circuit(n_qubits)

    def _generate_fixed_params(self, seed: int) -> ReservoirParams:
        """Generate fixed random Ising coupling strengths and rotation angles.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            ReservoirParams with randomly generated coupling strengths and
            rotation angles that remain constant across all calls.
        """
        rng = np.random.default_rng(seed)
        coupling_strengths = rng.uniform(
            -np.pi, np.pi, (self.n_layers, self.n_qubits, self.n_qubits)
        )
        rotation_angles = rng.uniform(
            -np.pi, np.pi, (self.n_layers, self.n_qubits)
        )
        return ReservoirParams(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            coupling_strengths=coupling_strengths,
            rotation_angles=rotation_angles,
            seed=seed,
        )

    def _build_and_run_circuit(self, data: np.ndarray) -> np.ndarray:
        """Build and execute a full reservoir circuit for one timestep.

        Uses backend-agnostic methods to support multiple quantum frameworks.
        For PennyLane, constructs a QNode. For CUDA-Q and others, uses the
        backend's native execution path.

        Args:
            data: Input feature vector of shape (d,) with values in [0, 1].

        Returns:
            NumPy array of Pauli-Z expectation values with shape (n_qubits,).
        """
        # Check if backend has native _device (PennyLane) or not (CUDA-Q, Qiskit)
        if hasattr(self.backend, '_device') and self.backend._device is not None:
            # PennyLane path: use qml.qnode
            dev = self.backend._device

            @qml.qnode(dev, interface="numpy")
            def circuit() -> list:
                # Encode input data
                angle_encode(data, self.n_qubits)

                # Apply reservoir unitary evolution_steps times
                for _ in range(self.evolution_steps):
                    self.backend.apply_reservoir(None, self.params)

                # Measure Pauli-Z observables
                return pauli_z_observables(self.n_qubits)

            result = circuit()
            return np.array(result)
        else:
            # Backend-agnostic path: use backend methods directly
            # This works for CUDA-Q, Qiskit, and other backends
            # Note: create_circuit was already called in __init__
            self.backend.apply_encoding(None, data, strategy="angle")
            
            # Store params for the kernel (evolution_steps handled internally)
            # For backends that support multiple evolution steps, we'd loop here
            # For now, we apply once (CUDA-Q kernel handles the layers internally)
            self.backend.apply_reservoir(None, self.params)
            
            result = self.backend.measure_observables(None, obs_set="pauli_z")
            return np.array(result)

    def encode(self, x: np.ndarray) -> None:
        """Encode input data into the quantum state via angle encoding.

        Delegates to the angle encoder. The actual encoding is applied when
        the circuit is built and executed.

        Args:
            x: Input feature vector of shape (d,) with values in [0, 1].

        Raises:
            ValueError: If input dimension exceeds n_qubits.
        """
        # Validate input dimension eagerly
        if len(x) > self.n_qubits:
            raise ValueError(
                f"Input dimension {len(x)} exceeds qubit count {self.n_qubits}"
            )
        self._current_input = x

    def evolve(self, steps: int) -> None:
        """Set the number of evolution steps for the next measurement.

        In the gate-based QRC model, evolution is applied as part of the
        circuit execution. This stores the step count for the next measure() call.

        Args:
            steps: Number of times to apply the reservoir unitary.
        """
        self._evolution_steps_override = steps

    def measure(self) -> np.ndarray:
        """Execute the circuit and extract Pauli-Z observables.

        Builds and runs the full circuit (encode → evolve → measure) using
        the most recently encoded input and evolution steps.

        Returns:
            NumPy array of ⟨Zᵢ⟩ expectation values with shape (n_qubits,).
        """
        steps = getattr(self, "_evolution_steps_override", self.evolution_steps)
        data = getattr(self, "_current_input", None)

        if data is None:
            # No input encoded — measure the |0⟩ state
            data = np.array([])

        dev = qml.device(
            self.backend.device_name,
            wires=self.n_qubits,
            shots=self.backend.shots,
        )

        @qml.qnode(dev, interface="numpy")
        def circuit() -> list:
            if len(data) > 0:
                angle_encode(data, self.n_qubits)
            for _ in range(steps):
                self.backend.apply_reservoir(None, self.params)
            return pauli_z_observables(self.n_qubits)

        result = circuit()
        # Clean up override
        self._evolution_steps_override = self.evolution_steps
        return np.array(result)

    def process(self, time_series: np.ndarray) -> np.ndarray:
        """Process a time-series through the reservoir.

        For each timestep, resets the reservoir state, encodes the input,
        applies the reservoir unitary, and measures Pauli-Z observables.

        Args:
            time_series: Input array of shape (T, d) where T is the number
                of timesteps and d is the feature dimension (d ≤ n_qubits).

        Returns:
            Feature array of shape (T, n_qubits) containing Pauli-Z
            expectation values for each timestep.
        """
        features = []
        for t in range(time_series.shape[0]):
            result = self._build_and_run_circuit(time_series[t])
            features.append(result)
        return np.array(features)

    def reset(self) -> None:
        """Reset the reservoir to the initial |0⟩⊗ⁿ state.

        Clears any stored input data and resets evolution steps to default.
        The fixed random Ising parameters remain unchanged.
        """
        self._current_input = None
        self._evolution_steps_override = self.evolution_steps
