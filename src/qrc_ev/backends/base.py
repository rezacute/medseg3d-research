"""Backend abstraction layer for quantum reservoir computing.

This module defines the abstract base classes and data structures that decouple
quantum circuit logic from specific simulator libraries (PennyLane, Qiskit, CUDA-Q).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ReservoirParams:
    """Parameters defining a fixed random quantum reservoir.

    Attributes:
        n_qubits: Number of qubits in the reservoir.
        n_layers: Number of reservoir unitary layers.
        coupling_strengths: Ising coupling strengths Jᵢⱼ for each layer.
            Shape: (n_layers, n_qubits, n_qubits)
        rotation_angles: Single-qubit rotation angles θᵢ for each layer.
            Shape: (n_layers, n_qubits)
        seed: Random seed used to generate these parameters.
    """

    n_qubits: int
    n_layers: int
    coupling_strengths: np.ndarray
    rotation_angles: np.ndarray
    seed: int


class QuantumBackend(ABC):
    """Abstract base class for quantum circuit backends.

    This interface decouples reservoir logic from specific quantum libraries,
    enabling the same experiment to run on PennyLane, Qiskit, or CUDA Quantum.
    """

    @abstractmethod
    def create_circuit(self, n_qubits: int) -> Any:
        """Initialize a quantum circuit or device with the specified qubit count.

        Args:
            n_qubits: Number of qubits for the circuit.

        Returns:
            Backend-specific circuit or device object.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_encoding(
        self, circuit: Any, data: np.ndarray, strategy: str = "angle"
    ) -> Any:
        """Apply data encoding to the circuit.

        Args:
            circuit: Backend-specific circuit object.
            data: Input data vector to encode.
            strategy: Encoding strategy name (e.g., "angle", "amplitude", "iqp").

        Returns:
            Modified circuit with encoding gates applied.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_reservoir(self, circuit: Any, params: ReservoirParams) -> Any:
        """Apply the reservoir unitary to the circuit.

        Args:
            circuit: Backend-specific circuit object.
            params: Fixed random reservoir parameters.

        Returns:
            Modified circuit with reservoir gates applied.
        """
        raise NotImplementedError

    @abstractmethod
    def measure_observables(self, circuit: Any, obs_set: str = "pauli_z") -> Any:
        """Extract observable measurements from the circuit.

        Args:
            circuit: Backend-specific circuit object.
            obs_set: Observable set name (e.g., "pauli_z").

        Returns:
            Observable measurements (type depends on backend implementation).
        """
        raise NotImplementedError

    @abstractmethod
    def execute(self, circuit: Any, shots: int = 0) -> Any:
        """Execute the circuit.

        Args:
            circuit: Backend-specific circuit object.
            shots: Number of measurement shots. If 0, use exact statevector simulation.

        Returns:
            Backend-specific execution result.
        """
        raise NotImplementedError


class QuantumReservoir(ABC):
    """Abstract base class for quantum reservoir implementations.

    This interface defines the core operations for processing time-series data
    through a quantum reservoir: encoding, evolution, measurement, and reset.
    """

    @abstractmethod
    def __init__(self, backend: QuantumBackend, **kwargs: Any) -> None:
        """Initialize the quantum reservoir with a backend.

        Args:
            backend: Quantum backend for circuit execution.
            **kwargs: Additional reservoir-specific parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def encode(self, x: np.ndarray) -> None:
        """Encode input data into the quantum state.

        Args:
            x: Input feature vector of shape (d,).
        """
        raise NotImplementedError

    @abstractmethod
    def evolve(self, steps: int) -> None:
        """Evolve the reservoir state by applying the reservoir unitary.

        Args:
            steps: Number of evolution steps (unitary applications).
        """
        raise NotImplementedError

    @abstractmethod
    def measure(self) -> np.ndarray:
        """Measure observables from the current reservoir state.

        Returns:
            NumPy array of observable expectation values.
        """
        raise NotImplementedError

    @abstractmethod
    def process(self, time_series: np.ndarray) -> np.ndarray:
        """Process a time-series through the reservoir.

        For each timestep: reset → encode → evolve → measure.

        Args:
            time_series: Input time-series of shape (T, d) where T is the number
                of timesteps and d is the feature dimension.

        Returns:
            Feature array of shape (T, n_qubits) containing observable values
            for each timestep.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset the reservoir to its initial |0⟩⊗ⁿ state.

        The fixed random parameters should remain unchanged.
        """
        raise NotImplementedError
