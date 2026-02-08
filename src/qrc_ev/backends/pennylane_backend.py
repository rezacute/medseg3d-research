"""PennyLane backend implementation for quantum reservoir computing.

This module provides a concrete implementation of the QuantumBackend interface
using PennyLane's quantum simulation devices.
"""

from typing import Any, Callable

import numpy as np
import pennylane as qml

from qrc_ev.backends.base import QuantumBackend, ReservoirParams


class PennyLaneBackend(QuantumBackend):
    """PennyLane implementation of the quantum backend.

    Supports both exact statevector simulation (default.qubit, lightning.qubit)
    and shot-based sampling.

    Attributes:
        device_name: PennyLane device name (e.g., "default.qubit", "lightning.qubit").
        shots: Default number of measurement shots (0 for exact simulation).
    """

    def __init__(self, device_name: str = "default.qubit", shots: int | None = None):
        """Initialize the PennyLane backend.

        Args:
            device_name: PennyLane device name. Defaults to "default.qubit".
            shots: Default number of measurement shots. None for exact statevector
                simulation. Defaults to None.
        """
        self.device_name = device_name
        self.shots = shots
        self._device: Any = None
        self._n_qubits: int = 0

    def create_circuit(self, n_qubits: int) -> Any:
        """Initialize a PennyLane device with the specified qubit count.

        Args:
            n_qubits: Number of qubits for the device.

        Returns:
            PennyLane device object.
        """
        self._n_qubits = n_qubits
        self._device = qml.device(self.device_name, wires=n_qubits, shots=self.shots)
        return self._device

    def apply_encoding(
        self, circuit: Any, data: np.ndarray, strategy: str = "angle"
    ) -> Any:
        """Apply data encoding to the circuit.

        Currently supports "angle" encoding strategy, which applies Ry(π × xᵢ)
        to each qubit.

        Args:
            circuit: PennyLane device (not used directly, encoding applied in QNode).
            data: Input data vector of shape (d,) with values in [0, 1].
            strategy: Encoding strategy name. Only "angle" is supported.

        Returns:
            The circuit object (unchanged, as encoding is applied via gates).

        Raises:
            ValueError: If strategy is not "angle" or if data dimension exceeds
                qubit count.
        """
        if strategy != "angle":
            raise ValueError(f"Unsupported encoding strategy: {strategy}")

        if len(data) > self._n_qubits:
            raise ValueError(
                f"Input dimension {len(data)} exceeds qubit count {self._n_qubits}"
            )

        # Apply Ry(π × xᵢ) to each qubit
        for i, x in enumerate(data):
            qml.RY(np.pi * x, wires=i)

        return circuit

    def apply_reservoir(self, circuit: Any, params: ReservoirParams) -> Any:
        """Apply the Ising-type reservoir unitary to the circuit.

        The reservoir unitary consists of n_layers, where each layer applies:
        1. Single-qubit Rz rotations with angles from params.rotation_angles
        2. Two-qubit CNOT+Rz couplings with strengths from params.coupling_strengths

        Args:
            circuit: PennyLane device (not used directly, gates applied in QNode).
            params: Fixed random reservoir parameters.

        Returns:
            The circuit object (unchanged, as gates are applied directly).
        """
        for layer in range(params.n_layers):
            # Apply single-qubit Rz rotations
            for qubit in range(params.n_qubits):
                qml.RZ(params.rotation_angles[layer, qubit], wires=qubit)

            # Apply two-qubit couplings: CNOT followed by Rz
            for i in range(params.n_qubits):
                for j in range(i + 1, params.n_qubits):
                    coupling_strength = params.coupling_strengths[layer, i, j]
                    if not np.isclose(coupling_strength, 0.0):
                        qml.CNOT(wires=[i, j])
                        qml.RZ(coupling_strength, wires=j)

        return circuit

    def measure_observables(self, circuit: Any, obs_set: str = "pauli_z") -> list[Any]:
        """Extract Pauli-Z observable measurements from the circuit.

        Args:
            circuit: PennyLane device (not used directly, measurements defined in QNode).
            obs_set: Observable set name. Only "pauli_z" is supported.

        Returns:
            NumPy array of ⟨Zᵢ⟩ expectation values with length n_qubits.

        Raises:
            ValueError: If obs_set is not "pauli_z".
        """
        if obs_set != "pauli_z":
            raise ValueError(f"Unsupported observable set: {obs_set}")

        # Return list of Pauli-Z observables for each qubit
        # This will be used in the QNode's return statement
        return [qml.expval(qml.PauliZ(i)) for i in range(self._n_qubits)]

    def execute(self, circuit: Any, shots: int | None = None) -> Any:
        """Execute the circuit.

        Note: In PennyLane, execution happens through QNodes. This method
        is primarily for interface compliance. The actual execution is handled
        by creating and calling QNodes with the device.

        Args:
            circuit: PennyLane device.
            shots: Number of measurement shots. If None, use exact statevector
                simulation. If > 0, use shot-based sampling.

        Returns:
            The device object (execution happens via QNode calls).
        """
        # Update device shots if different from default
        if shots != self.shots and shots is not None:
            self._device = qml.device(
                self.device_name, wires=self._n_qubits, shots=shots
            )

        return self._device

    def create_qnode(
        self, circuit_func: Callable, interface: str = "numpy"
    ) -> qml.QNode:
        """Create a PennyLane QNode from a circuit function.

        This is a helper method for creating executable quantum circuits.

        Args:
            circuit_func: Python function defining the quantum circuit.
            interface: Interface for automatic differentiation ("numpy", "torch", etc.).

        Returns:
            Executable QNode that can be called with parameters.
        """
        return qml.QNode(circuit_func, self._device, interface=interface)
