"""Observable extraction for quantum reservoir states.

This module provides functions for extracting measurement observables
from quantum reservoir states.
"""

import pennylane as qml


def pauli_z_observables(n_qubits: int) -> list:
    """Return PennyLane observable list for single-qubit Z expectations.

    Creates a list of Pauli-Z expectation value observables, one for each qubit.
    Each observable measures ⟨Zᵢ⟩ for qubit i, returning values in [-1, 1].

    Args:
        n_qubits: Number of qubits in the quantum circuit.

    Returns:
        List of PennyLane expectation value observables for Pauli-Z on each qubit.
        The list has length n_qubits, with element i measuring qubit i.

    Example:
        >>> obs = pauli_z_observables(3)
        >>> len(obs)
        3
        >>> # Each observable measures ⟨Z⟩ on a single qubit
    """
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
