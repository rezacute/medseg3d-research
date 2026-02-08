"""Angle encoding for quantum reservoir computing.

This module provides angle encoding that maps classical data to quantum states
via single-qubit Ry rotations.
"""

import numpy as np

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


def angle_encode(data: np.ndarray, n_qubits: int) -> None:
    """Apply angle encoding to map classical data to quantum state.
    
    Applies Ry(π × xᵢ) rotation to qubit i for each feature xᵢ in the input data.
    Input values should be in [0, 1] and are scaled to [0, π] for the rotation angle.
    Unused qubits (when d < n_qubits) remain in the |0⟩ state.
    
    Args:
        data: Input vector of shape (d,) with values in [0, 1].
        n_qubits: Total number of qubits available in the circuit.
        
    Raises:
        ValueError: If input dimension d exceeds n_qubits.
        ImportError: If PennyLane is not installed.
        
    Example:
        >>> import pennylane as qml
        >>> dev = qml.device("default.qubit", wires=4)
        >>> @qml.qnode(dev)
        ... def circuit(x):
        ...     angle_encode(x, n_qubits=4)
        ...     return [qml.expval(qml.PauliZ(i)) for i in range(4)]
        >>> x = np.array([0.0, 0.5, 1.0])
        >>> circuit(x)  # Encodes 3 features on 4 qubits
    """
    if not HAS_PENNYLANE:
        raise ImportError(
            "PennyLane is required for angle encoding. "
            "Install it with: pip install pennylane"
        )
    
    d = len(data)
    if d > n_qubits:
        raise ValueError(
            f"Input dimension {d} exceeds qubit count {n_qubits}"
        )
    
    # Apply Ry(π × xᵢ) to each qubit for each feature
    for i, x in enumerate(data):
        qml.RY(np.pi * x, wires=i)
