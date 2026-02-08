"""Property-based tests for angle encoding.

Tests verify that angle encoding correctly maps classical data to quantum states
via Ry rotations, handles dimension mismatches, and leaves unused qubits in |0⟩.
"""

import numpy as np
import pennylane as qml
import pytest
from hypothesis import given, settings, strategies as st

from qrc_ev.encoding.angle import angle_encode


# Hypothesis strategies for generating test data
@st.composite
def valid_encoding_inputs(draw):
    """Generate valid (data, n_qubits) pairs where d <= n_qubits."""
    n_qubits = draw(st.integers(min_value=1, max_value=10))
    d = draw(st.integers(min_value=1, max_value=n_qubits))
    data = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=d,
            max_size=d,
        )
    )
    return np.array(data), n_qubits


@st.composite
def oversized_encoding_inputs(draw):
    """Generate invalid (data, n_qubits) pairs where d > n_qubits."""
    n_qubits = draw(st.integers(min_value=1, max_value=10))
    d = draw(st.integers(min_value=n_qubits + 1, max_value=n_qubits + 10))
    data = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=d,
            max_size=d,
        )
    )
    return np.array(data), n_qubits


# Feature: phase1-foundation-setup, Property 1: Angle encoding produces correct quantum state
@given(inputs=valid_encoding_inputs())
@settings(max_examples=100, deadline=None)
def test_angle_encoding_correctness(inputs):
    """Property 1: Angle encoding produces correct quantum state.
    
    **Validates: Requirements 4.1, 4.3, 4.4**
    
    For any input vector x with values in [0, 1] and dimension d ≤ n_qubits,
    the angle encoder should produce a quantum state where:
    - Qubit i has been rotated by Ry(π × xᵢ) for i < d
    - Unused qubits (i >= d) remain in |0⟩ state (⟨Z⟩ = 1.0)
    """
    data, n_qubits = inputs
    d = len(data)
    
    # Create a quantum circuit that applies angle encoding and measures Pauli-Z
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit():
        angle_encode(data, n_qubits)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    expectations = circuit()
    
    # Verify encoded qubits have correct expectation values
    # For Ry(θ) applied to |0⟩: ⟨Z⟩ = cos(θ)
    for i in range(d):
        expected_z = np.cos(np.pi * data[i])
        assert np.isclose(
            expectations[i], expected_z, atol=1e-6
        ), f"Qubit {i}: expected ⟨Z⟩ = {expected_z}, got {expectations[i]}"
    
    # Verify unused qubits remain in |0⟩ state (⟨Z⟩ = 1.0)
    for i in range(d, n_qubits):
        assert np.isclose(
            expectations[i], 1.0, atol=1e-6
        ), f"Unused qubit {i}: expected ⟨Z⟩ = 1.0, got {expectations[i]}"


# Feature: phase1-foundation-setup, Property 2: Angle encoding rejects oversized input
@given(inputs=oversized_encoding_inputs())
@settings(max_examples=100, deadline=None)
def test_angle_encoding_rejects_oversized_input(inputs):
    """Property 2: Angle encoding rejects oversized input.
    
    **Validates: Requirements 4.2**
    
    For any input vector x with dimension d > n_qubits,
    the angle encoder should raise a ValueError.
    """
    data, n_qubits = inputs
    d = len(data)
    
    # Create a device (needed for qml context)
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit():
        angle_encode(data, n_qubits)
        return qml.expval(qml.PauliZ(0))
    
    # Should raise ValueError when d > n_qubits
    with pytest.raises(ValueError) as exc_info:
        circuit()
    
    assert "exceeds qubit count" in str(exc_info.value).lower()
    assert str(d) in str(exc_info.value)
    assert str(n_qubits) in str(exc_info.value)


# Unit tests for specific edge cases
def test_angle_encoding_all_zeros():
    """Edge case: All zero input should leave all qubits in |0⟩."""
    n_qubits = 4
    data = np.array([0.0, 0.0, 0.0])
    
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit():
        angle_encode(data, n_qubits)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    expectations = circuit()
    
    # All qubits should have ⟨Z⟩ = 1.0 (Ry(0) is identity)
    assert np.allclose(expectations, 1.0, atol=1e-6)


def test_angle_encoding_all_ones():
    """Edge case: All ones input should rotate qubits by π."""
    n_qubits = 4
    data = np.array([1.0, 1.0, 1.0])
    
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit():
        angle_encode(data, n_qubits)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    expectations = circuit()
    
    # First 3 qubits: Ry(π) flips |0⟩ to |1⟩, so ⟨Z⟩ = -1.0
    assert np.allclose(expectations[:3], -1.0, atol=1e-6)
    # Last qubit unused: ⟨Z⟩ = 1.0
    assert np.isclose(expectations[3], 1.0, atol=1e-6)


def test_angle_encoding_half_values():
    """Edge case: Input of 0.5 should rotate by π/2."""
    n_qubits = 2
    data = np.array([0.5])
    
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit():
        angle_encode(data, n_qubits)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    expectations = circuit()
    
    # Ry(π/2) on |0⟩ gives equal superposition: ⟨Z⟩ = 0.0
    assert np.isclose(expectations[0], 0.0, atol=1e-6)
    # Second qubit unused: ⟨Z⟩ = 1.0
    assert np.isclose(expectations[1], 1.0, atol=1e-6)


def test_angle_encoding_exact_dimension_match():
    """Edge case: d = n_qubits (no unused qubits)."""
    n_qubits = 3
    data = np.array([0.2, 0.5, 0.8])
    
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit():
        angle_encode(data, n_qubits)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    expectations = circuit()
    
    # All qubits should be encoded
    for i in range(n_qubits):
        expected_z = np.cos(np.pi * data[i])
        assert np.isclose(expectations[i], expected_z, atol=1e-6)


def test_angle_encoding_single_qubit():
    """Edge case: Single qubit encoding."""
    n_qubits = 1
    data = np.array([0.3])
    
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit():
        angle_encode(data, n_qubits)
        return qml.expval(qml.PauliZ(0))
    
    expectation = circuit()
    expected_z = np.cos(np.pi * 0.3)
    
    assert np.isclose(expectation, expected_z, atol=1e-6)
