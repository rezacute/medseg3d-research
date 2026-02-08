"""Quantum backend abstraction layer.

This module provides abstract base classes and concrete implementations for
quantum circuit execution across different quantum computing frameworks.
"""

from qrc_ev.backends.base import QuantumBackend, QuantumReservoir, ReservoirParams
from qrc_ev.backends.pennylane_backend import PennyLaneBackend

__all__ = [
    "QuantumBackend",
    "QuantumReservoir",
    "ReservoirParams",
    "PennyLaneBackend",
]
