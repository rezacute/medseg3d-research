"""Quantum backend abstraction layer.

This module provides abstract base classes and concrete implementations for
quantum circuit execution across different quantum computing frameworks.

Available backends:
    - PennyLaneBackend: Default backend using PennyLane (always available)
    - CUDAQuantumBackend: GPU-accelerated backend using NVIDIA CUDA-Quantum
      (requires CUDA toolkit and compatible GPU)

Example:
    >>> from qrc_ev.backends import PennyLaneBackend, CUDAQ_AVAILABLE
    >>> backend = PennyLaneBackend()
    >>> if CUDAQ_AVAILABLE:
    ...     from qrc_ev.backends import CUDAQuantumBackend
    ...     gpu_backend = CUDAQuantumBackend(target="nvidia")
"""

from qrc_ev.backends.base import QuantumBackend, QuantumReservoir, ReservoirParams
from qrc_ev.backends.pennylane_backend import PennyLaneBackend
from qrc_ev.backends.cudaq_backend import CUDAQBackend

# Conditional import for CUDA-Quantum (requires GPU + CUDA toolkit)
from qrc_ev.backends.cudaq_backend import (
    is_cudaq_available,
    get_available_targets,
)

CUDAQ_AVAILABLE = is_cudaq_available()

__all__ = [
    "QuantumBackend",
    "QuantumReservoir",
    "ReservoirParams",
    "PennyLaneBackend",
    "CUDAQ_AVAILABLE",
    "is_cudaq_available",
    "get_available_targets",

]

# Only export CUDAQuantumBackend if available (avoids import errors)
if CUDAQ_AVAILABLE:
    from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend
    __all__.append("CUDAQuantumBackend")
