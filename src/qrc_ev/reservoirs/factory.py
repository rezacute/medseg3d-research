"""Architecture factory for creating quantum reservoir instances.

This module provides a factory function that instantiates reservoir objects
from string identifiers and configuration dictionaries, enabling
configuration-driven architecture selection.
"""

from typing import Any

from qrc_ev.backends.base import QuantumBackend, QuantumReservoir
from qrc_ev.reservoirs.standard import StandardReservoir
from qrc_ev.reservoirs.recurrence_free import RecurrenceFreeReservoir
from qrc_ev.reservoirs.polynomial import PolynomialReservoir
from qrc_ev.reservoirs.noise_aware import NoiseAwareReservoir

# Registry mapping architecture names to reservoir classes
_REGISTRY: dict[str, type[QuantumReservoir]] = {
    "standard": StandardReservoir,
    "a1": StandardReservoir,  # Alias
    "recurrence_free": RecurrenceFreeReservoir,
    "rf_qrc": RecurrenceFreeReservoir,  # Alias
    "a2": RecurrenceFreeReservoir,  # Alias
    "polynomial": PolynomialReservoir,
    "poly_qrc": PolynomialReservoir,  # Alias
    "a4": PolynomialReservoir,  # Alias
    "noise_aware": NoiseAwareReservoir,
    "noisy_qrc": NoiseAwareReservoir,  # Alias
    "a6": NoiseAwareReservoir,  # Alias
}


def create_reservoir(
    arch: str, backend: QuantumBackend, **kwargs: Any
) -> QuantumReservoir:
    """Create a quantum reservoir instance from an architecture name.

    Args:
        arch: Architecture name (e.g., "standard").
        backend: Quantum backend implementation to use.
        **kwargs: Additional configuration parameters passed to the
            reservoir constructor (e.g., n_qubits, n_layers, seed).

    Returns:
        Instantiated quantum reservoir of the specified architecture.

    Raises:
        ValueError: If the architecture name is not recognized.

    Example:
        >>> from qrc_ev.backends.pennylane_backend import PennyLaneBackend
        >>> backend = PennyLaneBackend()
        >>> reservoir = create_reservoir(
        ...     arch="standard",
        ...     backend=backend,
        ...     n_qubits=4,
        ...     n_layers=2,
        ...     seed=42
        ... )
    """
    if arch not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(
            f"Unknown architecture '{arch}'. Available: {available}"
        )

    reservoir_class = _REGISTRY[arch]
    return reservoir_class(backend=backend, **kwargs)
