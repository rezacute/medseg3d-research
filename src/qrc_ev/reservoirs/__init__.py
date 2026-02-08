"""Quantum reservoir implementations.

This module provides various quantum reservoir computing architectures.
"""

from qrc_ev.reservoirs.factory import create_reservoir
from qrc_ev.reservoirs.standard import StandardReservoir

__all__ = ["StandardReservoir", "create_reservoir"]
