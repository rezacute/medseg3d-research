"""Tests for the architecture factory.

This module tests the factory function that creates reservoir instances
from architecture names and configuration parameters.
"""

import pytest
from hypothesis import given, settings, strategies as st

from qrc_ev.backends.pennylane_backend import PennyLaneBackend
from qrc_ev.reservoirs.factory import create_reservoir
from qrc_ev.reservoirs.standard import StandardReservoir


class TestFactoryBasic:
    """Basic unit tests for the architecture factory."""

    def test_create_standard_reservoir(self, small_qubit_count: int, test_seed: int):
        """Test that factory creates StandardReservoir for 'standard' architecture."""
        backend = PennyLaneBackend()
        reservoir = create_reservoir(
            arch="standard",
            backend=backend,
            n_qubits=small_qubit_count,
            n_layers=2,
            evolution_steps=1,
            seed=test_seed,
        )

        assert isinstance(reservoir, StandardReservoir)
        assert reservoir.n_qubits == small_qubit_count
        assert reservoir.n_layers == 2
        assert reservoir.evolution_steps == 1

    def test_factory_passes_kwargs_to_constructor(
        self, small_qubit_count: int, test_seed: int
    ):
        """Test that factory passes all kwargs to the reservoir constructor."""
        backend = PennyLaneBackend()
        n_layers = 3
        evolution_steps = 2

        reservoir = create_reservoir(
            arch="standard",
            backend=backend,
            n_qubits=small_qubit_count,
            n_layers=n_layers,
            evolution_steps=evolution_steps,
            seed=test_seed,
        )

        assert reservoir.n_layers == n_layers
        assert reservoir.evolution_steps == evolution_steps

    def test_factory_rejects_unknown_architecture(self):
        """Test that factory raises ValueError for unknown architecture names."""
        backend = PennyLaneBackend()

        with pytest.raises(ValueError) as exc_info:
            create_reservoir(
                arch="nonexistent_architecture",
                backend=backend,
                n_qubits=4,
            )

        error_msg = str(exc_info.value)
        assert "Unknown architecture 'nonexistent_architecture'" in error_msg
        assert "Available:" in error_msg
        assert "standard" in error_msg


class TestFactoryProperties:
    """Property-based tests for the architecture factory."""

    # Feature: phase1-foundation-setup, Property 23: Factory rejects unknown architectures
    @given(
        arch_name=st.text(
            alphabet=st.characters(
                blacklist_categories=("Cs",), blacklist_characters="\x00"
            ),
            min_size=1,
            max_size=50,
        ).filter(lambda x: x not in [
            "standard", "a1",
            "recurrence_free", "a2", "rf_qrc",
            "polynomial", "a4", "poly_qrc",
            "noise_aware", "a6", "noisy_qrc",
        ])
    )
    @settings(max_examples=100, deadline=None)
    def test_property_factory_rejects_unknown_architectures(self, arch_name: str):
        """Property 23: Factory rejects unknown architectures.

        For any string that is not in the architecture registry,
        calling create_reservoir() should raise a ValueError listing
        available architecture names.

        Validates: Requirements 10.2
        """
        backend = PennyLaneBackend()

        with pytest.raises(ValueError) as exc_info:
            create_reservoir(
                arch=arch_name,
                backend=backend,
                n_qubits=4,
            )

        error_msg = str(exc_info.value)
        # Verify error message contains the unknown architecture name
        assert arch_name in error_msg or "Unknown architecture" in error_msg
        # Verify error message lists available architectures
        assert "Available:" in error_msg or "available" in error_msg.lower()
