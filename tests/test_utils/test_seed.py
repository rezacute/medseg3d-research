"""Tests for seed management."""

import random
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from qrc_ev.utils.seed import SeedManager
from qrc_ev.backends.pennylane_backend import PennyLaneBackend
from qrc_ev.reservoirs.standard import StandardReservoir


class TestSeedManagerInit:
    """Tests for SeedManager initialization."""

    def test_explicit_seed(self):
        """SeedManager accepts an explicit seed."""
        sm = SeedManager(42)
        assert sm.global_seed == 42

    def test_auto_generate_seed(self):
        """SeedManager auto-generates seed when None is provided."""
        sm = SeedManager(None)
        assert isinstance(sm.global_seed, int)
        assert 0 <= sm.global_seed < 2**31

    def test_auto_generated_seeds_differ(self):
        """Multiple auto-generated seeds are different."""
        sm1 = SeedManager(None)
        sm2 = SeedManager(None)
        # Very unlikely to be equal (1 in 2^31 chance)
        assert sm1.global_seed != sm2.global_seed


class TestSeedAll:
    """Tests for seed_all() method."""

    def test_seeds_python_random(self):
        """seed_all() seeds Python's random module."""
        sm = SeedManager(42)
        sm.seed_all()
        val1 = random.random()
        
        sm2 = SeedManager(42)
        sm2.seed_all()
        val2 = random.random()
        
        assert val1 == val2

    def test_seeds_numpy(self):
        """seed_all() seeds NumPy's random generator."""
        sm = SeedManager(42)
        sm.seed_all()
        val1 = np.random.random()
        
        sm2 = SeedManager(42)
        sm2.seed_all()
        val2 = np.random.random()
        
        assert val1 == val2


class TestDeriveSeed:
    """Tests for derive_seed() method."""

    def test_derive_seed_deterministic(self):
        """derive_seed() returns same value for same component."""
        sm = SeedManager(42)
        seed1 = sm.derive_seed("reservoir")
        seed2 = sm.derive_seed("reservoir")
        assert seed1 == seed2

    def test_derive_seed_range(self):
        """Derived seeds are in valid range [0, 2^31)."""
        sm = SeedManager(42)
        seed = sm.derive_seed("test_component")
        assert isinstance(seed, int)
        assert 0 <= seed < 2**31

    def test_different_components_different_seeds(self):
        """Different component names yield different seeds."""
        sm = SeedManager(42)
        seed1 = sm.derive_seed("reservoir")
        seed2 = sm.derive_seed("data_split")
        assert seed1 != seed2

    def test_different_global_seeds_different_derived(self):
        """Different global seeds yield different derived seeds."""
        sm1 = SeedManager(42)
        sm2 = SeedManager(43)
        seed1 = sm1.derive_seed("reservoir")
        seed2 = sm2.derive_seed("reservoir")
        assert seed1 != seed2


# Feature: phase1-foundation-setup, Property 12: Seed reproducibility — reservoir outputs
@given(
    seed=st.integers(min_value=0, max_value=2**30),
    n_qubits=st.integers(min_value=2, max_value=6),
    n_layers=st.integers(min_value=1, max_value=3),
    n_timesteps=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=100, deadline=None)
def test_property_seed_reproducibility_reservoir(seed, n_qubits, n_layers, n_timesteps):
    """Property 12: Seed reproducibility — reservoir outputs.
    
    For any seed, backend configuration, and input data, running the A1 reservoir
    twice with the same seed should produce identical feature arrays (element-wise equality).
    
    Validates: Requirements 9.2, 15.2
    """
    # Create input data with dimension <= n_qubits
    d = min(n_qubits, 4)  # Keep feature dimension reasonable
    rng = np.random.default_rng(seed)
    data = rng.random((n_timesteps, d))
    
    # Create two reservoirs with the same seed
    backend1 = PennyLaneBackend(device_name="default.qubit")
    reservoir1 = StandardReservoir(
        backend=backend1,
        n_qubits=n_qubits,
        n_layers=n_layers,
        evolution_steps=1,
        seed=seed,
    )
    
    backend2 = PennyLaneBackend(device_name="default.qubit")
    reservoir2 = StandardReservoir(
        backend=backend2,
        n_qubits=n_qubits,
        n_layers=n_layers,
        evolution_steps=1,
        seed=seed,
    )
    
    # Process the same data through both reservoirs
    output1 = reservoir1.process(data)
    output2 = reservoir2.process(data)
    
    # Outputs should be identical
    np.testing.assert_array_equal(output1, output2)


# Feature: phase1-foundation-setup, Property 13: Seed derivation produces distinct child seeds
@given(
    global_seed=st.integers(min_value=0, max_value=2**30),
    component1=st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
    component2=st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
)
@settings(max_examples=100)
def test_property_seed_derivation_distinctness(global_seed, component1, component2):
    """Property 13: Seed derivation produces distinct child seeds.
    
    For any global seed and two different component name strings, the Seed_Manager's
    derive_seed() should return different child seed values.
    
    Validates: Requirements 9.3
    """
    # Skip if component names are the same
    if component1 == component2:
        return
    
    sm = SeedManager(global_seed)
    seed1 = sm.derive_seed(component1)
    seed2 = sm.derive_seed(component2)
    
    # Different components should yield different seeds
    assert seed1 != seed2
    
    # Both seeds should be in valid range
    assert 0 <= seed1 < 2**31
    assert 0 <= seed2 < 2**31
