import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from qrc_ev.data.synthetic import SyntheticGenerator


# Feature: phase1-foundation-setup, Property 21: Synthetic data shape and format
@given(
    length=st.integers(min_value=10, max_value=1000),
    n_features=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=50, deadline=None)
def test_sinusoidal_shape(length, n_features):
    """Validate shape and format for sinusoidal data.
    
    Requirement 14.1, 14.4.
    """
    gen = SyntheticGenerator()
    features, targets = gen.sinusoidal(length=length, n_features=n_features)
    
    assert features.shape == (length, n_features)
    assert targets.shape == (length,)
    assert isinstance(features, np.ndarray)
    assert isinstance(targets, np.ndarray)


@given(
    length=st.integers(min_value=24, max_value=1000),
    n_features=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=50, deadline=None)
def test_ev_charging_pattern_shape(length, n_features):
    """Validate shape and format for EV charging pattern data.
    
    Requirement 14.2, 14.4.
    """
    gen = SyntheticGenerator()
    features, targets = gen.ev_charging_pattern(length=length, n_features=n_features)
    
    assert features.shape == (length, n_features)
    assert targets.shape == (length,)
    assert isinstance(features, np.ndarray)
    assert isinstance(targets, np.ndarray)
    # Target values should be non-negative
    assert np.all(targets >= 0)


# Feature: phase1-foundation-setup, Property 22: Synthetic data reproducibility
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    length=st.integers(min_value=10, max_value=100),
)
@settings(max_examples=30, deadline=None)
def test_reproducibility_sinusoidal(seed, length):
    """Validate that same seed produces identical sinusoidal data.
    
    Requirement 14.3.
    """
    gen1 = SyntheticGenerator(seed=seed)
    f1, t1 = gen1.sinusoidal(length=length)
    
    gen2 = SyntheticGenerator(seed=seed)
    f2, t2 = gen2.sinusoidal(length=length)
    
    np.testing.assert_allclose(f1, f2)
    np.testing.assert_allclose(t1, t2)


@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    length=st.integers(min_value=24, max_value=100),
)
@settings(max_examples=30, deadline=None)
def test_reproducibility_ev_charging(seed, length):
    """Validate that same seed produces identical EV charging data.
    
    Requirement 14.3.
    """
    gen1 = SyntheticGenerator(seed=seed)
    f1, t1 = gen1.ev_charging_pattern(length=length)
    
    gen2 = SyntheticGenerator(seed=seed)
    f2, t2 = gen2.ev_charging_pattern(length=length)
    
    np.testing.assert_allclose(f1, f2)
    np.testing.assert_allclose(t1, t2)
