"""Property-based tests for FeatureEngineer.

Validates Requirements 13.1, 13.2, 13.3, 13.4 using Hypothesis properties.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from qrc_ev.data.feature_engineer import FeatureEngineer


@st.composite
def datetime_index(draw):
    """Strategy for generating pandas DatetimeIndex."""
    length = draw(st.integers(min_value=1, max_value=100))
    start_date = draw(st.datetimes(
        min_value=pd.Timestamp("2020-01-01"),
        max_value=pd.Timestamp("2025-01-01")
    ))
    freq = draw(st.sampled_from(["h", "min", "D"]))
    return pd.date_range(start=start_date, periods=length, freq=freq)


# Feature: phase1-foundation-setup, Property 18: Temporal features bounded in [-1, 1]
@given(timestamps=datetime_index())
@settings(max_examples=100)
def test_temporal_features_bounded(timestamps):
    """Verify temporal sin/cos features are within [-1, 1].
    
    Validates: Requirement 13.1
    """
    fe = FeatureEngineer()
    features = fe.add_temporal_features(timestamps)
    
    # Check shape: (T, 4)
    assert features.shape == (len(timestamps), 4)
    
    # Check bounds: [-1, 1]
    # Use a small tolerance for floating point precision
    assert np.all(features >= -1.0 - 1e-9)
    assert np.all(features <= 1.0 + 1e-9)


# Feature: phase1-foundation-setup, Property 19: Lag feature correctness
@given(
    series=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=10, max_value=100),
        elements=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)
    ),
    lag_step=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=100)
def test_lag_feature_correctness(series, lag_step):
    """Verify lag features correctly shift values.
    
    Validates: Requirement 13.2
    """
    fe = FeatureEngineer(lag_steps=[lag_step])
    lags = fe.add_lag_features(series)
    
    # Check shape: (T, 1)
    assert lags.shape == (len(series), 1)
    
    lag_values = lags[:, 0]
    
    # Values before lag_step should be 0.0
    assert np.all(lag_values[:lag_step] == 0.0)
    
    # Values after lag_step should match original series shifted
    original_shifted = series[:-lag_step]
    assert np.all(lag_values[lag_step:] == original_shifted)


# Feature: phase1-foundation-setup, Property 20: Feature dimension consistency
@given(
    series_len=st.integers(min_value=10, max_value=50),
    lag_steps=st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=5, unique=True)
)
@settings(max_examples=100)
def test_feature_dimension_consistency(series_len, lag_steps):
    """Verify feature_dim matches output shape of engineer().
    
    Validates: Requirements 13.3, 13.4
    """
    fe = FeatureEngineer(lag_steps=lag_steps)
    
    # Generate dummy data
    series = np.zeros(series_len)
    timestamps = pd.date_range("2024-01-01", periods=series_len, freq="h")
    
    features = fe.engineer(series, timestamps)
    
    expected_dim = 4 + len(lag_steps)
    assert fe.feature_dim == expected_dim
    assert features.shape == (series_len, expected_dim)


def test_feature_engineer_default_lags():
    """Verify default lag steps are applied correctly."""
    fe = FeatureEngineer()
    assert fe.lag_steps == [1, 2, 4, 12, 24]
    assert fe.feature_dim == 9
