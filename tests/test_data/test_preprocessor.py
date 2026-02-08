"""Property-based and unit tests for the Preprocessor class.

Tests verify data aggregation, cleaning, normalization, splitting,
and windowing logic against requirements 12.1-12.8.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import column, data_frames

from qrc_ev.data.preprocessor import Preprocessor
from qrc_ev.utils.config import DataConfig


@pytest.fixture
def default_config():
    """Default data configuration fixture."""
    return DataConfig()


@pytest.fixture
def preprocessor(default_config):
    """Preprocessor instance with default config."""
    return Preprocessor(default_config)


class TestPreprocessorUnit:
    """Unit tests for Preprocessor methods."""

    def test_aggregate_sessions_hourly(self, preprocessor):
        """Test aggregation of sessions into hourly bins (Req 12.1)."""
        data = {
            "connectionTime": [
                "2023-01-01 08:05:00",
                "2023-01-01 08:30:00",
                "2023-01-01 09:15:00",
            ],
            "energyDone": [5.0, 10.0, 7.0],
        }
        df = pd.DataFrame(data)
        
        # Hourly aggregation
        agg = preprocessor.aggregate_sessions(df, resolution="1h")
        
        assert len(agg) == 2
        assert agg.iloc[0] == 15.0  # 5 + 10
        assert agg.iloc[1] == 7.0
        assert agg.index[0] == pd.Timestamp("2023-01-01 08:00:00")
        assert agg.index[1] == pd.Timestamp("2023-01-01 09:00:00")

    def test_handle_missing_forward_fill(self, preprocessor):
        """Test forward-fill interpolation (Req 12.2)."""
        series = pd.Series([1.0, np.nan, 3.0, np.nan, np.nan, 6.0])
        filled = preprocessor.handle_missing(series, max_gap=4)
        
        expected = pd.Series([1.0, 1.0, 3.0, 3.0, 3.0, 6.0])
        pd.testing.assert_series_equal(filled, expected)

    def test_handle_missing_gap_detection(self, preprocessor, caplog):
        """Test logging of large gaps (Req 12.2)."""
        series = pd.Series([1.0, np.nan, np.nan, np.nan, 4.0], index=pd.date_range("2023-01-01", periods=5, freq="h"))
        
        with caplog.at_level("WARNING"):
            preprocessor.handle_missing(series, max_gap=2)
            assert "Gap exceeding threshold 2 detected" in caplog.text

    def test_normalize_raises_before_fit(self, preprocessor):
        """Test that normalize raises RuntimeError if not fitted."""
        with pytest.raises(RuntimeError, match="must be called before normalize"):
            preprocessor.normalize(np.array([1.0, 2.0]))

    def test_normalize_clips_out_of_range(self, preprocessor):
        """Test that normalization clips values outside training range (Req 12.8)."""
        train_data = np.array([[10.0], [20.0]])
        preprocessor.fit_normalize(train_data)
        
        test_data = np.array([[5.0], [25.0]])
        normalized = preprocessor.normalize(test_data)
        
        # 5.0 -> 0.0 (clipped from -0.5), 25.0 -> 1.0 (clipped from 1.5)
        expected = np.array([[0.0], [1.0]])
        np.testing.assert_array_equal(normalized, expected)


class TestPreprocessorProperties:
    """Property-based tests for Preprocessor invariants."""

    # Feature: phase1-foundation-setup, Property 14: Outlier clipping invariant
    @given(
        series=st.lists(
            st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
            min_size=5,
        )
    )
    @settings(max_examples=100)
    def test_clip_outliers_invariant(self, series):
        """Verify clipped values are within ±3σ (Req 12.3)."""
        preprocessor = Preprocessor(DataConfig())
        s = pd.Series(series)
        mean = s.mean()
        std = s.std()

        if np.isnan(std) or std == 0:
            return  # Skip cases where clipping doesn't apply uniquely

        clipped = preprocessor.clip_outliers(s, n_sigma=3.0)

        lower_bound = mean - 3.0 * std - 1e-9
        upper_bound = mean + 3.0 * std + 1e-9

        assert (clipped >= lower_bound).all()
        assert (clipped <= upper_bound).all()

    # Feature: phase1-foundation-setup, Property 15: Chronological split preserves order and ratios
    @given(
        data_len=st.integers(min_value=10, max_value=1000),
        ratios=st.tuples(
            st.floats(min_value=0.1, max_value=0.8),
            st.floats(min_value=0.1, max_value=0.8),
        ).filter(lambda x: x[0] + x[1] < 0.95),
    )
    @settings(max_examples=100)
    def test_chronological_split_preserves_order(self, data_len, ratios):
        """Verify chronological split segments (Req 12.4)."""
        train_r, val_r = ratios
        test_r = 1.0 - train_r - val_r

        config = DataConfig(
            train_ratio=train_r, val_ratio=val_r, test_ratio=test_r
        )
        preprocessor = Preprocessor(config)

        data = np.arange(data_len)
        train, val, test = preprocessor.split_chronological(data)

        # Verify sizes (within 1 due to floor/int conversion)
        assert len(train) + len(val) + len(test) == data_len

        # Verify continuity and order
        if len(train) > 0 and len(val) > 0:
            assert train[-1] < val[0]
        if len(val) > 0 and len(test) > 0:
            assert val[-1] < test[0]

        # Verify no mixing
        all_data = np.concatenate([train, val, test])
        np.testing.assert_array_equal(all_data, data)

    # Feature: phase1-foundation-setup, Property 16: Normalization output range invariant
    @given(
        train_data=arrays(
            np.float64,
            shape=st.tuples(st.integers(5, 50), st.integers(1, 5)),
            elements=st.floats(-1000, 1000),
        ),
        test_data=arrays(
            np.float64,
            shape=st.tuples(st.integers(5, 50), st.integers(1, 5)),
            elements=st.floats(-2000, 2000),
        ),
    )
    @settings(max_examples=100)
    def test_normalization_output_range(self, train_data, test_data):
        """Verify normalized outputs are in [0, 1] (Req 12.5, 12.6, 12.8)."""
        preprocessor = Preprocessor(DataConfig())
        
        # Ensure test data has exactly the same number of features as train data
        n_features = train_data.shape[1]
        if test_data.shape[1] < n_features:
            # Pad with zeros if test_data has fewer features
            padding = np.zeros((test_data.shape[0], n_features - test_data.shape[1]))
            test_data = np.hstack([test_data, padding])
        else:
            # Slice if test_data has more features
            test_data = test_data[:, :n_features]

        preprocessor.fit_normalize(train_data)

        # Check train normalization
        norm_train = preprocessor.normalize(train_data)
        assert (norm_train >= -1e-12).all()
        assert (norm_train <= 1.000000000001).all()

        # Check test normalization (with clipping)
        norm_test = preprocessor.normalize(test_data)
        assert (norm_test >= 0.0).all()
        assert (norm_test <= 1.0).all()

    # Feature: phase1-foundation-setup, Property 17: Windowed sample shapes
    @given(
        T=st.integers(min_value=20, max_value=100),
        d=st.integers(min_value=1, max_value=5),
        W=st.integers(min_value=1, max_value=10),
        h=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    def test_windowed_sample_shapes(self, T, d, W, h):
        """Verify sliding window shapes (Req 12.7)."""
        preprocessor = Preprocessor(DataConfig())
        features = np.random.randn(T, d)
        targets = np.random.randn(T)

        # num_samples = T - W - h + 1
        X, y = preprocessor.create_windows(features, targets, window_size=W, horizon=h)

        expected_n = T - W - h + 1
        if expected_n > 0:
            assert X.shape == (expected_n, W, d)
            assert y.shape == (expected_n, h)
        else:
            assert X.size == 0
            assert y.size == 0
