"""Tests for A2 Recurrence-Free QRC implementation."""

import numpy as np
import pytest

from qrc_ev.backends.base import ReservoirParams
from qrc_ev.reservoirs.recurrence_free import RecurrenceFreeReservoir


# Try to import CUDA-Q backend
try:
    from qrc_ev.backends.cudaq_backend import CUDAQuantumBackend, is_cudaq_available
    CUDAQ_AVAILABLE = is_cudaq_available()
except ImportError:
    CUDAQ_AVAILABLE = False


# Try to import PennyLane backend
try:
    from qrc_ev.backends.pennylane_backend import PennyLaneBackend
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


@pytest.fixture
def pennylane_backend():
    """Create a PennyLane backend for testing."""
    if not PENNYLANE_AVAILABLE:
        pytest.skip("PennyLane not available")
    return PennyLaneBackend(device_name="default.qubit", shots=None)


@pytest.fixture
def cudaq_backend():
    """Create a CUDA-Q backend for testing."""
    if not CUDAQ_AVAILABLE:
        pytest.skip("CUDA-Q not available")
    return CUDAQuantumBackend(target="nvidia", shots=None)


class TestRecurrenceFreeReservoir:
    """Test suite for RecurrenceFreeReservoir."""

    def test_initialization(self, pennylane_backend):
        """Test reservoir initialization."""
        reservoir = RecurrenceFreeReservoir(
            backend=pennylane_backend,
            n_qubits=4,
            n_layers=2,
            leak_rate=0.3,
            seed=42
        )
        
        assert reservoir.n_qubits == 4
        assert reservoir.n_layers == 2
        assert reservoir.leak_rate == 0.3
        assert reservoir.params is not None

    def test_process_single_sample(self, pennylane_backend):
        """Test processing a single timestep."""
        reservoir = RecurrenceFreeReservoir(
            backend=pennylane_backend,
            n_qubits=4,
            n_layers=2,
            seed=42
        )
        
        # Single timestep
        data = np.array([[0.1, 0.2, 0.3, 0.4]])
        result = reservoir.process(data)
        
        assert result.shape == (1, 4)
        assert all(-1 <= v <= 1 for v in result[0])

    def test_process_time_series(self, pennylane_backend):
        """Test processing a time series."""
        reservoir = RecurrenceFreeReservoir(
            backend=pennylane_backend,
            n_qubits=4,
            n_layers=2,
            leak_rate=0.3,
            seed=42
        )
        
        # Time series: 10 timesteps, 4 features
        data = np.random.rand(10, 4)
        result = reservoir.process(data)
        
        assert result.shape == (10, 4)
        # All values should be in [-1, 1] range
        assert np.all(result >= -1) and np.all(result <= 1)

    def test_leaky_integration(self, pennylane_backend):
        """Test that leaky integration creates temporal memory."""
        reservoir = RecurrenceFreeReservoir(
            backend=pennylane_backend,
            n_qubits=4,
            n_layers=2,
            leak_rate=0.5,  # High leak rate for visible effect
            seed=42
        )
        
        # Constant input should converge to stable output
        data = np.ones((20, 4)) * 0.5
        result = reservoir.process(data)
        
        # Later timesteps should be more similar (leaky integration smooths)
        diff_early = np.abs(result[1] - result[0]).mean()
        diff_late = np.abs(result[-1] - result[-2]).mean()
        
        # Late differences should be smaller (converged)
        assert diff_late <= diff_early + 0.1  # Allow some tolerance

    def test_svd_denoising(self, pennylane_backend):
        """Test SVD-based denoising."""
        reservoir = RecurrenceFreeReservoir(
            backend=pennylane_backend,
            n_qubits=4,
            n_layers=2,
            svd_rank=2,  # Keep only 2 singular values
            seed=42
        )
        
        data = np.random.rand(20, 4)
        result = reservoir.process(data)
        
        # Result should have reduced effective rank
        assert result.shape == (20, 4)
        
        # Verify SVD was applied by checking rank
        U, S, Vt = np.linalg.svd(result, full_matrices=False)
        # Most energy should be in first 2 singular values
        energy_ratio = S[:2].sum() / S.sum()
        assert energy_ratio > 0.9

    def test_reproducibility(self, pennylane_backend):
        """Test that same seed produces same results."""
        data = np.random.rand(10, 4)
        
        reservoir1 = RecurrenceFreeReservoir(
            backend=pennylane_backend,
            n_qubits=4,
            n_layers=2,
            seed=42
        )
        result1 = reservoir1.process(data)
        
        reservoir2 = RecurrenceFreeReservoir(
            backend=pennylane_backend,
            n_qubits=4,
            n_layers=2,
            seed=42
        )
        result2 = reservoir2.process(data)
        
        np.testing.assert_array_almost_equal(result1, result2)

    def test_reset_clears_state(self, pennylane_backend):
        """Test that reset clears internal state."""
        reservoir = RecurrenceFreeReservoir(
            backend=pennylane_backend,
            n_qubits=4,
            n_layers=2,
            seed=42
        )
        
        # Process some data
        data = np.random.rand(10, 4)
        reservoir.process(data)
        
        # Reset
        reservoir.reset()
        
        # Internal state should be cleared
        assert reservoir._leaky_state is None


@pytest.mark.skipif(not CUDAQ_AVAILABLE, reason="CUDA-Q not available")
class TestRecurrenceFreeReservoirCUDAQ:
    """Test suite for RecurrenceFreeReservoir with CUDA-Q backend."""

    def test_cudaq_initialization(self, cudaq_backend):
        """Test reservoir initialization with CUDA-Q."""
        reservoir = RecurrenceFreeReservoir(
            backend=cudaq_backend,
            n_qubits=6,
            n_layers=2,
            leak_rate=0.3,
            seed=42
        )
        
        assert reservoir.n_qubits == 6
        assert reservoir.n_layers == 2

    def test_cudaq_process_time_series(self, cudaq_backend):
        """Test processing time series with CUDA-Q."""
        reservoir = RecurrenceFreeReservoir(
            backend=cudaq_backend,
            n_qubits=6,
            n_layers=2,
            leak_rate=0.3,
            seed=42
        )
        
        data = np.random.rand(20, 6)
        result = reservoir.process(data)
        
        assert result.shape == (20, 6)
        assert np.all(result >= -1) and np.all(result <= 1)

    def test_cudaq_reproducibility(self, cudaq_backend):
        """Test reproducibility with CUDA-Q."""
        data = np.random.rand(10, 6)
        
        reservoir1 = RecurrenceFreeReservoir(
            backend=cudaq_backend,
            n_qubits=6,
            n_layers=2,
            seed=42
        )
        result1 = reservoir1.process(data)
        
        # Create new backend instance for second reservoir
        backend2 = CUDAQuantumBackend(target="nvidia", shots=None)
        reservoir2 = RecurrenceFreeReservoir(
            backend=backend2,
            n_qubits=6,
            n_layers=2,
            seed=42
        )
        result2 = reservoir2.process(data)
        
        np.testing.assert_array_almost_equal(result1, result2, decimal=5)


class TestRecurrenceFreeFactory:
    """Test factory creation of RF-QRC."""

    def test_factory_creation(self, pennylane_backend):
        """Test creating RF-QRC via factory."""
        from qrc_ev.reservoirs.factory import create_reservoir
        
        reservoir = create_reservoir(
            arch="recurrence_free",
            backend=pennylane_backend,
            n_qubits=4,
            n_layers=2,
            leak_rate=0.3,
            seed=42
        )
        
        assert isinstance(reservoir, RecurrenceFreeReservoir)
        assert reservoir.n_qubits == 4

    def test_factory_aliases(self, pennylane_backend):
        """Test factory aliases for RF-QRC."""
        from qrc_ev.reservoirs.factory import create_reservoir
        
        # Test rf_qrc alias
        r1 = create_reservoir(
            arch="rf_qrc",
            backend=pennylane_backend,
            n_qubits=4,
            n_layers=2,
            seed=42
        )
        assert isinstance(r1, RecurrenceFreeReservoir)
        
        # Test a2 alias
        r2 = create_reservoir(
            arch="a2",
            backend=pennylane_backend,
            n_qubits=4,
            n_layers=2,
            seed=42
        )
        assert isinstance(r2, RecurrenceFreeReservoir)
