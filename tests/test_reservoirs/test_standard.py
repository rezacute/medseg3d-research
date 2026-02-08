"""Tests for the A1 Standard Gate-Based QRC reservoir."""

import numpy as np
import pytest

from qrc_ev.backends.pennylane_backend import PennyLaneBackend
from qrc_ev.reservoirs.standard import StandardReservoir


@pytest.fixture
def backend():
    """Create a PennyLane backend for testing."""
    return PennyLaneBackend(device_name="default.qubit")


@pytest.fixture
def reservoir(backend):
    """Create a standard reservoir with 4 qubits for testing."""
    return StandardReservoir(
        backend=backend, n_qubits=4, n_layers=2, evolution_steps=1, seed=42
    )


class TestStandardReservoirInit:
    """Tests for reservoir initialization."""

    def test_fixed_params_generated_from_seed(self, backend):
        """Params are deterministic given the same seed."""
        r1 = StandardReservoir(backend=backend, n_qubits=3, n_layers=2, seed=99)
        r2 = StandardReservoir(backend=backend, n_qubits=3, n_layers=2, seed=99)
        np.testing.assert_array_equal(
            r1.params.coupling_strengths, r2.params.coupling_strengths
        )
        np.testing.assert_array_equal(
            r1.params.rotation_angles, r2.params.rotation_angles
        )

    def test_different_seeds_produce_different_params(self, backend):
        """Different seeds yield different reservoir parameters."""
        r1 = StandardReservoir(backend=backend, n_qubits=3, n_layers=2, seed=1)
        r2 = StandardReservoir(backend=backend, n_qubits=3, n_layers=2, seed=2)
        assert not np.array_equal(
            r1.params.coupling_strengths, r2.params.coupling_strengths
        )

    def test_params_shape(self, reservoir):
        """Coupling strengths and rotation angles have correct shapes."""
        assert reservoir.params.coupling_strengths.shape == (2, 4, 4)
        assert reservoir.params.rotation_angles.shape == (2, 4)


class TestStandardReservoirProcess:
    """Tests for the process() pipeline."""

    def test_process_output_shape(self, reservoir):
        """process() returns (T, n_qubits) for (T, d) input."""
        data = np.random.default_rng(0).random((5, 3))
        result = reservoir.process(data)
        assert result.shape == (5, 4)

    def test_process_values_bounded(self, reservoir):
        """All Pauli-Z expectations are in [-1, 1]."""
        data = np.random.default_rng(0).random((3, 4))
        result = reservoir.process(data)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)

    def test_process_reproducibility(self, backend):
        """Same seed + same input → identical output."""
        data = np.array([[0.1, 0.5], [0.3, 0.8]])
        r1 = StandardReservoir(backend=backend, n_qubits=3, n_layers=2, seed=7)
        r2 = StandardReservoir(backend=backend, n_qubits=3, n_layers=2, seed=7)
        np.testing.assert_array_equal(r1.process(data), r2.process(data))

    def test_process_single_timestep(self, reservoir):
        """process() works with a single timestep."""
        data = np.array([[0.2, 0.4, 0.6]])
        result = reservoir.process(data)
        assert result.shape == (1, 4)


class TestStandardReservoirEncodeMeasure:
    """Tests for encode/evolve/measure individual methods."""

    def test_encode_rejects_oversized_input(self, reservoir):
        """encode() raises ValueError when d > n_qubits."""
        with pytest.raises(ValueError, match="exceeds qubit count"):
            reservoir.encode(np.ones(5))

    def test_reset_restores_initial_state(self, reservoir):
        """After reset, measuring the |0⟩ state returns ⟨Z⟩ = 1.0 for all qubits."""
        # Process some data first
        reservoir.encode(np.array([0.5, 0.5]))
        reservoir.reset()
        result = reservoir.measure()
        np.testing.assert_allclose(result, np.ones(4), atol=1e-7)

    def test_reset_preserves_params(self, reservoir):
        """reset() does not change the fixed random parameters."""
        params_before = reservoir.params.coupling_strengths.copy()
        reservoir.encode(np.array([0.3, 0.7]))
        reservoir.reset()
        np.testing.assert_array_equal(
            reservoir.params.coupling_strengths, params_before
        )

    def test_encode_evolve_measure_cycle(self, reservoir):
        """Manual encode→evolve→measure produces n_qubits values in [-1, 1]."""
        reservoir.encode(np.array([0.2, 0.8, 0.5]))
        reservoir.evolve(1)
        result = reservoir.measure()
        assert result.shape == (4,)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)
