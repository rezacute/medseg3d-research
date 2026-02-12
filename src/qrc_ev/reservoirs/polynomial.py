"""A4 Polynomial-Enhanced Quantum Reservoir Computing.

Extends standard QRC by adding polynomial feature expansion to the
quantum observables. This enables capturing nonlinear relationships
in the data that linear readout would miss.

Based on literature showing polynomial features significantly improve
QRC performance on time-series tasks.

Key features:
- Standard quantum reservoir for base features
- Polynomial expansion (degree 2 and 3) of observables
- Efficient computation avoiding redundant terms
- Combines quantum nonlinearity with classical polynomial features
"""

from typing import TYPE_CHECKING, Optional
from itertools import combinations_with_replacement

import numpy as np

from qrc_ev.backends.base import QuantumBackend, QuantumReservoir, ReservoirParams

if TYPE_CHECKING:
    pass


class PolynomialReservoir(QuantumReservoir):
    """A4 Polynomial-Enhanced QRC.

    Processes time-series data through a quantum reservoir, then expands
    the measured observables with polynomial features. This creates a
    richer feature space for the classical readout.

    Attributes:
        backend: Quantum backend for circuit execution.
        n_qubits: Number of qubits in the reservoir.
        n_layers: Number of layers in the reservoir unitary.
        poly_degree: Maximum polynomial degree (2 or 3).
        include_bias: Whether to include bias term (constant 1).
        params: Fixed random reservoir parameters.
    """

    def __init__(
        self,
        backend: QuantumBackend,
        n_qubits: int,
        n_layers: int = 4,
        poly_degree: int = 2,
        include_bias: bool = True,
        evolution_steps: int = 1,
        seed: int = 42,
    ):
        """Initialize the polynomial-enhanced reservoir.

        Args:
            backend: Quantum backend implementation.
            n_qubits: Number of qubits in the reservoir.
            n_layers: Number of layers in the Ising unitary. Default: 4.
            poly_degree: Maximum polynomial degree (2 or 3). Default: 2.
            include_bias: Include constant bias term. Default: True.
            evolution_steps: Number of reservoir evolution steps. Default: 1.
            seed: Random seed for generating fixed Ising parameters. Default: 42.
        """
        self.backend = backend
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.poly_degree = min(poly_degree, 3)  # Cap at 3 for tractability
        self.include_bias = include_bias
        self.evolution_steps = evolution_steps
        self.params = self._generate_fixed_params(seed)
        
        # Compute output feature dimension
        self._n_poly_features = self._count_poly_features()
        
        # Initialize the backend
        self.backend.create_circuit(n_qubits)

    def _generate_fixed_params(self, seed: int) -> ReservoirParams:
        """Generate fixed random Ising coupling strengths and rotation angles."""
        rng = np.random.default_rng(seed)
        coupling_strengths = rng.uniform(
            -np.pi, np.pi, (self.n_layers, self.n_qubits, self.n_qubits)
        )
        rotation_angles = rng.uniform(
            -np.pi, np.pi, (self.n_layers, self.n_qubits)
        )
        return ReservoirParams(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            coupling_strengths=coupling_strengths,
            rotation_angles=rotation_angles,
            seed=seed,
        )

    def _count_poly_features(self) -> int:
        """Count total number of polynomial features."""
        n = self.n_qubits
        count = 0
        
        # Bias term
        if self.include_bias:
            count += 1
        
        # Degree 1 (original features)
        count += n
        
        # Degree 2: n*(n+1)/2 unique terms (with replacement)
        if self.poly_degree >= 2:
            count += n * (n + 1) // 2
        
        # Degree 3: n*(n+1)*(n+2)/6 unique terms
        if self.poly_degree >= 3:
            count += n * (n + 1) * (n + 2) // 6
        
        return count

    def _polynomial_expand(self, features: np.ndarray) -> np.ndarray:
        """Expand features with polynomial terms.

        Args:
            features: Input array of shape (T, n_qubits) with values in [-1, 1].

        Returns:
            Expanded array of shape (T, n_poly_features).
        """
        T, n = features.shape
        expanded = []
        
        # Bias term
        if self.include_bias:
            expanded.append(np.ones((T, 1)))
        
        # Degree 1 (original)
        expanded.append(features)
        
        # Degree 2: all unique pairs including self-products
        if self.poly_degree >= 2:
            degree2 = []
            for i, j in combinations_with_replacement(range(n), 2):
                degree2.append(features[:, i] * features[:, j])
            expanded.append(np.column_stack(degree2))
        
        # Degree 3: all unique triplets including self-products
        if self.poly_degree >= 3:
            degree3 = []
            for i, j, k in combinations_with_replacement(range(n), 3):
                degree3.append(features[:, i] * features[:, j] * features[:, k])
            expanded.append(np.column_stack(degree3))
        
        return np.hstack(expanded)

    def _measure_single_timestep(self, data: np.ndarray) -> np.ndarray:
        """Process a single timestep through the quantum circuit.

        Args:
            data: Input feature vector of shape (d,) with values in [0, 1].

        Returns:
            NumPy array of Pauli-Z expectation values.
        """
        # Pad data if needed
        if len(data) < self.n_qubits:
            padded = np.zeros(self.n_qubits)
            padded[:len(data)] = data
            data = padded
        elif len(data) > self.n_qubits:
            data = data[:self.n_qubits]
        
        # Check if backend has native _device (PennyLane) or not (CUDA-Q)
        if hasattr(self.backend, '_device') and self.backend._device is not None:
            # PennyLane path
            import pennylane as qml
            from qrc_ev.encoding.angle import angle_encode
            from qrc_ev.readout.observables import pauli_z_observables
            
            dev = self.backend._device
            
            @qml.qnode(dev, interface="numpy")
            def circuit() -> list:
                angle_encode(data, self.n_qubits)
                for _ in range(self.evolution_steps):
                    self.backend.apply_reservoir(None, self.params)
                return pauli_z_observables(self.n_qubits)
            
            result = circuit()
            return np.array(result)
        else:
            # CUDA-Q / backend-agnostic path
            self.backend.apply_encoding(None, data, strategy="angle")
            for _ in range(self.evolution_steps):
                self.backend.apply_reservoir(None, self.params)
            result = self.backend.measure_observables(None, obs_set="pauli_z")
            return np.array(result)

    def encode(self, x: np.ndarray) -> None:
        """Encode input data (kept for interface compatibility)."""
        pass

    def evolve(self, steps: int) -> None:
        """Evolve reservoir (kept for interface compatibility)."""
        pass

    def measure(self) -> np.ndarray:
        """Measure observables (kept for interface compatibility)."""
        return np.array([])

    def reset(self) -> None:
        """Reset the reservoir state."""
        pass

    def process(self, time_series: np.ndarray) -> np.ndarray:
        """Process a time-series through the polynomial-enhanced reservoir.

        Each timestep is processed through the quantum circuit, then
        polynomial feature expansion is applied to the observables.

        Args:
            time_series: Input array of shape (T, d) where T is the number
                of timesteps and d is the feature dimension.

        Returns:
            Feature array of shape (T, n_poly_features) containing
            polynomial-expanded observable values.
        """
        T = time_series.shape[0]
        
        # Step 1: Get raw quantum features
        raw_features = np.zeros((T, self.n_qubits))
        for t in range(T):
            raw_features[t] = self._measure_single_timestep(time_series[t])
        
        # Step 2: Apply polynomial expansion
        expanded_features = self._polynomial_expand(raw_features)
        
        return expanded_features

    @property
    def n_features(self) -> int:
        """Return the number of output features."""
        return self._n_poly_features
